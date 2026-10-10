//! Decomposition: splits a minimal DFA the model cannot convert into
//! independent pieces, converts each one (by the model or state elimination,
//! whichever is simpler) and combines their regexes.
//!
//! For a trimmed minimal DFA with start state `s`, these splits are exact:
//!
//! - **star**, when `s` has incoming transitions: `L = R* E`, with `R` the
//!   strings leading from `s` back to `s` without visiting it in between, and
//!   `E` the strings accepted without returning to `s`;
//! - **concatenation**, when `s` has none and a state `v` lies on every
//!   accepting path: `L = F L_v`, with `F` the strings leading from `s` to its
//!   first visit of `v` and `L_v` the language of `v`;
//! - **frontier**, over a region `B` closed under successors, without `s`,
//!   entered at a few states `x`: `L = L_A ∪ ⋃ F_x L_x`, with `L_A` what is
//!   accepted without entering `B` and `F_x` the strings entering it first at
//!   `x`;
//! - **union**, when `s` has none and removing it leaves several components:
//!   the union of the components' languages;
//! - **hubs**, which breaks up a large strongly connected component: for a
//!   few states `H` whose removal shatters it, the paths between `s`, the
//!   hubs and acceptance that visit no hub in between are pieces without the
//!   component's cycles, and the regex is state elimination over the small
//!   graph of `s`, `H` and acceptance, labelled by them.
//!
//! Each piece is a DFA, minimized on its own, so it has fewer states (or
//! smaller cycles) and usually far fewer bases than the whole.
//!
//! A conversion first splits the DFA into its tree of pieces, sharing equal
//! ones, then runs the model on every piece small enough, in batches, then
//! keeps for each piece, from the leaves up, the simplest of state
//! elimination, the model and the composition of its pieces.

use ahash::AHashMap;

use super::{NeuralSynthesizer, minimal_dfa};
use crate::{
    CharRange, IntSet,
    error::EngineError,
    execution_profile::ExecutionProfile,
    fast_automaton::{FastAutomaton, State},
    regex::RegularExpression,
};

mod analysis;
#[cfg(test)]
mod bench;
mod pieces;

/// Most pieces the model runs on per conversion.
const NEURAL_BUDGET: usize = 256;
/// Most pieces per conversion.
const MAX_PIECES: usize = 1024;
/// Pieces per model batch.
const BATCH: usize = 128;
/// Pieces whose state elimination regex is at most this complex are not
/// worth a model run.
const SKIP_COMPLEXITY: f64 = 4.0;

/// What a conversion did, for tracing.
#[derive(Clone, Debug, Default)]
pub(super) struct Stats {
    /// Distinct pieces, including the whole.
    pieces: usize,
    /// Splits by kind, over all the pieces.
    star_splits: usize,
    optional_splits: usize,
    concat_splits: usize,
    frontier_splits: usize,
    union_splits: usize,
    hub_splits: usize,
    /// Pieces the model ran on, and those it found a correct regex for.
    neural_runs: usize,
    neural_correct: usize,
    /// Pieces in the result whose regex is the model's.
    neural_kept: usize,
    /// Whether the result is a composition.
    composed: bool,
    /// The largest piece in the result not split further, in states.
    max_piece_states: usize,
}

/// How a piece splits; the `usize`s are pieces.
#[derive(Clone, Debug)]
enum Plan {
    Leaf,
    Star {
        returns: usize,
        exits: usize,
    },
    Optional {
        rest: usize,
    },
    Concat {
        prefix: usize,
        suffix: usize,
    },
    Frontier {
        stay: Option<usize>,
        crossings: Vec<(usize, usize)>,
    },
    Union {
        branches: Vec<usize>,
    },
    /// `paths[x][y]`: the paths from source `x` (the start, then the hubs)
    /// to target `y` (the hubs, then acceptance) visiting no hub in between;
    /// `None` when there are none.
    Hubs {
        hubs: usize,
        paths: Vec<Vec<Option<usize>>>,
    },
}

impl Plan {
    /// The pieces the plan combines.
    fn children(&self) -> Vec<usize> {
        match self {
            Plan::Leaf => vec![],
            Plan::Star { returns, exits } => vec![*returns, *exits],
            Plan::Optional { rest } => vec![*rest],
            Plan::Concat { prefix, suffix } => vec![*prefix, *suffix],
            Plan::Frontier { stay, crossings } => stay
                .iter()
                .copied()
                .chain(
                    crossings
                        .iter()
                        .flat_map(|(prefix, suffix)| [*prefix, *suffix]),
                )
                .collect(),
            Plan::Union { branches } => branches.clone(),
            Plan::Hubs { paths, .. } => paths.iter().flatten().flatten().copied().collect(),
        }
    }

    /// The regex the plan makes of its pieces' regexes, `None` for a leaf.
    fn compose(&self, regexes: &[RegularExpression]) -> Option<RegularExpression> {
        let regex = |piece: &usize| &regexes[*piece];
        Some(match self {
            Plan::Leaf => return None,
            Plan::Star { returns, exits } => {
                regex(returns).repeat(0, None).concat(regex(exits), true)
            }
            Plan::Optional { rest } => regex(rest).repeat(0, Some(1)),
            Plan::Concat { prefix, suffix } => regex(prefix).concat(regex(suffix), true),
            Plan::Frontier { stay, crossings } => {
                let branches: Vec<RegularExpression> = stay
                    .iter()
                    .map(|piece| regex(piece).clone())
                    .chain(
                        crossings
                            .iter()
                            .map(|(prefix, suffix)| regex(prefix).concat(regex(suffix), true)),
                    )
                    .collect();
                RegularExpression::union_all(&branches)
            }
            Plan::Union { branches } => RegularExpression::union_all(branches.iter().map(regex)),
            Plan::Hubs { hubs, paths } => {
                let paths: Vec<Vec<Option<RegularExpression>>> = paths
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|path| path.map(|p| regex(&p).clone()))
                            .collect()
                    })
                    .collect();
                eliminate_hubs(*hubs, &paths)
            }
        })
    }
}

/// A minimal DFA, its state elimination regex and how it splits.
struct Piece {
    dfa: FastAutomaton,
    eliminated: RegularExpression,
    plan: Plan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Eliminated,
    Neural,
    Composed,
}

/// A minimal DFA's states numbered by BFS from the start, following
/// transitions in the order of their characters: equal for isomorphic DFAs.
type CanonicalForm = Vec<(bool, Vec<(CharRange, usize)>)>;

pub(super) struct Decomposer<'a> {
    synthesizer: Option<&'a NeuralSynthesizer>,
    /// Whether the model already failed on the whole automaton, so is not
    /// run on it again.
    pub(super) whole_tried: bool,
    pub(super) stats: Stats,
    /// Children come before their parents.
    pieces: Vec<Piece>,
    /// The pieces of a split often share sub-pieces.
    index: AHashMap<CanonicalForm, usize>,
    /// Pieces started, finished or not: bounds the exploration.
    started: usize,
}

impl<'a> Decomposer<'a> {
    /// A decomposer running `synthesizer` on the pieces, or state elimination
    /// alone without one.
    pub(super) fn new(synthesizer: Option<&'a NeuralSynthesizer>) -> Self {
        Self {
            synthesizer,
            whole_tried: false,
            stats: Stats::default(),
            pieces: vec![],
            index: AHashMap::new(),
            started: 0,
        }
    }

    /// Converts `automaton`, keeping for every piece the simplest of state
    /// elimination, the model and the composition of its pieces.
    #[tracing::instrument(level = "debug", skip_all, fields(states = automaton.number_of_states()))]
    pub(super) fn convert(
        &mut self,
        automaton: &FastAutomaton,
    ) -> Result<RegularExpression, EngineError> {
        let root = self.explore(minimal_dfa(automaton)?)?;
        self.stats.pieces = self.pieces.len();
        let proposed = self.propose(root)?;

        let mut regexes: Vec<RegularExpression> = Vec::with_capacity(self.pieces.len());
        let mut origins: Vec<Origin> = Vec::with_capacity(self.pieces.len());
        for (piece, proposed) in self.pieces.iter().zip(proposed) {
            let mut regex = piece.eliminated.clone();
            let mut origin = Origin::Eliminated;
            if let Some(proposed) = proposed
                && proposed.evaluate_complexity() <= regex.evaluate_complexity()
            {
                regex = proposed;
                origin = Origin::Neural;
            }
            if let Some(composed) = piece.plan.compose(&regexes)
                && composed.evaluate_complexity() < regex.evaluate_complexity()
            {
                regex = composed;
                origin = Origin::Composed;
            }
            regexes.push(regex);
            origins.push(origin);
        }
        self.count_result(root, &origins);
        Ok(regexes.swap_remove(root))
    }

    /// The model's proposal for each piece small enough, within the budget,
    /// the largest pieces (the closest to the whole) first.
    fn propose(&mut self, root: usize) -> Result<Vec<Option<RegularExpression>>, EngineError> {
        let mut proposed = vec![None; self.pieces.len()];
        let Some(synthesizer) = self.synthesizer else {
            return Ok(proposed);
        };
        let eligible: Vec<usize> = (0..self.pieces.len())
            .rev()
            .filter(|&piece| {
                let Piece {
                    dfa, eliminated, ..
                } = &self.pieces[piece];
                !(self.whole_tried && piece == root)
                    && dfa.number_of_states() <= synthesizer.max_states()
                    && eliminated.evaluate_complexity() > SKIP_COMPLEXITY
            })
            .take(NEURAL_BUDGET)
            .collect();
        for batch in eligible.chunks(BATCH) {
            let dfas: Vec<&FastAutomaton> =
                batch.iter().map(|piece| &self.pieces[*piece].dfa).collect();
            for (piece, regex) in batch.iter().zip(synthesizer.propose_batch(&dfas)?) {
                self.stats.neural_runs += 1;
                if regex.is_some() {
                    self.stats.neural_correct += 1;
                }
                proposed[*piece] = regex;
            }
        }
        Ok(proposed)
    }

    fn count_result(&mut self, root: usize, origins: &[Origin]) {
        self.stats.composed = origins[root] == Origin::Composed;
        let mut seen = IntSet::default();
        let mut stack = vec![root];
        while let Some(piece) = stack.pop() {
            if !seen.insert(piece) {
                continue;
            }
            match origins[piece] {
                Origin::Composed => stack.extend(self.pieces[piece].plan.children()),
                origin => {
                    if origin == Origin::Neural {
                        self.stats.neural_kept += 1;
                    }
                    let states = self.pieces[piece].dfa.number_of_states();
                    self.stats.max_piece_states = self.stats.max_piece_states.max(states);
                }
            }
        }
    }

    /// The piece for `dfa`, a minimal DFA, split recursively.
    fn explore(&mut self, dfa: FastAutomaton) -> Result<usize, EngineError> {
        ExecutionProfile::get().assert_not_timed_out()?;
        let canonical_form = pieces::canonical_form(&dfa)?;
        if let Some(piece) = self.index.get(&canonical_form) {
            return Ok(*piece);
        }
        self.started += 1;
        let eliminated = dfa.to_regex()?;
        let plan = if self.started > MAX_PIECES {
            Plan::Leaf
        } else {
            self.plan(&dfa)?
        };
        let piece = self.pieces.len();
        self.pieces.push(Piece {
            dfa,
            eliminated,
            plan,
        });
        self.index.insert(canonical_form, piece);
        Ok(piece)
    }

    /// How `dfa` splits, its pieces explored.
    fn plan(&mut self, dfa: &FastAutomaton) -> Result<Plan, EngineError> {
        let number_of_states = dfa.number_of_states();
        if number_of_states <= 1 || dfa.is_empty() {
            return Ok(Plan::Leaf);
        }
        let start_state = dfa.start_state();
        let smaller = |piece: &FastAutomaton| piece.number_of_states() < number_of_states;

        if dfa.in_degree(start_state) > 0 {
            let returns = pieces::first_return(dfa, start_state)?;
            let exits = pieces::without_return(dfa, start_state)?;
            if smaller(&returns) && smaller(&exits) {
                self.stats.star_splits += 1;
                return Ok(Plan::Star {
                    returns: self.explore(returns)?,
                    exits: self.explore(exits)?,
                });
            }
            if let Some(plan) = self.frontier(dfa)? {
                return Ok(plan);
            }
            return self.hubs(dfa);
        }

        if dfa.is_accepted(start_state) {
            // The same states without the empty string, which lets the other
            // splits apply.
            let mut rest = dfa.clone();
            rest.unaccept(start_state);
            self.stats.optional_splits += 1;
            return Ok(Plan::Optional {
                rest: self.explore(minimal_dfa(&rest)?)?,
            });
        }

        if let Some(state) = analysis::closest_dominator(dfa) {
            let prefix = pieces::up_to(dfa, state)?;
            let suffix = pieces::language_of(dfa, state)?;
            if smaller(&prefix) && smaller(&suffix) {
                self.stats.concat_splits += 1;
                return Ok(Plan::Concat {
                    prefix: self.explore(prefix)?,
                    suffix: self.explore(suffix)?,
                });
            }
        }

        if let Some(plan) = self.frontier(dfa)? {
            return Ok(plan);
        }

        let components = analysis::components_without(dfa, start_state);
        if components.len() >= 2 {
            self.stats.union_splits += 1;
            let mut branches = Vec::with_capacity(components.len());
            for component in &components {
                let branch = pieces::start_into(dfa, component)?;
                branches.push(self.explore(branch)?);
            }
            return Ok(Plan::Union { branches });
        }

        self.hubs(dfa)
    }

    fn frontier(&mut self, dfa: &FastAutomaton) -> Result<Option<Plan>, EngineError> {
        let number_of_states = dfa.number_of_states();
        let smaller = |piece: &FastAutomaton| piece.number_of_states() < number_of_states;
        let Some(analysis::Cut { region, entries }) = analysis::best_cut(dfa) else {
            return Ok(None);
        };
        let stay = pieces::outside(dfa, &region)?;
        if !smaller(&stay) {
            return Ok(None);
        }
        let mut crossings = Vec::with_capacity(entries.len());
        for &entry in &entries {
            let prefix = pieces::entering_at(dfa, &region, entry)?;
            let suffix = pieces::language_of(dfa, entry)?;
            if !smaller(&prefix) || !smaller(&suffix) {
                return Ok(None);
            }
            crossings.push((prefix, suffix));
        }

        self.stats.frontier_splits += 1;
        let stay = if stay.is_empty() {
            None
        } else {
            Some(self.explore(stay)?)
        };
        let mut explored = Vec::with_capacity(crossings.len());
        for (prefix, suffix) in crossings {
            explored.push((self.explore(prefix)?, self.explore(suffix)?));
        }
        Ok(Some(Plan::Frontier {
            stay,
            crossings: explored,
        }))
    }

    fn hubs(&mut self, dfa: &FastAutomaton) -> Result<Plan, EngineError> {
        let Some(hubs) = analysis::choose_hubs(dfa) else {
            return Ok(Plan::Leaf);
        };
        // Every piece must have smaller cycles, or the split need not end.
        let weight = analysis::cyclic_weight(dfa);
        let sources: Vec<State> = std::iter::once(dfa.start_state())
            .chain(hubs.iter().copied())
            .collect();
        let targets: Vec<Option<State>> = hubs
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None))
            .collect();
        let mut paths = Vec::with_capacity(sources.len());
        for &source in &sources {
            let mut row = Vec::with_capacity(targets.len());
            for &target in &targets {
                let path = pieces::between_hubs(dfa, source, &hubs, target)?;
                if !path.is_empty() && analysis::cyclic_weight(&path) >= weight {
                    return Ok(Plan::Leaf);
                }
                row.push(path);
            }
            paths.push(row);
        }

        self.stats.hub_splits += 1;
        let mut explored = Vec::with_capacity(paths.len());
        for row in paths {
            let mut explored_row = Vec::with_capacity(row.len());
            for path in row {
                explored_row.push(if path.is_empty() {
                    None
                } else {
                    Some(self.explore(path)?)
                });
            }
            explored.push(explored_row);
        }
        Ok(Plan::Hubs {
            hubs: hubs.len(),
            paths: explored,
        })
    }
}

/// State elimination over the start (node 0), the `hubs` (1 to `hubs`) and
/// acceptance (`hubs + 1`), with `paths[x][y - 1]` from `x` to `y`: the
/// simplest result over the elimination orders.
fn eliminate_hubs(hubs: usize, paths: &[Vec<Option<RegularExpression>>]) -> RegularExpression {
    let accept = hubs + 1;
    let mut orders: Vec<Vec<usize>> = vec![vec![]];
    for _ in 0..hubs {
        let mut longer = Vec::with_capacity(orders.len() * hubs);
        for order in &orders {
            for hub in (1..=hubs).filter(|hub| !order.contains(hub)) {
                let mut order = order.clone();
                order.push(hub);
                longer.push(order);
            }
        }
        orders = longer;
    }

    let mut best: Option<RegularExpression> = None;
    for order in orders {
        // `edges[x][y]`, from node `x` to node `y`.
        let mut edges: Vec<Vec<Option<RegularExpression>>> = paths
            .iter()
            .map(|row| std::iter::once(None).chain(row.iter().cloned()).collect())
            .collect();
        let mut remaining: Vec<usize> = (1..=hubs).collect();
        for hub in order {
            remaining.retain(|node| *node != hub);
            let around = edges[hub][hub]
                .as_ref()
                .map_or_else(RegularExpression::new_empty_string, |cycle| {
                    cycle.repeat(0, None)
                });
            let from_nodes: Vec<usize> = std::iter::once(0)
                .chain(remaining.iter().copied())
                .collect();
            let to_nodes: Vec<usize> = remaining
                .iter()
                .copied()
                .chain(std::iter::once(accept))
                .collect();
            for &from_node in &from_nodes {
                let Some(into) = edges[from_node][hub].clone() else {
                    continue;
                };
                for &to_node in &to_nodes {
                    let Some(out) = &edges[hub][to_node] else {
                        continue;
                    };
                    let through = into.concat(&around, true).concat(out, true);
                    edges[from_node][to_node] = Some(match &edges[from_node][to_node] {
                        Some(direct) => direct.union(&through),
                        None => through,
                    });
                }
            }
        }
        let regex = edges[0][accept]
            .take()
            .unwrap_or_else(RegularExpression::new_empty);
        if best
            .as_ref()
            .is_none_or(|best| regex.evaluate_complexity() < best.evaluate_complexity())
        {
            best = Some(regex);
        }
    }
    best.unwrap_or_else(RegularExpression::new_empty)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use regex_charclass::char::Char;

    use super::*;

    fn assert_keeps_the_language(automaton: &FastAutomaton) -> Stats {
        let mut decomposer = Decomposer::new(None);
        let regex = decomposer.convert(automaton).unwrap();
        assert!(
            regex.to_automaton().unwrap().equivalent(automaton).unwrap(),
            "{regex}"
        );
        decomposer.stats
    }

    #[test]
    fn keeps_the_language() {
        for pattern in [
            "a",
            "",
            "a*",
            "(ab|cd)*efg",
            "(abc|def)xyz",
            "(abc|def)(x|yz)+",
            "x(ab)*y(cd)*z",
            "(a|b)*abb",
            ".*abc.*def.*",
            "(ab)?c",
            "a(b|c)*d(e|f)*",
            "((ab)*c)*d",
            "[a-z]{2,6}q",
            "(a+|b+)c(d|e)?",
        ] {
            let automaton = RegularExpression::new(pattern)
                .unwrap()
                .to_automaton()
                .unwrap();
            assert_keeps_the_language(&automaton);
        }
    }

    /// Intersections, whose largest strongly connected component the other
    /// splits leave whole.
    #[test]
    fn hub_splits_keep_the_language() {
        let mut hub_splits = 0;
        for (a, b) in [
            ("(a|b)*a(a|b){3}", ".*"),
            ("((ab|ba)*c)*", ".*"),
            (".*abc.*", ".*def.*"),
            ("[a-e]*", ".*a.*b.*c.*"),
            ("(xa*b*)*", ".*ab.*ba.*"),
        ] {
            let a = RegularExpression::new(a).unwrap().to_automaton().unwrap();
            let b = RegularExpression::new(b).unwrap().to_automaton().unwrap();
            let automaton = a.intersection(&b).unwrap();
            hub_splits += assert_keeps_the_language(&automaton).hub_splits;
        }
        assert!(hub_splits > 0);
    }

    /// The stars state elimination of the minimal DFA tangles, the splits
    /// find back.
    #[test]
    fn simplifies_nested_stars() {
        let automaton = RegularExpression::new("((ab)*c(de)*f)*g")
            .unwrap()
            .to_automaton()
            .unwrap();
        let dfa = minimal_dfa(&automaton).unwrap();
        assert_eq!(
            dfa.to_regex().unwrap().to_string(),
            "((ab)*c(de|fab(ab)*c|fc)*f)?g"
        );
        let regex = Decomposer::new(None).convert(&dfa).unwrap();
        assert_eq!(regex.to_string(), "((ab)*c(de)*f)*g");
    }

    /// A small palette of character ranges for random automata.
    fn palette(index: usize) -> CharRange {
        let bounds = [
            ('a', 'a'),
            ('b', 'b'),
            ('c', 'c'),
            ('a', 'c'),
            ('b', 'd'),
            ('x', 'z'),
            ('\u{0}', '\u{10FFFF}'),
        ];
        let (low, high) = bounds[index % bounds.len()];
        CharRange::new_from_range(Char::new(low)..=Char::new(high))
    }

    fn arb_automaton() -> impl Strategy<Value = FastAutomaton> {
        (
            2usize..7,
            proptest::collection::vec((0usize..6, 0usize..6, 0usize..7), 1..15),
            0u8..=255,
        )
            .prop_map(|(number_of_states, edges, accept_mask)| {
                let mut automaton = FastAutomaton::new_empty();
                for _ in 1..number_of_states {
                    automaton.new_state();
                }
                for (from_state, to_state, range) in edges {
                    automaton
                        .add_transition_from_range(
                            from_state % number_of_states,
                            to_state % number_of_states,
                            &palette(range),
                        )
                        .unwrap();
                }
                for state in 0..number_of_states {
                    if accept_mask & (1 << state) != 0 {
                        automaton.accept(state);
                    }
                }
                automaton
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn keeps_the_language_of_random_automata(automaton in arb_automaton()) {
            let regex = Decomposer::new(None).convert(&automaton).unwrap();
            prop_assert!(regex.to_automaton().unwrap().equivalent(&automaton).unwrap());
        }
    }
}
