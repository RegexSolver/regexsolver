//! Property-based tests built on `proptest` strategies that generate random
//! DFAs, NFAs and regular expressions.
//!
//! # Coverage guarantee
//!
//! The strategies are parameterized by a fixed finite alphabet ([`ALPHABET`])
//! and a maximum number of states ([`MAX_STATES`]). Within those bounds every
//! structure has a strictly positive probability of being generated:
//!
//! * [`arb_dfa`] — every deterministic automaton over the alphabet with
//!   `1..=MAX_STATES` states (start state fixed to `0`, any accepting subset,
//!   any total transition function) can be produced. A transition function maps
//!   each `(state, letter)` to at most one target, which is exactly the
//!   definition of a DFA, so the whole DFA space is covered.
//! * [`arb_nfa`] — every nondeterministic automaton is reachable: each ordered
//!   `(from, to)` pair may carry any subset of the alphabet letters as its
//!   label, plus optional epsilon transitions. Since labels may overlap, this
//!   spans all NFAs (and, as a subset, all DFAs).
//! * [`arb_regex`] — every regular expression built from the four
//!   [`RegularExpression`] variants up to the configured recursion depth and
//!   bound sizes is reachable, including the empty language (`[]`) and `.`.
//!
//! The alphabet is small on purpose so the spaces are finite and the
//! determinization / set operations under test stay cheap.
//!
//! While *coverage* is uniform-in-support, the *distribution* is deliberately
//! shaped: per-automaton edge/epsilon/accept densities are sampled and the
//! character-class strategy favors single letters, so that degenerate
//! languages (∅, {""}) and near-complete blobs are occasional rather than
//! dominant. All weights stay strictly inside (0, 1), preserving the
//! non-null-probability guarantee.

use proptest::prelude::*;
use regex_charclass::char::Char;
use regexsolver::CharRange;
use regexsolver::error::EngineError;
use regexsolver::execution_profile::ExecutionProfileBuilder;
use regexsolver::fast_automaton::FastAutomaton;
use regexsolver::fast_automaton::condition::Condition;
use regexsolver::fast_automaton::spanning_set::SpanningSet;
use regexsolver::regex::RegularExpression;

/// Fixed alphabet the strategies draw transition labels from.
pub const ALPHABET: &[char] = &['a', 'b'];

/// Maximum number of states a generated automaton can have.
pub const MAX_STATES: usize = 4;

/// The single-character range for the `i`-th alphabet letter.
fn letter(i: usize) -> CharRange {
    let c = Char::new(ALPHABET[i]);
    CharRange::new_from_range(c..=c)
}

/// The spanning set induced by the alphabet (one base per letter + a "rest").
fn spanning_set() -> SpanningSet {
    let ranges: Vec<CharRange> = (0..ALPHABET.len()).map(letter).collect();
    SpanningSet::compute_spanning_set(&ranges)
}

/// The transition-label bases: one [`CharRange`] per alphabet letter.
///
/// We deliberately exclude the spanning set's "rest" range. The spanning set's
/// contract is that the rest holds exactly the characters that **no** transition
/// uses, so a label may only be a subset of the alphabet letters; that is also
/// precisely the standard "automaton over Σ" model.
fn bases(ss: &SpanningSet) -> Vec<CharRange> {
    let _ = ss;
    (0..ALPHABET.len()).map(letter).collect()
}

/// Number of transition-label bases (one per alphabet letter).
fn num_bases() -> usize {
    ALPHABET.len()
}

/// Number of states, biased toward larger automata (the maximum of two uniform
/// draws). Tiny automata are still generated, but the n = 1 space is almost
/// entirely degenerate, so uniform sampling would waste a quarter of all cases
/// on it.
fn arb_num_states() -> impl Strategy<Value = usize> {
    (1usize..=MAX_STATES, 1usize..=MAX_STATES).prop_map(|(a, b)| a.max(b))
}

/// Builds an automaton from a structural description using only the public API.
///
/// `accepts[s]` marks state `s` accepting; each `(from, to, mask)` adds a
/// transition whose label is the union of the bases selected by `mask`; each
/// `(from, to)` in `eps` adds an epsilon transition. The start state is `0`.
fn build(
    n: usize,
    accepts: &[bool],
    char_edges: &[(usize, usize, Vec<bool>)],
    eps: &[(usize, usize)],
) -> FastAutomaton {
    let ss = spanning_set();
    let bs = bases(&ss);

    let mut a = FastAutomaton::new_empty();
    // `new_empty` already owns state 0; the spanning set is applied before any
    // transition so the conditions we add line up with it.
    a.apply_new_spanning_set(&ss)
        .expect("applying a spanning set to an empty automaton never fails");
    for _ in 1..n {
        a.new_state();
    }

    for (s, &acc) in accepts.iter().enumerate() {
        if acc {
            a.accept(s);
        }
    }

    for (from, to, mask) in char_edges {
        let mut range = CharRange::empty();
        for (i, &on) in mask.iter().enumerate() {
            if on {
                range = range.union(&bs[i]);
            }
        }
        if range.is_empty() {
            continue;
        }
        let cond = Condition::from_range(&range, &ss)
            .expect("a union of full spanning bases is always a valid condition");
        a.add_transition(*from, *to, &cond);
    }

    // Epsilon transitions are added last: `add_epsilon_transition` eagerly
    // folds the target's current transitions into the source.
    for (from, to) in eps {
        a.add_epsilon_transition(*from, *to);
    }

    a
}

/// Strategy producing every DFA over the alphabet with `1..=MAX_STATES` states.
///
/// Determinism is structural: for each state and each letter we choose at most
/// one target, so transitions leaving a state always carry disjoint labels.
///
/// An edge density and an accept density are sampled per automaton (both
/// bounded away from 0 and 1, so every DFA keeps a positive probability).
/// Mostly-total transition functions keep the states connected, which makes
/// degenerate (∅ / {""}) languages the exception rather than the rule.
pub fn arb_dfa() -> impl Strategy<Value = FastAutomaton> {
    (arb_num_states(), 0.6f64..0.97, 0.4f64..0.9)
        .prop_flat_map(|(n, edge_density, accept_density)| {
            let accepts = prop::collection::vec(prop::bool::weighted(accept_density), n);
            // transition function: tf[state][base] = optional target state
            let tf = prop::collection::vec(
                prop::collection::vec(
                    prop::option::weighted(edge_density, 0usize..n),
                    num_bases(),
                ),
                n,
            );
            (Just(n), accepts, tf)
        })
        .prop_map(|(n, accepts, tf)| {
            let nb = num_bases();
            let mut edges: Vec<(usize, usize, Vec<bool>)> = Vec::new();
            for (from, row) in tf.iter().enumerate() {
                // Group the bases by their chosen target so each (from, to)
                // edge gets a single, disjoint-from-its-siblings label.
                let mut by_target: std::collections::BTreeMap<usize, Vec<bool>> =
                    std::collections::BTreeMap::new();
                for (base, target) in row.iter().enumerate() {
                    if let Some(t) = target {
                        by_target.entry(*t).or_insert_with(|| vec![false; nb])[base] = true;
                    }
                }
                for (to, mask) in by_target {
                    edges.push((from, to, mask));
                }
            }
            build(n, &accepts, &edges, &[])
        })
}

/// Strategy producing every NFA over the alphabet with `1..=MAX_STATES` states
/// (overlapping labels allowed, plus optional epsilon transitions).
///
/// Instead of a fixed per-bit probability (which blobs large automata and
/// starves small ones), a per-state branching target is sampled and converted
/// into a bit density of `target / (n · |Σ|)`, so the *local* structure is
/// comparable across sizes. Epsilon and accept densities are sampled too. All
/// densities stay strictly inside (0, 1), so every NFA keeps a positive
/// probability.
pub fn arb_nfa() -> impl Strategy<Value = FastAutomaton> {
    (
        arb_num_states(),
        1.0f64..2.8,
        0.02f64..0.18,
        0.4f64..0.9,
    )
        .prop_flat_map(|(n, target_out_degree, eps_density, accept_density)| {
            let label_density =
                (target_out_degree / (n as f64 * num_bases() as f64)).clamp(0.02, 0.95);
            let accepts = prop::collection::vec(prop::bool::weighted(accept_density), n);
            // labels[from][to] = mask over the alphabet letters
            let labels = prop::collection::vec(
                prop::collection::vec(
                    prop::collection::vec(prop::bool::weighted(label_density), num_bases()),
                    n,
                ),
                n,
            );
            // eps[from][to] = whether an epsilon transition is present
            let eps = prop::collection::vec(
                prop::collection::vec(prop::bool::weighted(eps_density), n),
                n,
            );
            (Just(n), accepts, labels, eps)
        })
        .prop_map(|(n, accepts, labels, eps)| {
            let mut char_edges = Vec::new();
            for (from, row) in labels.iter().enumerate() {
                for (to, mask) in row.iter().enumerate() {
                    if mask.iter().any(|&b| b) {
                        char_edges.push((from, to, mask.clone()));
                    }
                }
            }
            let mut eps_edges = Vec::new();
            for (from, row) in eps.iter().enumerate() {
                for (to, &on) in row.iter().enumerate() {
                    if on && from != to {
                        eps_edges.push((from, to));
                    }
                }
            }
            build(n, &accepts, &char_edges, &eps_edges)
        })
}

/// Strategy for a character class: any subset of the alphabet (including the
/// empty language) and, occasionally, the total range `.`.
///
/// Single letters dominate: the unbiased subset mask would produce the empty
/// class `[]` a quarter of the time, and a single `[]` anywhere in a
/// concatenation collapses the whole expression to the empty language. The
/// mask branch keeps every subset (including `[]`) at a positive probability.
fn arb_charrange() -> impl Strategy<Value = CharRange> {
    prop_oneof![
        8 => (0..ALPHABET.len()).prop_map(letter),
        3 => prop::collection::vec(any::<bool>(), ALPHABET.len()).prop_map(|mask| {
            let mut r = CharRange::empty();
            for (i, &on) in mask.iter().enumerate() {
                if on {
                    r = r.union(&letter(i));
                }
            }
            r
        }),
        1 => Just(CharRange::total()),
    ]
}

/// Strategy producing regular expressions over the four [`RegularExpression`]
/// variants up to a bounded recursion depth.
pub fn arb_regex() -> impl Strategy<Value = RegularExpression> {
    let leaf = arb_charrange().prop_map(RegularExpression::Character);
    leaf.prop_recursive(3, 24, 3, |inner| {
        prop_oneof![
            (inner.clone(), 0u32..=2, 0u32..=2, any::<bool>()).prop_map(
                |(r, min, extra, has_max)| {
                    let max = if has_max { Some(min + extra) } else { None };
                    RegularExpression::Repetition(Box::new(r), min, max)
                }
            ),
            prop::collection::vec(inner.clone(), 1..=3)
                .prop_map(|v| RegularExpression::Concat(v.into_iter().collect())),
            prop::collection::vec(inner, 1..=3).prop_map(RegularExpression::Alternation),
        ]
    })
}

/// Runs `f` under a bounded execution profile, returning `None` when the
/// operation legitimately exceeds the state/time budget (which is not a bug).
fn bounded<T, F: FnOnce() -> Result<T, EngineError>>(f: F) -> Option<T> {
    ExecutionProfileBuilder::new()
        .max_number_of_states(8192)
        .execution_timeout(3000)
        .build()
        .run(|| match f() {
            Ok(v) => Some(v),
            Err(EngineError::AutomatonHasTooManyStates)
            | Err(EngineError::OperationTimeOutError) => None,
            Err(e) => panic!("unexpected engine error: {e:?}"),
        })
}

fn determinized(a: &FastAutomaton) -> Option<FastAutomaton> {
    bounded(|| a.determinize().map(|c| c.into_owned()))
}

fn complemented(a: &FastAutomaton) -> Option<FastAutomaton> {
    bounded(|| {
        let mut c = a.clone();
        c.complement()?;
        Ok(c)
    })
}

/// All strings up to length 4 over the alphabet (plus the empty string).
fn probes() -> Vec<String> {
    let mut all = vec![String::new()];
    let mut frontier = vec![String::new()];
    for _ in 0..4 {
        let mut next = Vec::new();
        for w in &frontier {
            for &c in ALPHABET {
                let mut s = w.clone();
                s.push(c);
                next.push(s);
            }
        }
        all.extend(next.iter().cloned());
        frontier = next;
    }
    all
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    /// The DFA strategy really does produce deterministic automata.
    #[test]
    fn dfa_strategy_is_deterministic(a in arb_dfa()) {
        prop_assert!(a.is_deterministic(), "arb_dfa produced a non-deterministic automaton");
    }

    /// `a` and `determinize(a)` accept the same language.
    #[test]
    fn determinize_preserves_language(a in arb_nfa()) {
        if let Some(d) = determinized(&a) {
            prop_assert!(d.is_deterministic());
            if let Some(eq) = bounded(|| a.equivalent(&d)) {
                prop_assert!(eq, "determinize changed the language");
            }
        }
    }

    /// Minimizing a DFA preserves its language.
    #[test]
    fn minimize_preserves_language(a in arb_nfa()) {
        if let Some(d) = determinized(&a) {
            let mut m = d.clone();
            if bounded(|| m.minimize()).is_some()
                && let Some(eq) = bounded(|| d.equivalent(&m))
            {
                prop_assert!(eq, "minimize changed the language");
            }
        }
    }

    /// Complement laws: membership flips, `a ∩ ¬a = ∅`, `a ∪ ¬a = Σ*`.
    #[test]
    fn complement_laws(a in arb_dfa()) {
        let d = match determinized(&a) { Some(d) => d, None => return Ok(()) };
        let c = match complemented(&d) { Some(c) => c, None => return Ok(()) };

        for s in probes() {
            prop_assert_eq!(d.is_match(&s), !c.is_match(&s), "complement membership for {:?}", s);
        }

        if let Some(inter) = bounded(|| d.intersection(&c))
            && let Some(empty) = bounded(|| inter.equivalent(&FastAutomaton::new_empty()))
        {
            prop_assert!(empty, "a ∩ ¬a is not empty");
        }
        if let Some(union) = bounded(|| d.union(&c))
            && let Some(total) = bounded(|| union.equivalent(&FastAutomaton::new_total()))
        {
            prop_assert!(total, "a ∪ ¬a is not total");
        }
    }

    /// Membership of intersection / union / difference matches the boolean
    /// combination of the operands on every probe string.
    #[test]
    fn set_ops_membership(a in arb_nfa(), b in arb_nfa()) {
        let ps = probes();

        if let Some(inter) = bounded(|| a.intersection(&b)) {
            for s in &ps {
                prop_assert_eq!(
                    inter.is_match(s), a.is_match(s) && b.is_match(s),
                    "intersection membership for {:?}", s
                );
            }
        }
        if let Some(union) = bounded(|| a.union(&b)) {
            for s in &ps {
                prop_assert_eq!(
                    union.is_match(s), a.is_match(s) || b.is_match(s),
                    "union membership for {:?}", s
                );
            }
        }
        // `difference` determinizes the subtrahend itself.
        if let Some(diff) = bounded(|| a.difference(&b)) {
            for s in &ps {
                prop_assert_eq!(
                    diff.is_match(s), a.is_match(s) && !b.is_match(s),
                    "difference membership for {:?}", s
                );
            }
        }
    }

    /// `subset` and `equivalent` agree: mutual subset iff equivalent; both are
    /// reflexive.
    #[test]
    fn subset_equivalent_consistency(a in arb_nfa(), b in arb_nfa()) {
        if let Some(refl) = bounded(|| a.equivalent(&a)) {
            prop_assert!(refl, "equivalent is not reflexive");
        }
        if let Some(refl) = bounded(|| a.subset(&a)) {
            prop_assert!(refl, "subset is not reflexive");
        }
        if let (Some(ab), Some(ba), Some(eq)) = (
            bounded(|| a.subset(&b)),
            bounded(|| b.subset(&a)),
            bounded(|| a.equivalent(&b)),
        ) {
            prop_assert_eq!(ab && ba, eq, "mutual subset disagrees with equivalent");
        }
    }

    /// `a -> regex -> a` round-trips: the regular expression extracted from an
    /// automaton compiles back to an equivalent automaton.
    #[test]
    fn automaton_to_regex_roundtrip(a in arb_nfa()) {
        let r = a.to_regex();
        if let Some(a2) = bounded(|| r.to_automaton())
            && let Some(eq) = bounded(|| a.equivalent(&a2))
        {
            prop_assert!(eq, "automaton -> regex -> automaton changed the language: {}", r);
        }
    }

    /// Regular expressions round-trip through an automaton and agree with the
    /// reference `regex` crate on every probe string.
    #[test]
    fn regex_roundtrip_and_oracle(r in arb_regex()) {
        let a = match bounded(|| r.to_automaton()) { Some(a) => a, None => return Ok(()) };

        // regex -> automaton -> regex -> automaton preserves the language.
        let r2 = a.to_regex();
        if let Some(a2) = bounded(|| r2.to_automaton())
            && let Some(eq) = bounded(|| a.equivalent(&a2))
        {
            prop_assert!(eq, "regex round-trip changed the language: {} -> {}", r, r2);
        }

        // Cross-check membership against the standard regex engine (anchored,
        // dot-matches-newline). Patterns denoting the empty language ("[]") are
        // rejected by the `regex` crate, so we only compare when it accepts the
        // pattern.
        let pattern = r.to_string();
        if let Ok(re) = regex::Regex::new(&format!("(?s)^(?:{})$", pattern)) {
            for s in probes() {
                prop_assert_eq!(
                    a.is_match(&s), re.is_match(&s),
                    "pattern {:?} disagrees with reference engine on {:?}", pattern, s
                );
            }
        }
    }
}

#[cfg(test)]
mod inspect {
    use super::*;
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::TestRunner;

    fn samples<S: Strategy>(strat: S, n: usize) -> Vec<S::Value> {
        let mut runner = TestRunner::deterministic();
        (0..n)
            .map(|_| strat.new_tree(&mut runner).unwrap().current())
            .collect()
    }

    #[test]
    fn show_regex() {
        for (i, r) in samples(arb_regex(), 30).into_iter().enumerate() {
            println!("regex[{i:02}] = {r}");
        }
    }

    #[test]
    fn show_dfa() {
        for (i, a) in samples(arb_dfa(), 30).into_iter().enumerate() {
            println!("dfa[{i:02}] det={} Graphviz={}", a.is_deterministic(), a);
        }
    }

    #[test]
    fn show_nfa() {
        for (i, a) in samples(arb_nfa(), 30).into_iter().enumerate() {
            println!("nfa[{i:02}] det={} Graphviz={}", a.is_deterministic(), a);
        }
    }

    /// Classifies the language of an automaton for the quality summary.
    fn classify(a: &FastAutomaton) -> &'static str {
        if a.is_empty() {
            "empty"
        } else if a.is_empty_string() {
            "{\"\"}"
        } else if a
            .determinize()
            .map(|d| d.is_total())
            .unwrap_or(false)
        {
            "total"
        } else {
            "interesting"
        }
    }

    fn automaton_stats(name: &str, autos: &[FastAutomaton]) {
        let n = autos.len() as f64;
        let mut counts = std::collections::BTreeMap::new();
        let mut states = 0usize;
        let mut edges = 0usize;
        for a in autos {
            *counts.entry(classify(a)).or_insert(0usize) += 1;
            states += a.get_number_of_states();
            edges += a
                .states_vec()
                .iter()
                .map(|&s| a.transitions_from_vec(s).len())
                .sum::<usize>();
        }
        let pct = |k: &str| 100.0 * *counts.get(k).unwrap_or(&0) as f64 / n;
        println!(
            "{name}: empty {:>5.1}% | {{\"\"}} {:>5.1}% | total {:>5.1}% | interesting {:>5.1}% | avg states {:.2} | avg edges {:.2}",
            pct("empty"), pct("{\"\"}"), pct("total"), pct("interesting"),
            states as f64 / n, edges as f64 / n,
        );
    }

    /// Quantitative quality summary over a larger sample.
    #[test]
    fn stats() {
        const N: usize = 300;

        let regexes = samples(arb_regex(), N);
        let regex_autos: Vec<FastAutomaton> = regexes
            .iter()
            .map(|r| r.to_automaton().expect("small regexes always convert"))
            .collect();
        let avg_len =
            regexes.iter().map(|r| r.to_string().len()).sum::<usize>() as f64 / N as f64;
        automaton_stats("regex", &regex_autos);
        println!("regex: avg pattern length {avg_len:.1}");

        automaton_stats("dfa  ", &samples(arb_dfa(), N));
        automaton_stats("nfa  ", &samples(arb_nfa(), N));
    }
}
