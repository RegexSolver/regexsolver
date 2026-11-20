use std::ops::Range;

use condition::converter::{ConditionConverter, SharedSpanningSet};

use crate::{error::EngineError, execution_profile::ExecutionProfile};

use super::state_sets::{Id, Marks, Operand, SetTrie};
use super::*;

impl FastAutomaton {
    /// Returns `true` if both automata accept the same strings.
    ///
    /// Neither automaton is determinized.
    ///
    /// Close to linear in the number of states for two deterministic automata.
    /// Exponential in the worst case otherwise: the timeout of the execution
    /// profile applies, and its `max_number_of_states` caps the number of pairs
    /// of sets of states visited.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn equivalent(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        // `is_empty` is exact, so a mismatch proves the languages differ.
        // (`is_total` must NOT be part of this fast path: it is conservative
        // on non-deterministic automata, returning `false` for an automaton
        // that actually accepts every string, so an `is_total` mismatch alone
        // proves nothing.)
        if self.is_empty() != other.is_empty() {
            return Ok(false);
        } else if self == other {
            return Ok(true);
        }
        if self.deterministic && other.deterministic {
            self.equivalent_deterministic(other)
        } else {
            self.equivalent_over_state_sets(other)
        }
    }

    /// The algorithm of Hopcroft & Karp, *A linear algorithm for testing
    /// equivalence of finite automata* (1971), for two deterministic
    /// automata.
    ///
    /// A word reaches one state on each side, so the walk visits pairs of
    /// states, and a pair is skipped when its states are already in the same
    /// class of [`Classes`]. Each pair walked merges two classes, so there
    /// are at most as many as there are states.
    ///
    /// A missing transition leads to a sink that both automata share, so the
    /// states that cannot reach an accept state need no special care: they
    /// are merged with the sink like any other state equivalent to it.
    pub(super) fn equivalent_deterministic(
        &self,
        other: &FastAutomaton,
    ) -> Result<bool, EngineError> {
        let shared = SharedSpanningSet::new(&self.spanning_set, &other.spanning_set);
        match shared.converters()? {
            (Some(self_converter), Some(other_converter)) => {
                let project = |automaton: &FastAutomaton, converter: &ConditionConverter| {
                    (0..automaton.transitions.len())
                        .map(|state| {
                            automaton
                                .transitions_from(state)
                                .map(|(condition, to)| Ok((converter.convert(condition)?, *to)))
                                .collect()
                        })
                        .collect::<Result<Vec<Vec<(Condition, State)>>, EngineError>>()
                };
                let (left, right) = (
                    project(self, &self_converter)?,
                    project(other, &other_converter)?,
                );
                self.walk_deterministic(
                    other,
                    shared.spanning_set(),
                    |state| left[state].iter().map(|(condition, to)| (condition, *to)),
                    |state| right[state].iter().map(|(condition, to)| (condition, *to)),
                )
            }
            _ => self.walk_deterministic(
                other,
                shared.spanning_set(),
                |state| {
                    self.transitions_from(state)
                        .map(|(condition, to)| (condition, *to))
                },
                |state| {
                    other
                        .transitions_from(state)
                        .map(|(condition, to)| (condition, *to))
                },
            ),
        }
    }

    /// The walk of [`equivalent_deterministic`](Self::equivalent_deterministic),
    /// over the transitions of both automata expressed in `spanning_set`.
    fn walk_deterministic<'a, L, R>(
        &self,
        other: &FastAutomaton,
        spanning_set: &SpanningSet,
        left_transitions: impl Fn(State) -> L,
        right_transitions: impl Fn(State) -> R,
    ) -> Result<bool, EngineError>
    where
        L: Iterator<Item = (&'a Condition, State)> + Clone,
        R: Iterator<Item = (&'a Condition, State)> + Clone,
    {
        let execution_profile = ExecutionProfile::get();

        // Both automata and the sink share one id space, `self` first.
        let other_offset = self.transitions.len();
        let sink = other_offset + other.transitions.len();
        if u32::try_from(sink).is_err() {
            return Err(EngineError::AutomatonHasTooManyStates);
        }
        let accepts = |id: usize| {
            if id < other_offset {
                self.accept_states.contains(&id)
            } else {
                id < sink && other.accept_states.contains(&(id - other_offset))
            }
        };

        let start = (self.start_state, other_offset + other.start_state);
        if accepts(start.0) != accepts(start.1) {
            return Ok(false);
        }
        let mut classes = Classes::new(sink + 1);
        classes.merge(start.0, start.1);
        let mut worklist = VecDeque::new();
        worklist.push_back(start);
        let mut pairs = 1;

        // The characters each state of the pair has a transition for.
        let mut left_covered = Condition::empty(spanning_set);
        let mut right_covered = Condition::empty(spanning_set);

        while let Some((left, right)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(pairs)?;

            // Returns `false` on a counterexample.
            let mut relate = |left: usize, right: usize| {
                if accepts(left) != accepts(right) {
                    return false;
                }
                if classes.merge(left, right) {
                    worklist.push_back((left, right));
                    pairs += 1;
                }
                true
            };

            let left_transitions = (left != sink)
                .then(|| left_transitions(left))
                .into_iter()
                .flatten();
            let right_transitions = (right != sink)
                .then(|| right_transitions(right - other_offset))
                .into_iter()
                .flatten();
            left_covered.clear();
            for (condition, _) in left_transitions.clone() {
                left_covered.union_with(condition);
            }
            right_covered.clear();
            for (condition, _) in right_transitions.clone() {
                right_covered.union_with(condition);
            }

            for (left_condition, left_to) in left_transitions {
                for (right_condition, right_to) in right_transitions.clone() {
                    if left_condition.has_intersection(right_condition)
                        && !relate(left_to, other_offset + right_to)
                    {
                        return Ok(false);
                    }
                }
                if !left_condition.is_subset_of(&right_covered) && !relate(left_to, sink) {
                    return Ok(false);
                }
            }
            for (right_condition, right_to) in right_transitions {
                if !right_condition.is_subset_of(&left_covered)
                    && !relate(sink, other_offset + right_to)
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Algorithm HKC of Bonchi & Pous, *Checking NFA equivalence with
    /// bisimulations up to congruence* (POPL 2013).
    ///
    /// The walk visits the pairs `(X, Y)` of state sets that one word reaches
    /// in `self` and in `other`. A pair is a counterexample when exactly one
    /// side accepts, or exactly one side is empty. It is skipped when it
    /// follows from the pairs met so far by [`Congruence`], which can hold
    /// for a pair neither of whose sets was met before.
    pub(super) fn equivalent_over_state_sets(
        &self,
        other: &FastAutomaton,
    ) -> Result<bool, EngineError> {
        let execution_profile = ExecutionProfile::get();

        // Both operands share one id space, `self` first: the sets a rule
        // rewrites mix states of the two.
        let other_offset = self.transitions.len();
        let (left, right, shared) = Operand::pair(self, 0, other, other_offset)?;
        let bases = shared.spanning_set().spanning_ranges_with_rest_len();

        let (Some(left_start), Some(right_start)) = (left.start, right.start) else {
            return Ok(left.start.is_none() && right.start.is_none());
        };
        if left.accepts(left_start) != right.accepts(right_start) {
            return Ok(false);
        }

        let mut congruence = Congruence::new(other_offset + other.transitions.len());
        let mut worklist = VecDeque::new();
        worklist.push_back(congruence.relate(&[left_start], &[right_start]));

        let mut left_successors = vec![Vec::new(); bases];
        let mut right_successors = vec![Vec::new(); bases];
        let mut pairs = 1;

        while let Some((x, y)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(pairs)?;

            left.successors(&congruence.sets[x], &mut left_successors);
            right.successors(&congruence.sets[y], &mut right_successors);

            for (x, y) in left_successors.iter().zip(&right_successors) {
                if x.is_empty() && y.is_empty() {
                    continue;
                }
                // Only live states are kept, so a non-empty set accepts some
                // word, which an empty one does not.
                if x.is_empty() != y.is_empty() || left.is_accepting(x) != right.is_accepting(y) {
                    return Ok(false);
                }
                if congruence.relates(x, y) {
                    continue;
                }
                worklist.push_back(congruence.relate(x, y));
                pairs += 1;
            }
        }
        Ok(true)
    }
}

/// The classes of the states related so far (union-find).
struct Classes {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Classes {
    fn new(states: usize) -> Self {
        Classes {
            parent: (0..states as u32).collect(),
            size: vec![1; states],
        }
    }

    fn find(&mut self, mut state: usize) -> usize {
        while self.parent[state] as usize != state {
            let parent = self.parent[state] as usize;
            self.parent[state] = self.parent[parent];
            state = parent;
        }
        state
    }

    /// Puts both states in the same class; returns `false` if they already
    /// were.
    fn merge(&mut self, a: usize, b: usize) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        let (small, large) = if self.size[a] < self.size[b] {
            (a, b)
        } else {
            (b, a)
        };
        self.parent[small] = large as u32;
        self.size[large] += self.size[small];
        true
    }
}

/// The number of trie nodes one normal form search may visit: `BUDGET_BASE`,
/// plus `BUDGET_PER_STATE` for each state of the two sets compared.
const BUDGET_BASE: usize = 256;
const BUDGET_PER_STATE: usize = 16;

/// The pairs of sets met by the walk, read as rewriting rules: a pair
/// `(X, Y)` lets any set containing `X` grow by `Y`, and any set containing
/// `Y` grow by `X`. Two sets are related by the congruence closure of the
/// pairs iff rewriting takes them to the same normal form.
struct Congruence {
    /// The sets of the pairs, back to back.
    sets: Vec<Id>,
    /// Each side of each pair, valued with where the other side is in `sets`.
    rules: SetTrie<Range<usize>>,
    root: usize,
    // Scratch state of `normal_form_contains`, kept to reuse the allocations.
    normal_form: Marks,
    /// By state, the trie nodes waiting for that state to enter the normal
    /// form; `touched` lists the states with a non-empty entry.
    parked: Vec<Vec<usize>>,
    touched: Vec<Id>,
    /// The visited trie nodes with too many children to park one by one.
    wide: Vec<usize>,
    stack: Vec<usize>,
}

impl Congruence {
    fn new(states: usize) -> Self {
        let mut rules = SetTrie::new();
        let root = rules.new_root();
        Congruence {
            sets: Vec::new(),
            rules,
            root,
            normal_form: Marks::new(states),
            parked: vec![Vec::new(); states],
            touched: Vec::new(),
            wide: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Records the pair `(x, y)` and returns where it is stored in `sets`.
    fn relate(&mut self, x: &[Id], y: &[Id]) -> (Range<usize>, Range<usize>) {
        let start = self.sets.len();
        self.sets.extend_from_slice(x);
        self.sets.extend_from_slice(y);
        let stored = (start..start + x.len(), start + x.len()..self.sets.len());
        self.rules.insert(self.root, x, stored.1.clone());
        self.rules.insert(self.root, y, stored.0.clone());
        stored
    }

    /// Returns `true` if `x` and `y` are known to be related: they have the
    /// same normal form, which holds iff each is contained in the normal
    /// form of the other.
    ///
    /// `false` is always a safe answer (the pair is then walked like any new
    /// one), and is the answer when the search for a normal form runs out of
    /// budget. A recorded pair is recognised whatever the budget, which
    /// keeps the walk finite.
    fn relates(&mut self, x: &[Id], y: &[Id]) -> bool {
        let recorded = self
            .rules
            .get(self.root, x)
            .iter()
            .any(|other| self.sets[other.clone()] == *y);
        recorded || (self.normal_form_contains(x, y) && self.normal_form_contains(y, x))
    }

    /// Returns `true` if the normal form of `set`, the smallest superset
    /// closed under every rule, is found to contain `target`.
    ///
    /// A trie node is reached once every state on its path is in the normal
    /// form, at which point the rules ending there fire. A child whose state
    /// is missing is parked on that state and resumed if a later rule brings
    /// the state in, so no part of the trie is walked twice and the parts the
    /// normal form never covers are not walked at all.
    ///
    /// The number of rules contained in a set can be exponential in its size
    /// (when determinization itself blows up and nothing can be skipped), so
    /// the search stops after a number of nodes proportional to the sets at
    /// hand, and answers from the part of the normal form built so far.
    fn normal_form_contains(&mut self, set: &[Id], target: &[Id]) -> bool {
        let Congruence {
            sets,
            rules,
            root,
            normal_form,
            parked,
            touched,
            wide,
            stack,
        } = self;

        for &id in set {
            normal_form.insert(id);
        }
        stack.push(*root);
        for _ in 0..BUDGET_BASE + BUDGET_PER_STATE * (set.len() + target.len()) {
            let Some(index) = stack.pop() else {
                break;
            };
            let node = rules.node(index);
            for rhs in &node.values {
                for &id in &sets[rhs.clone()] {
                    if normal_form.insert(id) {
                        stack.append(&mut parked[id as usize]);
                        stack.extend(wide.iter().filter_map(|&wide| rules.node(wide).child(id)));
                    }
                }
            }
            match node.few_children() {
                Some(children) => {
                    for &(id, child) in children {
                        if normal_form.contains(id) {
                            stack.push(child);
                        } else {
                            if parked[id as usize].is_empty() {
                                touched.push(id);
                            }
                            parked[id as usize].push(child);
                        }
                    }
                }
                // Too many children to park: look up the states the normal
                // form holds now, and the ones it gains later as they come.
                None => {
                    stack.extend(
                        normal_form
                            .members()
                            .iter()
                            .filter_map(|&id| node.child(id)),
                    );
                    wide.push(index);
                }
            }
        }

        let contained = target.iter().all(|&id| normal_form.contains(id));
        normal_form.clear();
        stack.clear();
        wide.clear();
        for id in touched.drain(..) {
            parked[id as usize].clear();
        }
        contained
    }
}

#[cfg(test)]
mod tests {
    use super::super::state_sets::{Id, WIDE};
    use super::Congruence;
    use crate::regex::RegularExpression;

    // {1} ~ {2} and {2, 3} ~ {4}: relating {1, 3} to {4} takes the first
    // rule to bring 2 in before the second one can fire, i.e. a trie branch
    // that was not coverable when first met has to be resumed.
    #[test]
    fn normal_form_chains_rules() {
        let mut congruence = Congruence::new(8);
        congruence.relate(&[1], &[2]);
        congruence.relate(&[2, 3], &[4]);

        assert!(congruence.relates(&[1, 3], &[4]));
        assert!(congruence.relates(&[1, 3], &[1, 2, 3, 4]));
        assert!(congruence.relates(&[2, 3], &[4]));
        // Containment in one direction only is not enough.
        assert!(!congruence.relates(&[1, 3], &[2]));
        assert!(!congruence.relates(&[1], &[4]));
        assert!(!congruence.relates(&[3], &[4]));
        assert!(!congruence.relates(&[5], &[6]));
    }

    // Same chain through a node with more children than are scanned.
    #[test]
    fn normal_form_chains_rules_through_wide_nodes() {
        let states = 4 * WIDE as Id;
        let mut congruence = Congruence::new(2 * states as usize + 2);
        // i ~ states + i for every i: the root gets 2 * states children.
        for id in 0..states {
            congruence.relate(&[id], &[states + id]);
        }
        let last = 2 * states;
        congruence.relate(&[states + 3, states + 5], &[last]);
        assert!(
            congruence
                .rules
                .node(congruence.root)
                .few_children()
                .is_none()
        );

        assert!(congruence.relates(&[7], &[states + 7]));
        assert!(congruence.relates(&[3, 5], &[last]));
        assert!(congruence.relates(&[3, states + 5], &[5, states + 3, last]));
        assert!(!congruence.relates(&[3], &[last]));
        assert!(!congruence.relates(&[3, 6], &[last]));
    }

    #[test]
    fn test_equivalent() -> Result<(), String> {
        assert_equivalent(
            &RegularExpression::new_empty(),
            &RegularExpression::new_empty_string(),
            false,
        );

        assert_equivalent(
            &RegularExpression::new_total(),
            &RegularExpression::new_empty_string(),
            false,
        );

        let regex_1 = RegularExpression::parse("cd", false).unwrap();
        let regex_2 = RegularExpression::parse("cd", false).unwrap();
        assert_equivalent(&regex_1, &regex_2, true);

        let regex_1 = RegularExpression::parse("test.*other", false).unwrap();
        let regex_2 = RegularExpression::parse("test.*othew", false).unwrap();

        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::parse("test.{0,50}other", false).unwrap();
        let regex_2 = RegularExpression::parse("test.{0,49}other", false).unwrap();

        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::parse("[0]", false).unwrap();
        let regex_2 = RegularExpression::parse("[01]", false).unwrap();
        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::parse("(b+a+)*", false).unwrap();
        let regex_2 = RegularExpression::parse("(b[a-b]*a)?", false).unwrap();
        assert_equivalent(&regex_1, &regex_2, true);

        Ok(())
    }

    fn assert_equivalent(regex_1: &RegularExpression, regex_2: &RegularExpression, expected: bool) {
        println!("{regex_1} and {regex_2}");
        let automaton_1 = regex_1.to_automaton().unwrap();
        assert!(automaton_1.equivalent(&automaton_1).unwrap());

        let automaton_2 = regex_2.to_automaton().unwrap();
        assert!(automaton_2.equivalent(&automaton_2).unwrap());

        assert_eq!(expected, automaton_1.equivalent(&automaton_2).unwrap());
    }
}
