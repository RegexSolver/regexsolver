use std::hash::BuildHasherDefault;
use std::ops::Range;

use condition::converter::ConditionConverter;

use crate::fast_automaton::analyze::state_sets::{Id, Operand};
use crate::{EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Totalize the automaton. Precondition: `self.deterministic` is true
    /// (the only caller, `complement`, determinizes first).
    fn totalize(&mut self) -> Result<(), EngineError> {
        debug_assert!(self.deterministic, "totalize requires a DFA");

        let crash_state = self.new_state();
        let mut transitions_to_crash_state: IntMap<State, Condition> =
            IntMap::with_capacity_and_hasher(
                self.number_of_states(),
                BuildHasherDefault::default(),
            );

        let mut ranges = Vec::with_capacity(self.number_of_states());
        for from_state in self.states() {
            let mut new_condition = Condition::empty(&self.spanning_set);
            for (condition, _) in self.transitions_from(from_state) {
                new_condition = new_condition.union(condition);
                ranges.push(condition.to_range(self.spanning_set())?);
            }

            new_condition = new_condition.complement();

            transitions_to_crash_state.insert(from_state, new_condition);
        }

        for (from_state, condition) in &transitions_to_crash_state {
            self.add_transition(*from_state, crash_state, condition);
            ranges.push(condition.to_range(self.spanning_set())?);
        }

        let new_spanning_set = SpanningSet::compute_spanning_set(&ranges);
        self.apply_new_spanning_set(&new_spanning_set)?;

        if self.in_degree(crash_state) == 1 {
            // Only the self-loop points to crash; nothing else needs it.
            self.remove_state(crash_state);
        }
        Ok(())
    }

    /// Replaces the automaton with its complement: the strings it does not
    /// accept.
    ///
    /// A non-deterministic automaton is determinized first, which is
    /// exponential in the worst case.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::DeterministicAutomatonRequired`] for a
    /// non-deterministic automaton when the execution profile disables implicit
    /// determinization.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states(), deterministic = self.is_deterministic()))]
    pub fn complement(&mut self) -> Result<(), EngineError> {
        if !self.deterministic {
            *self = self.determinize_implicit()?.into_owned();
        }
        self.totalize()?;

        let mut new_accept_states = IntSet::default();
        for state in self.states() {
            if self.accept_states.contains(&state) {
                continue;
            }
            new_accept_states.insert(state);
        }

        self.accept_states = new_accept_states;
        Ok(())
    }

    /// Returns the difference of `self` and `other`: the strings that `self`
    /// accepts and `other` does not.
    ///
    /// Neither automaton is determinized. The result is deterministic when
    /// `self` is.
    ///
    /// With `n` and `m` the numbers of states of `self` and `other`, the result
    /// has up to `n × 2^m` states. The timeout of the execution profile
    /// applies, and its `max_number_of_states` caps the number of states built,
    /// including those left out of the result because no accept state can be
    /// reached from them.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn difference(&self, other: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        if self.is_empty() || other.is_total() || self == other {
            return Ok(Self::new_empty());
        } else if other.is_empty() {
            return Ok(self.clone());
        }
        let (left, right, shared) = Operand::pair(self, 0, other, 0)?;
        let spanning_set = shared.into_spanning_set();
        let Some(start) = left.start else {
            return Ok(Self::new_empty());
        };

        let mut difference = Draft::default();
        let mut pairs = Pairs::new(&left, &right);
        pairs.state(&mut difference, start, right.start.as_slice());
        pairs.walk(&mut difference, &spanning_set)?;
        Ok(difference.into_automaton(spanning_set))
    }

    /// Returns the symmetric difference of `self` and `other`: the strings that
    /// exactly one of the two accepts.
    ///
    /// Neither automaton is determinized. The result is deterministic when both
    /// are.
    ///
    /// With `n` and `m` the numbers of states of `self` and `other`, the result
    /// has up to `n × m + n × 2^m + m × 2^n` states, and about `n × m` when
    /// both are deterministic. The timeout of the execution profile applies,
    /// and its `max_number_of_states` caps the number of states built,
    /// including those left out of the result because no accept state can be
    /// reached from them.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn symmetric_difference(
        &self,
        other: &FastAutomaton,
    ) -> Result<FastAutomaton, EngineError> {
        match (self.is_empty(), other.is_empty()) {
            (true, true) => return Ok(Self::new_empty()),
            (true, false) => return Ok(other.clone()),
            (false, true) => return Ok(self.clone()),
            (false, false) if self == other => return Ok(Self::new_empty()),
            (false, false) => {}
        }
        let (left, right, shared) = Operand::pair(self, 0, other, 0)?;
        let spanning_set = shared.into_spanning_set();
        // Neither operand is empty, so both start states are live. An
        // operand whose start state is not live would be empty all the same.
        let (start, other_start) = match (left.start, right.start) {
            (Some(start), Some(other_start)) => (start, other_start),
            (None, _) => return Ok(other.clone()),
            (_, None) => return Ok(self.clone()),
        };
        let execution_profile = ExecutionProfile::get();

        let mut difference = Draft::default();
        difference.new_state(left.accepts(start) != right.accepts(other_start));
        // The state of each pair of one state of `self` and one state of
        // `other`: the strings that exactly one of the two states accepts.
        let mut states: AHashMap<(Id, Id), usize> = AHashMap::new();
        states.insert((start, other_start), Draft::START);
        let mut worklist = VecDeque::new();
        worklist.push_back((Draft::START, start, other_start));
        // The pairs met once a word reaches several states of an operand, or
        // none: the two differences.
        let (mut pairs, mut other_pairs) = (Pairs::new(&left, &right), Pairs::new(&right, &left));

        let bases = spanning_set.spanning_ranges_with_rest_len();
        let mut successors = vec![Vec::new(); bases];
        let mut other_successors = vec![Vec::new(); bases];
        // A base for each pair of sets the two states lead to, with all the
        // bases that lead to it.
        let mut targets: Vec<(usize, Condition)> = Vec::new();

        while let Some((state, from, other_from)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(difference.number_of_states())?;

            left.successors(&[from], &mut successors);
            right.successors(&[other_from], &mut other_successors);

            for base in 0..bases {
                let (set, other_set) = (&successors[base], &other_successors[base]);
                if set.is_empty() && other_set.is_empty() {
                    continue;
                }
                match targets.iter_mut().find(|(other_base, _)| {
                    successors[*other_base] == *set && other_successors[*other_base] == *other_set
                }) {
                    Some((_, bases)) => bases.insert_base(base),
                    None => targets.push((base, Condition::single_base(base, &spanning_set))),
                }
            }
            for (base, bases) in targets.drain(..) {
                let (set, other_set) = (&successors[base], &other_successors[base]);
                if let (&[to], &[other_to]) = (set.as_slice(), other_set.as_slice()) {
                    let target = *states.entry((to, other_to)).or_insert_with(|| {
                        let target =
                            difference.new_state(left.accepts(to) != right.accepts(other_to));
                        worklist.push_back((target, to, other_to));
                        target
                    });
                    difference.add_transition(state, target, bases);
                    continue;
                }
                for &to in set {
                    let target = pairs.state(&mut difference, to, other_set);
                    difference.add_transition(state, target, bases.clone());
                }
                for &to in other_set {
                    let target = other_pairs.state(&mut difference, to, set);
                    difference.add_transition(state, target, bases.clone());
                }
            }
        }
        pairs.walk(&mut difference, &spanning_set)?;
        other_pairs.walk(&mut difference, &spanning_set)?;
        Ok(difference.into_automaton(spanning_set))
    }

    /// Returns the symmetric difference of `automata`: the strings that an odd
    /// number of them accepts.
    ///
    /// The order of the automata does not change the language of the result. An
    /// empty iterator gives the automaton that accepts no string, and two
    /// automata are combined as
    /// [`symmetric_difference`](Self::symmetric_difference) does.
    ///
    /// No automaton is determinized first. The result is deterministic when
    /// every automaton is.
    ///
    /// Exponential in the worst case: with `n` the total number of states of
    /// the automata, the result has up to `2^n` states. The timeout of the
    /// execution profile applies, and its `max_number_of_states` caps the
    /// number of states built, including those left out of the result because
    /// no accept state can be reached from them.
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn symmetric_difference_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let automata: Vec<&FastAutomaton> = automata.into_iter().collect();
        match automata[..] {
            [] => Ok(FastAutomaton::new_empty()),
            [automaton] => Ok(automaton.clone()),
            [automaton, other] => automaton.symmetric_difference(other),
            _ => Self::symmetric_difference_over_state_sets(&automata),
        }
    }

    /// Returns the symmetric difference of `automata`, as
    /// [`symmetric_difference_all`](Self::symmetric_difference_all) does,
    /// working on groups of automata in parallel.
    ///
    /// It can take less time than `symmetric_difference_all` on large automata,
    /// and takes more on small ones: measure before choosing it.
    ///
    /// Only available with the `parallel` feature (enabled by default), and not
    /// on `wasm`.
    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn symmetric_difference_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let execution_profile = ExecutionProfile::get();

        // ∅ is the identity.
        let empty = FastAutomaton::new_empty();

        automata
            .into_par_iter()
            .fold(Vec::new, |mut group, automaton| {
                group.push(automaton);
                group
            })
            .map(|group| {
                execution_profile.apply(|| Self::symmetric_difference_over_state_sets(&group))
            })
            .try_reduce(
                || empty.clone(),
                |acc, next| {
                    execution_profile
                        .apply(|| Self::symmetric_difference_over_state_sets(&[&acc, &next]))
                },
            )
    }

    /// The symmetric difference of `automata` in one walk, which
    /// [`symmetric_difference_all`](Self::symmetric_difference_all)
    /// describes.
    ///
    /// The automata share one space of ids, so the sets of states a word
    /// reaches in each of them are one sorted set: the walk is the subset
    /// construction of all the automata side by side, where a set accepts
    /// when an odd number of automata have an accept state in it.
    fn symmetric_difference_over_state_sets(
        automata: &[&FastAutomaton],
    ) -> Result<FastAutomaton, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let mut spanning_set = Cow::Owned(SpanningSet::new_empty());
        for (i, automaton) in automata.iter().enumerate() {
            if i == 0 {
                spanning_set = Cow::Borrowed(&automaton.spanning_set);
            } else if automaton.spanning_set != *spanning_set {
                spanning_set = Cow::Owned(spanning_set.merge(&automaton.spanning_set));
            }
        }
        let mut operands = Vec::with_capacity(automata.len());
        // The end of the ids of each operand.
        let mut ends: Vec<Id> = Vec::with_capacity(automata.len());
        let mut offset = 0;
        for automaton in automata {
            let converter = (automaton.spanning_set != *spanning_set)
                .then(|| ConditionConverter::new(&automaton.spanning_set, &spanning_set))
                .transpose()?;
            operands.push(Operand::new(automaton, offset, converter.as_ref())?);
            offset += automaton.transitions.len();
            ends.push(offset as Id);
        }
        let accepts = |set: &[Id]| {
            let accepting = operands.iter().zip(parts(set, &ends));
            accepting
                .filter(|(operand, part)| operand.is_accepting(part))
                .count()
                % 2
                == 1
        };

        // The sets met, back to back.
        let mut sets: Vec<Id> = operands
            .iter()
            .filter_map(|operand| operand.start)
            .collect();
        let mut difference = Draft::default();
        difference.new_state(accepts(&sets));
        let mut states: AHashMap<Vec<Id>, usize> = AHashMap::new();
        states.insert(sets.clone(), Draft::START);
        let mut worklist = VecDeque::new();
        worklist.push_back((Draft::START, 0..sets.len()));
        // What is left of each operand where a word only reaches states of
        // that operand: the operand minus nothing, which is not determinized.
        let mut alone: Vec<Pairs> = operands
            .iter()
            .map(|operand| Pairs::new(operand, operand))
            .collect();

        let bases = spanning_set.spanning_ranges_with_rest_len();
        let mut successors = vec![Vec::new(); bases];
        let mut operand_successors = vec![Vec::new(); bases];
        // A base for each set the current one leads to, with all the bases
        // that lead to it.
        let mut targets: Vec<(usize, Condition)> = Vec::new();

        while let Some((state, set)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(difference.number_of_states())?;

            successors.iter_mut().for_each(Vec::clear);
            for (operand, part) in operands.iter().zip(parts(&sets[set], &ends)) {
                if part.is_empty() {
                    continue;
                }
                operand.successors(part, &mut operand_successors);
                // The ids of an operand follow those of the previous ones.
                for (all, of_operand) in successors.iter_mut().zip(&operand_successors) {
                    all.extend_from_slice(of_operand);
                }
            }

            for base in 0..bases {
                let set = &successors[base];
                if set.is_empty() {
                    continue;
                }
                match targets
                    .iter_mut()
                    .find(|(other_base, _)| successors[*other_base] == *set)
                {
                    Some((_, bases)) => bases.insert_base(base),
                    None => targets.push((base, Condition::single_base(base, &spanning_set))),
                }
            }
            for (base, bases) in targets.drain(..) {
                let set = &successors[base];
                let operand = ends.partition_point(|end| *end <= set[0]);
                if set[set.len() - 1] < ends[operand] {
                    for &to in set {
                        let target = alone[operand].state(&mut difference, to, &[]);
                        difference.add_transition(state, target, bases.clone());
                    }
                    continue;
                }
                let target = match states.get(set.as_slice()) {
                    Some(target) => *target,
                    None => {
                        let target = difference.new_state(accepts(set));
                        states.insert(set.clone(), target);
                        worklist.push_back((target, sets.len()..sets.len() + set.len()));
                        sets.extend_from_slice(set);
                        target
                    }
                };
                difference.add_transition(state, target, bases);
            }
        }
        for pairs in &mut alone {
            pairs.walk(&mut difference, &spanning_set)?;
        }
        Ok(difference.into_automaton(spanning_set.into_owned()))
    }
}

/// The states of each operand in `set`, a sorted set of states of operands
/// that share one space of ids: `ends` holds the end of the ids of each.
fn parts<'a>(mut set: &'a [Id], ends: &'a [Id]) -> impl Iterator<Item = &'a [Id]> {
    ends.iter().map(move |end| {
        let (part, others) = set.split_at(set.partition_point(|id| id < end));
        set = others;
        part
    })
}

/// A difference as its walk builds it. The pairs that reach no accept state
/// are common, since they are the strings the subtrahend covers: as entries
/// of a list they are left out of the automaton, which costs less than
/// removing them from it.
#[derive(Default)]
struct Draft {
    /// Whether each state accepts.
    accept: Vec<bool>,
    /// `(from, to, bases)`.
    transitions: Vec<(usize, usize, Condition)>,
}

impl Draft {
    /// The start state: the first one to be added.
    const START: usize = 0;

    #[inline]
    fn number_of_states(&self) -> usize {
        self.accept.len()
    }

    #[inline]
    fn new_state(&mut self, accept: bool) -> usize {
        self.accept.push(accept);
        self.accept.len() - 1
    }

    #[inline]
    fn add_transition(&mut self, from: usize, to: usize, bases: Condition) {
        self.transitions.push((from, to, bases));
    }

    /// The automaton of the states that reach an accept state, over
    /// `spanning_set`. All of them are reached from the start state.
    fn into_automaton(self, spanning_set: SpanningSet) -> FastAutomaton {
        // The predecessors of each state, back to back: those of `state`
        // are at `starts[state]..starts[state + 1]`.
        let mut starts = vec![0; self.accept.len() + 1];
        for (_, to, _) in &self.transitions {
            starts[*to] += 1;
        }
        let mut end = 0;
        for start in &mut starts {
            end += *start;
            *start = end;
        }
        let mut predecessors = vec![0; end];
        for (from, to, _) in &self.transitions {
            starts[*to] -= 1;
            predecessors[starts[*to]] = *from;
        }

        let mut live = self.accept.clone();
        let mut worklist: Vec<usize> = (0..live.len()).filter(|&state| live[state]).collect();
        while let Some(state) = worklist.pop() {
            for &from in &predecessors[starts[state]..starts[state + 1]] {
                if !live[from] {
                    live[from] = true;
                    worklist.push(from);
                }
            }
        }

        let mut automaton = FastAutomaton::new_empty();
        if live.first() != Some(&true) {
            return automaton;
        }
        automaton.spanning_set = spanning_set;
        let states: Vec<State> = (0..live.len())
            .map(|state| match state {
                Self::START => automaton.start_state,
                _ if live[state] => automaton.new_state(),
                _ => usize::MAX,
            })
            .collect();
        for state in (0..live.len()).filter(|&state| self.accept[state]) {
            automaton.accept(states[state]);
        }
        for (from, to, bases) in self.transitions {
            if live[from] && live[to] {
                automaton.add_owned_transition(states[from], states[to], bases);
            }
        }
        automaton
    }
}

/// The pairs of a state of an operand, `left`, and the set of states of the
/// other, `right`, that the same word reaches: the states of `left` minus
/// `right`, which [`difference`](FastAutomaton::difference) describes.
struct Pairs<'a> {
    left: &'a Operand,
    right: &'a Operand,
    /// The sets of the pairs, back to back.
    sets: Vec<Id>,
    /// The state of each pair, keyed by the state of `left` followed by the
    /// set of states of `right`.
    states: AHashMap<Vec<Id>, usize>,
    /// The pairs whose transitions are still to be added: their state, their
    /// state of `left` and their set in `sets`.
    worklist: VecDeque<(usize, Id, Range<usize>)>,
    key: Vec<Id>,
}

impl<'a> Pairs<'a> {
    fn new(left: &'a Operand, right: &'a Operand) -> Self {
        Pairs {
            left,
            right,
            sets: Vec::new(),
            states: AHashMap::new(),
            worklist: VecDeque::new(),
            key: Vec::new(),
        }
    }

    /// The state of the pair of `from` and `set`, which is added to
    /// `difference` if the pair was not met yet.
    fn state(&mut self, difference: &mut Draft, from: Id, set: &[Id]) -> usize {
        self.key.clear();
        self.key.push(from);
        self.key.extend_from_slice(set);
        if let Some(state) = self.states.get(self.key.as_slice()) {
            return *state;
        }
        let state = difference.new_state(self.left.accepts(from) && !self.right.is_accepting(set));
        self.states.insert(self.key.clone(), state);
        self.worklist
            .push_back((state, from, self.sets.len()..self.sets.len() + set.len()));
        self.sets.extend_from_slice(set);
        state
    }

    /// Adds to `difference` the transitions of the pairs met, and the pairs
    /// they lead to.
    fn walk(
        &mut self,
        difference: &mut Draft,
        spanning_set: &SpanningSet,
    ) -> Result<(), EngineError> {
        let execution_profile = ExecutionProfile::get();

        let mut successors = vec![Vec::new(); spanning_set.spanning_ranges_with_rest_len()];
        // The pairs one transition of `left` leads to: the state of the
        // pair, a base that leads to it, and all the bases that do.
        let mut targets: Vec<(usize, usize, Condition)> = Vec::new();

        while let Some((state, from, set)) = self.worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(difference.number_of_states())?;

            self.right.successors(&self.sets[set], &mut successors);

            for (condition, to) in self.left.transitions_from(from) {
                for base in condition.iter_set_bits() {
                    let set = &successors[base];
                    if let Some((_, _, bases)) = targets
                        .iter_mut()
                        .find(|(_, other_base, _)| successors[*other_base] == *set)
                    {
                        bases.insert_base(base);
                        continue;
                    }
                    let target = self.state(difference, *to, set);
                    targets.push((target, base, Condition::single_base(base, spanning_set)));
                }
                for (target, _, bases) in targets.drain(..) {
                    difference.add_transition(state, target, bases);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::error::EngineError;
    use crate::execution_profile::ExecutionProfileBuilder;
    use crate::fast_automaton::FastAutomaton;
    use crate::regex::RegularExpression;

    // `totalize` adds a `crash_state` with a total self-loop, so the complement
    // of a finite language is infinite (it matches arbitrarily long strings via
    // the crash-state loop).
    #[test]
    fn complement_of_finite_is_infinite() {
        let mut a = RegularExpression::parse("abc", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        a.complement().unwrap();

        assert!(!a.is_match("abc"), "complement must not match 'abc'");
        assert!(a.is_match("x"));
        assert!(a.is_match("xx"));
        assert!(a.is_match("xxxxxxxxxx"));
    }

    fn automaton(pattern: &str) -> FastAutomaton {
        RegularExpression::new(pattern)
            .unwrap()
            .to_automaton()
            .unwrap()
    }

    #[test]
    fn symmetric_difference_all_keeps_what_an_odd_number_accepts() {
        let (a, b, c) = (
            automaton("a|ab|abc"),
            automaton("ab|abc|b"),
            automaton("abc|b|c"),
        );

        assert!(
            FastAutomaton::symmetric_difference_all([])
                .unwrap()
                .is_empty()
        );
        assert!(
            FastAutomaton::symmetric_difference_all([&a])
                .unwrap()
                .equivalent(&a)
                .unwrap()
        );
        // An automaton given twice cancels itself.
        assert!(
            FastAutomaton::symmetric_difference_all([&a, &b, &a])
                .unwrap()
                .equivalent(&b)
                .unwrap()
        );

        // `a` and `c` are in one automaton, `abc` in the three of them.
        let expected = automaton("a|abc|c");
        for automata in [[&a, &b, &c], [&c, &a, &b], [&b, &c, &a]] {
            let all = FastAutomaton::symmetric_difference_all(automata).unwrap();
            assert!(all.equivalent(&expected).unwrap());
            #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
            assert!(
                FastAutomaton::symmetric_difference_all_par(automata)
                    .unwrap()
                    .equivalent(&expected)
                    .unwrap()
            );
        }
    }

    // Combined two by two, each intermediate result is walked again, and
    // these 16 automata of 2 states take thousands of states, or hundreds of
    // thousands when two intermediate results are combined with each other.
    #[test]
    fn symmetric_difference_all_walks_all_the_automata_together() {
        let automata: Vec<FastAutomaton> = ('a'..='p')
            .map(|letter| automaton(&format!(".*{letter}")))
            .collect();
        // No string ends with two letters: an odd number is one.
        let expected = FastAutomaton::union_all(&automata).unwrap();

        ExecutionProfileBuilder::new()
            .max_number_of_states(100)
            .build()
            .run(|| {
                // The states reached after each last letter, and the start.
                let all = FastAutomaton::symmetric_difference_all(&automata).unwrap();
                assert_eq!(17, all.number_of_states());
                assert!(all.is_deterministic());
                assert!(all.equivalent(&expected).unwrap());

                #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
                assert!(
                    FastAutomaton::symmetric_difference_all_par(&automata)
                        .unwrap()
                        .equivalent(&expected)
                        .unwrap()
                );
            });
    }

    // Past the first letter, a word only reaches states of one automaton,
    // which is followed as it is: none can be determinized under the cap.
    #[test]
    fn symmetric_difference_all_does_not_determinize_an_automaton_left_alone() {
        let automata: Vec<FastAutomaton> = ['x', 'y', 'z']
            .map(|letter| automaton(&format!("{letter}(a|b)*a(a|b){{11}}")))
            .into();
        let expected = FastAutomaton::union_all(&automata).unwrap();

        let all = ExecutionProfileBuilder::new()
            .max_number_of_states(200)
            .build()
            .run(|| {
                for automaton in &automata {
                    assert_eq!(
                        automaton.determinize().unwrap_err(),
                        EngineError::AutomatonHasTooManyStates
                    );
                }
                FastAutomaton::symmetric_difference_all(&automata).unwrap()
            });
        assert!(!all.is_deterministic());
        // Compared outside of the cap, which the comparison would exceed.
        assert!(all.equivalent(&expected).unwrap());
    }

    // empty.complement() = Σ*.
    #[test]
    fn complement_of_empty_is_total() {
        let mut a = FastAutomaton::new_empty();
        a.complement().unwrap();
        assert!(a.is_total(), "complement of ∅ must be Σ*");
    }
}
