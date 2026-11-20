use crate::{error::EngineError, execution_profile::ExecutionProfile};

use super::state_sets::{Marks, Operand, SetTrie};
use super::*;

impl FastAutomaton {
    /// Returns `true` if every string that `self` accepts is also accepted by
    /// `other`.
    ///
    /// Neither automaton is determinized.
    ///
    /// Exponential in the size of `other` in the worst case: the timeout of the
    /// execution profile applies, and its `max_number_of_states` caps the
    /// number of pairs visited, each made of a state of `self` and a set of
    /// states of `other`.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn subset(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() || other.is_total() || self == other {
            return Ok(true);
        } else if other.is_empty() {
            return Ok(false);
        }
        self.subset_over_state_sets(other)
    }

    /// The antichain algorithm of De Wulf, Doyen, Henzinger & Raskin,
    /// *Antichains: A New Algorithm for Checking Universality of Finite
    /// Automata* (CAV 2006, section 5), in the forward form that Bonchi & Pous
    /// describe (section 4).
    ///
    /// The walk visits the pairs `(x, Y)` of a state of `self` and the set of
    /// `other`'s states that one word reaches. A pair is a counterexample
    /// when `x` accepts and `Y` does not, or when `Y` is empty (`x` is live).
    /// It is skipped when `(x, Y')` was met for some `Y' ⊆ Y`. The pairs
    /// kept are not a strict antichain: a pair met earlier with a superset
    /// of `Y` is not removed, which costs memory but skips the same pairs.
    ///
    /// Inclusion is also the equivalence `self ∪ other ≡ other`, but
    /// [`equivalent_over_state_sets`](Self::equivalent_over_state_sets)
    /// would skip no more pairs: between two automata with no state in
    /// common, the two criteria match (Bonchi & Pous, section 4.2.2).
    pub(super) fn subset_over_state_sets(
        &self,
        other: &FastAutomaton,
    ) -> Result<bool, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let (left, right, shared) = Operand::pair(self, 0, other, 0)?;
        let bases = shared.spanning_set().spanning_ranges_with_rest_len();

        let Some(start) = left.start else {
            return Ok(true);
        };
        let Some(right_start) = right.start else {
            return Ok(false);
        };
        if left.accepts(start) && !right.accepts(right_start) {
            return Ok(false);
        }

        // The sets each state of `self` was met with, one collection per
        // state, created on first use.
        let mut visited = SetTrie::new();
        let mut roots: Vec<Option<usize>> = vec![None; self.transitions.len()];
        let mut worklist = VecDeque::new();
        let root = visited.new_root();
        roots[start as usize] = Some(root);
        visited.insert(root, &[right_start], ());
        // The sets of the queued pairs, back to back.
        let mut sets = vec![right_start];
        worklist.push_back((start, 0..1));

        let mut successors = vec![Vec::new(); bases];
        let mut query = Marks::new(other.transitions.len());
        // The nodes `has_subset_of` has to visit, reused by each call.
        let mut stack = Vec::new();
        let mut pairs = 1;

        while let Some((state, set)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(pairs)?;

            right.successors(&sets[set], &mut successors);

            for (condition, to_state) in left.transitions_from(state) {
                for base in condition.iter_set_bits() {
                    let set = &successors[base];
                    if set.is_empty() || (left.accepts(*to_state) && !right.is_accepting(set)) {
                        return Ok(false);
                    }
                    let root = *roots[*to_state as usize].get_or_insert_with(|| visited.new_root());
                    set.iter().for_each(|&id| {
                        query.insert(id);
                    });
                    let met = visited.has_subset_of(root, &query, &mut stack);
                    query.clear();
                    if met {
                        continue;
                    }
                    visited.insert(root, set, ());
                    worklist.push_back((*to_state, sets.len()..sets.len() + set.len()));
                    sets.extend_from_slice(set);
                    pairs += 1;
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {

    use crate::CharRange;
    use crate::fast_automaton::FastAutomaton;
    use crate::regex::RegularExpression;
    use regex_charclass::char::Char;

    // One `a` reaches every one of the 20,000 states of `other`, which makes
    // a stored set that deep: looking it up must not recurse per state,
    // which overflowed the stack of a thread.
    #[test]
    fn subset_with_a_large_set_of_states() {
        let character = |c| CharRange::new_from_range(Char::new(c)..=Char::new(c));

        let mut a_star = FastAutomaton::new_empty();
        a_star.accept(0);
        a_star
            .add_transition_from_range(0, 0, &character('a'))
            .unwrap();

        let mut other = FastAutomaton::new_empty();
        other.accept(0);
        other
            .add_transition_from_range(0, 0, &character('b'))
            .unwrap();
        for _ in 0..20_000 {
            let state = other.new_state();
            other.accept(state);
            other
                .add_transition_from_range(0, state, &character('a'))
                .unwrap();
            other
                .add_transition_from_range(state, state, &character('a'))
                .unwrap();
        }

        assert!(a_star.subset(&other).unwrap());
        assert!(!other.subset(&a_star).unwrap());
    }

    #[test]
    fn test_subset() -> Result<(), String> {
        assert_subset(
            &RegularExpression::new_empty(),
            &RegularExpression::new_empty_string(),
            true,
            false,
        );

        assert_subset(
            &RegularExpression::new_total(),
            &RegularExpression::new_empty_string(),
            false,
            true,
        );

        let regex1 = RegularExpression::parse("test.*other", false).unwrap();
        let regex2 = RegularExpression::parse("test.*othew", false).unwrap();

        assert_subset(&regex1, &regex2, false, false);

        let regex1 = RegularExpression::parse("test.{0,50}other", false).unwrap();
        let regex2 = RegularExpression::parse("test.{0,49}other", false).unwrap();

        assert_subset(&regex1, &regex2, false, true);

        let regex1 = RegularExpression::parse("(abc|def)", false).unwrap();
        let regex2 = RegularExpression::parse("(abc|def|xyz)", false).unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::parse("[0]", false).unwrap();
        let regex2 = RegularExpression::parse("[01]", false).unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::parse("a.*b.*c.*", false).unwrap();
        let regex2 = RegularExpression::parse("a.*b.*", false).unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::parse("1..", false).unwrap();
        let regex2 = RegularExpression::parse("...", false).unwrap();

        assert_subset(&regex1, &regex2, true, false);

        Ok(())
    }

    fn assert_subset(
        regex_1: &RegularExpression,
        regex_2: &RegularExpression,
        expected_1_2: bool,
        expected_2_1: bool,
    ) {
        println!("{regex_1} and {regex_2}");
        let automaton_1 = regex_1.to_automaton().unwrap();
        assert!(automaton_1.subset(&automaton_1).unwrap());

        let automaton_2 = regex_2.to_automaton().unwrap();
        assert!(automaton_2.subset(&automaton_2).unwrap());

        assert_eq!(expected_1_2, automaton_1.subset(&automaton_2).unwrap());
        assert_eq!(expected_2_1, automaton_2.subset(&automaton_1).unwrap());
    }
}
