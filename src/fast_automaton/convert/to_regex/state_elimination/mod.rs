use super::*;

mod builder;
mod eliminate;

struct Gnfa {
    start_state: usize,
    accept_state: usize,
    transitions: Vec<IntMap<usize, RegularExpression>>,
    transitions_in: IntMap<usize, IntSet<usize>>,
    removed_states: IntSet<usize>,
    empty: bool,
}

impl Display for Gnfa {
    fn fmt(&self, sb: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(sb, "digraph GNFA {{")?;
        writeln!(sb, "\trankdir = LR;")?;
        for from_state in self.all_states_iter() {
            write!(sb, "\t{from_state}")?;
            if self.accept_state == from_state {
                writeln!(sb, "\t[shape=doublecircle,label=\"{from_state}\"];")?;
            } else {
                writeln!(sb, "\t[shape=circle,label=\"{from_state}\"];")?;
            }

            if self.start_state == from_state {
                writeln!(sb, "\tinitial [shape=plaintext,label=\"\"];")?;
                writeln!(sb, "\tinitial -> {from_state}")?;
            }
            for (regex, to_state) in self.transitions_from_iter(from_state) {
                writeln!(sb, "\t{from_state} -> {to_state} [label=\"{regex}\"]")?;
            }
        }
        write!(sb, "}}")
    }
}

impl Gnfa {
    fn get_transition(&self, from_state: State, to_state: State) -> Option<&RegularExpression> {
        self.transitions.get(from_state)?.get(&to_state)
    }

    #[inline]
    fn all_states_iter(&self) -> impl Iterator<Item = State> + '_ {
        (0..self.transitions.len()).filter(|s| !self.removed_states.contains(s))
    }

    /// Incoming transitions by reference: `transitions_in` gives the exact
    /// predecessor set, so each edge label is one direct map lookup.
    fn transitions_to_iter(
        &self,
        state: State,
    ) -> impl Iterator<Item = (State, &RegularExpression)> {
        self.transitions_in
            .get(&state)
            .into_iter()
            .flatten()
            .filter_map(move |&from_state| {
                self.get_transition(from_state, state)
                    .map(|regex| (from_state, regex))
            })
    }

    /// Outgoing transitions by reference.
    #[inline]
    fn transitions_from_iter(
        &self,
        state: State,
    ) -> impl Iterator<Item = (&RegularExpression, State)> {
        self.transitions[state]
            .iter()
            .filter(|(s, _)| !self.removed_states.contains(*s))
            .map(|(s, c)| (c, *s))
    }

    #[inline]
    fn has_self_loop(&self, state: State) -> bool {
        self.get_transition(state, state).is_some()
    }
}

pub(super) fn convert_to_regex(
    automaton: &FastAutomaton,
) -> Result<RegularExpression, EngineError> {
    let mut gnfa = Gnfa::from_automaton(automaton)?;
    gnfa.convert()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CharRange;
    use proptest::prelude::*;
    use regex_charclass::char::Char;

    /// Every produced pattern must be byte-identical to what full
    /// re-scoring produces: the score cache must never go stale.
    fn assert_same_pattern_as_reference(automaton: &FastAutomaton) {
        let incremental = Gnfa::from_automaton(automaton).unwrap().convert().unwrap();
        let reference = Gnfa::from_automaton(automaton)
            .unwrap()
            .convert_reference()
            .unwrap();
        assert_eq!(
            incremental.to_string(),
            reference.to_string(),
            "incremental scoring changed the elimination order"
        );
    }

    #[test]
    fn incremental_scoring_matches_full_rescoring() {
        let patterns = [
            "abc",
            ".*de",
            "(a*ba*)*",
            ".*u(ab|de)",
            "a(bcfe|bcdg|mkv)*(abc){2,3}",
            "(aad|ads|a)*abc.*def(x|q)",
            "(a|b)*a(a|b){3}",
            "[a-z]{1,6}",
            "(ab|xy){2}",
            "x*|(xxx)*|y",
            "a{0,3}b{2}(c|d)?",
        ];
        for pattern in patterns {
            let automaton = RegularExpression::new(pattern)
                .unwrap()
                .to_automaton()
                .unwrap();
            assert_same_pattern_as_reference(&automaton);
            assert_same_pattern_as_reference(&automaton.determinize().unwrap());
        }
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
                for (from, to, range) in edges {
                    automaton
                        .add_transition_from_range(
                            from % number_of_states,
                            to % number_of_states,
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
        fn incremental_scoring_matches_full_rescoring_on_random_automata(
            automaton in arb_automaton()
        ) {
            let incremental = Gnfa::from_automaton(&automaton).unwrap().convert().unwrap();
            let reference = Gnfa::from_automaton(&automaton)
                .unwrap()
                .convert_reference()
                .unwrap();
            prop_assert_eq!(incremental.to_string(), reference.to_string());
        }
    }

    #[test]
    fn test_state_elimination() -> Result<(), String> {
        test_correct("abc");
        test_correct(".*de");
        test_correct(".*def");
        test_correct("(a*ba*)*");
        test_correct(".*u(ab|d)");
        test_correct(".*u(ab|de)");
        Ok(())
    }

    fn test_correct(pattern: &str) {
        println!("Pattern: {pattern}");

        let automaton = RegularExpression::new(pattern)
            .unwrap()
            .to_automaton()
            .unwrap();

        let regex = Gnfa::from_automaton(&automaton).unwrap().convert().unwrap();
        println!("-> {regex}");

        let new_automaton = regex.to_automaton().unwrap();

        assert!(automaton.equivalent(&new_automaton).unwrap());

        let automaton = automaton.determinize().unwrap().into_owned();

        let regex = Gnfa::from_automaton(&automaton).unwrap().convert().unwrap();
        println!("-> {regex}");

        let new_automaton = regex.to_automaton().unwrap();

        assert!(automaton.equivalent(&new_automaton).unwrap());
    }
}
