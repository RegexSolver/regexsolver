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
            for (regex, to_state) in self.transitions_from_vec(from_state) {
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

    fn transitions_to_vec(&self, state: State) -> Vec<(State, RegularExpression)> {
        let mut in_transitions = vec![];
        for from_state in self.transitions_in.get(&state).unwrap_or(&IntSet::new()) {
            for (condition, to_state) in self.transitions_from_vec(*from_state) {
                if to_state == state {
                    in_transitions.push((*from_state, condition));
                    break;
                }
            }
        }
        in_transitions
    }

    #[inline]
    fn transitions_from_vec(&self, state: State) -> Vec<(RegularExpression, State)> {
        self.transitions[state]
            .iter()
            .map(|(s, c)| (c.clone(), *s))
            .filter(|s| !self.removed_states.contains(&s.1))
            .collect()
    }

    #[inline]
    fn has_self_loop(&self, state: State) -> bool {
        self.get_transition(state, state).is_some()
    }
}

pub(super) fn convert_to_regex(automaton: &FastAutomaton) -> RegularExpression {
    let mut gnfa = Gnfa::from_automaton(automaton);
    gnfa.convert()
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let regex = Gnfa::from_automaton(&automaton).convert();
        println!("-> {regex}");

        let new_automaton = regex.to_automaton().unwrap();

        assert!(automaton.equivalent(&new_automaton).unwrap());

        let automaton = automaton.determinize().unwrap().into_owned();

        let regex = Gnfa::from_automaton(&automaton).convert();
        println!("-> {regex}");

        let new_automaton = regex.to_automaton().unwrap();

        assert!(automaton.equivalent(&new_automaton).unwrap());
    }
}
