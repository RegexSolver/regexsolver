use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    /// Returns `true` if both automata accept the same language.
    ///
    /// Non-deterministic operands are determinized internally, unless the
    /// execution profile disables implicit determinization, in which case
    /// [`EngineError::DeterministicAutomatonRequired`] is returned.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn equivalent(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() != other.is_empty() && self.is_total() != other.is_total() {
            return Ok(false);
        } else if self == other {
            return Ok(true);
        }

        let mut other_complement = other.determinize_implicit()?.into_owned();
        other_complement.complement()?;

        if self.has_intersection(&other_complement)? {
            return Ok(false);
        }

        let mut self_complement = self.determinize_implicit()?.into_owned();
        self_complement.complement()?;

        Ok(!self_complement.has_intersection(other)?)
    }
}

#[cfg(test)]
mod tests {

    use crate::regex::RegularExpression;

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
