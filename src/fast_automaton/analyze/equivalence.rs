use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    pub fn is_equivalent_of(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() != other.is_empty() && self.is_total() != other.is_total() {
            return Ok(false);
        } else if self == other {
            return Ok(true);
        }

        let mut other_complement = other.determinize()?;
        other_complement.complement()?;

        if self.has_intersection(&other_complement)? {
            return Ok(false);
        }

        let mut self_complement = self.determinize()?;
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

        let regex_1 = RegularExpression::new("cd").unwrap();
        let regex_2 = RegularExpression::new("cd").unwrap();
        assert_equivalent(&regex_1, &regex_2, true);

        let regex_1 = RegularExpression::new("test.*other").unwrap();
        let regex_2 = RegularExpression::new("test.*othew").unwrap();

        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::new("test.{0,50}other").unwrap();
        let regex_2 = RegularExpression::new("test.{0,49}other").unwrap();

        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::new("[0]").unwrap();
        let regex_2 = RegularExpression::new("[01]").unwrap();
        assert_equivalent(&regex_1, &regex_2, false);

        let regex_1 = RegularExpression::new("(b+a+)*").unwrap();
        let regex_2 = RegularExpression::new("(b[a-b]*a)?").unwrap();
        assert_equivalent(&regex_1, &regex_2, true);

        Ok(())
    }

    fn assert_equivalent(regex_1: &RegularExpression, regex_2: &RegularExpression, expected: bool) {
        println!("{regex_1} and {regex_2}");
        let automaton_1 = regex_1.to_automaton().unwrap();
        assert_eq!(true, automaton_1.is_equivalent_of(&automaton_1).unwrap());

        let automaton_2 = regex_2.to_automaton().unwrap();
        assert_eq!(true, automaton_2.is_equivalent_of(&automaton_2).unwrap());

        assert_eq!(
            expected,
            automaton_1.is_equivalent_of(&automaton_2).unwrap()
        );
    }
}
