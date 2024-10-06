use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    pub fn is_subset_of(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() || other.is_total() || self == other {
            return Ok(true);
        } else if other.is_empty() || self.is_total() {
            return Ok(false);
        }

        let mut other = other.determinize()?;
        other.complement()?;

        Ok(!self.has_intersection(&other)?)
    }
}

#[cfg(test)]
mod tests {

    use crate::regex::RegularExpression;

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

        let regex1 = RegularExpression::new("test.*other").unwrap();
        let regex2 = RegularExpression::new("test.*othew").unwrap();

        assert_subset(&regex1, &regex2, false, false);

        let regex1 = RegularExpression::new("test.{0,50}other").unwrap();
        let regex2 = RegularExpression::new("test.{0,49}other").unwrap();

        assert_subset(&regex1, &regex2, false, true);

        let regex1 = RegularExpression::new("(abc|def)").unwrap();
        let regex2 = RegularExpression::new("(abc|def|xyz)").unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::new("[0]").unwrap();
        let regex2 = RegularExpression::new("[01]").unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::new("a.*b.*c.*").unwrap();
        let regex2 = RegularExpression::new("a.*b.*").unwrap();

        assert_subset(&regex1, &regex2, true, false);

        let regex1 = RegularExpression::new("1..").unwrap();
        let regex2 = RegularExpression::new("...").unwrap();

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
        assert_eq!(true, automaton_1.is_subset_of(&automaton_1).unwrap());

        let automaton_2 = regex_2.to_automaton().unwrap();
        assert_eq!(true, automaton_2.is_subset_of(&automaton_2).unwrap());

        assert_eq!(
            expected_1_2,
            automaton_1.is_subset_of(&automaton_2).unwrap()
        );
        assert_eq!(
            expected_2_1,
            automaton_2.is_subset_of(&automaton_1).unwrap()
        );
    }
}
