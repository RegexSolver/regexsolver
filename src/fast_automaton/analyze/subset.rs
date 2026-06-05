use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    /// Returns `true` if all strings accepted by `self` are also accepted by `other`.
    ///
    /// A non-deterministic `other` is determinized internally — unless the
    /// execution profile disables implicit determinization, in which case
    /// [`EngineError::DeterministicAutomatonRequired`] is returned.
    pub fn subset(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() || other.is_total() || self == other {
            return Ok(true);
        } else if other.is_empty() {
            return Ok(false);
        } else if self.is_total() {
            // self ⊆ other iff Σ* ⊆ other iff other = Σ*. We already failed
            // the cheap `other.is_total()` check above; that check is sound
            // but conservative on NFAs, so retry on the determinized form.
            return Ok(other.determinize_implicit()?.is_total());
        }

        let mut other = other.determinize_implicit()?.into_owned();
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
