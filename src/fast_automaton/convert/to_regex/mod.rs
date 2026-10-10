use super::*;

#[cfg(feature = "neural-synthesis")]
pub mod neural_synthesis;
mod state_elimination;

/// A way to convert an automaton to a [`RegularExpression`], for
/// [`FastAutomaton::to_regex_with`] and
/// [`Term::to_regex_with`](crate::Term::to_regex_with).
///
/// The crate provides [`StateElimination`], what
/// [`FastAutomaton::to_regex`] uses, and, with the `neural-synthesis` feature,
/// `neural_synthesis::NeuralSynthesizer`, which asks a learned model for a
/// simpler regex.
pub trait RegexSynthesizer {
    /// A regular expression describing exactly the language of `automaton`.
    fn synthesize(&self, automaton: &FastAutomaton) -> Result<RegularExpression, EngineError>;
}

/// Conversion by state elimination, what [`FastAutomaton::to_regex`] uses:
/// the automaton's states are removed one at a time, each removal folding the
/// paths through the state into regexes on the transitions around it, until
/// one transition from the start to acceptance remains. The result is always
/// exact, but often long and hard to read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StateElimination;

impl RegexSynthesizer for StateElimination {
    fn synthesize(&self, automaton: &FastAutomaton) -> Result<RegularExpression, EngineError> {
        automaton.to_regex()
    }
}

impl FastAutomaton {
    /// Converts the automaton to a [`RegularExpression`] by state
    /// elimination (see [`StateElimination`]).
    ///
    /// The result describes exactly the automaton's language, but can be long
    /// and hard to read; [`to_regex_with`](Self::to_regex_with) takes other
    /// ways to convert it.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states()))]
    pub fn to_regex(&self) -> Result<RegularExpression, EngineError> {
        state_elimination::convert_to_regex(self)
    }

    /// Converts the automaton to a [`RegularExpression`] with `synthesizer`,
    /// for instance [`StateElimination`] (as [`to_regex`](Self::to_regex)
    /// does) or, with the `neural-synthesis` feature, a
    /// `neural_synthesis::NeuralSynthesizer`.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states()))]
    pub fn to_regex_with<S: RegexSynthesizer + ?Sized>(
        &self,
        synthesizer: &S,
    ) -> Result<RegularExpression, EngineError> {
        synthesizer.synthesize(self)
    }
}

#[cfg(test)]
mod tests {
    use ::regex::Regex;

    use super::*;

    #[test]
    fn test_convert() -> Result<(), String> {
        assert_convert(".*u(ab|de)");
        assert_convert(".*sf.*uif(ab|de)");

        assert_convert("(a+|,)*");
        assert_convert("((ab)*,(cd)*)*");
        assert_convert("(a*,a*,a*)*");
        assert_convert("(a*,a*)*");

        assert_convert("(ac|ads|a)*");
        assert_convert(".*sf");
        assert_convert(".*sf.*uif(ab|de)");

        assert_convert(".*ab");

        assert_convert("(abc|fg){2}");
        assert_convert("a{2,3}");
        assert_convert("a(bcfe|bcdg|mkv){1,2}");
        assert_convert("a(bcfe|bcdg|mkv){5,6}");
        assert_convert("a(bcfe|bcdg|mkv){0,8}");
        assert_convert("a(bcfe|bcdg|mkv){3,20}");

        assert_convert("a+");

        assert_convert("a*bc*");
        assert_convert("(abc)*(a|ab)");
        assert_convert("a(bcfe|bcdg)*");
        assert_convert(".*abc");
        assert_convert(".*abc.*def");
        assert_convert("a(bcfe|bcdg|mkv)*");

        assert_convert("(bc|a)*");

        assert_convert(".*a(bc|d)");
        assert_convert("abc.*def.*uif(ab|de)");

        assert_convert("(b+a+)*");
        assert_convert("a+(ba+)*");
        Ok(())
    }

    fn assert_convert(regex: &str) {
        let input_regex = RegularExpression::parse(regex, false).unwrap();
        println!("IN                     : {}", input_regex);
        let input_automaton = input_regex.to_automaton().unwrap();

        let output_regex = input_automaton.to_regex().unwrap();
        println!("OUT (non deterministic): {}", output_regex);
        let output_automaton = output_regex.to_automaton().unwrap();

        assert!(input_automaton.equivalent(&output_automaton).unwrap());

        let input_automaton = input_automaton.determinize().unwrap();

        let output_regex = input_automaton.to_regex().unwrap();
        println!("OUT (deterministic)    : {}", output_regex);
        let output_automaton = output_regex.to_automaton().unwrap();

        assert!(input_automaton.equivalent(&output_automaton).unwrap());
    }

    #[test]
    fn test_convert_after_operation_1() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("(ab|cd)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("ab", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = automaton2.determinize().unwrap();

        let result = automaton1.difference(&automaton2).unwrap();

        result.print_dot();

        let output_regex = result.to_regex().unwrap();
        assert_eq!("cd", output_regex.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_2() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("a*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("b*", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let result = automaton1.intersection(&automaton2).unwrap();

        result.print_dot();

        let output_regex = result.to_regex().unwrap();
        assert_eq!("", output_regex.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_3() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("x*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(xxx)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = automaton2.determinize().unwrap();

        let result = automaton1.difference(&automaton2).unwrap();
        result.print_dot();

        let result = result.to_regex().unwrap();

        assert_eq!("x(x{3})*x?", result.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_4() -> Result<(), String> {
        let automaton1 = RegularExpression::parse(".*abc.*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse(".*def.*", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let result = automaton1.intersection(&automaton2).unwrap();

        let result = result.to_regex().unwrap();

        assert_eq!(".*(abc.*def|def.*abc).*", result.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_5() -> Result<(), String> {
        let automaton = RegularExpression::parse(".*abc.*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let mut automaton = automaton.determinize().unwrap().into_owned();

        automaton.complement().unwrap();

        let result = format!("^{}$", automaton.to_regex().unwrap());

        println!("{result}");

        let result = Regex::new(&result).unwrap();

        assert!(!result.is_match("abc"));
        assert!(!result.is_match("2374abc012"));

        assert!(result.is_match("bc"));
        assert!(result.is_match("237a4bc012"));

        Ok(())
    }

    #[test]
    fn test_automaton() -> Result<(), String> {
        let automaton = RegularExpression::parse("a*ba*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();

        let automaton1 = RegularExpression::parse("(a*ba*)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton1.print_dot();

        automaton1.determinize().unwrap().print_dot();

        // (a*b[ab]*)?
        // a*b+a+b+

        let automaton2 = RegularExpression::parse("(a*b[ab]*)?", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        assert!(automaton1.equivalent(&automaton2).unwrap());

        Ok(())
    }
}
