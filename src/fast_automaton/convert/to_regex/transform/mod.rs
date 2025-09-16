use crate::fast_automaton::{
    FastAutomaton, convert::to_regex::transform::shape::dotstar::dot_star,
};

mod shape;

const TRANSFORM_FUNCTION: &[fn(&FastAutomaton) -> FastAutomaton] = &[dot_star];

pub fn transform(automaton: &FastAutomaton) -> FastAutomaton {
    let mut automaton = automaton.clone();
    for transform in TRANSFORM_FUNCTION {
        automaton = transform(&automaton);
    }

    automaton
}

#[cfg(test)]
mod tests {
    use crate::{
        fast_automaton::convert::to_regex::transform::transform, regex::RegularExpression,
    };

    #[test]
    fn test_equivalence() -> Result<(), String> {
        assert_equivalent("abc");
        assert_equivalent(".*abc");
        assert_equivalent(".*abc.*def");
        assert_equivalent(".*abc.*def(ab|fr)");
        assert_equivalent(".*abc.*def(ab|fr).*mpa");

        Ok(())
    }

    fn assert_equivalent(pattern: &str) {
        let before = RegularExpression::parse(pattern, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let before = before.determinize().unwrap();

        let after = transform(&before);

        assert!(before.equivalent(&after).unwrap());
    }
}
