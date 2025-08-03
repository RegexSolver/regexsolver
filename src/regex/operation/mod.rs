use super::*;

mod concat;
mod repeat;
mod simplify;
mod union;

#[cfg(test)]
mod tests {

    use regex_charclass::char::Char;

    use crate::{regex::RegularExpression, CharRange};

    #[test]
    fn test_parse_and_simplify() -> Result<(), String> {
        assert_parse_and_simplify("(xxx)*", "(x{3})*");
        assert_parse_and_simplify("(x*){3}", "x*");
        assert_parse_and_simplify("(x+)?", "x*");
        assert_parse_and_simplify("(x?)+", "x*");
        assert_parse_and_simplify("(x{0,3})+", "x*");
        assert_parse_and_simplify("(x{2,3})+", "x{2,}");
        assert_parse_and_simplify("(x{7,9})+", "(x{7,9})+");
        assert_parse_and_simplify("(x+)*", "x*");
        assert_parse_and_simplify(".*abc", ".*abc");
        assert_parse_and_simplify(".*a(b|cd)", ".*a(b|cd)");
        assert_parse_and_simplify(
            "a(bcfe|bcdg|mkv)*(abc){2,3}(abc){2}",
            "a(bc(dg|fe)|mkv)*(abc){4,5}",
        );
        assert_parse_and_simplify("((abc|fg)abc|(abc|fg)fg)", "(abc|fg){2}");
        assert_parse_and_simplify("(a{2}|a{3})", "a{2,3}");
        assert_parse_and_simplify("(a|b)", "[ab]");
        assert_parse_and_simplify("(ab|a|cd|b|ef)", "(b|ab?|cd|ef)");
        assert_parse_and_simplify("(ab|ab)", "ab");
        assert_parse_and_simplify("(ab)(ab)(ab)", "(ab){3}");
        assert_parse_and_simplify("aaaabbbbbccc", "a{4}b{5}c{3}");
        assert_parse_and_simplify("((ab))?(ab)(((ab)))((((ab)){3}))", "(ab){5,6}");
        assert_parse_and_simplify("(cd|ab)*(ab|cd)*", "(ab|cd)*");
        assert_parse_and_simplify(".*q(ab|ab|abc|ca)x", ".*q(abc?|ca)x");
        assert_parse_and_simplify(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}",
            "(q|(a|ads|a{2}d)*abc.*def.*uif(x|ads|a{2}d)*abc.*oxs.*def(ads|ax|a{2}d)*abc.*def.*ksd){1,100}",
        );

        assert_parse_and_simplify("(a{2,4}){2,4}", "a{4,16}");
        Ok(())
    }

    fn assert_parse_and_simplify(regex: &str, regex_simplified: &str) {
        let regex_parsed = RegularExpression::new(regex).unwrap();
        assert_eq!(regex_simplified, regex_parsed.to_string());
    }

    #[test]
    fn test_repeat_simplify() -> Result<(), String> {
        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            2,
            Some(4),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(3),
            0,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            Some(2),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(4),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            7,
            Some(8),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            None,
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            None,
            0,
            Some(1),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(1),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(4),
            2,
            Some(4),
        );

        Ok(())
    }

    fn assert_repeat_simplify(
        range: &CharRange,
        min1: u32,
        max1: Option<u32>,
        min2: u32,
        max2: Option<u32>,
    ) {
        let repeat = RegularExpression::Repetition(
            Box::new(RegularExpression::Repetition(
                Box::new(RegularExpression::Character(range.clone())),
                min1,
                max1,
            )),
            min2,
            max2,
        );

        let got = RegularExpression::new(&repeat.to_string()).unwrap();

        println!("{} -> {}", repeat, got);

        let repeat = repeat.to_automaton().unwrap();

        //repeat.to_dot();

        let result = got.to_automaton().unwrap();

        assert!(repeat.are_equivalent(&result).unwrap());
    }
}
