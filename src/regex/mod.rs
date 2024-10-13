use std::{cmp, collections::VecDeque, fmt::Display};

use crate::Range;
use execution_profile::ThreadLocalParams;
use regex_charclass::CharacterClass;
use regex_syntax::hir::{Class, ClassBytes, ClassUnicode, Hir, HirKind};

use self::fast_automaton::FastAutomaton;

use super::*;

mod analyze;
mod builder;
mod operation;
mod serializer;

/// Represent a regular expression.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum RegularExpression {
    Character(Range),
    Repetition(Box<RegularExpression>, u32, Option<u32>),
    Concat(VecDeque<RegularExpression>),
    Alternation(Vec<RegularExpression>),
}

impl Display for RegularExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            RegularExpression::Character(range) => {
                if range.is_empty() {
                    return write!(f, "[]");
                }
                range.to_regex()
            }
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                let regex_part = regular_expression.to_string();
                let multiplicator_part;
                if *min == 0 && max_opt.is_none() {
                    multiplicator_part = String::from("*");
                } else if *min == 1 && max_opt.is_none() {
                    multiplicator_part = String::from("+");
                } else if *min == 0 && max_opt.is_some() && max_opt.unwrap() == 1 {
                    multiplicator_part = String::from("?");
                } else if let Some(max) = max_opt {
                    if max == min {
                        multiplicator_part = format!("{{{}}}", max);
                    } else {
                        multiplicator_part = format!("{{{},{}}}", min, max);
                    }
                } else {
                    multiplicator_part = format!("{{{},}}", min);
                }
                match **regular_expression {
                    RegularExpression::Repetition(_, _, _) => {
                        format!("({}){}", regex_part, multiplicator_part)
                    }
                    RegularExpression::Concat(_) => {
                        format!("({}){}", regex_part, multiplicator_part)
                    }
                    _ => format!("{}{}", regex_part, multiplicator_part),
                }
            }
            RegularExpression::Concat(concat) => {
                let mut sb = String::new();
                for regex in concat.iter() {
                    sb.push_str(regex.to_string().as_str());
                }
                sb
            }
            RegularExpression::Alternation(alternation) => {
                if alternation.is_empty() {
                    return write!(f, "[]");
                }
                let mut sb = String::new();
                for i in 0..alternation.len() {
                    sb.push_str(alternation[i].to_string().as_str());
                    if i != alternation.len() - 1 {
                        sb.push('|');
                    }
                }
                if alternation.len() == 1 {
                    sb
                } else {
                    format!("({})", sb)
                }
            }
        };
        write!(f, "{}", str)
    }
}

impl RegularExpression {
    pub fn is_empty(&self) -> bool {
        match self {
            RegularExpression::Alternation(alternation) => alternation.is_empty(),
            RegularExpression::Character(range) => range.is_empty(),
            _ => false,
        }
    }

    pub fn is_empty_string(&self) -> bool {
        match self {
            RegularExpression::Concat(concat) => concat.is_empty(),
            _ => false,
        }
    }

    pub fn is_total(&self) -> bool {
        match self {
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                if min != &0 || max_opt.is_some() {
                    false
                } else {
                    match &**regular_expression {
                        RegularExpression::Character(range) => range.is_total(),
                        _ => false,
                    }
                }
            }
            _ => false,
        }
    }

    pub fn to_automaton(&self) -> Result<FastAutomaton, EngineError> {
        if self.get_number_of_states_in_nfa() >= ThreadLocalParams::get_max_number_of_states() {
            return Err(EngineError::AutomatonHasTooManyStates);
        }
        match self {
            RegularExpression::Character(range) => FastAutomaton::make_from_range(range),
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                let mut automaton = regular_expression.to_automaton()?;
                automaton.repeat(*min, *max_opt)?;
                Ok(automaton)
            }
            RegularExpression::Concat(concat) => {
                let mut concats = Vec::with_capacity(concat.len());
                for c in concat.iter() {
                    concats.push(c.to_automaton()?);
                }
                FastAutomaton::concatenate(concats)
            }
            RegularExpression::Alternation(alternation) => {
                let mut concats = Vec::with_capacity(alternation.len());
                for c in alternation.iter() {
                    concats.push(c.to_automaton()?);
                }
                FastAutomaton::alternation(concats)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() -> Result<(), String> {
        let automaton = RegularExpression::new_empty();
        assert!(automaton.is_empty());
        assert!(!automaton.is_empty_string());
        assert!(!automaton.is_total());
        Ok(())
    }

    #[test]
    fn test_empty_string() -> Result<(), String> {
        let automaton = RegularExpression::new_empty_string();
        assert!(!automaton.is_empty());
        assert!(automaton.is_empty_string());
        assert!(!automaton.is_total());
        Ok(())
    }

    #[test]
    fn test_total() -> Result<(), String> {
        let automaton = RegularExpression::new_total();
        assert!(!automaton.is_empty());
        assert!(!automaton.is_empty_string());
        assert!(automaton.is_total());
        Ok(())
    }
}
