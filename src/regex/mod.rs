use std::{cmp, collections::VecDeque, fmt::Display};

use crate::execution_profile::ExecutionProfile;
use regex_charclass::CharacterClass;
use regex_syntax::hir::{Class, ClassBytes, ClassUnicode, Hir, HirKind};

use self::fast_automaton::FastAutomaton;

use super::*;

mod analyze;
mod builder;
mod operation;
#[cfg(feature = "serializable")]
mod serializer;

/// Represent a regular expression.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum RegularExpression {
    Character(CharRange),
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
                        multiplicator_part = format!("{{{max}}}");
                    } else {
                        multiplicator_part = format!("{{{min},{max}}}");
                    }
                } else {
                    multiplicator_part = format!("{{{min},}}");
                }
                match **regular_expression {
                    RegularExpression::Repetition(_, _, _) => {
                        format!("({regex_part}){multiplicator_part}")
                    }
                    RegularExpression::Concat(_) => {
                        format!("({regex_part}){multiplicator_part}")
                    }
                    _ => format!("{regex_part}{multiplicator_part}"),
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
                    format!("({sb})")
                }
            }
        };
        write!(f, "{str}")
    }
}

impl RegularExpression {
    /// Checks if the regular expression matches the empty language.
    pub fn is_empty(&self) -> bool {
        match self {
            RegularExpression::Alternation(alternation) => alternation.is_empty(),
            RegularExpression::Character(range) => range.is_empty(),
            _ => false,
        }
    }

    /// Checks if the regular expression only matches the empty string `""`.
    pub fn is_empty_string(&self) -> bool {
        match self {
            RegularExpression::Concat(concat) => concat.is_empty(),
            _ => false,
        }
    }

    /// Checks if the regular expression matches all possible strings.
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

    /// Converts the regular expression to an equivalent [`FastAutomaton`].
    pub fn to_automaton(&self) -> Result<FastAutomaton, EngineError> {
        ExecutionProfile::get().assert_max_number_of_states(self.get_number_of_states_in_nfa())?;

        match self {
            RegularExpression::Character(range) => Ok(FastAutomaton::new_from_range(range)),
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                let mut automaton = regular_expression.to_automaton()?;
                automaton.repeat_mut(*min, *max_opt)?;
                Ok(automaton)
            }
            RegularExpression::Concat(concat) => {
                let mut concats = Vec::with_capacity(concat.len());
                for c in concat.iter() {
                    concats.push(c.to_automaton()?);
                }
                FastAutomaton::concat_all(&concats)
            }
            RegularExpression::Alternation(alternation) => {
                let mut alternates = Vec::with_capacity(alternation.len());
                for c in alternation.iter() {
                    alternates.push(c.to_automaton()?);
                }
                FastAutomaton::union_all(&alternates)
            }
        }
    }

    pub fn evaluate_complexity(&self) -> f64 {
        let (score, depth, _) = self.eval_inner();
        score + Self::depth_penalty(depth)
    }

    // returns: (score, max_depth, contains_repetition)
    fn eval_inner(&self) -> (f64, usize, bool) {
        match self {
            RegularExpression::Character(range) => {
                let len = range.to_regex().len() as f64;
                // small, capped cost for raw length
                let base = 1.0 + 0.05 * len.min(40.0);
                (base, 1, false)
            }

            RegularExpression::Repetition(inner, min, max_opt) => {
                let (inner_score, inner_depth, inner_has_rep) = inner.eval_inner();

                // multipliers tuned for readability impact
                let mut m = match max_opt {
                    None => 1.6,
                    Some(max) if max > min => 1.3,
                    Some(max) if max == min && *min > 1 => 1.1,
                    _ => 1.0,
                };

                // nested quantifiers like (...+)+ are harder
                if inner_has_rep {
                    m *= 1.5;
                }

                (inner_score * m, inner_depth + 1, true)
            }

            RegularExpression::Concat(items) => {
                let mut sum = 0.0;
                let mut max_depth = 0usize;
                let mut has_rep = false;

                for (i, it) in items.iter().enumerate() {
                    let (s, d, h) = it.eval_inner();
                    sum += s;
                    if i > 0 {
                        // tiny discount: linear sequences are relatively easy to read
                        sum *= 0.98;
                    }
                    if d > max_depth {
                        max_depth = d;
                    }
                    has_rep |= h;
                }

                (sum, max_depth + 1, has_rep)
            }

            RegularExpression::Alternation(branches) => {
                if branches.is_empty() {
                    return (0.0, 1, false);
                }
                let mut sum = 0.0;
                let mut max_depth = 0usize;
                let mut has_rep = false;

                for b in branches {
                    let (s, d, h) = b.eval_inner();
                    sum += s;
                    if d > max_depth {
                        max_depth = d;
                    }
                    has_rep |= h;
                }

                // branching cost: more alternatives = harder to scan
                let k = branches.len() as f64;
                let multiplier = 1.0 + 0.15 * (k - 1.0);

                (sum * multiplier, max_depth + 1, has_rep)
            }
        }
    }

    fn depth_penalty(depth: usize) -> f64 {
        // no penalty up to depth 2, then quadratic growth
        if depth <= 2 {
            0.0
        } else {
            ((depth - 2) as f64).powi(2) * 0.8
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
