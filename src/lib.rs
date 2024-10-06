use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};

use cardinality::Cardinality;
use error::EngineError;
use execution_profile::ThreadLocalParams;
use fast_automaton::FastAutomaton;
use nohash_hasher::NoHashHasher;
use regex::RegularExpression;
use regex_charclass::{char::Char, irange::RangeSet};
use serde::{Deserialize, Serialize};

pub mod cardinality;
pub mod condition;
pub mod error;
pub mod execution_profile;
pub mod fast_automaton;
pub mod regex;
pub mod tokenizer;
pub mod used_bases;

pub type IntMap<Key, Value> = HashMap<Key, Value, BuildHasherDefault<NoHashHasher<Key>>>;
pub type IntSet<Key> = HashSet<Key, BuildHasherDefault<NoHashHasher<Key>>>;
pub type Range = RangeSet<Char>;

/// A term is either:
/// - a regular expression
/// - an automaton
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Term {
    #[serde(rename = "regex")]
    RegularExpression(RegularExpression),
    #[serde(rename = "fair")]
    Automaton(FastAutomaton),
}

/// A details contains the following information about a term:
/// - cardinality: the number of unique strings matched
/// - length: the minimum and the maximum length of matched strings
/// - empty: if it does not match any string
/// - total: if it match all possible strings
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename = "details")]
pub struct Details {
    cardinality: Option<Cardinality<u32>>,
    length: (Option<u32>, Option<u32>),
    empty: bool,
    total: bool,
}

/// Compute the union of the given collection of terms.
/// Returns the resulting term.
pub fn union(operands: &[Term]) -> Result<Term, EngineError> {
    check_number_of_terms(operands)?;
    let mut return_regex = RegularExpression::new_empty();
    let mut return_automaton = FastAutomaton::new_empty();
    for operand in operands {
        match operand {
            Term::RegularExpression(regex) => {
                return_regex = return_regex.union(regex);
                if return_regex.is_total() {
                    return Ok(Term::RegularExpression(RegularExpression::new_total()));
                }
            }
            Term::Automaton(automaton) => {
                return_automaton = return_automaton.union(automaton)?;
                if return_automaton.is_total() {
                    return Ok(Term::RegularExpression(RegularExpression::new_total()));
                }
            }
        }
    }

    if return_automaton.is_empty() {
        Ok(Term::RegularExpression(return_regex))
    } else {
        if !return_regex.is_empty() {
            return_automaton = return_automaton.union(&return_regex.to_automaton()?)?;
        }

        if let Some(regex) = return_automaton.to_regex() {
            Ok(Term::RegularExpression(regex))
        } else {
            Ok(Term::Automaton(return_automaton))
        }
    }
}

/// Compute the intersection of the given collection of terms.
/// Returns the resulting term.
pub fn intersection(terms: &[Term]) -> Result<Term, EngineError> {
    check_number_of_terms(terms)?;
    let mut return_automaton = FastAutomaton::new_total();
    for term in terms {
        let automaton = get_automaton_from_term(term)?;
        return_automaton = return_automaton.intersection(&automaton)?;
        if return_automaton.is_empty() {
            return Ok(Term::RegularExpression(RegularExpression::new_empty()));
        }
    }

    if let Some(regex) = return_automaton.to_regex() {
        Ok(Term::RegularExpression(regex))
    } else {
        Ok(Term::Automaton(return_automaton))
    }
}

/// Compute the subtraction/difference of the two given terms.
/// Returns the resulting term.
pub fn subtraction(minuend: &Term, subtrahend: &Term) -> Result<Term, EngineError> {
    let minuend_automaton = get_automaton_from_term(minuend)?;
    let subtrahend_automaton = get_automaton_from_term(subtrahend)?;
    let subtrahend_automaton = determinize_subtrahend(&minuend_automaton, &subtrahend_automaton)?;
    let return_automaton = minuend_automaton.subtraction(&subtrahend_automaton)?;

    if let Some(regex) = return_automaton.to_regex() {
        Ok(Term::RegularExpression(regex))
    } else {
        Ok(Term::Automaton(return_automaton))
    }
}

/// Returns the Details of the given term.
pub fn get_details(term: &Term) -> Result<Details, EngineError> {
    match term {
        Term::RegularExpression(regex) => Ok(Details {
            cardinality: Some(regex.get_cardinality()),
            length: regex.get_length(),
            empty: regex.is_empty(),
            total: regex.is_total(),
        }),
        Term::Automaton(automaton) => Ok(Details {
            cardinality: automaton.get_cardinality(),
            length: automaton.get_length(),
            empty: automaton.is_empty(),
            total: automaton.is_total(),
        }),
    }
}

/// Generate strings matched by the given term.
pub fn generate_strings(term: &Term, count: usize) -> Result<Vec<String>, EngineError> {
    Ok(get_automaton_from_term(term)?
        .generate_strings(count)?
        .into_iter()
        .collect())
}

/// Compute if the two given terms are equivalent.
pub fn are_equivalent(this: &Term, that: &Term) -> Result<bool, EngineError> {
    if this == that {
        return Ok(true);
    }

    let automaton_1 = get_automaton_from_term(this)?;
    let automaton_2 = get_automaton_from_term(that)?;
    automaton_1.is_equivalent_of(&automaton_2)
}

/// Compute if the first term is a subset of the second one.
pub fn is_subset_of(this: &Term, that: &Term) -> Result<bool, EngineError> {
    if this == that {
        return Ok(true);
    }

    let automaton_1 = get_automaton_from_term(this)?;
    let automaton_2 = get_automaton_from_term(that)?;
    automaton_1.is_subset_of(&automaton_2)
}

fn check_number_of_terms(terms: &[Term]) -> Result<(), EngineError> {
    let number_of_terms = terms.len();
    let max_number_of_terms = ThreadLocalParams::get_max_number_of_terms();
    if number_of_terms > max_number_of_terms {
        Err(EngineError::TooMuchTerms(
            max_number_of_terms,
            number_of_terms,
        ))
    } else {
        Ok(())
    }
}

fn get_automaton_from_term(term: &Term) -> Result<Cow<FastAutomaton>, EngineError> {
    Ok(match term {
        Term::RegularExpression(regex) => Cow::Owned(regex.to_automaton()?),
        Term::Automaton(automaton) => Cow::Borrowed(automaton),
    })
}

fn determinize_subtrahend<'a>(
    minuend: &FastAutomaton,
    subtrahend: &'a FastAutomaton,
) -> Result<Cow<'a, FastAutomaton>, EngineError> {
    if subtrahend.is_determinitic() {
        Ok(Cow::Borrowed(subtrahend))
    } else if !minuend.is_cyclic() && subtrahend.is_cyclic() {
        Ok(Cow::Owned(minuend.intersection(subtrahend)?.determinize()?))
    } else {
        Ok(Cow::Owned(subtrahend.determinize()?))
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_details() -> Result<(), String> {
        let regex1 = RegularExpression::new("a").unwrap();
        let regex2 = RegularExpression::new("b").unwrap();

        let details = intersection(&vec![
            Term::RegularExpression(regex1),
            Term::RegularExpression(regex2),
        ]);
        assert!(details.is_ok());

        Ok(())
    }

    #[test]
    fn test_subtraction_1() -> Result<(), String> {
        let regex1 = RegularExpression::new("a*").unwrap();
        let regex2 = RegularExpression::new("").unwrap();

        let result = subtraction(
            &Term::RegularExpression(regex1),
            &Term::RegularExpression(regex2),
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("a+").unwrap()),
            result
        );

        Ok(())
    }

    /*#[test]
    fn test_subtraction_2() -> Result<(), String> {
        let regex1 = RegularExpression::new("x*").unwrap();
        let regex2 = RegularExpression::new("(xxx)*").unwrap();

        let result = subtraction(
            &Term::RegularExpression(regex1),
            &Term::RegularExpression(regex2),
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("(xxx)*(x|xx)").unwrap()),
            result
        );

        Ok(())
    }*/

    #[test]
    fn test_intersection_1() -> Result<(), String> {
        let regex1 = RegularExpression::new("a*").unwrap();
        let regex2 = RegularExpression::new("b*").unwrap();

        let result = intersection(&vec![
            Term::RegularExpression(regex1),
            Term::RegularExpression(regex2),
        ]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("").unwrap()),
            result
        );

        Ok(())
    }

    #[test]
    fn test_intersection_2() -> Result<(), String> {
        let regex1 = RegularExpression::new("x*").unwrap();
        let regex2 = RegularExpression::new("(xxx)*").unwrap();

        let result = intersection(&vec![
            Term::RegularExpression(regex1),
            Term::RegularExpression(regex2),
        ]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("(x{3})*").unwrap()),
            result
        );

        Ok(())
    }
}
