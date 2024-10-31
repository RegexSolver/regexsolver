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
pub mod error;
pub mod execution_profile;
pub mod fast_automaton;
pub mod regex;
pub mod tokenizer;

type IntMap<Key, Value> = HashMap<Key, Value, BuildHasherDefault<NoHashHasher<Key>>>;
type IntSet<Key> = HashSet<Key, BuildHasherDefault<NoHashHasher<Key>>>;
type Range = RangeSet<Char>;

/// Represents a term that can be either a regular expression or a finite automaton. This term can be manipulated with a wide range of operations.
///
/// To put constraint and limitation on the execution of operations please refer to [`execution_profile::ExecutionProfile`].
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Term {
    #[serde(rename = "regex")]
    RegularExpression(RegularExpression),
    #[serde(rename = "fair")]
    Automaton(FastAutomaton),
}

impl Term {
    /// Create a term based on the given pattern.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_regex(".*abc.*").unwrap();
    /// ```
    pub fn from_regex(regex: &str) -> Result<Self, EngineError> {
        Ok(Term::RegularExpression(RegularExpression::new(regex)?))
    }

    /// Compute the union of the given collection of terms.
    /// Returns the resulting term.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_regex("abc").unwrap();
    /// let term2 = Term::from_regex("de").unwrap();
    /// let term3 = Term::from_regex("fghi").unwrap();
    ///
    /// let union = term1.union(&[term2, term3]).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = union {
    ///     assert_eq!("(abc|de|fghi)", regex.to_string());
    /// }
    /// ```
    pub fn union(&self, terms: &[Term]) -> Result<Term, EngineError> {
        Self::check_number_of_terms(terms)?;

        let mut return_regex = RegularExpression::new_empty();
        let mut return_automaton = FastAutomaton::new_empty();
        match self {
            Term::RegularExpression(regular_expression) => {
                return_regex = regular_expression.clone();
            }
            Term::Automaton(fast_automaton) => {
                return_automaton = fast_automaton.clone();
            }
        }
        for operand in terms {
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
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_regex("(abc|de){2}").unwrap();
    /// let term2 = Term::from_regex("de.*").unwrap();
    /// let term3 = Term::from_regex(".*abc").unwrap();
    ///
    /// let intersection = term1.intersection(&[term2, term3]).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = intersection {
    ///     assert_eq!("deabc", regex.to_string());
    /// }
    /// ```
    pub fn intersection(&self, terms: &[Term]) -> Result<Term, EngineError> {
        Self::check_number_of_terms(terms)?;
        let mut return_automaton = self.get_automaton()?;
        for term in terms {
            let automaton = term.get_automaton()?;
            return_automaton = Cow::Owned(return_automaton.intersection(&automaton)?);
            if return_automaton.is_empty() {
                return Ok(Term::RegularExpression(RegularExpression::new_empty()));
            }
        }

        if let Some(regex) = return_automaton.to_regex() {
            Ok(Term::RegularExpression(regex))
        } else {
            Ok(Term::Automaton(return_automaton.into_owned()))
        }
    }

    /// Compute the subtraction/difference of the two given terms.
    /// Returns the resulting term.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_regex("(abc|de)").unwrap();
    /// let term2 = Term::from_regex("de").unwrap();
    ///
    /// let subtraction = term1.subtraction(&term2).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = subtraction {
    ///     assert_eq!("abc", regex.to_string());
    /// }
    /// ```
    pub fn subtraction(&self, subtrahend: &Term) -> Result<Term, EngineError> {
        let minuend_automaton = self.get_automaton()?;
        let subtrahend_automaton = subtrahend.get_automaton()?;
        let subtrahend_automaton =
            Self::determinize_subtrahend(&minuend_automaton, &subtrahend_automaton)?;
        let return_automaton = minuend_automaton.subtraction(&subtrahend_automaton)?;

        if let Some(regex) = return_automaton.to_regex() {
            Ok(Term::RegularExpression(regex))
        } else {
            Ok(Term::Automaton(return_automaton))
        }
    }

    /// See [`Self::subtraction`].
    #[inline]
    pub fn difference(&self, subtrahend: &Term) -> Result<Term, EngineError> {
        self.subtraction(subtrahend)
    }

    /// Returns the Details of the given term.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::{Term, cardinality::Cardinality};
    ///
    /// let term = Term::from_regex("(abc|de)").unwrap();
    ///
    /// let details = term.get_details().unwrap();
    ///
    /// assert_eq!(Some(Cardinality::Integer(2)), *details.get_cardinality());
    /// assert_eq!((Some(2), Some(3)), *details.get_length());
    /// assert!(!details.is_empty());
    /// assert!(!details.is_total());
    /// ```
    pub fn get_details(&self) -> Result<Details, EngineError> {
        match self {
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
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_regex("(abc|de){2}").unwrap();
    ///
    /// let strings = term.generate_strings(3).unwrap();
    ///
    /// assert_eq!(3, strings.len()); // ex: ["deabc", "dede", "abcde"]
    /// ```
    pub fn generate_strings(&self, count: usize) -> Result<Vec<String>, EngineError> {
        Ok(self
            .get_automaton()?
            .generate_strings(count)?
            .into_iter()
            .collect())
    }

    /// Compute if the two given terms are equivalent.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_regex("(abc|de)").unwrap();
    /// let term2 = Term::from_regex("(abc|de)*").unwrap();
    ///
    /// assert!(!term1.are_equivalent(&term2).unwrap());
    /// ```
    pub fn are_equivalent(&self, that: &Term) -> Result<bool, EngineError> {
        if self == that {
            return Ok(true);
        }

        let automaton_1 = self.get_automaton()?;
        let automaton_2 = that.get_automaton()?;
        automaton_1.is_equivalent_of(&automaton_2)
    }

    /// Compute if the first term is a subset of the second one.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_regex("de").unwrap();
    /// let term2 = Term::from_regex("(abc|de)").unwrap();
    ///
    /// assert!(term1.is_subset_of(&term2).unwrap());
    /// ```
    pub fn is_subset_of(&self, that: &Term) -> Result<bool, EngineError> {
        if self == that {
            return Ok(true);
        }

        let automaton_1 = self.get_automaton()?;
        let automaton_2 = that.get_automaton()?;
        automaton_1.is_subset_of(&automaton_2)
    }

    fn check_number_of_terms(terms: &[Term]) -> Result<(), EngineError> {
        let number_of_terms = terms.len() + 1;
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

    fn get_automaton(&self) -> Result<Cow<FastAutomaton>, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => Cow::Owned(regex.to_automaton()?),
            Term::Automaton(automaton) => Cow::Borrowed(automaton),
        })
    }
}

/// Represents details about a [Term].
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename = "details")]
pub struct Details {
    cardinality: Option<Cardinality<u32>>,
    length: (Option<u32>, Option<u32>),
    empty: bool,
    total: bool,
}

impl Details {
    /// Return the number of unique strings matched.
    pub fn get_cardinality(&self) -> &Option<Cardinality<u32>> {
        &self.cardinality
    }

    /// Return the minimum and the maximum length of matched strings.
    pub fn get_length(&self) -> &(Option<u32>, Option<u32>) {
        &self.length
    }

    /// Return `true` if it does not match any string.
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    /// Return `true` if it match all possible strings.
    pub fn is_total(&self) -> bool {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_details() -> Result<(), String> {
        let regex1 = Term::from_regex("a").unwrap();
        let regex2 = Term::from_regex("b").unwrap();

        let details = regex1.intersection(&vec![regex2]);
        assert!(details.is_ok());

        Ok(())
    }

    #[test]
    fn test_subtraction_1() -> Result<(), String> {
        let regex1 = Term::from_regex("a*").unwrap();
        let regex2 = Term::from_regex("").unwrap();

        let result = regex1.subtraction(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("a+").unwrap()),
            result
        );

        Ok(())
    }

    #[test]
    fn test_subtraction_2() -> Result<(), String> {
        let regex1 = Term::from_regex("x*").unwrap();
        let regex2 = Term::from_regex("(xxx)*").unwrap();

        let result = regex1.subtraction(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("(xxx)*(x|xx)").unwrap()),
            result
        );

        Ok(())
    }

    #[test]
    fn test_intersection_1() -> Result<(), String> {
        let regex1 = Term::from_regex("a*").unwrap();
        let regex2 = Term::from_regex("b*").unwrap();

        let result = regex1.intersection(&vec![regex2]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(Term::from_regex("").unwrap(), result);

        Ok(())
    }

    #[test]
    fn test_intersection_2() -> Result<(), String> {
        let regex1 = Term::from_regex("x*").unwrap();
        let regex2 = Term::from_regex("(xxx)*").unwrap();

        let result = regex1.intersection(&vec![regex2]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("(x{3})*").unwrap()),
            result
        );

        Ok(())
    }

    #[test]
    fn test__() -> Result<(), String> {
        let term = Term::from_regex("(abc|de){2}").unwrap();

        let strings = term.generate_strings(3).unwrap();

        println!("strings={:?}", strings);

        Ok(())
    }
}
