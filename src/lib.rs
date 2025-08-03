use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::BuildHasherDefault,
};

use cardinality::Cardinality;
use error::EngineError;
use fast_automaton::FastAutomaton;
use nohash_hasher::NoHashHasher;
use rayon::prelude::*;
use regex::RegularExpression;
use regex_charclass::{char::Char, irange::RangeSet};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::execution_profile::ExecutionProfile;

pub mod cardinality;
pub mod error;
pub mod execution_profile;
pub mod fast_automaton;
pub mod regex;
pub mod tokenizer;

pub type IntMap<Key, Value> = HashMap<Key, Value, BuildHasherDefault<NoHashHasher<Key>>>;
pub type IntSet<Key> = HashSet<Key, BuildHasherDefault<NoHashHasher<Key>>>;
pub type CharRange = RangeSet<Char>;

/// Represents a term that can be either a regular expression or a finite automaton. This term can be manipulated with a wide range of operations.
///
/// To put constraint and limitation on the execution of operations please refer to [`execution_profile::ExecutionProfile`].
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "value"))]
pub enum Term {
    #[cfg_attr(feature = "serde", serde(rename = "regex"))]
    RegularExpression(RegularExpression),
    #[cfg_attr(feature = "serde", serde(rename = "fair"))]
    Automaton(FastAutomaton),
}

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::RegularExpression(regular_expression) => write!(f, "{regular_expression}"),
            Term::Automaton(fast_automaton) => write!(f, "{fast_automaton}"),
        }
    }
}

impl Term {
    /// Creates a term that matches the empty language.
    pub fn new_empty() -> Self {
        Term::RegularExpression(RegularExpression::new_empty())
    }

    /// Creates a term that matches all possible strings.
    pub fn new_total() -> Self {
        Term::RegularExpression(RegularExpression::new_total())
    }

    /// Creates a term that only matches the empty string `""`.
    pub fn new_empty_string() -> Self {
        Term::RegularExpression(RegularExpression::new_empty_string())
    }

    /// Parses the provided pattern and returns a new `Term` holding the resulting `RegularExpression`.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern(".*abc.*").unwrap();
    /// ```
    pub fn from_pattern(pattern: &str) -> Result<Self, EngineError> {
        Ok(Term::RegularExpression(RegularExpression::new(pattern)?))
    }

    /// Creates a new `Term` holding the provided `RegularExpression`.
    pub fn from_regex(regex: RegularExpression) -> Self {
        Term::RegularExpression(regex)
    }

    /// Creates a new `Term` holding the provided `FastAutomaton`.
    pub fn from_automaton(automaton: FastAutomaton) -> Self {
        Term::Automaton(automaton)
    }

    /// Computes the concatenation of the given terms.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("abc").unwrap();
    /// let term2 = Term::from_pattern("d.").unwrap();
    /// let term3 = Term::from_pattern(".*").unwrap();
    ///
    /// let concat = term1.concat(&[term2, term3]).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = concat {
    ///     assert_eq!("abcd.+", regex.to_string());
    /// }
    /// ```
    pub fn concat(&self, terms: &[Term]) -> Result<Term, EngineError> {
        let mut return_regex = RegularExpression::new_empty();
        let mut return_automaton = FastAutomaton::new_empty();
        let mut has_automaton = false;
        match self {
            Term::RegularExpression(regular_expression) => {
                return_regex = regular_expression.clone()
            }
            Term::Automaton(fast_automaton) => {
                has_automaton = true;
                return_automaton = fast_automaton.clone();
            }
        }
        for term in terms {
            if has_automaton {
                return_automaton = return_automaton.concat(term.to_automaton()?.as_ref())?;
            } else {
                match term {
                    Term::RegularExpression(regular_expression) => {
                        return_regex = return_regex.concat(regular_expression, true);
                    }
                    Term::Automaton(fast_automaton) => {
                        has_automaton = true;
                        return_automaton = return_regex.to_automaton()?.concat(fast_automaton)?;
                    }
                }
            }
        }

        if !has_automaton {
            Ok(Term::RegularExpression(return_regex))
        } else {
            Ok(Term::Automaton(return_automaton))
        }
    }

    /// Computes the union of the given terms.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("abc").unwrap();
    /// let term2 = Term::from_pattern("de").unwrap();
    /// let term3 = Term::from_pattern("fghi").unwrap();
    ///
    /// let union = term1.union(&[term2, term3]).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = union {
    ///     assert_eq!("(abc|de|fghi)", regex.to_string());
    /// }
    /// ```
    pub fn union(&self, terms: &[Term]) -> Result<Term, EngineError> {
        if self.is_total() {
            return Ok(Term::new_total());
        }

        let mut has_automaton = matches!(self, Term::Automaton(_));
        if !has_automaton {
            for term in terms {
                if term.is_total() {
                    return Ok(Term::new_total());
                }
                if matches!(term, Term::Automaton(_)) {
                    has_automaton = true;
                    break;
                }
            }
        }

        if has_automaton {
            let parallel = terms.len() > 3;

            let automaton_list = self.get_automata(terms, parallel)?;

            let automaton_list = automaton_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

            let return_automaton = if parallel {
                FastAutomaton::union_all_par(automaton_list)
            } else {
                FastAutomaton::union_all(automaton_list)
            }?;

            Ok(Term::Automaton(return_automaton))
        } else {
            let regexes_list = self
                .get_regexes(terms)
                .expect("No automaton should be here so this operation is not supposed to fail.");

            let regexes_list = regexes_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

            Ok(Term::RegularExpression(RegularExpression::union_all(
                regexes_list,
            )))
        }
    }

    /// Computes the intersection of the given terms.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de){2}").unwrap();
    /// let term2 = Term::from_pattern("de.*").unwrap();
    /// let term3 = Term::from_pattern(".*abc").unwrap();
    ///
    /// let intersection = term1.intersection(&[term2, term3]).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = intersection {
    ///     assert_eq!("deabc", regex.to_string());
    /// }
    /// ```
    pub fn intersection(&self, terms: &[Term]) -> Result<Term, EngineError> {
        if self.is_empty() || terms.iter().any(|t| t.is_empty()) {
            return Ok(Term::new_empty());
        }

        let parallel = terms.len() > 3;

        let automaton_list = self.get_automata(terms, parallel)?;

        let automaton_list = automaton_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

        let return_automaton = if parallel {
            FastAutomaton::intersection_all_par(automaton_list)
        } else {
            FastAutomaton::intersection_all(automaton_list)
        }?;

        Ok(Term::Automaton(return_automaton))
    }

    /// Computes the difference between `self` and the given subtrahend.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de)").unwrap();
    /// let term2 = Term::from_pattern("de").unwrap();
    ///
    /// let subtraction = term1.subtraction(&term2).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = subtraction {
    ///     assert_eq!("abc", regex.to_string());
    /// }
    /// ```
    pub fn subtraction(&self, subtrahend: &Term) -> Result<Term, EngineError> {
        let minuend_automaton = self.to_automaton()?;
        let subtrahend_automaton = subtrahend.to_automaton()?;
        let subtrahend_automaton =
            Self::determinize_subtrahend(&minuend_automaton, &subtrahend_automaton)?;
        let return_automaton = minuend_automaton.subtraction(&subtrahend_automaton)?;

        Ok(Term::Automaton(return_automaton))
    }

    /// See [`Self::subtraction`].
    #[inline]
    pub fn difference(&self, subtrahend: &Term) -> Result<Term, EngineError> {
        self.subtraction(subtrahend)
    }

    /// Computes the repetition of the current term between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("abc").unwrap();
    ///
    /// let repeat = term.repeat(1, None).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = repeat {
    ///     assert_eq!("(abc)+", regex.to_string());
    /// }
    ///
    /// let repeat = term.repeat(3, Some(5)).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = repeat {
    ///     assert_eq!("(abc){3,5}", regex.to_string());
    /// }
    /// ```
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> Result<Term, EngineError> {
        match self {
            Term::RegularExpression(regular_expression) => Ok(Term::RegularExpression(
                regular_expression.repeat(min, max_opt),
            )),
            Term::Automaton(fast_automaton) => {
                let repeat_automaton = fast_automaton.repeat(min, max_opt)?;
                Ok(Term::Automaton(repeat_automaton))
            }
        }
    }

    /// Generates `count` strings matched by the term.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("(abc|de){2}").unwrap();
    ///
    /// let strings = term.generate_strings(3).unwrap();
    ///
    /// assert_eq!(3, strings.len()); // ex: ["deabc", "dede", "abcde"]
    /// ```
    pub fn generate_strings(&self, count: usize) -> Result<Vec<String>, EngineError> {
        self.to_automaton()?.generate_strings(count)
    }

    /// Returns `true` if both terms accept the same language.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de)").unwrap();
    /// let term2 = Term::from_pattern("(abc|de)*").unwrap();
    ///
    /// assert!(!term1.are_equivalent(&term2).unwrap());
    /// ```
    pub fn are_equivalent(&self, that: &Term) -> Result<bool, EngineError> {
        if self == that {
            return Ok(true);
        }

        let automaton_1 = self.to_automaton()?;
        let automaton_2 = that.to_automaton()?;
        automaton_1.are_equivalent(&automaton_2)
    }

    /// Returns `true` if all strings matched by the current term are also matched by the given term.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("de").unwrap();
    /// let term2 = Term::from_pattern("(abc|de)").unwrap();
    ///
    /// assert!(term1.is_subset_of(&term2).unwrap());
    /// ```
    pub fn is_subset_of(&self, that: &Term) -> Result<bool, EngineError> {
        if self == that {
            return Ok(true);
        }

        let automaton_1 = self.to_automaton()?;
        let automaton_2 = that.to_automaton()?;
        automaton_1.is_subset_of(&automaton_2)
    }

    /// Checks if the term matches the empty language.
    pub fn is_empty(&self) -> bool {
        match self {
            Term::RegularExpression(regular_expression) => regular_expression.is_empty(),
            Term::Automaton(fast_automaton) => fast_automaton.is_empty(),
        }
    }

    /// Checks if the term matches all possible strings.
    pub fn is_total(&self) -> bool {
        match self {
            Term::RegularExpression(regular_expression) => regular_expression.is_total(),
            Term::Automaton(fast_automaton) => fast_automaton.is_total(),
        }
    }

    /// Checks if the term matches only the empty string `""`.
    pub fn is_empty_string(&self) -> bool {
        match self {
            Term::RegularExpression(regular_expression) => regular_expression.is_empty_string(),
            Term::Automaton(fast_automaton) => fast_automaton.is_empty_string(),
        }
    }

    /// Returns the minimum and maximum length of matched strings.
    pub fn get_length(&self) -> (Option<u32>, Option<u32>) {
        match self {
            Term::RegularExpression(regex) => regex.get_length(),
            Term::Automaton(automaton) => automaton.get_length(),
        }
    }

    /// Returns the cardinality of the term (i.e., the number of possible matched strings).
    pub fn get_cardinality(&self) -> Result<Cardinality<u32>, EngineError> {
        match self {
            Term::RegularExpression(regex) => Ok(regex.get_cardinality()),
            Term::Automaton(automaton) => {
                let cardinality = if !automaton.is_determinitic() {
                    automaton.determinize()?.get_cardinality()
                } else {
                    automaton.get_cardinality()
                };

                if let Some(cardinality) = cardinality {
                    Ok(cardinality)
                } else {
                    Err(EngineError::CannotComputeAutomatonCardinality)
                }
            }
        }
    }

    /// Converts the term to a `FastAutomaton`.
    pub fn to_automaton(&self) -> Result<Cow<FastAutomaton>, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => Cow::Owned(regex.to_automaton()?),
            Term::Automaton(automaton) => Cow::Borrowed(automaton),
        })
    }

    /// Converts the term to a RegularExpression; returns `None` if conversion isn’t possible.
    pub fn to_regex(&self) -> Option<Cow<RegularExpression>> {
        Some(match self {
            Term::RegularExpression(regex) => Cow::Borrowed(regex),
            Term::Automaton(automaton) => Cow::Owned(automaton.to_regex()?),
        })
    }

    /// Converts the term to a regular expression pattern; returns `None` if conversion isn’t possible.
    pub fn to_pattern(&self) -> Option<String> {
        Some(self.to_regex()?.to_string())
    }

    fn determinize_subtrahend<'a>(
        minuend: &FastAutomaton,
        subtrahend: &'a FastAutomaton,
    ) -> Result<Cow<'a, FastAutomaton>, EngineError> {
        if subtrahend.is_determinitic() {
            Ok(Cow::Borrowed(subtrahend))
        } else if !minuend.is_cyclic() && subtrahend.is_cyclic() {
            Ok(Cow::Owned(minuend.intersection(subtrahend)?.determinize()?.into_owned()))
        } else {
            Ok(subtrahend.determinize()?)
        }
    }

    fn get_automata<'a>(
        &'a self,
        terms: &'a [Term],
        parallel: bool,
    ) -> Result<Vec<Cow<'a, FastAutomaton>>, EngineError> {
        let mut automaton_list = Vec::with_capacity(terms.len() + 1);
        automaton_list.push(self.to_automaton()?);

        let mut terms_automata = if parallel {
            let execution_profile = ExecutionProfile::get();
            terms
                .par_iter()
                .map(|a| execution_profile.apply(|| a.to_automaton()))
                .collect::<Result<Vec<_>, _>>()
        } else {
            terms
                .iter()
                .map(Term::to_automaton)
                .collect::<Result<Vec<_>, _>>()
        }?;
        automaton_list.append(&mut terms_automata);

        Ok(automaton_list)
    }

    fn get_regexes<'a>(&'a self, terms: &'a [Term]) -> Option<Vec<Cow<'a, RegularExpression>>> {
        let mut regex_list = Vec::with_capacity(terms.len() + 1);
        regex_list.push(self.to_regex()?);

        let mut terms_regexes = terms
            .iter()
            .map(Term::to_regex)
            .collect::<Option<Vec<_>>>()?;
        regex_list.append(&mut terms_regexes);

        Some(regex_list)
    }
}

#[cfg(test)]
mod tests {
    use crate::{execution_profile::ExecutionProfileBuilder, regex::RegularExpression};

    use super::*;

    #[test]
    fn test_details() -> Result<(), String> {
        let regex1 = Term::from_pattern("a").unwrap();
        let regex2 = Term::from_pattern("b").unwrap();

        let details = regex1.intersection(&vec![regex2]);
        assert!(details.is_ok());

        Ok(())
    }

    #[test]
    fn test_subtraction_1() -> Result<(), String> {
        let regex1 = Term::from_pattern("a*").unwrap();
        let regex2 = Term::from_pattern("").unwrap();

        let result = regex1.subtraction(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("a+", result);

        Ok(())
    }

    #[test]
    fn test_subtraction_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.subtraction(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_regex().unwrap().into_owned();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("(xxx)*(x|xx)").unwrap()),
            Term::RegularExpression(result)
        );

        Ok(())
    }

    #[test]
    fn test_intersection_1() -> Result<(), String> {
        let regex1 = Term::from_pattern("a*").unwrap();
        let regex2 = Term::from_pattern("b*").unwrap();

        let result = regex1.intersection(&vec![regex2]);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("", result);

        Ok(())
    }

    #[test]
    fn test_intersection_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.intersection(&vec![regex2]);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("(x{3})*", result);

        Ok(())
    }

    #[test]
    fn test_readme_code_1() -> Result<(), String> {
        // Create terms from regex
        let t1 = Term::from_pattern("abc.*").unwrap();
        let t2 = Term::from_pattern(".*xyz").unwrap();

        // Concatenate
        let concat = t1.concat(&[t2]).unwrap();
        assert_eq!(concat.to_pattern().unwrap(), "abc.*xyz");

        // Union
        let union = t1.union(&[Term::from_pattern("fgh").unwrap()]).unwrap(); // (abc.*|fgh)
        assert_eq!(union.to_pattern().unwrap(), "(abc.*|fgh)");

        // Intersection
        let inter = Term::from_pattern("(ab|xy){2}")
            .unwrap()
            .intersection(&[Term::from_pattern(".*xy").unwrap()])
            .unwrap(); // (ab|xy)xy
        assert_eq!(inter.to_pattern().unwrap(), "(ab|xy)xy");

        // Subtraction
        let diff = Term::from_pattern("a*")
            .unwrap()
            .subtraction(&Term::from_pattern("").unwrap())
            .unwrap();
        assert_eq!(diff.to_pattern().unwrap(), "a+");

        // Repetition
        let rep = Term::from_pattern("abc")
            .unwrap()
            .repeat(2, Some(4))
            .unwrap(); // (abc){2,4}
        assert_eq!(rep.to_pattern().unwrap(), "(abc){2,4}");

        // Analyze
        assert_eq!(rep.get_length(), (Some(6), Some(12)));
        assert!(!rep.is_empty());

        // Generate examples
        let samples = Term::from_pattern("(x|y){1,3}")
            .unwrap()
            .generate_strings(5)
            .unwrap();
        println!("Some matches: {:?}", samples);

        // Equivalence & subset
        let a = Term::from_pattern("a+").unwrap();
        let b = Term::from_pattern("a*").unwrap();
        assert!(!a.are_equivalent(&b).unwrap());
        assert!(a.is_subset_of(&b).unwrap());

        Ok(())
    }

    #[test]
    fn test_readme_code_2() -> Result<(), String> {
        let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();

        let execution_profile = ExecutionProfileBuilder::new()
            .execution_timeout(5) // We set the limit (5ms)
            .build();

        // We run the operation with the defined limitation
        execution_profile.run(|| {
            assert_eq!(
                EngineError::OperationTimeOutError,
                term.generate_strings(1000).unwrap_err()
            );
        });

        Ok(())
    }

    #[test]
    fn test_readme_code_3() -> Result<(), String> {
        let term1 = Term::from_pattern(".*abcdef.*").unwrap();
        let term2 = Term::from_pattern(".*defabc.*").unwrap();

        let execution_profile = ExecutionProfileBuilder::new()
            .max_number_of_states(5) // We set the limit
            .build();

        // We run the operation with the defined limitation
        execution_profile.run(|| {
            assert_eq!(
                EngineError::AutomatonHasTooManyStates,
                term1.intersection(&[term2]).unwrap_err()
            );
        });

        Ok(())
    }
}
