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

use crate::execution_profile::ExecutionProfile;

pub mod cardinality;
pub mod error;
pub mod execution_profile;
pub mod fast_automaton;
pub mod regex;

pub type IntMap<Key, Value> = HashMap<Key, Value, BuildHasherDefault<NoHashHasher<Key>>>;
pub type IntSet<Key> = HashSet<Key, BuildHasherDefault<NoHashHasher<Key>>>;
pub type CharRange = RangeSet<Char>;

/// Represents a term that can be either a regular expression or a finite automaton. This term can be manipulated with a wide range of operations.
///
/// # Example
/// ```rust
/// use regexsolver::Term;
/// use regexsolver::error::EngineError;
///
/// fn main() -> Result<(), EngineError> {
///     // Create terms from regex
///     let t1 = Term::from_pattern("abc.*")?;
///     let t2 = Term::from_pattern(".*xyz")?;
///
///     // Concatenate
///     let concat = t1.concat(&[t2])?;
///     assert_eq!(concat.to_pattern(), "abc.*xyz");
///
///     // Union
///     let union = t1.union(&[Term::from_pattern("fgh")?])?;
///     assert_eq!(union.to_pattern(), "(abc.*|fgh)");
///
///     // Intersection
///     let inter = Term::from_pattern("(ab|xy){2}")?
///         .intersection(&[Term::from_pattern(".*xy")?])?;
///     assert_eq!(inter.to_pattern(), "(ab|xy)xy");
///
///     // Difference
///     let diff = Term::from_pattern("a*")?
///         .difference(&Term::from_pattern("")?)?;
///     assert_eq!(diff.to_pattern(), "a+");
///
///     // Repetition
///     let rep = Term::from_pattern("abc")?
///         .repeat(2, Some(4))?;
///     assert_eq!(rep.to_pattern(), "(abc){2,4}");
///
///     // Analyze
///     assert_eq!(rep.get_length(), (Some(6), Some(12)));
///     assert!(!rep.is_empty().unwrap());
///
///     // Generate examples
///     let samples = Term::from_pattern("(x|y){1,3}")?
///         .generate_strings(5, 0, false)?;
///     println!("Some matches: {:?}", samples);
///
///     // Equivalence & subset
///     let a = Term::from_pattern("a+")?;
///     let b = Term::from_pattern("a*")?;
///     assert!(!a.equivalent(&b)?);
///     assert!(a.subset(&b)?);
///
///     Ok(())
/// }
/// # main();
/// ```
///
/// To put constraint and limitation on the execution of operations please refer to [`ExecutionProfile`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Term {
    RegularExpression(RegularExpression),
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

    /// Parses and simplifies the provided pattern and returns a new [`Term`] holding the resulting [`RegularExpression`].
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

    /// Creates a new `Term` holding the provided [`RegularExpression`].
    pub fn from_regex(regex: RegularExpression) -> Self {
        Term::RegularExpression(regex)
    }

    /// Creates a new `Term` holding the provided [`FastAutomaton`].
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
        let mut has_automaton = matches!(self, Term::Automaton(_));
        if !has_automaton {
            for term in terms {
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

    /// Computes the difference between `self` and `other`.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de)").unwrap();
    /// let term2 = Term::from_pattern("de").unwrap();
    ///
    /// let difference = term1.difference(&term2).unwrap();
    ///
    /// if let Term::RegularExpression(regex) = difference {
    ///     assert_eq!("abc", regex.to_string());
    /// }
    /// ```
    pub fn difference(&self, other: &Term) -> Result<Term, EngineError> {
        let minuend_automaton = self.to_automaton()?;
        let subtrahend_automaton = other.to_automaton()?;
        // `FastAutomaton::difference` determinizes the subtrahend itself.
        let return_automaton = minuend_automaton.difference(&subtrahend_automaton)?;

        Ok(Term::Automaton(return_automaton))
    }

    /// Computes the complement of `self`.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("(abc|de)").unwrap();
    ///
    /// let complement = term.complement().unwrap();
    ///
    /// assert!(term.intersection(&[complement.clone()]).unwrap().is_empty().unwrap());
    /// assert!(term.union(&[complement]).unwrap().is_total().unwrap());
    /// ```
    pub fn complement(&self) -> Result<Term, EngineError> {
        // `FastAutomaton::complement` determinizes `self` itself.
        let mut automaton = self.to_automaton()?.into_owned();
        automaton.complement()?;

        Ok(Term::Automaton(automaton))
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

    /// Generates up to `limit` distinct strings matched by the term, skipping the first `offset` strings.
    ///
    /// When paginating through a large set of generated strings, you should set `return_stable_term`
    /// to `true` on the initial call. This instructs the engine to compile the term into a deterministic
    /// and minimized state (a "stable term").
    ///
    /// Replacing your current term with this returned stable term for subsequent calls guarantees
    /// that no strings are repeated.
    ///
    /// The stable term is returned as the first element of the tuple (`Some(Term)`). If the term is
    /// already stable, or if `return_stable_term` is `false`, it returns `None` to save resources.
    ///
    /// # Example:
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let mut term = Term::from_pattern("(abc|de){2}").unwrap();
    ///
    /// // Generate the first 2 matched strings and request a stable term
    /// let (stable_term, batch) = term.generate_strings(2, 0, true).unwrap();
    /// assert_eq!(2, batch.len()); // ["dede", "deabc"]
    ///
    /// // Update the term if a newly compiled stable term was returned
    /// if let Some(t) = stable_term {
    ///     term = t;
    /// }
    ///
    /// // Generate the next 2 matched strings by setting the offset using the stable term
    /// let (_, batch) = term.generate_strings(2, 2, false).unwrap();
    /// assert_eq!(2, batch.len()); // ["abcde", "abcabc"]
    /// ```
    pub fn generate_strings(
        &self,
        limit: usize,
        offset: usize,
        return_stable_term: bool,
    ) -> Result<(Option<Term>, Vec<String>), EngineError> {
        let automaton = self.to_automaton()?;
        if !return_stable_term || automaton.is_deterministic() {
            Ok((None, self.to_automaton()?.generate_strings(limit, offset)?))
        } else {
            // `minimize` determinizes first, yielding the deterministic,
            // minimal "stable" automaton.
            let mut automaton = automaton.into_owned();
            if !automaton.is_minimal() {
                automaton.minimize()?;
            }

            let generated_strings = automaton.generate_strings(limit, offset)?;
            Ok((Some(Term::Automaton(automaton)), generated_strings))
        }
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
    /// assert!(!term1.equivalent(&term2).unwrap());
    /// ```
    pub fn equivalent(&self, term: &Term) -> Result<bool, EngineError> {
        if self == term {
            return Ok(true);
        }

        let automaton_1 = self.to_automaton()?;
        let automaton_2 = term.to_automaton()?;
        automaton_1.equivalent(&automaton_2)
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
    /// assert!(term1.subset(&term2).unwrap());
    /// ```
    pub fn subset(&self, term: &Term) -> Result<bool, EngineError> {
        if self == term {
            return Ok(true);
        }

        let automaton_1 = self.to_automaton()?;
        let automaton_2 = term.to_automaton()?;
        automaton_1.subset(&automaton_2)
    }

    /// Checks if the term matches the empty language.
    pub fn is_empty(&self) -> Result<bool, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => regex.is_empty(),
            Term::Automaton(automaton) => automaton.is_empty(),
        })
    }

    /// Checks if the term matches all possible strings.
    pub fn is_total(&self) -> Result<bool, EngineError> {
        match self {
            Term::RegularExpression(regex) => Ok(regex.is_total()),
            Term::Automaton(automaton) => {
                if automaton.is_total() {
                    Ok(true)
                } else if automaton.is_deterministic() {
                    Ok(false)
                } else {
                    Ok(automaton.determinize()?.is_total())
                }
            }
        }
    }

    /// Checks if the term matches only the empty string `""`.
    pub fn is_empty_string(&self) -> Result<bool, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => regex.is_empty_string(),
            Term::Automaton(automaton) => automaton.is_empty_string(),
        })
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
            Term::Automaton(automaton) => Ok(if !automaton.is_deterministic() {
                automaton.determinize()?.get_cardinality()
            } else {
                automaton.get_cardinality()
            }),
        }
    }

    /// Converts the term to a [`FastAutomaton`].
    pub fn to_automaton(&self) -> Result<Cow<'_, FastAutomaton>, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => Cow::Owned(regex.to_automaton()?),
            Term::Automaton(automaton) => Cow::Borrowed(automaton),
        })
    }

    /// Converts the term to a [`RegularExpression`].
    pub fn to_regex(&self) -> Cow<'_, RegularExpression> {
        match self {
            Term::RegularExpression(regex) => Cow::Borrowed(regex),
            Term::Automaton(automaton) => Cow::Owned(automaton.to_regex()),
        }
    }

    /// Converts the term to a regular expression pattern.
    pub fn to_pattern(&self) -> String {
        self.to_regex().to_string()
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
        regex_list.push(self.to_regex());

        let mut terms_regexes = terms.iter().map(Term::to_regex).collect::<Vec<_>>();
        regex_list.append(&mut terms_regexes);

        Some(regex_list)
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_complement() -> Result<(), String> {
        let term = Term::from_pattern("(abc|de)").unwrap();

        let complement = term.complement().unwrap();

        assert!(
            term.intersection(std::slice::from_ref(&complement))
                .unwrap()
                .is_empty()
                .unwrap()
        );

        println!("term: {}", term.to_automaton().unwrap().as_dot());

        if let Term::Automaton(complement) = &complement {
            println!("complement: {}", complement.as_dot());
        }

        let union = term.union(&[complement]).unwrap();
        if let Term::Automaton(union) = &union {
            println!("{}", union.as_dot());
            let union = union.determinize().unwrap();
            println!("{}", union.as_dot());
        }

        assert!(union.is_total().unwrap());

        Ok(())
    }

    #[test]
    fn test_intersection() -> Result<(), String> {
        let regex1 = Term::from_pattern("a").unwrap();
        let regex2 = Term::from_pattern("b").unwrap();

        let intersection = regex1.intersection(&[regex2]).unwrap();
        assert!(intersection.is_empty().unwrap());
        assert_eq!("[]", intersection.to_pattern());

        Ok(())
    }

    #[test]
    fn test_difference_1() -> Result<(), String> {
        let regex1 = Term::from_pattern("a*").unwrap();
        let regex2 = Term::from_pattern("").unwrap();

        let result = regex1.difference(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern();
        assert_eq!("a+", result);

        Ok(())
    }

    #[test]
    fn test_difference_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.difference(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_regex().into_owned();
        assert_eq!(
            Term::RegularExpression(RegularExpression::new("x(x{3})*x?").unwrap()),
            Term::RegularExpression(result)
        );

        Ok(())
    }

    #[test]
    fn test_intersection_1() -> Result<(), String> {
        let regex1 = Term::from_pattern("a*").unwrap();
        let regex2 = Term::from_pattern("b*").unwrap();

        let result = regex1.intersection(&[regex2]);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern();
        assert_eq!("", result);

        Ok(())
    }

    #[test]
    fn test_intersection_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.intersection(&[regex2]);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern();
        assert_eq!("(x{3})*", result);

        Ok(())
    }
}
