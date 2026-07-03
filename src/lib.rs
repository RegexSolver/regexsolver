//! RegexSolver treats regular expressions as the **sets of strings they
//! match**, so you can intersect, subtract, compare, complement, repeat, and
//! enumerate them — and convert the result back into a regex pattern.
//!
//! # Quick start
//!
//! [`Term`] is the main entry point: it wraps either a [`RegularExpression`]
//! or a [`FastAutomaton`] and picks the cheaper representation for each
//! operation.
//!
//! ```
//! use regexsolver::Term;
//!
//! let a: Term = "(ab|xy){2}".parse()?;
//! let b: Term = ".*xy".parse()?;
//!
//! // Which strings match BOTH patterns? Get the answer back as a regex:
//! let both = a.intersection([&b])?;
//! assert_eq!(both.to_pattern()?, "(ab|xy)xy");
//!
//! // Matching is anchored (whole-string):
//! assert!(both.matches("abxy")?);
//! # Ok::<(), regexsolver::error::EngineError>(())
//! ```
//!
//! # Semantics
//!
//! RegexSolver implements **pure regular languages**, which differs from a
//! typical regex engine in two ways: matching is always **anchored** (a pattern
//! describes whole strings, so `abc` matches only `"abc"`), and `.` matches any
//! character including line feed. Constructs that a regular language can't
//! represent — backreferences, look-around, inline flags, and anchors/word
//! boundaries in non-redundant positions — return an [`EngineError`] rather
//! than being applied incorrectly. See the crate README for the full list.
//!
//! # Bounding execution
//!
//! Automaton operations can blow up on adversarial input, so a thread-local
//! [`ExecutionProfile`] can cap runtime and
//! state count and control implicit determinization; hitting a limit returns a
//! specific [`EngineError`] instead of hanging.
//!
//! # Modules
//!
//! Most users only need [`Term`]. The lower-level building blocks live in
//! [`regex`] (the parsed-pattern AST), [`fast_automaton`] (finite automata),
//! [`execution_profile`] (resource limits), [`cardinality`], and [`error`].

#![warn(missing_docs)]

use std::{
    borrow::{Borrow, Cow},
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    hash::BuildHasherDefault,
    ops::{Bound, RangeBounds},
    str::FromStr,
};

use cardinality::Cardinality;
use error::EngineError;
use fast_automaton::FastAutomaton;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use regex::RegularExpression;
use regex_charclass::{char::Char, irange::RangeSet};

use crate::execution_profile::ExecutionProfile;

/// Cardinality of a language ([`Cardinality`]): a finite count, a count too
/// large for `u32`, or infinite.
pub mod cardinality;
/// The [`EngineError`] type returned by fallible operations.
pub mod error;
/// Resource limits: the thread-local [`ExecutionProfile`] governing timeouts,
/// state caps, and implicit determinization.
pub mod execution_profile;
/// Finite automata: [`FastAutomaton`] and its building blocks (conditions,
/// spanning sets).
pub mod fast_automaton;
/// The parsed-pattern AST: [`RegularExpression`].
pub mod regex;

/// Re-export of [`regex-charclass`](https://docs.rs/regex-charclass), the
/// crate behind [`CharRange`]: everything needed to build transition labels
/// by hand (`Char`, range sets) without adding a separately version-matched
/// dependency.
pub use regex_charclass;

/// A no-op [`Hasher`](std::hash::Hasher) for integer keys that are already
/// well distributed, such as state ids: the key's value is used as the hash
/// directly. Only the integer key types it is implemented for can be hashed
/// with it; anything else does not compile.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoHashHasher<Key>(u64, std::marker::PhantomData<Key>);

macro_rules! impl_no_hash_hasher {
    ($($int:ty => $write:ident),* $(,)?) => {
        $(
            impl std::hash::Hasher for NoHashHasher<$int> {
                #[inline]
                fn finish(&self) -> u64 {
                    self.0
                }

                fn write(&mut self, _: &[u8]) {
                    unreachable!("NoHashHasher hashes integer keys through their value");
                }

                #[inline]
                fn $write(&mut self, n: $int) {
                    self.0 = n as u64;
                }
            }
        )*
    };
}
impl_no_hash_hasher!(u32 => write_u32, u64 => write_u64, usize => write_usize);

/// A hash map keyed by integer state ids using a no-op hasher. Internal.
pub(crate) type IntMap<Key, Value> = HashMap<Key, Value, BuildHasherDefault<NoHashHasher<Key>>>;
/// A hash set of integer state ids using a no-op hasher (the hasher is fast
/// because state ids are already well-distributed small integers). Returned by
/// [`FastAutomaton::accept_states`] and related inspection methods.
pub type IntSet<Key> = HashSet<Key, BuildHasherDefault<NoHashHasher<Key>>>;
/// A set of character ranges (the transition-label alphabet type), re-exported
/// from [`regex-charclass`](https://docs.rs/regex-charclass).
pub type CharRange = RangeSet<Char>;

/// Represents a term that can be either a regular expression or a finite automaton. This term can be manipulated with a wide range of operations.
///
/// # Examples
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
///     assert_eq!(concat.to_pattern()?, "abc.*xyz");
///
///     // Union
///     let union = t1.union(&[Term::from_pattern("fgh")?])?;
///     assert_eq!(union.to_pattern()?, "(abc.*|fgh)");
///
///     // Intersection
///     let inter = Term::from_pattern("(ab|xy){2}")?
///         .intersection(&[Term::from_pattern(".*xy")?])?;
///     assert_eq!(inter.to_pattern()?, "(ab|xy)xy");
///
///     // Difference
///     let diff = Term::from_pattern("a*")?
///         .difference(&Term::from_pattern("")?)?;
///     assert_eq!(diff.to_pattern()?, "a+");
///
///     // Repetition
///     let rep = Term::from_pattern("abc")?
///         .repeat(2..=4)?;
///     assert_eq!(rep.to_pattern()?, "(abc){2,4}");
///
///     // Analyze
///     assert_eq!(rep.length(), (Some(6), Some(12)));
///     assert!(!rep.is_empty()?);
///
///     // Generate examples
///     let samples = Term::from_pattern("(x|y){1,3}")?
///         .generate_strings(5, 0)?;
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
///
/// # Tracing
///
/// The core operations on [`Term`], [`FastAutomaton`], and [`RegularExpression`]
/// are instrumented with [`tracing`](https://docs.rs/tracing) spans (mostly at
/// `debug` level). Install a [`tracing-subscriber`](https://docs.rs/tracing-subscriber)
/// (or any other `tracing` subscriber) in your application to observe them; if
/// no subscriber is installed, instrumentation has negligible overhead and
/// produces no output.
///
/// # Equality
///
/// `PartialEq`/`Eq` (`==`) compare the **underlying representation**, not the
/// language. Two terms that match exactly the same strings can compare
/// unequal (for example, an automaton and an equivalent regular expression, or
/// two differently-written regexes for the same language). To compare
/// *languages*, use [`equivalent`](Self::equivalent); for `self ⊆ other`, use
/// [`subset`](Self::subset).
#[derive(Clone, PartialEq, Eq, Debug)]
#[must_use = "terms are immutable; operations return a new term"]
pub enum Term {
    /// The term is backed by a parsed regular-expression AST.
    RegularExpression(RegularExpression),
    /// The term is backed by a finite automaton.
    Automaton(FastAutomaton),
}

/// The default term is the empty language (matches nothing), the identity for
/// [`union`](Term::union). See [`new_empty`](Term::new_empty).
impl Default for Term {
    fn default() -> Self {
        Term::new_empty()
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::RegularExpression(regular_expression) => write!(f, "{regular_expression}"),
            Term::Automaton(fast_automaton) => write!(f, "{fast_automaton}"),
        }
    }
}

/// Parses a pattern into a [`Term`], so patterns can be built with
/// [`str::parse`].
///
/// # Examples
///
/// ```
/// use regexsolver::Term;
///
/// let term: Term = ".*abc.*".parse().unwrap();
/// ```
impl FromStr for Term {
    type Err = EngineError;

    fn from_str(pattern: &str) -> Result<Self, Self::Err> {
        Term::from_pattern(pattern)
    }
}

impl From<RegularExpression> for Term {
    fn from(regex: RegularExpression) -> Self {
        Term::RegularExpression(regex)
    }
}

impl From<FastAutomaton> for Term {
    fn from(automaton: FastAutomaton) -> Self {
        Term::Automaton(automaton)
    }
}

impl Term {
    /// `Term` operations manage the underlying representation themselves, so
    /// the determinizations they perform are by definition explicit:
    /// they run with the profile's `implicit_determinization` setting
    /// re-enabled (that knob targets direct [`FastAutomaton`] usage). The
    /// rest of the profile is preserved.
    fn run_with_implicit_determinization<R>(f: impl FnOnce() -> R) -> R {
        ExecutionProfile::get()
            .with_implicit_determinization(true)
            .apply(f)
    }

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
    /// # Examples
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
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("abc").unwrap();
    /// let term2 = Term::from_pattern("d.").unwrap();
    /// let term3 = Term::from_pattern(".*").unwrap();
    ///
    /// let concat = term1.concat([&term2, &term3]).unwrap();
    ///
    /// assert_eq!("abcd.+", concat.to_pattern().unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn concat(
        &self,
        terms: impl IntoIterator<Item = impl Borrow<Term>>,
    ) -> Result<Term, EngineError> {
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
            let term = term.borrow();
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
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("abc").unwrap();
    /// let term2 = Term::from_pattern("de").unwrap();
    /// let term3 = Term::from_pattern("fghi").unwrap();
    ///
    /// let union = term1.union([&term2, &term3]).unwrap();
    ///
    /// assert_eq!("(abc|de|fghi)", union.to_pattern().unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn union(
        &self,
        terms: impl IntoIterator<Item = impl Borrow<Term>>,
    ) -> Result<Term, EngineError> {
        let terms: Vec<_> = terms.into_iter().collect();
        let terms: Vec<&Term> = terms.iter().map(Borrow::borrow).collect();

        let mut has_automaton = matches!(self, Term::Automaton(_));
        if !has_automaton {
            for term in &terms {
                if matches!(term, Term::Automaton(_)) {
                    has_automaton = true;
                    break;
                }
            }
        }

        if has_automaton {
            let parallel = cfg!(feature = "parallel") && terms.len() > 3;

            let automaton_list = self.get_automata(&terms, parallel)?;

            let automaton_list = automaton_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

            #[cfg(feature = "parallel")]
            let return_automaton = if parallel {
                FastAutomaton::union_all_par(automaton_list)
            } else {
                FastAutomaton::union_all(automaton_list)
            }?;
            #[cfg(not(feature = "parallel"))]
            let return_automaton = FastAutomaton::union_all(automaton_list)?;

            Ok(Term::Automaton(return_automaton))
        } else {
            let regexes_list = self.get_regexes(&terms)?;

            let regexes_list = regexes_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

            Ok(Term::RegularExpression(RegularExpression::union_all(
                regexes_list,
            )))
        }
    }

    /// Computes the intersection of the given terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de){2}").unwrap();
    /// let term2 = Term::from_pattern("de.*").unwrap();
    /// let term3 = Term::from_pattern(".*abc").unwrap();
    ///
    /// let intersection = term1.intersection([&term2, &term3]).unwrap();
    ///
    /// assert_eq!("deabc", intersection.to_pattern().unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn intersection(
        &self,
        terms: impl IntoIterator<Item = impl Borrow<Term>>,
    ) -> Result<Term, EngineError> {
        let terms: Vec<_> = terms.into_iter().collect();
        let terms: Vec<&Term> = terms.iter().map(Borrow::borrow).collect();

        let parallel = cfg!(feature = "parallel") && terms.len() > 3;

        let automaton_list = self.get_automata(&terms, parallel)?;

        let automaton_list = automaton_list.iter().map(AsRef::as_ref).collect::<Vec<_>>();

        #[cfg(feature = "parallel")]
        let return_automaton = if terms.len() > 3 {
            FastAutomaton::intersection_all_par(automaton_list)
        } else {
            FastAutomaton::intersection_all(automaton_list)
        }?;
        #[cfg(not(feature = "parallel"))]
        let return_automaton = FastAutomaton::intersection_all(automaton_list)?;

        Ok(Term::Automaton(return_automaton))
    }

    /// Computes the difference between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de)").unwrap();
    /// let term2 = Term::from_pattern("de").unwrap();
    ///
    /// let difference = term1.difference(&term2).unwrap();
    ///
    /// assert_eq!("abc", difference.to_pattern().unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic(), other_deterministic = other.is_deterministic()))]
    pub fn difference(&self, other: &Term) -> Result<Term, EngineError> {
        Self::run_with_implicit_determinization(|| {
            let minuend_automaton = self.to_automaton()?;
            let subtrahend_automaton = other.to_automaton()?;
            // `FastAutomaton::difference` determinizes the subtrahend itself.
            let return_automaton = minuend_automaton.difference(&subtrahend_automaton)?;

            Ok(Term::Automaton(return_automaton))
        })
    }

    /// Computes the complement of `self`.
    ///
    /// # Examples
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
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic()))]
    pub fn complement(&self) -> Result<Term, EngineError> {
        Self::run_with_implicit_determinization(|| {
            // `FastAutomaton::complement` determinizes `self` itself.
            let mut automaton = self.to_automaton()?.into_owned();
            automaton.complement()?;

            Ok(Term::Automaton(automaton))
        })
    }

    /// Computes the repetition of the current term over the given range of
    /// counts.
    ///
    /// An unbounded end (`n..`) means unlimited repetition; an unset start
    /// (`..n` or `..=n`) means zero. Exclusive bounds are normalized to inclusive.
    /// A range containing no count at all (`0..0`, `3..3`, `5..2`) yields the
    /// empty language.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("abc").unwrap();
    ///
    /// assert_eq!("(abc)+", term.repeat(1..).unwrap().to_pattern().unwrap());
    /// assert_eq!("(abc){3,5}", term.repeat(3..=5).unwrap().to_pattern().unwrap());
    /// assert_eq!("(abc){3,5}", term.repeat(3..6).unwrap().to_pattern().unwrap());
    /// assert_eq!("(abc){0,2}", term.repeat(..=2).unwrap().to_pattern().unwrap());
    /// assert!(term.repeat(0..0).unwrap().is_empty().unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic(), min = tracing::field::Empty, max = tracing::field::Empty))]
    pub fn repeat(&self, range: impl RangeBounds<u32>) -> Result<Term, EngineError> {
        let mut min = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let max_opt = match range.end_bound() {
            Bound::Included(&n) => Some(n),
            Bound::Excluded(&n) => Some(n.saturating_sub(1)),
            Bound::Unbounded => None,
        };
        if matches!(range.end_bound(), Bound::Excluded(&0)) {
            min = min.max(1);
        }
        let span = tracing::Span::current();
        span.record("min", min);
        span.record("max", tracing::field::debug(max_opt));
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
    /// Strings are only guaranteed to be distinct **within a single call**:
    /// the offset fast-skips by counting paths, and in a non-deterministic
    /// automaton the same string can be reached through several paths, so
    /// calls with different offsets may repeat strings (or skip some). The
    /// enumeration order also depends on the automaton's structure, so
    /// offsets are only consistent across calls made on the same term.
    ///
    /// For pagination without repetition or skipped strings, make the term deterministic once and generate
    /// from it. To check if a term is deterministic use [`is_deterministic`](Self::is_deterministic).
    /// To determinize run [`determinize`](Self::determinize).
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// // Minimize once, then paginate with consistent offsets.
    /// let term = Term::from_pattern("(abc|de){2}").unwrap().minimize().unwrap();
    ///
    /// let batch = term.generate_strings(2, 0).unwrap();
    /// assert_eq!(2, batch.len()); // ["dede", "deabc"]
    ///
    /// let batch = term.generate_strings(2, 2).unwrap();
    /// assert_eq!(2, batch.len()); // ["abcde", "abcabc"]
    /// ```
    #[tracing::instrument(level = "debug", skip(self), fields(self_deterministic = self.is_deterministic(), limit = limit, offset = offset))]
    pub fn generate_strings(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<String>, EngineError> {
        self.to_automaton()?.generate_strings(limit, offset)
    }

    /// Returns a lazy iterator over the strings matched by the term, fetched in
    /// batches behind the scenes so you can stop early without choosing a limit
    /// up front.
    ///
    /// The underlying automaton is computed once at construction time, not on
    /// every batch. Each item is a `Result`: a construction or generation error
    /// (e.g. a timeout from the active [`ExecutionProfile`]) surfaces as an
    /// `Err`, after which the iterator ends. The same determinism caveat as
    /// [`generate_strings`](Self::generate_strings) applies: call
    /// [`determinize`](Self::determinize) (or [`minimize`](Self::minimize))
    /// first for distinct, stable enumeration.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("(abc|de){2}").unwrap().minimize().unwrap();
    ///
    /// // Take the first three matches lazily.
    /// let first_three = term
    ///     .iter_strings()
    ///     .take(3)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(3, first_three.len());
    /// ```
    pub fn iter_strings(&self) -> StringGenerator<'_> {
        match self.to_automaton() {
            Ok(automaton) => StringGenerator {
                automaton: Some(automaton),
                pending_error: None,
                offset: 0,
                buffer: VecDeque::new(),
            },
            Err(e) => StringGenerator {
                automaton: None,
                pending_error: Some(e),
                offset: 0,
                buffer: VecDeque::new(),
            },
        }
    }

    /// Returns an equivalent term backed by a deterministic automaton.
    ///
    /// Already-deterministic terms are returned as-is.
    ///
    /// Determinization is always explicit, so it runs regardless of the
    /// profile's [`implicit_determinization`](crate::execution_profile::ExecutionProfileBuilder::implicit_determinization)
    /// setting.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern(".*abc").unwrap();
    /// assert!(!term.is_deterministic());
    ///
    /// let dfa = term.determinize().unwrap();
    /// assert!(dfa.is_deterministic());
    /// assert!(term.equivalent(&dfa).unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic()))]
    pub fn determinize(&self) -> Result<Term, EngineError> {
        let automaton = self.to_automaton()?;
        let determinized = automaton.determinize()?.into_owned();
        Ok(Term::Automaton(determinized))
    }

    /// Returns an equivalent term backed by the minimal deterministic
    /// automaton.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern(".*abc").unwrap();
    /// let minimal = term.minimize().unwrap();
    /// assert!(minimal.is_minimal());
    /// assert!(term.equivalent(&minimal).unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic(), self_minimal = self.is_minimal()))]
    pub fn minimize(&self) -> Result<Term, EngineError> {
        Self::run_with_implicit_determinization(|| {
            let mut automaton = self.to_automaton()?.into_owned();
            automaton.minimize()?;
            Ok(Term::Automaton(automaton))
        })
    }

    /// Returns `true` if both terms accept the same language.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("(abc|de)").unwrap();
    /// let term2 = Term::from_pattern("(abc|de)*").unwrap();
    ///
    /// assert!(!term1.equivalent(&term2).unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic(), other_deterministic = other.is_deterministic()))]
    pub fn equivalent(&self, other: &Term) -> Result<bool, EngineError> {
        if self == other {
            return Ok(true);
        }

        Self::run_with_implicit_determinization(|| {
            let automaton_1 = self.to_automaton()?;
            let automaton_2 = other.to_automaton()?;
            automaton_1.equivalent(&automaton_2)
        })
    }

    /// Returns `true` if all strings matched by the current term are also matched by the given term.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term1 = Term::from_pattern("de").unwrap();
    /// let term2 = Term::from_pattern("(abc|de)").unwrap();
    ///
    /// assert!(term1.subset(&term2).unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic(), other_deterministic = other.is_deterministic()))]
    pub fn subset(&self, other: &Term) -> Result<bool, EngineError> {
        if self == other {
            return Ok(true);
        }

        Self::run_with_implicit_determinization(|| {
            let automaton_1 = self.to_automaton()?;
            let automaton_2 = other.to_automaton()?;
            automaton_1.subset(&automaton_2)
        })
    }

    /// Returns `true` if the term matches the given string.
    ///
    /// Matching is **anchored** (full-string), consistent with the rest of the
    /// crate: the whole input must be accepted, not just a substring.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// let term = Term::from_pattern("abc.*").unwrap();
    ///
    /// assert!(term.matches("abcdef").unwrap());
    /// assert!(!term.matches("xyzabc").unwrap());
    /// ```
    #[tracing::instrument(level = "debug", skip(self, input), fields(self_deterministic = self.is_deterministic(), input_len = input.len()))]
    pub fn matches(&self, input: &str) -> Result<bool, EngineError> {
        Ok(self.to_automaton()?.is_match(input))
    }

    /// Returns `true` if the term matches the empty language (no strings at all).
    ///
    /// Note: the empty language is distinct from the language containing only
    /// the empty string `""`. Use [`is_empty_string`](Self::is_empty_string) to
    /// test for the latter.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// assert!(Term::new_empty().is_empty().unwrap());
    /// assert!(!Term::new_empty_string().is_empty().unwrap()); // matches ""
    /// assert!(!Term::from_pattern("abc").unwrap().is_empty().unwrap());
    /// ```
    pub fn is_empty(&self) -> Result<bool, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => regex.is_empty(),
            Term::Automaton(automaton) => automaton.is_empty(),
        })
    }

    /// Returns `true` if the term matches all possible strings.
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

    /// Returns `true` if the term matches only the empty string `""`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// assert!(Term::new_empty_string().is_empty_string().unwrap());
    /// assert!(!Term::new_empty().is_empty_string().unwrap());
    /// assert!(!Term::from_pattern("a*").unwrap().is_empty_string().unwrap());
    /// ```
    pub fn is_empty_string(&self) -> Result<bool, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => regex.is_empty_string(),
            Term::Automaton(automaton) => automaton.is_empty_string(),
        })
    }

    /// Returns `true` if the term is *already backed by* a deterministic
    /// automaton.
    ///
    /// A deterministic automaton has one path per accepted string.
    ///
    /// To determinize a term call [`determinize`](Self::determinize).
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        match self {
            Term::RegularExpression(_) => false,
            Term::Automaton(automaton) => automaton.is_deterministic(),
        }
    }

    /// Returns `true` if the term is *already backed by* the minimal
    /// deterministic automaton.
    ///
    /// The minimal deterministic automaton of a given language is unique.
    ///
    /// To minimize a term call [`minimize`](Self::minimize).
    #[must_use]
    pub fn is_minimal(&self) -> bool {
        match self {
            Term::RegularExpression(_) => false,
            Term::Automaton(automaton) => automaton.is_minimal(),
        }
    }

    /// Returns the minimum and maximum length of matched strings.
    ///
    /// `None` for the minimum means the language is empty (no strings are
    /// matched). `None` for the maximum means the language is infinite
    /// (unbounded match length).
    #[must_use]
    pub fn length(&self) -> (Option<u32>, Option<u32>) {
        match self {
            Term::RegularExpression(regex) => regex.length(),
            Term::Automaton(automaton) => automaton.length(),
        }
    }

    /// Returns the cardinality of the term (the number of distinct matched strings).
    ///
    /// The exact count is represented as `u32`. If the exact count exceeds
    /// `u32::MAX`, the result is `Cardinality::BigInteger` rather than a
    /// truncated value. Infinite languages return `Cardinality::Infinite`.
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic()))]
    pub fn cardinality(&self) -> Result<Cardinality<u32>, EngineError> {
        Self::run_with_implicit_determinization(|| self.to_automaton()?.cardinality())
    }

    /// Returns `true` if the term matches a finite number of strings.
    ///
    /// A finite language is one with no unbounded repetition (`*`, `+`, ...).
    /// Convenience over [`cardinality`](Self::cardinality) when only the
    /// finite/infinite distinction matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::Term;
    ///
    /// assert!(Term::from_pattern("(ab|c){2}").unwrap().is_finite().unwrap());
    /// assert!(!Term::from_pattern("a+").unwrap().is_finite().unwrap());
    /// ```
    pub fn is_finite(&self) -> Result<bool, EngineError> {
        Ok(!matches!(self.cardinality()?, Cardinality::Infinite))
    }

    /// Converts the term to a [`FastAutomaton`].
    ///
    /// Returns a [`Cow`]: borrows the automaton when the term is already
    /// automaton-backed, and allocates a new one when converting from a
    /// [`RegularExpression`].
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic()))]
    pub fn to_automaton(&self) -> Result<Cow<'_, FastAutomaton>, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => Cow::Owned(regex.to_automaton()?),
            Term::Automaton(automaton) => Cow::Borrowed(automaton),
        })
    }

    /// Converts the term to a [`RegularExpression`].
    ///
    /// Returns a [`Cow`]: borrows the expression when the term is already
    /// regex-backed, and allocates a new one when converting from a
    /// [`FastAutomaton`] via state elimination.
    #[tracing::instrument(level = "debug", skip_all, fields(self_deterministic = self.is_deterministic()))]
    pub fn to_regex(&self) -> Result<Cow<'_, RegularExpression>, EngineError> {
        Ok(match self {
            Term::RegularExpression(regex) => Cow::Borrowed(regex),
            Term::Automaton(automaton) => Cow::Owned(automaton.to_regex()?),
        })
    }

    /// Converts the term to a regular expression pattern.
    pub fn to_pattern(&self) -> Result<String, EngineError> {
        Ok(self.to_regex()?.to_string())
    }

    fn get_automata<'a>(
        &'a self,
        terms: &[&'a Term],
        parallel: bool,
    ) -> Result<Vec<Cow<'a, FastAutomaton>>, EngineError> {
        let mut automaton_list = Vec::with_capacity(terms.len() + 1);
        automaton_list.push(self.to_automaton()?);

        #[cfg(feature = "parallel")]
        let mut terms_automata = if parallel {
            let execution_profile = ExecutionProfile::get();
            terms
                .par_iter()
                .map(|a| execution_profile.apply(|| a.to_automaton()))
                .collect::<Result<Vec<_>, _>>()
        } else {
            terms
                .iter()
                .map(|a| a.to_automaton())
                .collect::<Result<Vec<_>, _>>()
        }?;
        #[cfg(not(feature = "parallel"))]
        let mut terms_automata = {
            let _ = parallel;
            terms
                .iter()
                .map(|a| a.to_automaton())
                .collect::<Result<Vec<_>, EngineError>>()?
        };
        automaton_list.append(&mut terms_automata);

        Ok(automaton_list)
    }

    fn get_regexes<'a>(
        &'a self,
        terms: &[&'a Term],
    ) -> Result<Vec<Cow<'a, RegularExpression>>, EngineError> {
        let mut regex_list = Vec::with_capacity(terms.len() + 1);
        regex_list.push(self.to_regex()?);
        for term in terms {
            regex_list.push(term.to_regex()?);
        }
        Ok(regex_list)
    }
}

/// Lazy iterator over the strings matched by a [`Term`], created by
/// [`Term::iter_strings`].
///
/// The underlying automaton is computed once at construction. Yields
/// `Result<String, EngineError>`: errors (from construction or generation)
/// are surfaced as `Err` items, after which the iterator ends.
#[derive(Debug)]
pub struct StringGenerator<'a> {
    automaton: Option<Cow<'a, FastAutomaton>>,
    pending_error: Option<EngineError>,
    offset: usize,
    buffer: VecDeque<String>,
}

// Every terminal state (language exhausted, or error yielded) drops the
// automaton, after which `next` returns `None` forever.
impl std::iter::FusedIterator for StringGenerator<'_> {}

impl Iterator for StringGenerator<'_> {
    type Item = Result<String, EngineError>;

    fn next(&mut self) -> Option<Self::Item> {
        const BATCH: usize = 32;

        if let Some(s) = self.buffer.pop_front() {
            return Some(Ok(s));
        }
        if let Some(e) = self.pending_error.take() {
            return Some(Err(e));
        }
        let automaton = self.automaton.as_ref()?;
        match automaton.generate_strings(BATCH, self.offset) {
            Ok(batch) => {
                if batch.len() < BATCH {
                    self.automaton = None;
                }
                self.offset += batch.len();
                self.buffer.extend(batch);
                self.buffer.pop_front().map(Ok)
            }
            Err(e) => {
                self.automaton = None;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    // A range containing no count at all (`0..0`, `3..3`, `5..2`) is the
    // empty language, while a range containing exactly the count 0 (`0..=0`,
    // `0..1`) is the empty-string language.
    #[test]
    #[allow(clippy::reversed_empty_ranges)] // deliberately empty ranges are the point
    fn repeat_empty_ranges_yield_the_empty_language() {
        let regex_term = Term::from_pattern("abc").unwrap();
        let automaton_term = regex_term.determinize().unwrap();
        assert!(matches!(automaton_term, Term::Automaton(..)));

        for term in [regex_term, automaton_term] {
            // Ranges containing no count at all: the empty language.
            assert!(term.repeat(0..0).unwrap().is_empty().unwrap());
            assert!(term.repeat(3..3).unwrap().is_empty().unwrap());
            assert!(term.repeat(5..2).unwrap().is_empty().unwrap());

            // Ranges containing exactly the count 0: the empty-string language.
            assert!(term.repeat(0..=0).unwrap().is_empty_string().unwrap());
            assert!(term.repeat(0..1).unwrap().is_empty_string().unwrap());
        }
    }

    // Pins the intentional `Display` behavior: regex-backed terms render
    // their pattern; automaton-backed terms render Graphviz DOT. Use
    // `to_pattern` to obtain a parseable pattern for either kind.
    #[test]
    fn display_is_pattern_for_regexes_and_dot_for_automata() {
        let regex_term = Term::from_pattern("(abc){2}").unwrap();
        assert_eq!("(abc){2}", regex_term.to_string());

        let automaton_term = regex_term.determinize().unwrap();
        assert!(matches!(automaton_term, Term::Automaton(..)));
        assert!(automaton_term.to_string().starts_with("digraph"));
        let reparsed: Term = automaton_term.to_pattern().unwrap().parse().unwrap();
        assert!(reparsed.equivalent(&automaton_term).unwrap());
    }

    // `to_pattern` (state elimination) can grow super-polynomially, so it
    // must honor the execution deadline and fail with a timeout rather than
    // run unbudgeted.
    #[test]
    fn to_pattern_honors_the_execution_deadline() {
        let term = Term::from_pattern(".*abc.*def.*")
            .unwrap()
            .determinize()
            .unwrap();

        crate::execution_profile::ExecutionProfileBuilder::new()
            .execution_timeout(0)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term.to_pattern().unwrap_err()
                );
            });

        // Without the 0ms deadline the very same conversion succeeds.
        assert!(term.to_pattern().is_ok());
    }

    #[test]
    fn test_complement() -> Result<(), String> {
        let term = Term::from_pattern("(abc|de)").unwrap();

        let complement = term.complement().unwrap();

        assert!(
            term.intersection([&complement])
                .unwrap()
                .is_empty()
                .unwrap()
        );

        println!("term: {}", term.to_automaton().unwrap().to_dot());

        if let Term::Automaton(complement) = &complement {
            println!("complement: {}", complement.to_dot());
        }

        let union = term.union(&[complement]).unwrap();
        if let Term::Automaton(union) = &union {
            println!("{}", union.to_dot());
            let union = union.determinize().unwrap();
            println!("{}", union.to_dot());
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
        assert_eq!("[]", intersection.to_pattern().unwrap());

        Ok(())
    }

    #[test]
    fn test_difference_1() -> Result<(), String> {
        let regex1 = Term::from_pattern("a*").unwrap();
        let regex2 = Term::from_pattern("").unwrap();

        let result = regex1.difference(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("a+", result);

        Ok(())
    }

    #[test]
    fn test_difference_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.difference(&regex2);
        assert!(result.is_ok());
        let result = result.unwrap().to_regex().unwrap().into_owned();
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
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("", result);

        Ok(())
    }

    #[test]
    fn test_intersection_2() -> Result<(), String> {
        let regex1 = Term::from_pattern("x*").unwrap();
        let regex2 = Term::from_pattern("(xxx)*").unwrap();

        let result = regex1.intersection(&[regex2]);
        assert!(result.is_ok());
        let result = result.unwrap().to_pattern().unwrap();
        assert_eq!("(x{3})*", result);

        Ok(())
    }

    #[test]
    fn test_default_is_empty_language() {
        assert!(Term::default().is_empty().unwrap());
        assert_eq!(Term::default(), Term::new_empty());
    }

    #[test]
    fn test_iter_strings_exhaustive_matches_generate_strings() {
        // A finite, deterministic term: lazy iteration must yield exactly the
        // same multiset as a single large `generate_strings` call, with no
        // duplicates or omissions across batch boundaries.
        let term = Term::from_pattern("[A-Za-z0-9]")
            .unwrap()
            .minimize()
            .unwrap();

        let eager = term.generate_strings(1000, 0).unwrap();
        let lazy = term.iter_strings().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(eager.len(), lazy.len());
        assert_eq!(eager, lazy);
        assert_eq!(62, lazy.len());
    }

    #[test]
    fn test_is_finite() {
        assert!(
            Term::from_pattern("(ab|c){2}")
                .unwrap()
                .is_finite()
                .unwrap()
        );
        assert!(!Term::from_pattern("a+").unwrap().is_finite().unwrap());
    }

    #[test]
    fn test_matches_is_anchored() {
        let term = Term::from_pattern("abc.*").unwrap();
        assert!(term.matches("abc").unwrap());
        assert!(term.matches("abcdef").unwrap());
        // Anchored: a prefix/suffix match is not enough.
        assert!(!term.matches("xyzabc").unwrap());

        let exact = Term::from_pattern("abc").unwrap();
        assert!(exact.matches("abc").unwrap());
        assert!(!exact.matches("abcd").unwrap());

        // Works on an automaton-backed term too.
        let automaton_backed = exact.intersection([&term]).unwrap();
        assert!(matches!(automaton_backed, Term::Automaton(_)));
        assert!(automaton_backed.matches("abc").unwrap());
        assert!(!automaton_backed.matches("abcd").unwrap());

        // The empty language matches nothing; the empty string matches only "".
        assert!(!Term::new_empty().matches("").unwrap());
        assert!(Term::new_empty_string().matches("").unwrap());
        assert!(!Term::new_empty_string().matches("a").unwrap());
    }

    #[test]
    fn test_from_str_and_from_conversions() {
        // `FromStr` agrees with `from_pattern`.
        let parsed: Term = "abc".parse().unwrap();
        assert_eq!(parsed, Term::from_pattern("abc").unwrap());

        // Invalid patterns surface as parse errors (backreferences are not regular).
        assert!(r"(a)\1".parse::<Term>().is_err());

        // `From<RegularExpression>` / `From<FastAutomaton>` match the explicit constructors.
        let regex = RegularExpression::new("abc").unwrap();
        let from_into: Term = regex.clone().into();
        assert_eq!(from_into, Term::from_regex(regex));

        let automaton = Term::from_pattern("abc")
            .unwrap()
            .to_automaton()
            .unwrap()
            .into_owned();
        let from_into: Term = automaton.clone().into();
        assert_eq!(from_into, Term::from_automaton(automaton));
    }

    #[test]
    fn test_is_deterministic_and_determinize() {
        // A pattern-backed term is never reported deterministic (NFA form).
        let regex_term = Term::from_pattern("(abc|de){2}").unwrap();
        assert!(!regex_term.is_deterministic());

        // `determinize` produces a deterministic, language-equivalent term.
        let dfa = regex_term.determinize().unwrap();
        assert!(dfa.is_deterministic());
        assert!(regex_term.equivalent(&dfa).unwrap());

        // Determinizing an already-deterministic term keeps it deterministic
        // and equivalent.
        let dfa2 = dfa.determinize().unwrap();
        assert!(dfa2.is_deterministic());
        assert!(dfa.equivalent(&dfa2).unwrap());
    }

    #[test]
    fn test_is_minimal_and_minimize() {
        // A pattern-backed term is never reported minimal.
        let regex_term = Term::from_pattern("(abc|de){2}").unwrap();
        assert!(!regex_term.is_minimal());

        // `minimize` produces a minimal, language-equivalent term.
        let minimal = regex_term.minimize().unwrap();
        assert!(minimal.is_minimal());
        assert!(minimal.is_deterministic()); // minimal implies deterministic
        assert!(regex_term.equivalent(&minimal).unwrap());
    }

    #[test]
    fn test_eq_is_structural_not_language() {
        // Same language, different representation: structurally unequal, but
        // language-equivalent. `==` must not be mistaken for `equivalent`.
        let regex_term = Term::from_pattern("(a|b)*").unwrap();
        let automaton_term = Term::from_automaton(regex_term.to_automaton().unwrap().into_owned());

        assert_ne!(regex_term, automaton_term);
        assert!(regex_term.equivalent(&automaton_term).unwrap());
    }

    #[test]
    fn test_repeat_range_edges() {
        let term = Term::from_pattern("abc").unwrap();

        // Unbounded / unset bounds.
        assert_eq!("(abc)*", term.repeat(..).unwrap().to_pattern().unwrap());
        assert_eq!("(abc){2,}", term.repeat(2..).unwrap().to_pattern().unwrap());
        assert_eq!(
            "(abc){0,2}",
            term.repeat(..3).unwrap().to_pattern().unwrap()
        );

        // Zero repetitions is the empty string.
        assert!(term.repeat(0..=0).unwrap().is_empty_string().unwrap());

        // A range whose normalized max < min denotes no valid repetition count,
        // so the simplifier reduces it to the empty language (matches nothing).
        // (Bounds from variables: a literal reversed range trips a lint.)
        let (min, max) = (5u32, 3u32);
        assert!(term.repeat(min..max).unwrap().is_empty().unwrap());
    }

    #[test]
    fn test_iter_strings_is_lazy_on_infinite_language() {
        // Must not hang on an infinite language: take a finite prefix.
        let term = Term::from_pattern("a+").unwrap();
        let first = term
            .iter_strings()
            .take(5)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(5, first.len());
    }

    #[test]
    fn test_iter_strings_propagates_error_then_ends() {
        use crate::execution_profile::ExecutionProfileBuilder;

        // A tight state budget makes the underlying `to_automaton` fail; the
        // iterator must surface that error once and then terminate.
        let term = Term::from_pattern("abcdef").unwrap();
        let profile = ExecutionProfileBuilder::new()
            .max_number_of_states(1)
            .build();

        profile.run(|| {
            let mut it = term.iter_strings();
            assert!(matches!(
                it.next(),
                Some(Err(EngineError::AutomatonHasTooManyStates))
            ));
            assert!(it.next().is_none());
        });
    }

    #[test]
    fn test_variadic_ops_with_no_operands_equal_self() {
        let term = Term::from_pattern("abc").unwrap();

        assert!(
            term.concat(std::iter::empty::<&Term>())
                .unwrap()
                .equivalent(&term)
                .unwrap()
        );
        assert!(
            term.union(std::iter::empty::<&Term>())
                .unwrap()
                .equivalent(&term)
                .unwrap()
        );
        assert!(
            term.intersection(std::iter::empty::<&Term>())
                .unwrap()
                .equivalent(&term)
                .unwrap()
        );
    }
}
