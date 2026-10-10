use std::{cmp, collections::VecDeque, fmt::Display};

use crate::execution_profile::ExecutionProfile;
use regex_charclass::CharacterClass;
use regex_syntax::hir::{Class, ClassBytes, ClassUnicode, Hir, HirKind, Look};

use self::fast_automaton::FastAutomaton;

use super::*;

mod analyze;
mod builder;
mod operation;

/// Represents a regular expression.
///
/// The variants are public and freely constructible and matchable. Values
/// can also be built with the parser ([`new`](Self::new) /
/// [`parse`](Self::parse)) or the simplifying combinators
/// ([`concat`](Self::concat), [`union`](Self::union),
/// [`repeat`](Self::repeat)). A directly-constructed repetition whose
/// maximum is below its minimum denotes no valid language and is rejected
/// with [`EngineError::InvalidRepetitionBounds`] when converted by
/// [`to_automaton`](Self::to_automaton).
///
/// ```
/// use regexsolver::regex::RegularExpression;
///
/// let regex = RegularExpression::new("a{2,3}").unwrap();
/// if let RegularExpression::Repetition(inner, min, max) = &regex {
///     assert_eq!((*min, *max), (2, Some(3)));
///     assert_eq!(inner.to_string(), "a");
/// }
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[must_use = "regular expressions are immutable; operations return a new expression"]
pub enum RegularExpression {
    /// A single character drawn from the given range; an empty range denotes
    /// the empty language `[]`.
    Character(CharRange),
    /// `r{min,max}`; `None` means unbounded. Expected invariant: `max >= min`
    /// when bounded (checked by [`to_automaton`](Self::to_automaton)).
    Repetition(Box<RegularExpression>, u32, Option<u32>),
    /// The concatenation of the parts in order; no parts denotes the empty
    /// string `""`.
    Concat(VecDeque<RegularExpression>),
    /// The union of the parts; no parts denotes the empty language `[]`.
    Alternation(Vec<RegularExpression>),
}

impl Display for RegularExpression {
    /// Streams the pattern via an explicit work stack instead of recursion,
    /// so printing a pathologically deep hand-built tree cannot overflow the
    /// call stack.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        enum Frame<'a> {
            Node(&'a RegularExpression),
            Literal(&'static str),
            Quantifier(u32, Option<u32>),
        }

        // Frames pop in reverse push order, so children are pushed
        // right-to-left and trailing literals before them.
        let mut stack = vec![Frame::Node(self)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Literal(literal) => f.write_str(literal)?,
                Frame::Quantifier(min, max_opt) => {
                    if min == 0 && max_opt.is_none() {
                        write!(f, "*")?;
                    } else if min == 1 && max_opt.is_none() {
                        write!(f, "+")?;
                    } else if min == 0 && max_opt == Some(1) {
                        write!(f, "?")?;
                    } else if let Some(max) = max_opt {
                        if max == min {
                            write!(f, "{{{max}}}")?;
                        } else {
                            write!(f, "{{{min},{max}}}")?;
                        }
                    } else {
                        write!(f, "{{{min},}}")?;
                    }
                }
                Frame::Node(RegularExpression::Character(range)) => {
                    if range.is_empty() {
                        write!(f, "[]")?;
                    } else {
                        write!(f, "{}", range.to_regex())?;
                    }
                }
                Frame::Node(RegularExpression::Repetition(regular_expression, min, max_opt)) => {
                    stack.push(Frame::Quantifier(*min, *max_opt));
                    if RegularExpression::quantifier_needs_parens(regular_expression) {
                        stack.push(Frame::Literal(")"));
                        stack.push(Frame::Node(regular_expression));
                        stack.push(Frame::Literal("("));
                    } else {
                        stack.push(Frame::Node(regular_expression));
                    }
                }
                Frame::Node(RegularExpression::Concat(concat)) => {
                    stack.extend(concat.iter().rev().map(Frame::Node));
                }
                Frame::Node(RegularExpression::Alternation(alternation)) => {
                    match alternation.as_slice() {
                        [] => write!(f, "[]")?,
                        [single] => stack.push(Frame::Node(single)),
                        parts => {
                            stack.push(Frame::Literal(")"));
                            for (i, regex) in parts.iter().enumerate().rev() {
                                stack.push(Frame::Node(regex));
                                if i != 0 {
                                    stack.push(Frame::Literal("|"));
                                }
                            }
                            stack.push(Frame::Literal("("));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl RegularExpression {
    /// Whether applying a quantifier to the printed form of `r` requires
    /// wrapping it in a group. Singleton `Concat`/`Alternation` wrappers
    /// print transparently, so the decision must look through them instead
    /// of matching on the direct child's variant, and do so iteratively,
    /// since a hand-built tree can chain such wrappers arbitrarily deep.
    fn quantifier_needs_parens(mut r: &RegularExpression) -> bool {
        loop {
            match r {
                // Prints as a single char or a [class]: one token.
                RegularExpression::Character(..) => return false,
                RegularExpression::Repetition(..) => return true,
                RegularExpression::Concat(parts) => match parts.len() {
                    1 => r = &parts[0],
                    // Covers both the empty concatenation (which prints as ""
                    // and needs the explicit group; `()*` is valid but a bare
                    // `*` is not) and real multi-part concatenations.
                    _ => return true,
                },
                RegularExpression::Alternation(parts) => match parts.len() {
                    // The empty alternation prints as "[]": one token.
                    0 => return false,
                    1 => r = &parts[0],
                    // Multi-part alternations print self-parenthesized.
                    _ => return false,
                },
            }
        }
    }

    /// Returns `true` if the regular expression matches the empty language.
    pub fn is_empty(&self) -> bool {
        match self {
            RegularExpression::Alternation(alternation) => alternation.is_empty(),
            RegularExpression::Character(range) => range.is_empty(),
            _ => false,
        }
    }

    /// Returns `true` if the regular expression matches only the empty string `""`.
    pub fn is_empty_string(&self) -> bool {
        match self {
            RegularExpression::Concat(concat) => concat.is_empty(),
            _ => false,
        }
    }

    /// Returns `true` if the regular expression matches all possible strings.
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

    /// The deepest a directly-constructed expression tree may nest before
    /// [`to_automaton`](Self::to_automaton) refuses to convert it. Parsed
    /// patterns never approach this (`regex-syntax` caps parse nesting far
    /// lower); it only bounds the recursion so a pathologically deep hand-built
    /// tree returns an error instead of overflowing the stack.
    pub const MAX_NESTING_DEPTH: usize = 1000;

    /// Returns an error if the tree nests deeper than [`MAX_NESTING_DEPTH`](Self::MAX_NESTING_DEPTH).
    ///
    /// Uses an explicit stack (not recursion) so measuring a deep tree cannot
    /// itself overflow, and bails as soon as the limit is exceeded.
    fn assert_depth_within_limit(&self) -> Result<(), EngineError> {
        let mut stack = vec![(self, 1usize)];
        while let Some((node, depth)) = stack.pop() {
            if depth > Self::MAX_NESTING_DEPTH {
                return Err(EngineError::RegexTooDeeplyNested(Self::MAX_NESTING_DEPTH));
            }
            match node {
                RegularExpression::Character(_) => {}
                RegularExpression::Repetition(inner, _, _) => stack.push((inner, depth + 1)),
                RegularExpression::Concat(parts) => {
                    stack.extend(parts.iter().map(|p| (p, depth + 1)));
                }
                RegularExpression::Alternation(parts) => {
                    stack.extend(parts.iter().map(|p| (p, depth + 1)));
                }
            }
        }
        Ok(())
    }

    /// Converts the regular expression to an equivalent [`FastAutomaton`].
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn to_automaton(&self) -> Result<FastAutomaton, EngineError> {
        // Both whole-tree checks run once here: a subtree can never nest
        // deeper than the tree it came from, and the execution profile's
        // thread-locals don't change mid-conversion.
        self.assert_depth_within_limit()?;
        self.to_automaton_inner(&ExecutionProfile::get())
    }

    fn to_automaton_inner(
        &self,
        execution_profile: &ExecutionProfile,
    ) -> Result<FastAutomaton, EngineError> {
        // The per-node state estimate is load-bearing (a subtree like the
        // inner of `big{0,0}` can exceed the budget even when the root's
        // estimate doesn't), but is only worth its O(subtree) walk when a
        // state limit is actually configured.
        if execution_profile.limits_number_of_states() {
            execution_profile.assert_max_number_of_states(self.get_number_of_states_in_nfa())?;
        }

        match self {
            RegularExpression::Character(range) => Ok(FastAutomaton::new_from_range(range)),
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                // The variants are freely constructible; invalid bounds are
                // rejected at this boundary instead.
                if let Some(max) = max_opt
                    && max < min
                {
                    return Err(EngineError::InvalidRepetitionBounds(*min, *max));
                }
                let mut automaton = regular_expression.to_automaton_inner(execution_profile)?;
                automaton.repeat_mut(*min, *max_opt)?;
                Ok(automaton)
            }
            RegularExpression::Concat(concat) => {
                let mut concats = Vec::with_capacity(concat.len());
                for c in concat.iter() {
                    concats.push(c.to_automaton_inner(execution_profile)?);
                }
                FastAutomaton::concat_all(&concats)
            }
            RegularExpression::Alternation(alternation) => {
                let mut alternates = Vec::with_capacity(alternation.len());
                for c in alternation.iter() {
                    alternates.push(c.to_automaton_inner(execution_profile)?);
                }
                FastAutomaton::union_all(&alternates)
            }
        }
    }

    /// Returns the cognitive complexity of the pattern: a readability score,
    /// independent of the characters used.
    ///
    /// - class: 1.
    /// - concatenation: the sum of its parts.
    /// - alternation: its branches read one level deeper, plus `1 + depth` per
    ///   extra branch.
    /// - quantifier: its operand, plus the quantifier's weight × `(1 + depth)`;
    ///   `?`, `*`, `+` are cheapest, then `{n}`, `{n,}`, `{m,n}`. A quantified
    ///   group reads one level deeper, and a quantifier with another quantifier
    ///   inside it pays an extra `1 + depth`.
    ///
    /// The empty string costs 0; the empty language `[]` is a class (1).
    pub fn evaluate_complexity(&self) -> f64 {
        const CLASS: f64 = 1.0;
        const NESTED_QUANTIFIER: f64 = 1.0;

        // Iterative, since a hand-built tree can nest arbitrarily deep. Each
        // node adds its own cost at its depth. A quantifier's extra cost for
        // holding another one is settled at the end: each quantifier marks
        // the nearest one around it, which suffices, as that one is in turn
        // inside any further ones.
        let mut score = 0.0;
        // Per quantifier: its `1 + depth`, and whether it holds another one.
        let mut quantifiers: Vec<(f64, bool)> = Vec::new();
        let mut stack = vec![(self, 0usize, None::<usize>)];
        while let Some((node, depth, enclosing)) = stack.pop() {
            let nesting = 1.0 + depth as f64;
            match node {
                RegularExpression::Character(_) => score += CLASS,
                RegularExpression::Concat(items) => {
                    stack.extend(items.iter().map(|item| (item, depth, enclosing)));
                }
                RegularExpression::Alternation(branches) => {
                    score += branches.len().saturating_sub(1) as f64 * nesting;
                    stack.extend(branches.iter().map(|branch| (branch, depth + 1, enclosing)));
                }
                RegularExpression::Repetition(inner, min, max) => {
                    score += Self::quantifier_weight(*min, *max) * nesting;
                    if let Some(enclosing) = enclosing {
                        quantifiers[enclosing].1 = true;
                    }
                    quantifiers.push((nesting, false));
                    // A single class needs no group; anything else is a group
                    // the reader has to enter.
                    let inner_depth = match **inner {
                        RegularExpression::Character(_) => depth,
                        _ => depth + 1,
                    };
                    stack.push((inner, inner_depth, Some(quantifiers.len() - 1)));
                }
            }
        }
        score
            + quantifiers
                .iter()
                .filter(|(_, nested)| *nested)
                .map(|(nesting, _)| NESTED_QUANTIFIER * nesting)
                .sum::<f64>()
    }

    fn quantifier_weight(min: u32, max: Option<u32>) -> f64 {
        /// `?`, `*`, `+`.
        const SIMPLE_QUANTIFIER: f64 = 0.5;
        /// `{n}`.
        const EXACT_COUNT: f64 = 1.0;
        /// `{n,}`.
        const OPEN_COUNT: f64 = 1.5;
        /// `{m,n}`.
        const RANGE_COUNT: f64 = 2.0;

        match (min, max) {
            (0, Some(1)) | (0, None) | (1, None) => SIMPLE_QUANTIFIER,
            (_, None) => OPEN_COUNT,
            (min, Some(max)) if min == max => EXACT_COUNT,
            _ => RANGE_COUNT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_complexity() {
        let complexity = |pattern: &str| {
            RegularExpression::parse(pattern, false)
                .unwrap()
                .evaluate_complexity()
        };
        assert_eq!(complexity(""), 0.0);
        assert_eq!(complexity("[]"), 1.0);
        assert_eq!(complexity("abc"), 3.0);
        assert_eq!(complexity("[a-c]*"), 1.5);
        assert_eq!(complexity("(ab|c)"), 4.0);
        assert_eq!(complexity("(ab)*"), 2.5);
        assert_eq!(complexity("(a*b)+"), 4.5);
        assert!(complexity("[a-egh]*hbh?") < complexity("(h+b)+h?"));
    }

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

    /// Drops a deep chain-shaped tree level by level. `Box`'s drop glue
    /// recurses (see [`RegularExpression::MAX_NESTING_DEPTH`]), so the deep
    /// trees below must not be dropped whole.
    fn drop_chain_iteratively(mut regex: RegularExpression) {
        loop {
            regex = match regex {
                RegularExpression::Repetition(inner, _, _) => *inner,
                RegularExpression::Concat(mut parts) if parts.len() == 1 => {
                    parts.pop_front().expect("len() == 1")
                }
                RegularExpression::Alternation(mut parts) if parts.len() == 1 => {
                    parts.pop().expect("len() == 1")
                }
                _ => return,
            };
        }
    }

    // Display streams via an explicit work stack: a hand-built tree far
    // deeper than any recursive formatter could survive must still print.
    #[test]
    fn display_does_not_recurse_on_deep_trees() {
        const DEPTH: usize = 100_000;

        // `((...(a*)*...)*)*`: every level goes through the parenthesization
        // decision and the quantifier path.
        let mut regex = RegularExpression::new("a").unwrap();
        for _ in 0..DEPTH {
            regex = RegularExpression::Repetition(Box::new(regex), 0, None);
        }
        let printed = regex.to_string();
        assert_eq!(3 * DEPTH - 1, printed.len());
        assert!(printed.starts_with("((("));
        assert!(printed.ends_with(")*)*)*"));
        drop_chain_iteratively(regex);

        // A deep chain of singleton wrappers prints transparently, and the
        // quantifier's parenthesization must look through all of them
        // without recursing.
        let mut regex = RegularExpression::new("ab").unwrap();
        for i in 0..DEPTH {
            regex = if i % 2 == 0 {
                RegularExpression::Concat(VecDeque::from([regex]))
            } else {
                RegularExpression::Alternation(vec![regex])
            };
        }
        let regex = RegularExpression::Repetition(Box::new(regex), 0, None);
        assert_eq!("(ab)*", regex.to_string());
        drop_chain_iteratively(regex);
    }
}
