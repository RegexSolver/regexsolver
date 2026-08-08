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
    /// of matching on the direct child's variant — iteratively, since a
    /// hand-built tree can chain such wrappers arbitrarily deep.
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

    /// Returns a heuristic score for the readability of the pattern.
    pub fn evaluate_complexity(&self) -> f64 {
        let (score, depth, _) = self.eval_inner();
        score + Self::depth_penalty(depth)
    }

    /// Returns: (score, max_depth, contains_repetition)
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
