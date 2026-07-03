use std::fmt::{self};

/// An error thrown by the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// Invalid character used in regex.
    InvalidCharacterInRegex,
    /// The operation took too much time.
    OperationTimeOutError,
    /// The automaton has too many states.
    AutomatonHasTooManyStates,
    /// The regular expression cannot be parsed.
    RegexSyntaxError(String),
    /// The provided range cannot be built from the spanning set.
    ConditionInvalidRange,
    /// The repetition bounds are invalid: the maximum is below the minimum.
    InvalidRepetitionBounds(u32, u32),
    /// The condition does not match the spanning set it is evaluated against.
    IncompatibleSpanningSet,
    /// The operation requires a deterministic automaton, and implicit
    /// determinization is disabled by the execution profile.
    DeterministicAutomatonRequired,
    /// The pattern uses a regex feature the engine cannot represent (an
    /// unsupported anchor/boundary position, or inline flags). The string
    /// describes the specific feature.
    UnsupportedRegexFeature(String),
    /// A directly-constructed [`RegularExpression`](crate::regex::RegularExpression)
    /// tree nests deeper than the engine converts safely (the payload is the
    /// limit). Parsed patterns never hit this; it only guards against
    /// stack-overflowing on pathologically deep hand-built trees.
    RegexTooDeeplyNested(usize),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidCharacterInRegex => {
                write!(f, "invalid character used in regex")
            }
            EngineError::OperationTimeOutError => write!(f, "the operation timed out"),
            EngineError::AutomatonHasTooManyStates => {
                write!(f, "the automaton has too many states")
            }
            EngineError::RegexSyntaxError(err) => write!(f, "invalid regex syntax: {err}"),
            EngineError::ConditionInvalidRange => write!(
                f,
                "the provided range cannot be built from the spanning set"
            ),
            EngineError::InvalidRepetitionBounds(min, max) => write!(
                f,
                "the repetition maximum ({max}) is below its minimum ({min})"
            ),
            EngineError::IncompatibleSpanningSet => write!(
                f,
                "the condition does not match the spanning set it is evaluated against"
            ),
            EngineError::DeterministicAutomatonRequired => write!(
                f,
                "the operation requires a deterministic automaton, and implicit determinization is disabled by the execution profile"
            ),
            EngineError::UnsupportedRegexFeature(feature) => {
                write!(f, "unsupported regex feature: {feature}")
            }
            EngineError::RegexTooDeeplyNested(limit) => write!(
                f,
                "the regular expression is nested more than {limit} levels deep"
            ),
        }
    }
}

impl std::error::Error for EngineError {}
