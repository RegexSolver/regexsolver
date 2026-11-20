use std::fmt::{self};

/// An error returned by an operation of the crate.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// The pattern holds a literal that is not valid UTF-8.
    InvalidCharacterInRegex,
    /// The operation ran past the timeout of the
    /// [`ExecutionProfile`](crate::execution_profile::ExecutionProfile).
    OperationTimeOutError,
    /// The operation needed more states than the `max_number_of_states` of the
    /// [`ExecutionProfile`](crate::execution_profile::ExecutionProfile).
    AutomatonHasTooManyStates,
    /// The pattern cannot be parsed; the string describes why.
    RegexSyntaxError(String),
    /// The spanning set cannot express the range.
    ConditionInvalidRange,
    /// The maximum of a repetition, the second value, is below its minimum, the
    /// first.
    InvalidRepetitionBounds(u32, u32),
    /// The condition was not built over the spanning set it is read with.
    IncompatibleSpanningSet,
    /// The operation needs a deterministic automaton, and the execution profile
    /// disables implicit determinization.
    DeterministicAutomatonRequired,
    /// The pattern uses a construct the engine cannot represent: an anchor or a
    /// word boundary where it is not supported, or an inline flag. The string
    /// names it.
    UnsupportedRegexFeature(String),
    /// A [`RegularExpression`](crate::regex::RegularExpression) built by hand
    /// nests deeper than the limit, which the value holds. A parsed pattern
    /// never does.
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
