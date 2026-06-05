use std::fmt::{self};

/// An error thrown by the engine.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// Invalid character used in regex.
    InvalidCharacterInRegex,
    /// The operation took too much time.
    OperationTimeOutError,
    /// The automaton has too many states.
    AutomatonHasTooManyStates,
    /// The regular expression can not be parsed.
    RegexSyntaxError(String),
    /// The provided range can not be built from the spanning set.
    ConditionInvalidRange,
    /// The repetition bounds are invalid: the maximum is below the minimum.
    InvalidRepetitionBounds(u32, u32),
    /// The condition does not match the spanning set it is evaluated against.
    IncompatibleSpanningSet,
    /// The operation requires a deterministic automaton, and implicit
    /// determinization is disabled by the execution profile.
    DeterministicAutomatonRequired,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidCharacterInRegex => write!(f, "Invalid character used in regex."),
            EngineError::OperationTimeOutError => write!(f, "The operation took too much time."),
            EngineError::AutomatonHasTooManyStates => {
                write!(f, "The automaton has too many states.")
            }
            EngineError::RegexSyntaxError(err) => write!(f, "{err}."),
            EngineError::ConditionInvalidRange => write!(
                f,
                "The provided range can not be built from the spanning set."
            ),
            EngineError::InvalidRepetitionBounds(min, max) => write!(
                f,
                "The repetition maximum ({max}) is below its minimum ({min})."
            ),
            EngineError::IncompatibleSpanningSet => write!(
                f,
                "The condition does not match the spanning set it is evaluated against."
            ),
            EngineError::DeterministicAutomatonRequired => write!(
                f,
                "The operation requires a deterministic automaton, and implicit determinization is disabled by the execution profile."
            ),
        }
    }
}

impl std::error::Error for EngineError {}
