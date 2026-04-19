use std::fmt::{self};

/// An error thrown by the engine.
#[derive(Debug, PartialEq, Eq)]
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
        }
    }
}

impl std::error::Error for EngineError {}
