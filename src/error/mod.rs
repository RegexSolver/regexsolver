use std::fmt::{self};

#[cfg(feature = "serializable")]
use crate::tokenizer::token::TokenError;

/// An error thrown by the engine.
#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    /// Invalid character used in regex.
    InvalidCharacterInRegex,
    /// The operation took too much time.
    OperationTimeOutError,
    /// The given automaton should be deterministic.
    AutomatonShouldBeDeterministic,
    /// The automaton has too many states.
    AutomatonHasTooManyStates,
    /// The regular expression can not be parsed.
    RegexSyntaxError(String),
    /// The provided range can not be built from the spanning set.
    ConditionInvalidRange,
    /// The provided index is out of bound of the condition.
    ConditionIndexOutOfBound,
    #[cfg(feature = "serializable")]
    /// There is an error with one of the token.
    TokenError(TokenError),
    /// Computing the cardinality of the provided automaton failed.
    CannotComputeAutomatonCardinality,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidCharacterInRegex => write!(f, "Invalid character used in regex."),
            EngineError::OperationTimeOutError => write!(f, "The operation took too much time."),
            EngineError::AutomatonShouldBeDeterministic => {
                write!(f, "The given automaton should be deterministic.")
            }
            EngineError::AutomatonHasTooManyStates => {
                write!(f, "The automaton has too many states.")
            }
            EngineError::RegexSyntaxError(err) => write!(f, "{err}."),
            #[cfg(feature = "serializable")]
            EngineError::TokenError(err) => write!(f, "{err}."),
            EngineError::ConditionInvalidRange => write!(
                f,
                "The provided range can not be built from the spanning set."
            ),
            EngineError::ConditionIndexOutOfBound => {
                write!(f, "The provided index is out of bound of the condition.")
            }
            EngineError::CannotComputeAutomatonCardinality => write!(
                f,
                "Computing the cardinality of the provided automaton failed."
            ),
        }
    }
}

impl std::error::Error for EngineError {}
