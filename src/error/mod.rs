use std::fmt::{self};

use crate::tokenizer::token::TokenError;

#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    InvalidCharacterInRegex,
    OperationTimeOutError,
    AutomatonShouldBeDeterministic,
    AutomatonHasTooManyStates,
    RegexSyntaxError(String),
    TooMuchTerms(usize, usize),
    ConditionInvalidRange,
    ConditionIndexOutOfBound,
    TokenError(TokenError),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidCharacterInRegex => write!(f, "Invalid character used in regex."),
            EngineError::OperationTimeOutError => write!(f, "The operation took too much time."),
            EngineError::AutomatonShouldBeDeterministic => write!(f, "The given automaton should be deterministic."),
            EngineError::AutomatonHasTooManyStates => write!(f, "The automaton has too many states."),
            EngineError::RegexSyntaxError(err) => write!(f, "{err}."),
            EngineError::TooMuchTerms(max, got) => write!(f, "Too many terms are used in this operation, the maximum allowed for your plan is {max} and you used {got}."),
            EngineError::TokenError(err) =>  write!(f, "{err}."),
            EngineError::ConditionInvalidRange => write!(f, "The provided range can not be built from bases."),
            EngineError::ConditionIndexOutOfBound => write!(f, "The provided index is out of bound of the condition."),
        }
    }
}

impl std::error::Error for EngineError {}

impl EngineError {
    /// Determine if the error is a server error.
    /// A server error should not be shown to the end user.
    pub fn is_server_error(&self) -> bool {
        match self {
            EngineError::InvalidCharacterInRegex => false,
            EngineError::OperationTimeOutError => false,
            EngineError::AutomatonShouldBeDeterministic => true,
            EngineError::AutomatonHasTooManyStates => false,
            EngineError::RegexSyntaxError(_) => false,
            EngineError::TooMuchTerms(_, _) => false,
            EngineError::TokenError(_) => false,
            EngineError::ConditionInvalidRange => true,
            EngineError::ConditionIndexOutOfBound => true,
        }
    }
}
