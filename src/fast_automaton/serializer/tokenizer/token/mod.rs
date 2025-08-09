use std::fmt::Display;

use super::*;

pub mod automaton_token;
pub mod range_token;

#[derive(Debug, PartialEq, Eq)]
pub enum TokenError {
    TokenOutOfBound(&'static str, usize, usize),
    UnknownToken,
    SyntaxError,
}

impl Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenError::TokenOutOfBound(token, expected, got) => write!(
                f,
                "TokenOutOfBound: {token}, expected: {expected}, got: {got}."
            ),
            TokenError::UnknownToken => write!(f, "UnknownToken"),
            TokenError::SyntaxError => write!(f, "SyntaxError"),
        }
    }
}