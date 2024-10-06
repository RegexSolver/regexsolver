use std::fmt::Display;

use super::*;

pub mod automaton_token;
pub mod range_token;
pub mod regex_operations_token;
pub mod regex_token;

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

pub trait Token {
    fn from_ai_token(token: u8) -> Self;

    fn to_ai_token(&self) -> Result<u8, TokenError>;

    fn to_ai_tokens(tokens: &[Self]) -> Result<Vec<u8>, TokenError>
    where
        Self: Sized,
    {
        let mut vec = Vec::with_capacity(tokens.len());
        for token in tokens {
            vec.push(token.to_ai_token()?);
        }
        Ok(vec)
    }

    fn from_fair_token(token: u16) -> Self;

    fn to_fair_token(&self) -> Result<u16, TokenError>;

    fn to_fair_tokens(tokens: &[Self]) -> Result<Vec<u16>, TokenError>
    where
        Self: Sized,
    {
        let mut vec = Vec::with_capacity(tokens.len());
        for token in tokens {
            vec.push(token.to_fair_token()?);
        }
        Ok(vec)
    }
}
