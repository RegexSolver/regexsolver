use self::regex_token::RegexToken;

use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum RegexOperationsToken {
    RegexToken(RegexToken),
    And,
    Not,
    Error,
}

impl Ord for RegexOperationsToken {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.to_ai_token().unwrap()).cmp(&other.to_ai_token().unwrap())
    }
}

impl PartialOrd for RegexOperationsToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl RegexOperationsToken {
    const TK_AI_REGEX_TOKEN: u8 = 0;
    const TK_AI_AND: u8 = Self::TK_AI_REGEX_TOKEN + RegexToken::AI_VOCABULARY_SIZE;
    const TK_AI_NOT: u8 = Self::TK_AI_AND + 1;

    pub const AI_VOCABULARY_SIZE: u8 = Self::TK_AI_NOT + 1;
}

impl Token for RegexOperationsToken {
    fn from_ai_token(token: u8) -> RegexOperationsToken {
        if (Self::TK_AI_REGEX_TOKEN..Self::TK_AI_REGEX_TOKEN + RegexToken::AI_VOCABULARY_SIZE)
            .contains(&token)
        {
            RegexOperationsToken::RegexToken(RegexToken::from_ai_token(token))
        } else if token == Self::TK_AI_AND {
            RegexOperationsToken::And
        } else if token == Self::TK_AI_NOT {
            RegexOperationsToken::Not
        } else {
            RegexOperationsToken::Error
        }
    }

    fn to_ai_token(&self) -> Result<u8, TokenError> {
        Ok(match self {
            RegexOperationsToken::RegexToken(regex_token) => regex_token.to_ai_token()?,
            RegexOperationsToken::And => Self::TK_AI_AND,
            RegexOperationsToken::Not => Self::TK_AI_NOT,
            RegexOperationsToken::Error => return Err(TokenError::UnknownToken),
        })
    }

    fn from_fair_token(_: u16) -> RegexOperationsToken {
        panic!("A RegexOperationsToken does not have a FAIR representation.")
    }

    fn to_fair_token(&self) -> Result<u16, TokenError> {
        panic!("A RegexOperationsToken does not have a FAIR representation.")
    }
}
