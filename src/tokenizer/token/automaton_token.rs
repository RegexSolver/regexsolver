use self::range_token::RangeToken;

use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum AutomatonToken {
    Range(RangeToken),
    State(u16),
    AcceptState,
    SeparatorState,
    Error,
}

impl Ord for AutomatonToken {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.to_fair_token().unwrap()).cmp(&other.to_fair_token().unwrap())
    }
}

impl PartialOrd for AutomatonToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl AutomatonToken {
    const TK_AI_RANGE: u8 = 0;
    const TK_AI_STATE: u8 = Self::TK_AI_RANGE + RangeToken::AI_VOCABULARY_SIZE;
    const TK_AI_ACCEPT_STATE: u8 = Self::TK_AI_STATE + Self::AI_MAX_NUMBER_OF_STATES;
    const TK_AI_SEPARATOR_STATE: u8 = Self::TK_AI_ACCEPT_STATE + 1;

    pub const AI_MAX_NUMBER_OF_STATES: u8 = 100;

    pub const AI_VOCABULARY_SIZE: u8 = Self::TK_AI_SEPARATOR_STATE + 1;

    const TK_FAIR_RANGE: u16 = 0;
    const TK_FAIR_STATE: u16 = Self::TK_FAIR_RANGE + RangeToken::FAIR_VOCABULARY_SIZE;
    const TK_FAIR_ACCEPT_STATE: u16 = Self::TK_FAIR_STATE + Self::FAIR_MAX_NUMBER_OF_STATES;
    const TK_FAIR_SEPARATOR_STATE: u16 = Self::TK_FAIR_ACCEPT_STATE + 1;

    pub const FAIR_MAX_NUMBER_OF_STATES: u16 = 65_000;

    pub const FAIR_VOCABULARY_SIZE: u16 = Self::TK_FAIR_SEPARATOR_STATE + 1;
}

impl Token for AutomatonToken {
    fn from_ai_token(token: u8) -> AutomatonToken {
        if (Self::TK_AI_RANGE..Self::TK_AI_RANGE + RangeToken::AI_VOCABULARY_SIZE).contains(&token)
        {
            AutomatonToken::Range(RangeToken::from_ai_token(token))
        } else if (Self::TK_AI_STATE..Self::TK_AI_STATE + Self::AI_MAX_NUMBER_OF_STATES)
            .contains(&token)
        {
            AutomatonToken::State((token - Self::TK_AI_STATE) as u16)
        } else if token == Self::TK_AI_ACCEPT_STATE {
            AutomatonToken::AcceptState
        } else if token == Self::TK_AI_SEPARATOR_STATE {
            AutomatonToken::SeparatorState
        } else {
            AutomatonToken::Error
        }
    }

    fn to_ai_token(&self) -> Result<u8, TokenError> {
        Ok(match self {
            AutomatonToken::Range(r) => r.to_ai_token()?,
            AutomatonToken::State(s) => {
                let max = Self::AI_MAX_NUMBER_OF_STATES;
                let s = *s as u8;
                if s > max {
                    return Err(TokenError::TokenOutOfBound("State", max.into(), s.into()));
                }
                s + Self::TK_AI_STATE
            }
            AutomatonToken::AcceptState => Self::TK_AI_ACCEPT_STATE,
            AutomatonToken::SeparatorState => Self::TK_AI_SEPARATOR_STATE,
            AutomatonToken::Error => return Err(TokenError::UnknownToken),
        })
    }

    fn from_fair_token(token: u16) -> AutomatonToken {
        if (Self::TK_FAIR_RANGE..Self::TK_FAIR_RANGE + RangeToken::FAIR_VOCABULARY_SIZE)
            .contains(&token)
        {
            AutomatonToken::Range(RangeToken::from_fair_token(token))
        } else if (Self::TK_FAIR_STATE..Self::TK_FAIR_STATE + Self::FAIR_MAX_NUMBER_OF_STATES)
            .contains(&token)
        {
            AutomatonToken::State(token - Self::TK_FAIR_STATE)
        } else if token == Self::TK_FAIR_ACCEPT_STATE {
            AutomatonToken::AcceptState
        } else if token == Self::TK_FAIR_SEPARATOR_STATE {
            AutomatonToken::SeparatorState
        } else {
            AutomatonToken::Error
        }
    }

    fn to_fair_token(&self) -> Result<u16, TokenError> {
        Ok(match self {
            AutomatonToken::Range(r) => r.to_fair_token()?,
            AutomatonToken::State(s) => {
                let max = Self::FAIR_MAX_NUMBER_OF_STATES;
                let s = *s;
                if s > max {
                    return Err(TokenError::TokenOutOfBound("State", max.into(), s.into()));
                }
                s + Self::TK_FAIR_STATE
            }
            AutomatonToken::AcceptState => Self::TK_FAIR_ACCEPT_STATE,
            AutomatonToken::SeparatorState => Self::TK_FAIR_SEPARATOR_STATE,
            AutomatonToken::Error => return Err(TokenError::UnknownToken),
        })
    }
}
