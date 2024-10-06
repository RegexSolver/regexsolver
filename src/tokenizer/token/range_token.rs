use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum RangeToken {
    Total,
    Base(usize),
    Error,
}

impl RangeToken {
    const TK_AI_TOTAL: u8 = 0;
    const TK_AI_BASE: u8 = 1;

    pub const AI_MAX_NUMBER_OF_BASES: u8 = 10;

    pub const AI_VOCABULARY_SIZE: u8 = Self::TK_AI_BASE + Self::AI_MAX_NUMBER_OF_BASES + 1;

    const TK_FAIR_TOTAL: u16 = 0;
    const TK_FAIR_BASE: u16 = 1;

    pub const FAIR_MAX_NUMBER_OF_BASES: u16 = 127;

    pub const FAIR_VOCABULARY_SIZE: u16 = Self::TK_FAIR_BASE + Self::FAIR_MAX_NUMBER_OF_BASES + 1;
}

impl Ord for RangeToken {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.to_fair_token().unwrap()).cmp(&other.to_fair_token().unwrap())
    }
}

impl PartialOrd for RangeToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Token for RangeToken {
    fn from_ai_token(token: u8) -> RangeToken {
        if token == Self::TK_AI_TOTAL {
            RangeToken::Total
        } else if (Self::TK_AI_BASE..Self::TK_AI_BASE + Self::AI_MAX_NUMBER_OF_BASES)
            .contains(&token)
        {
            RangeToken::Base((token - Self::TK_AI_BASE) as usize)
        } else {
            RangeToken::Error
        }
    }

    fn to_ai_token(&self) -> Result<u8, TokenError> {
        Ok(match self {
            RangeToken::Total => Self::TK_AI_TOTAL,
            RangeToken::Base(b) => {
                let max = Self::AI_MAX_NUMBER_OF_BASES;
                let b = *b as u8;
                if b > max {
                    return Err(TokenError::TokenOutOfBound("Base", max.into(), b.into()));
                }
                b + Self::TK_AI_BASE
            }
            RangeToken::Error => return Err(TokenError::UnknownToken),
        })
    }

    fn from_fair_token(token: u16) -> RangeToken {
        if token == Self::TK_FAIR_TOTAL {
            RangeToken::Total
        } else if (Self::TK_FAIR_BASE..Self::TK_FAIR_BASE + Self::FAIR_MAX_NUMBER_OF_BASES)
            .contains(&token)
        {
            RangeToken::Base((token - Self::TK_FAIR_BASE) as usize)
        } else {
            RangeToken::Error
        }
    }

    fn to_fair_token(&self) -> Result<u16, TokenError> {
        Ok(match self {
            RangeToken::Total => Self::TK_FAIR_TOTAL,
            RangeToken::Base(b) => {
                let max = Self::FAIR_MAX_NUMBER_OF_BASES;
                let b = *b as u16;
                if b > max {
                    return Err(TokenError::TokenOutOfBound("Base", max.into(), b.into()));
                }
                b + Self::TK_FAIR_BASE
            }
            RangeToken::Error => return Err(TokenError::UnknownToken),
        })
    }
}
