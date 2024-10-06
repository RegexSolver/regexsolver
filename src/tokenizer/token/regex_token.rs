use self::range_token::RangeToken;

use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum RegexToken {
    Range(RangeToken),
    StartGroup,
    EndGroup,
    Alternation,
    RepetitionNone,
    Repetition(u16),
    Error,
}

impl Ord for RegexToken {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.to_fair_token().unwrap()).cmp(&other.to_fair_token().unwrap())
    }
}

impl PartialOrd for RegexToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl RegexToken {
    const TK_AI_RANGE: u8 = 0;
    const TK_AI_START_GROUP: u8 = Self::TK_AI_RANGE + RangeToken::AI_VOCABULARY_SIZE;
    const TK_AI_END_GROUP: u8 = Self::TK_AI_START_GROUP + 1;
    const TK_AI_ALTERNATION: u8 = Self::TK_AI_END_GROUP + 1;
    const TK_AI_REPETITION_NONE: u8 = Self::TK_AI_ALTERNATION + 1;
    const TK_AI_REPETITION: u8 = Self::TK_AI_REPETITION_NONE + 1;

    pub const AI_MAX_NUMBER_OF_REPETITION: u8 = 10;

    pub const AI_VOCABULARY_SIZE: u8 =
        Self::TK_AI_REPETITION + Self::AI_MAX_NUMBER_OF_REPETITION + 1;

    const TK_FAIR_RANGE: u16 = 0;
    const TK_FAIR_START_GROUP: u16 = Self::TK_FAIR_RANGE + RangeToken::FAIR_VOCABULARY_SIZE;
    const TK_FAIR_END_GROUP: u16 = Self::TK_FAIR_START_GROUP + 1;
    const TK_FAIR_ALTERNATION: u16 = Self::TK_FAIR_END_GROUP + 1;
    const TK_FAIR_REPETITION_NONE: u16 = Self::TK_FAIR_ALTERNATION + 1;
    const TK_FAIR_REPETITION: u16 = Self::TK_FAIR_REPETITION_NONE + 1;

    pub const FAIR_MAX_NUMBER_OF_REPETITION: u16 = 1024;

    pub const FAIR_VOCABULARY_SIZE: u16 =
        Self::TK_FAIR_REPETITION + Self::FAIR_MAX_NUMBER_OF_REPETITION + 1;
}

impl Token for RegexToken {
    fn from_ai_token(token: u8) -> RegexToken {
        if (Self::TK_AI_RANGE..Self::TK_AI_RANGE + RangeToken::AI_VOCABULARY_SIZE).contains(&token)
        {
            RegexToken::Range(RangeToken::from_ai_token(token))
        } else if token == Self::TK_AI_START_GROUP {
            RegexToken::StartGroup
        } else if token == Self::TK_AI_END_GROUP {
            RegexToken::EndGroup
        } else if token == Self::TK_AI_ALTERNATION {
            RegexToken::Alternation
        } else if token == Self::TK_AI_REPETITION_NONE {
            RegexToken::RepetitionNone
        } else if (Self::TK_AI_REPETITION
            ..Self::TK_AI_REPETITION + Self::AI_MAX_NUMBER_OF_REPETITION)
            .contains(&token)
        {
            RegexToken::Repetition((token - Self::TK_AI_REPETITION) as u16)
        } else {
            RegexToken::Error
        }
    }

    fn to_ai_token(&self) -> Result<u8, TokenError> {
        Ok(match self {
            RegexToken::Range(r) => r.to_ai_token()?,
            RegexToken::StartGroup => Self::TK_AI_START_GROUP,
            RegexToken::EndGroup => Self::TK_AI_END_GROUP,
            RegexToken::Alternation => Self::TK_AI_ALTERNATION,
            RegexToken::RepetitionNone => Self::TK_AI_REPETITION_NONE,
            RegexToken::Repetition(r) => {
                let max = Self::AI_MAX_NUMBER_OF_REPETITION;
                let r = *r as u8;
                if r > max {
                    return Err(TokenError::TokenOutOfBound("Repetition", max.into(), r.into()));
                }
                r + Self::TK_AI_REPETITION
            }
            RegexToken::Error => return Err(TokenError::UnknownToken),
        })
    }

    fn from_fair_token(token: u16) -> RegexToken {
        if (Self::TK_FAIR_RANGE..Self::TK_FAIR_RANGE + RangeToken::FAIR_VOCABULARY_SIZE)
            .contains(&token)
        {
            RegexToken::Range(RangeToken::from_fair_token(token))
        } else if token == Self::TK_FAIR_START_GROUP {
            RegexToken::StartGroup
        } else if token == Self::TK_FAIR_END_GROUP {
            RegexToken::EndGroup
        } else if token == Self::TK_FAIR_ALTERNATION {
            RegexToken::Alternation
        } else if token == Self::TK_FAIR_REPETITION_NONE {
            RegexToken::RepetitionNone
        } else if (Self::TK_FAIR_REPETITION
            ..Self::TK_FAIR_REPETITION + Self::FAIR_MAX_NUMBER_OF_REPETITION)
            .contains(&token)
        {
            RegexToken::Repetition(token - Self::TK_FAIR_REPETITION)
        } else {
            RegexToken::Error
        }
    }

    fn to_fair_token(&self) -> Result<u16, TokenError> {
        Ok(match self {
            RegexToken::Range(r) => r.to_fair_token()?,
            RegexToken::StartGroup => Self::TK_FAIR_START_GROUP,
            RegexToken::EndGroup => Self::TK_FAIR_END_GROUP,
            RegexToken::Alternation => Self::TK_FAIR_ALTERNATION,
            RegexToken::RepetitionNone => Self::TK_FAIR_REPETITION_NONE,
            RegexToken::Repetition(r) => {
                let max = Self::FAIR_MAX_NUMBER_OF_REPETITION;
                let r = *r;
                if r > max {
                    return Err(TokenError::TokenOutOfBound("Repetition", max.into(), r.into()));
                }
                r + Self::TK_FAIR_REPETITION
            }
            RegexToken::Error => return Err(TokenError::UnknownToken),
        })
    }
}
