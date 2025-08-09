use self::range_token::RangeToken;

use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum AutomatonToken {
    Range(RangeToken),
    State(usize),
    AcceptState,
    SeparatorState,
    Error,
}

impl Ord for AutomatonToken {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (AutomatonToken::Range(a), AutomatonToken::Range(b)) => a.cmp(b),
            (AutomatonToken::Range(_), _) => Ordering::Less,
            (_, AutomatonToken::Range(_)) => Ordering::Greater,

            (AutomatonToken::State(a), AutomatonToken::State(b)) => a.cmp(b),
            (AutomatonToken::State(_), _) => Ordering::Less,
            (_, AutomatonToken::State(_)) => Ordering::Greater,

            (AutomatonToken::AcceptState, AutomatonToken::AcceptState) => Ordering::Equal,
            (AutomatonToken::AcceptState, _) => Ordering::Less,
            (_, AutomatonToken::AcceptState) => Ordering::Greater,

            (AutomatonToken::SeparatorState, AutomatonToken::SeparatorState) => Ordering::Equal,
            (AutomatonToken::SeparatorState, _) => Ordering::Less,
            (_, AutomatonToken::SeparatorState) => Ordering::Greater,

            (AutomatonToken::Error, AutomatonToken::Error) => Ordering::Equal,
        }
    }
}

impl PartialOrd for AutomatonToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl AutomatonToken {
    pub fn from_token(
        token: usize,
        number_of_bases: usize,
        number_of_states: usize,
    ) -> AutomatonToken {
        let states = number_of_bases + 1;
        let accept_state = states + number_of_states;
        let separator_state = accept_state + 1;
        if (0..states).contains(&token) {
            AutomatonToken::Range(RangeToken::from_token(token, number_of_bases))
        } else if (states..accept_state).contains(&token) {
            AutomatonToken::State(token - states)
        } else if token == accept_state {
            AutomatonToken::AcceptState
        } else if token == separator_state {
            AutomatonToken::SeparatorState
        } else {
            AutomatonToken::Error
        }
    }

    pub fn to_token(
        &self,
        number_of_bases: usize,
        number_of_states: usize,
    ) -> Result<usize, TokenError> {
        let states = number_of_bases + 1;
        let accept_state = states + number_of_states;
        let separator_state = accept_state + 1;
        Ok(match self {
            AutomatonToken::Range(r) => r.to_token(number_of_bases)?,
            AutomatonToken::State(s) => s + states,
            AutomatonToken::AcceptState => accept_state,
            AutomatonToken::SeparatorState => separator_state,
            AutomatonToken::Error => return Err(TokenError::UnknownToken),
        })
    }

    pub fn to_tokens(
        tokens: &[Self],
        number_of_bases: usize,
        number_of_states: usize,
    ) -> Result<Vec<usize>, TokenError> {
        let mut vec = Vec::with_capacity(tokens.len());
        for token in tokens {
            vec.push(token.to_token(number_of_bases, number_of_states)?);
        }
        Ok(vec)
    }
}
