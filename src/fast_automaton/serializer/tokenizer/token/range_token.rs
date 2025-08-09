use super::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum RangeToken {
    Total,
    Base(usize),
    Error,
}

impl Ord for RangeToken {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (RangeToken::Total, RangeToken::Total) => Ordering::Equal,
            (RangeToken::Total, _) => Ordering::Less,
            (_, RangeToken::Total) => Ordering::Greater,
            (RangeToken::Base(a), RangeToken::Base(b)) => a.cmp(b),
            (RangeToken::Base(_), RangeToken::Error) => Ordering::Less,
            (RangeToken::Error, RangeToken::Base(_)) => Ordering::Greater,
            (RangeToken::Error, RangeToken::Error) => Ordering::Equal,
        }
    }
}

impl PartialOrd for RangeToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl RangeToken {
    pub fn from_token(token: usize, number_of_bases: usize) -> RangeToken {
        let max_number_of_bases = number_of_bases + 1;
        if token == 0 {
            RangeToken::Total
        } else if (1..max_number_of_bases).contains(&token) {
            RangeToken::Base(token - 1)
        } else {
            RangeToken::Error
        }
    }

    pub fn to_token(&self, number_of_bases: usize) -> Result<usize, TokenError> {
        let max_number_of_bases = number_of_bases + 1;
        Ok(match self {
            RangeToken::Total => 0,
            RangeToken::Base(b) => {
                if *b > max_number_of_bases {
                    return Err(TokenError::TokenOutOfBound("Base", max_number_of_bases, *b));
                }
                b + 1
            }
            RangeToken::Error => return Err(TokenError::UnknownToken),
        })
    }
}
