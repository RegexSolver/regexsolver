use crate::used_bases::UsedBases;

use self::token::range_token::RangeToken;

use super::*;

#[derive(Debug)]
pub struct RangeTokenizer<'a> {
    used_bases: &'a UsedBases,
    total: Range,
}

impl RangeTokenizer<'_> {
    pub fn get_used_bases(&self) -> &UsedBases {
        self.used_bases
    }

    pub fn new(used_bases: &UsedBases) -> RangeTokenizer<'_> {
        let total = used_bases.get_rest().complement();
        RangeTokenizer { used_bases, total }
    }

    pub fn range_to_embedding(&self, range: &Range) -> Option<Vec<RangeToken>> {
        if range == &self.total {
            return Some(vec![RangeToken::Total]);
        } else if !range.difference(&self.total).is_empty() {
            return None;
        }

        let mut vec = vec![];
        for (token, base) in self.used_bases.get_bases().enumerate() {
            if range.contains_all(base) {
                vec.push(RangeToken::Base(token));
            }
        }
        vec.sort_unstable();

        Some(vec)
    }

    pub fn embedding_to_range(&self, vec: &[RangeToken]) -> Option<Range> {
        if vec.is_empty() {
            return Some(Range::empty());
        }

        let mut range = Range::empty();
        if vec[0] == RangeToken::Total {
            return Some(self.total.clone());
        }

        for token in vec {
            if let Some(base) = self.token_to_range(token) {
                range = range.union(base);
            } else {
                return None;
            }
        }

        Some(range)
    }

    pub fn token_to_range(&self, token: &RangeToken) -> Option<&Range> {
        match token {
            RangeToken::Total => Some(&self.total),
            RangeToken::Base(b) => self.used_bases.get_base(*b),
            RangeToken::Error => panic!("error token"),
        }
    }

    pub fn get_number_of_bases(&self) -> usize {
        self.used_bases.get_number_of_bases()
    }
}
