use self::token::range_token::RangeToken;

use super::*;

#[derive(Debug)]
pub struct RangeTokenizer<'a> {
    spanning_set: &'a SpanningSet,
    total: Range,
}

impl RangeTokenizer<'_> {
    pub fn get_spanning_set(&self) -> &SpanningSet {
        self.spanning_set
    }

    pub fn new(spanning_set: &SpanningSet) -> RangeTokenizer<'_> {
        let total = spanning_set.get_rest().complement();
        RangeTokenizer {
            spanning_set,
            total,
        }
    }

    pub fn range_to_embedding(&self, range: &Range) -> Option<Vec<RangeToken>> {
        if range == &self.total {
            return Some(vec![RangeToken::Total]);
        } else if !range.difference(&self.total).is_empty() {
            return None;
        }

        let mut vec = vec![];
        for (token, base) in self.spanning_set.get_spanning_ranges().enumerate() {
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
            RangeToken::Base(b) => self.spanning_set.get_spanning_range(*b),
            RangeToken::Error => panic!("error token"),
        }
    }

    pub fn get_number_of_spanning_ranges(&self) -> usize {
        self.spanning_set.get_number_of_spanning_ranges()
    }
}
