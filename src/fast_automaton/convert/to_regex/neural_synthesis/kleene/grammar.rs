//! Constrains decoding to well-formed regex trees.

use super::tokens::{ALT, CHAR, CONCAT, COUNT_BASE, COUNT_INF, END, EOS, REPEAT};

/// What a sequence being decoded still expects; each costs at least one more
/// token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pending {
    Node,
    /// Children until `END`.
    List,
    Min,
    Max,
}

/// Tracks one sequence being decoded and says which tokens may come next, so
/// the output is always one tree (see [`super::tokens`]) closed within
/// `max_len` positions, BOS and EOS included.
#[derive(Clone, Debug)]
pub(super) struct PrefixGrammar {
    max_len: usize,
    stack: Vec<Pending>,
    /// Positions so far, BOS included.
    length: usize,
    /// The minimum of the repeat being closed.
    min: u32,
}

impl PrefixGrammar {
    pub(super) fn new(max_len: usize) -> Self {
        Self {
            max_len,
            stack: vec![Pending::Node],
            length: 1,
            min: 0,
        }
    }

    fn expands(token: u32) -> &'static [Pending] {
        match token {
            // Top last: the minimum comes first, then the maximum, then the node.
            REPEAT => &[Pending::Node, Pending::Max, Pending::Min],
            CONCAT | ALT => &[Pending::List],
            _ => &[],
        }
    }

    fn stack_size_after(&self, top: Pending, token: u32) -> usize {
        let pops = top != Pending::List || token == END;
        self.stack.len() - pops as usize + Self::expands(token).len()
    }

    /// Whether `token` may come next.
    pub(super) fn allows(&self, token: u32) -> bool {
        let Some(&top) = self.stack.last() else {
            return token == EOS;
        };
        let candidate = match top {
            Pending::Min => (COUNT_BASE..COUNT_INF).contains(&token),
            Pending::Max => (COUNT_BASE + self.min..=COUNT_INF).contains(&token),
            Pending::Node => matches!(token, CHAR | REPEAT | CONCAT | ALT),
            Pending::List => matches!(token, CHAR | REPEAT | CONCAT | ALT | END),
        };
        // The token, then the open stack, then EOS must fit.
        candidate && self.length + 1 + self.stack_size_after(top, token) < self.max_len
    }

    /// Applies `token`, which must be allowed.
    pub(super) fn step(&mut self, token: u32) {
        let Some(&top) = self.stack.last() else {
            return;
        };
        if top != Pending::List || token == END {
            self.stack.pop();
        }
        if top == Pending::Min {
            self.min = token - COUNT_BASE;
        }
        self.stack.extend_from_slice(Self::expands(token));
        self.length += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::neural_synthesis::kleene::tokens::VOCAB_SIZE;

    use super::*;

    fn allowed(grammar: &PrefixGrammar) -> Vec<u32> {
        (0..VOCAB_SIZE as u32)
            .filter(|t| grammar.allows(*t))
            .collect()
    }

    #[test]
    fn follows_the_tree() {
        let mut grammar = PrefixGrammar::new(128);
        assert_eq!(allowed(&grammar), vec![CONCAT, ALT, CHAR, REPEAT]);
        grammar.step(REPEAT);
        assert_eq!(
            allowed(&grammar),
            (COUNT_BASE..COUNT_INF).collect::<Vec<_>>()
        );
        grammar.step(COUNT_BASE + 3);
        assert_eq!(
            allowed(&grammar),
            (COUNT_BASE + 3..=COUNT_INF).collect::<Vec<_>>()
        );
        grammar.step(COUNT_INF);
        grammar.step(CONCAT);
        assert!(grammar.allows(END));
        grammar.step(CHAR);
        grammar.step(END);
        assert_eq!(allowed(&grammar), vec![EOS]);
    }

    #[test]
    fn closes_within_max_len() {
        // BOS, then the tree, then EOS: 4 positions leave 2 for the tree.
        let mut grammar = PrefixGrammar::new(4);
        assert!(grammar.allows(CONCAT) && !grammar.allows(REPEAT));
        grammar.step(CONCAT);
        assert_eq!(allowed(&grammar), vec![END]);
    }
}
