//! The model's output vocabulary, and the regex its tokens describe.
//!
//! A regex is written in prefix order:
//!
//! ```text
//! node := CHAR | REPEAT min max node | (CONCAT | ALT) node* END
//! ```
//!
//! Each `CHAR` carries a mask over the DFA's bases (bit `i` = base `i`).

use super::*;

pub(super) const PAD: u32 = 0;
pub(super) const BOS: u32 = 1;
pub(super) const EOS: u32 = 2;
pub(super) const CONCAT: u32 = 3;
pub(super) const ALT: u32 = 4;
pub(super) const END: u32 = 5;
pub(super) const CHAR: u32 = 6;
pub(super) const REPEAT: u32 = 7;
pub(super) const MAX_COUNT: u32 = 16;
pub(super) const COUNT_BASE: u32 = 8;
pub(super) const COUNT_INF: u32 = COUNT_BASE + MAX_COUNT + 1;
pub(super) const VOCAB_SIZE: usize = COUNT_INF as usize + 1;

/// The token names in id order, as `config.json` lists them.
pub(super) fn vocab() -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = [
        ("PAD", PAD),
        ("BOS", BOS),
        ("EOS", EOS),
        ("CONCAT", CONCAT),
        ("ALT", ALT),
        ("END", END),
        ("CHAR", CHAR),
        ("REPEAT", REPEAT),
    ]
    .into_iter()
    .map(|(name, id)| (name.to_string(), id))
    .collect();
    v.extend((0..=MAX_COUNT).map(|n| (format!("COUNT_{n}"), COUNT_BASE + n)));
    v.push(("COUNT_INF".to_string(), COUNT_INF));
    v
}

/// Rebuilds the regex of prefix-order `tokens` (BOS and EOS excluded), bit
/// `i` of a `CHAR`'s mask denoting `bases[i]`. Returns `None` if the tokens do
/// not form exactly one tree.
pub(super) fn decode(
    tokens: &[u32],
    masks: &[u64],
    bases: &[CharRange],
) -> Option<RegularExpression> {
    fn count(tokens: &[u32], pos: &mut usize) -> Option<Option<u32>> {
        let token = *tokens.get(*pos)?;
        *pos += 1;
        match token {
            COUNT_INF => Some(None),
            t if (COUNT_BASE..COUNT_INF).contains(&t) => Some(Some(t - COUNT_BASE)),
            _ => None,
        }
    }

    fn node(
        tokens: &[u32],
        masks: &[u64],
        bases: &[CharRange],
        pos: &mut usize,
    ) -> Option<RegularExpression> {
        let at = *pos;
        let token = *tokens.get(at)?;
        *pos += 1;
        match token {
            CHAR => {
                let range = bases
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| masks[at] >> i & 1 == 1)
                    .fold(CharRange::empty(), |range, (_, base)| range.union(base));
                Some(RegularExpression::Character(range))
            }
            REPEAT => {
                let min = count(tokens, pos)??;
                let max = count(tokens, pos)?;
                if max.is_some_and(|max| max < min) {
                    return None;
                }
                Some(node(tokens, masks, bases, pos)?.repeat(min, max))
            }
            CONCAT | ALT => {
                let mut children = Vec::new();
                while *tokens.get(*pos)? != END {
                    children.push(node(tokens, masks, bases, pos)?);
                }
                *pos += 1;
                Some(if token == CONCAT {
                    RegularExpression::concat_all(&children)
                } else {
                    RegularExpression::union_all(&children)
                })
            }
            _ => None,
        }
    }

    if tokens.len() != masks.len() {
        return None;
    }
    let mut pos = 0;
    let regex = node(tokens, masks, bases, &mut pos)?;
    (pos == tokens.len()).then_some(regex)
}

#[cfg(test)]
mod tests {
    use regex_charclass::char::Char;

    use super::*;

    fn base(c: char) -> CharRange {
        CharRange::new_from_range(Char::new(c)..=Char::new(c))
    }

    #[test]
    fn vocab_is_dense() {
        let vocab = vocab();
        assert_eq!(vocab.len(), VOCAB_SIZE);
        assert!(vocab.iter().enumerate().all(|(i, (_, id))| *id == i as u32));
    }

    #[test]
    fn decodes_prefix_order() {
        let bases = [base('a'), base('b'), base('c')];
        // (a|b)*c
        let tokens = [
            CONCAT, REPEAT, COUNT_BASE, COUNT_INF, ALT, CHAR, CHAR, END, CHAR, END,
        ];
        let masks = [0, 0, 0, 0, 0, 0b001, 0b010, 0, 0b100, 0];
        let regex = decode(&tokens, &masks, &bases).unwrap();
        assert_eq!(regex.to_string(), "[ab]*c");

        // a{2,5}
        let tokens = [REPEAT, COUNT_BASE + 2, COUNT_BASE + 5, CHAR];
        let regex = decode(&tokens, &[0, 0, 0, 0b001], &bases).unwrap();
        assert_eq!(regex.to_string(), "a{2,5}");
    }

    #[test]
    fn rejects_malformed_sequences() {
        let bases = [base('a')];
        for tokens in [
            &[CONCAT, CHAR][..],
            &[CHAR, CHAR],
            &[REPEAT, COUNT_INF, COUNT_INF, CHAR],
            &[REPEAT, COUNT_BASE + 3, COUNT_BASE + 2, CHAR],
            &[END],
            &[],
        ] {
            let masks = vec![1; tokens.len()];
            assert!(decode(tokens, &masks, &bases).is_none(), "{tokens:?}");
        }
    }
}
