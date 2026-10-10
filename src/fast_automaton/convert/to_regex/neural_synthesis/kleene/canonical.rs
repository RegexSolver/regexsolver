//! Canonical form of a minimal DFA, invariant under state renumbering and
//! base relabeling: the form the model was trained on.
//!
//! Color refinement partitions the bases into classes that no isomorphism can
//! mix; only bases tied within a class are permuted. For each candidate base
//! order the states are numbered by BFS from the start (so the start is always
//! 0), and the lexicographically smallest encoding wins.

use super::*;

/// Above this many candidate base orders, ties are broken by the original
/// base order.
const MAX_BASE_PERMUTATIONS: usize = 5040;

pub(super) const DEAD: u32 = u32::MAX;

/// A DFA in canonical form.
#[derive(Clone, Debug)]
pub(super) struct CanonicalDfa {
    pub n: usize,
    pub k: usize,
    /// Whether each canonical state accepts. The start is state 0.
    pub accept: Vec<bool>,
    /// `delta[s * k + b]`: target of canonical state `s` on canonical base
    /// `b`, or [`DEAD`].
    pub delta: Vec<u32>,
    /// Canonical base `i` as a character range.
    pub bases: Vec<CharRange>,
}

/// The bases of `spanning_set` in condition bit order (the rest first, when
/// non-empty).
fn bases(spanning_set: &SpanningSet) -> Vec<CharRange> {
    std::iter::once(spanning_set.rest())
        .filter(|rest| !rest.is_empty())
        .chain(spanning_set.spanning_ranges())
        .cloned()
        .collect()
}

/// Canonicalizes a minimal (hence deterministic and trimmed) automaton.
pub(super) fn canonicalize(automaton: &FastAutomaton) -> CanonicalDfa {
    debug_assert!(automaton.is_deterministic());
    let bases = bases(automaton.spanning_set());
    let k = bases.len();

    let raw_states = automaton.states_vec();
    let n = raw_states.len();
    let max_raw = raw_states.iter().max().map_or(0, |s| s + 1);
    let mut dense = vec![DEAD; max_raw];
    for (i, s) in raw_states.iter().enumerate() {
        dense[*s] = i as u32;
    }

    let mut delta = vec![DEAD; n * k];
    for (i, &from) in raw_states.iter().enumerate() {
        for (condition, &to) in automaton.transitions_from(from) {
            for b in condition.iter_set_bits() {
                delta[i * k + b] = dense[to];
            }
        }
    }
    let table = Table {
        n,
        k,
        delta,
        start: dense[automaton.start_state()] as usize,
        accept: raw_states
            .iter()
            .map(|s| automaton.is_accepted(*s))
            .collect(),
    };
    let (best, order) = table.canonical_encoding();

    let mut accept = vec![false; n];
    for s in best.accepts {
        accept[s as usize] = true;
    }
    CanonicalDfa {
        n,
        k,
        accept,
        delta: best.delta,
        bases: order.iter().map(|b| bases[*b].clone()).collect(),
    }
}

struct Table {
    n: usize,
    k: usize,
    /// `delta[s * k + b]`: target of state `s` on base `b`, or [`DEAD`].
    delta: Vec<u32>,
    start: usize,
    accept: Vec<bool>,
}

/// The DFA renumbered for one base order.
struct Relabeled {
    /// Sorted.
    accepts: Vec<u32>,
    /// Row-major over canonical states and bases.
    delta: Vec<u32>,
}

impl Relabeled {
    fn cmp_key(&self) -> (&[u32], &[u32]) {
        (&self.accepts, &self.delta)
    }
}

impl Table {
    fn target(&self, s: usize, b: usize) -> Option<usize> {
        let t = self.delta[s * self.k + b];
        (t != DEAD).then_some(t as usize)
    }

    /// Returns the smallest encoding and its base order.
    fn canonical_encoding(&self) -> (Relabeled, Vec<usize>) {
        let base_colors = self.refine();

        let mut order: Vec<usize> = (0..self.k).collect();
        order.sort_by_key(|b| (base_colors[*b], *b));
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut i = 0;
        while i < self.k {
            let mut j = i + 1;
            while j < self.k && base_colors[order[j]] == base_colors[order[i]] {
                j += 1;
            }
            if j - i > 1 {
                groups.push(i..j);
            }
            i = j;
        }

        let mut product: usize = 1;
        let exact = groups.iter().all(|g| {
            product = (1..=g.len()).fold(product, |p, f| p.saturating_mul(f));
            product <= MAX_BASE_PERMUTATIONS
        });
        if !exact {
            return (self.relabel(&order), order);
        }

        let mut best = self.relabel(&order);
        let mut best_order = order.clone();
        // Odometer over per-group permutations; each group starts sorted
        // (ascending original index) and `next_permutation` wraps it back.
        loop {
            let mut advanced = false;
            for g in &groups {
                if next_permutation(&mut order[g.clone()]) {
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                break;
            }
            let candidate = self.relabel(&order);
            if candidate.cmp_key() < best.cmp_key() {
                best = candidate;
                best_order.clone_from(&order);
            }
        }
        (best, best_order)
    }

    fn relabel(&self, order: &[usize]) -> Relabeled {
        let (n, k) = (self.n, self.k);
        let mut numbering = vec![DEAD; n];
        let mut queue = Vec::with_capacity(n);
        numbering[self.start] = 0;
        queue.push(self.start);
        let mut head = 0;
        while head < queue.len() {
            let s = queue[head];
            head += 1;
            for &b in order {
                if let Some(t) = self.target(s, b)
                    && numbering[t] == DEAD
                {
                    numbering[t] = queue.len() as u32;
                    queue.push(t);
                }
            }
        }
        // A minimal DFA has every state reachable; number any stragglers
        // anyway so the encoding stays total.
        for (s, number) in numbering.iter_mut().enumerate() {
            if *number == DEAD {
                *number = queue.len() as u32;
                queue.push(s);
            }
        }

        let mut delta = vec![DEAD; n * k];
        for (new_s, &s) in queue.iter().enumerate() {
            for (j, &b) in order.iter().enumerate() {
                if let Some(t) = self.target(s, b) {
                    delta[new_s * k + j] = numbering[t];
                }
            }
        }
        let mut accepts: Vec<u32> = (0..n)
            .filter(|s| self.accept[*s])
            .map(|s| numbering[s])
            .collect();
        accepts.sort_unstable();
        Relabeled { accepts, delta }
    }

    /// Alternating color refinement of states and bases; returns the stable
    /// base colors. Colors are ranks of sorted signatures, so they are
    /// invariant under renumbering and relabeling.
    fn refine(&self) -> Vec<u32> {
        let (n, k) = (self.n, self.k);
        let mut state_colors = ranks(
            &(0..n)
                .map(|s| {
                    let out = (0..k).filter(|b| self.target(s, *b).is_some()).count();
                    vec![(s == self.start) as u32, self.accept[s] as u32, out as u32]
                })
                .collect::<Vec<_>>(),
        );
        let mut base_colors = ranks(
            &(0..k)
                .map(|b| {
                    let (mut defined, mut loops, mut into_accept) = (0, 0, 0);
                    for s in 0..n {
                        if let Some(t) = self.target(s, b) {
                            defined += 1;
                            loops += (t == s) as u32;
                            into_accept += self.accept[t] as u32;
                        }
                    }
                    vec![defined, loops, into_accept]
                })
                .collect::<Vec<_>>(),
        );

        let classes = |colors: &[u32]| colors.iter().max().map_or(0, |m| m + 1);
        let mut total = classes(&state_colors) + classes(&base_colors);
        loop {
            let color_of = |t: Option<usize>, colors: &[u32]| t.map_or(DEAD, |t| colors[t]);
            state_colors = ranks(
                &(0..n)
                    .map(|s| {
                        let mut sig: Vec<(u32, u32)> = (0..k)
                            .map(|b| (base_colors[b], color_of(self.target(s, b), &state_colors)))
                            .collect();
                        sig.sort_unstable();
                        std::iter::once(state_colors[s])
                            .chain(sig.into_iter().flat_map(|(a, b)| [a, b]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            base_colors = ranks(
                &(0..k)
                    .map(|b| {
                        let mut sig: Vec<(u32, u32)> = (0..n)
                            .map(|s| (state_colors[s], color_of(self.target(s, b), &state_colors)))
                            .collect();
                        sig.sort_unstable();
                        std::iter::once(base_colors[b])
                            .chain(sig.into_iter().flat_map(|(a, b)| [a, b]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            let new_total = classes(&state_colors) + classes(&base_colors);
            if new_total == total {
                return base_colors;
            }
            total = new_total;
        }
    }
}

/// Maps each signature to its rank among the distinct sorted signatures.
fn ranks<T: Ord + Clone>(signatures: &[T]) -> Vec<u32> {
    let mut sorted = signatures.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    signatures
        .iter()
        .map(|s| sorted.binary_search(s).unwrap() as u32)
        .collect()
}

/// Advances `v` to its next lexicographic permutation; on the last one,
/// resets it to ascending order and returns `false`.
fn next_permutation(v: &mut [usize]) -> bool {
    if v.len() < 2 {
        return false;
    }
    let mut i = v.len() - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        v.reverse();
        return false;
    }
    let mut j = v.len() - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use crate::neural_synthesis::minimal_dfa;

    use super::*;

    fn canonical(pattern: &str) -> CanonicalDfa {
        canonicalize(
            &minimal_dfa(
                &RegularExpression::new(pattern)
                    .unwrap()
                    .to_automaton()
                    .unwrap(),
            )
            .unwrap(),
        )
    }

    fn key(pattern: &str) -> (Vec<bool>, Vec<u32>) {
        let c = canonical(pattern);
        (c.accept, c.delta)
    }

    #[test]
    fn invariant_under_relabeling() {
        assert_eq!(key("(ab|cd)+e"), key("(cd|ab)(ab|cd)*e"));
        assert_eq!(key("a(ba)*"), key("(ab)*a"));
        assert_eq!(key("a*b"), key("b*a"));
        assert_eq!(key("x(y|z)*w"), key("q(r|s)*t"));
        assert_ne!(key("ab"), key("ab*"));
        assert_ne!(key("a*b"), key("a*a"));
    }

    #[test]
    fn start_is_zero_and_states_are_bfs_numbered() {
        let c = canonical("a(b|c)*d");
        // Bases: the rest, `a`, `[bc]`, `d`.
        assert_eq!((c.n, c.k), (3, 4));
        // From the start, only `a` leads anywhere: to state 1.
        assert_eq!(
            (0..c.k)
                .filter(|b| c.delta[*b] != DEAD)
                .map(|b| c.delta[b])
                .collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn next_permutation_enumerates_all() {
        let mut v = vec![0, 1, 2, 3];
        let mut count = 1;
        while next_permutation(&mut v) {
            count += 1;
        }
        assert_eq!(count, 24);
        assert_eq!(v, vec![0, 1, 2, 3]);
    }
}
