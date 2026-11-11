//! Canonical form of a minimal DFA, invariant under state renumbering and
//! base relabeling: the form the model was trained on.
//!
//! Color refinement partitions the bases into classes that no isomorphism can
//! mix; only bases tied within a class are permuted. For each candidate base
//! order the states are numbered by BFS from the start (so the start is always
//! 0), and the lexicographically smallest encoding wins.

use std::cmp::Ordering;

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

/// The bases of `spanning_set` in condition bit order: the rest first, when
/// non-empty and `rest_is_base`, then the spanning ranges.
fn bases(spanning_set: &SpanningSet, rest_is_base: bool) -> Vec<CharRange> {
    std::iter::once(spanning_set.rest())
        .filter(|rest| rest_is_base && !rest.is_empty())
        .chain(spanning_set.spanning_ranges())
        .cloned()
        .collect()
}

/// Canonicalizes a minimal (hence deterministic and trimmed) automaton whose
/// spanning set is minimal, so that no transition uses its rest.
/// `rest_is_base`: the rest, when non-empty, is a base (format v1), one no
/// transition uses; format v2 leaves it out, so that `a*` and `.*` are the
/// same input.
pub(super) fn canonicalize(automaton: &FastAutomaton, rest_is_base: bool) -> CanonicalDfa {
    let (table, bases) = Table::new(automaton, rest_is_base);
    let (best, order) = table.canonical_encoding();
    table.into_canonical(best, &order, &bases)
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
    /// Whether each canonical state accepts.
    accept: Vec<bool>,
    /// Row-major over canonical states and bases.
    delta: Vec<u32>,
}

/// The BFS numbering of the states for one base order, reused across orders.
struct Numbering {
    /// Canonical number of each state.
    number: Vec<u32>,
    /// The states in canonical order: the inverse of `number`.
    queue: Vec<usize>,
}

impl Numbering {
    fn relabel(&self, t: u32) -> u32 {
        if t == DEAD {
            DEAD
        } else {
            self.number[t as usize]
        }
    }
}

impl Table {
    /// The transition table of `automaton` and its bases, in condition bit
    /// order.
    fn new(automaton: &FastAutomaton, rest_is_base: bool) -> (Table, Vec<CharRange>) {
        debug_assert!(automaton.is_deterministic());
        let spanning_set = automaton.spanning_set();
        let bases = bases(spanning_set, rest_is_base);
        let k = bases.len();
        // Conditions put the rest, when non-empty, at bit 0: without it, base
        // `b` is bit `b + 1`.
        let offset = usize::from(!rest_is_base && !spanning_set.rest().is_empty());

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
                for bit in condition.iter_set_bits() {
                    debug_assert!(bit >= offset, "a transition uses the rest");
                    if let Some(b) = bit.checked_sub(offset) {
                        delta[i * k + b] = dense[to];
                    }
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
        (table, bases)
    }

    fn into_canonical(self, best: Relabeled, order: &[usize], bases: &[CharRange]) -> CanonicalDfa {
        CanonicalDfa {
            n: self.n,
            k: self.k,
            accept: best.accept,
            delta: best.delta,
            bases: order.iter().map(|b| bases[*b].clone()).collect(),
        }
    }

    fn row(&self, s: usize) -> &[u32] {
        &self.delta[s * self.k..(s + 1) * self.k]
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

        let mut numbering = Numbering {
            number: vec![DEAD; self.n],
            queue: Vec::with_capacity(self.n),
        };
        self.number(&order, &mut numbering, None);
        let mut best = Relabeled {
            accept: vec![false; self.n],
            delta: vec![DEAD; self.n * self.k],
        };
        self.write(&order, &numbering, 0, &mut best);

        let mut product: usize = 1;
        let exact = groups.iter().all(|g| {
            product = (1..=g.len()).fold(product, |p, f| p.saturating_mul(f));
            product <= MAX_BASE_PERMUTATIONS
        });
        if !exact {
            return (best, order);
        }

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
            if self.improve(&order, &mut numbering, &mut best) {
                best_order.clone_from(&order);
            }
        }
        (best, best_order)
    }

    /// Numbers the states by BFS from the start, following bases in `order`.
    /// Given the acceptance of `best`, compares with it as states are
    /// numbered, and stops as soon as the candidate is larger: it then
    /// returns `Greater`, with the numbering unfinished.
    fn number(
        &self,
        order: &[usize],
        numbering: &mut Numbering,
        best: Option<&[bool]>,
    ) -> Ordering {
        numbering.number.fill(DEAD);
        numbering.queue.clear();
        let mut accepts = Ordering::Equal;
        if !self.visit(self.start, numbering, best, &mut accepts) {
            return Ordering::Greater;
        }
        let mut head = 0;
        while head < numbering.queue.len() {
            let row = self.row(numbering.queue[head]);
            head += 1;
            for &b in order {
                let t = row[b];
                if t != DEAD
                    && numbering.number[t as usize] == DEAD
                    && !self.visit(t as usize, numbering, best, &mut accepts)
                {
                    return Ordering::Greater;
                }
            }
        }
        // A minimal DFA has every state reachable; number any stragglers
        // anyway so the encoding stays total.
        while let Some(s) = numbering.number.iter().position(|n| *n == DEAD) {
            if !self.visit(s, numbering, best, &mut accepts) {
                return Ordering::Greater;
            }
        }
        accepts
    }

    /// Gives state `s` the next canonical number. While the acceptance so far
    /// ties with `best`, compares it: as both have as many accepting states,
    /// their sorted lists compare as the acceptance flags with accepting
    /// first. Returns `false` once the candidate is larger.
    fn visit(
        &self,
        s: usize,
        numbering: &mut Numbering,
        best: Option<&[bool]>,
        accepts: &mut Ordering,
    ) -> bool {
        let m = numbering.queue.len();
        numbering.number[s] = m as u32;
        numbering.queue.push(s);
        if let Some(best) = best
            && *accepts == Ordering::Equal
        {
            *accepts = best[m].cmp(&self.accept[s]);
        }
        *accepts != Ordering::Greater
    }

    /// Writes the candidate numbered by `numbering` into `best`: the
    /// acceptance in full, the transitions from entry `from` (row-major) on.
    fn write(&self, order: &[usize], numbering: &Numbering, from: usize, best: &mut Relabeled) {
        for (accept, &s) in best.accept.iter_mut().zip(&numbering.queue) {
            *accept = self.accept[s];
        }
        let k = self.k;
        if k == 0 {
            return;
        }
        for (new_s, best_row) in best.delta.chunks_exact_mut(k).enumerate().skip(from / k) {
            let row = self.row(numbering.queue[new_s]);
            let skip = if new_s == from / k { from % k } else { 0 };
            for (slot, &b) in best_row.iter_mut().zip(order).skip(skip) {
                *slot = numbering.relabel(row[b]);
            }
        }
    }

    /// Relabels for `order` and, when the result is smaller than `best`,
    /// writes it there. The key is `(accepts, delta)`: comparison stops at
    /// the first difference, and only the transitions from there on are
    /// written.
    fn improve(&self, order: &[usize], numbering: &mut Numbering, best: &mut Relabeled) -> bool {
        match self.number(order, numbering, Some(&best.accept)) {
            Ordering::Greater => return false,
            Ordering::Less => {
                self.write(order, numbering, 0, best);
                return true;
            }
            Ordering::Equal => {}
        }
        let k = self.k;
        if k == 0 {
            return false;
        }
        let mut from = None;
        'rows: for (new_s, best_row) in best.delta.chunks_exact(k).enumerate() {
            let row = self.row(numbering.queue[new_s]);
            for (j, (&b, best_t)) in order.iter().zip(best_row).enumerate() {
                match numbering.relabel(row[b]).cmp(best_t) {
                    Ordering::Equal => {}
                    Ordering::Greater => return false,
                    Ordering::Less => {
                        from = Some(new_s * k + j);
                        break 'rows;
                    }
                }
            }
        }
        let Some(from) = from else {
            return false;
        };
        self.write(order, numbering, from, best);
        true
    }

    /// Alternating color refinement of states and bases; returns the stable
    /// base colors. Colors are ranks of sorted signatures, so they are
    /// invariant under renumbering and relabeling.
    fn refine(&self) -> Vec<u32> {
        let (n, k) = (self.n, self.k);
        let mut ranker = Ranker::default();
        let mut signatures = Vec::with_capacity(n.max(k) * (1 + 2 * n.max(k)));
        for s in 0..n {
            let out = self.row(s).iter().filter(|t| **t != DEAD).count();
            signatures.extend([(s == self.start) as u32, self.accept[s] as u32, out as u32]);
        }
        let mut state_colors = Vec::new();
        ranker.ranks(&signatures, 3, &mut state_colors);
        signatures.clear();
        for b in 0..k {
            let (mut defined, mut loops, mut into_accept) = (0, 0, 0);
            for s in 0..n {
                let t = self.delta[s * k + b];
                if t != DEAD {
                    defined += 1;
                    loops += (t as usize == s) as u32;
                    into_accept += self.accept[t as usize] as u32;
                }
            }
            signatures.extend([defined, loops, into_accept]);
        }
        let mut base_colors = Vec::new();
        ranker.ranks(&signatures, 3, &mut base_colors);

        let classes = |colors: &[u32]| colors.iter().max().map_or(0, |m| m + 1);
        let mut total = classes(&state_colors) + classes(&base_colors);
        // Signatures are the previous color, then the sorted (color, target
        // color) pairs, packed into a `u64` so that they sort the same.
        let mut pairs: Vec<u64> = Vec::with_capacity(n.max(k));
        let mut next_colors = Vec::new();
        loop {
            let color_of =
                |t: u32, colors: &[u32]| if t == DEAD { DEAD } else { colors[t as usize] };
            let pack = |a: u32, b: u32| (u64::from(a) << 32) | u64::from(b);
            let unpack = |p: &u64| [(p >> 32) as u32, *p as u32];

            signatures.clear();
            for s in 0..n {
                pairs.clear();
                pairs.extend(
                    self.row(s)
                        .iter()
                        .zip(&base_colors)
                        .map(|(t, c)| pack(*c, color_of(*t, &state_colors))),
                );
                pairs.sort_unstable();
                signatures.push(state_colors[s]);
                signatures.extend(pairs.iter().flat_map(unpack));
            }
            ranker.ranks(&signatures, 1 + 2 * k, &mut next_colors);
            std::mem::swap(&mut state_colors, &mut next_colors);

            signatures.clear();
            for (b, &color) in base_colors.iter().enumerate() {
                pairs.clear();
                pairs.extend((0..n).map(|s| {
                    pack(
                        state_colors[s],
                        color_of(self.delta[s * k + b], &state_colors),
                    )
                }));
                pairs.sort_unstable();
                signatures.push(color);
                signatures.extend(pairs.iter().flat_map(unpack));
            }
            ranker.ranks(&signatures, 1 + 2 * n, &mut next_colors);
            std::mem::swap(&mut base_colors, &mut next_colors);

            let new_total = classes(&state_colors) + classes(&base_colors);
            if new_total == total {
                return base_colors;
            }
            total = new_total;
        }
    }
}

/// Ranks fixed-width signatures, reusing its buffer.
#[derive(Default)]
struct Ranker {
    order: Vec<usize>,
}

impl Ranker {
    /// Writes into `ranks` the rank of each `width`-long signature of the
    /// flat `signatures` among the distinct sorted signatures.
    fn ranks(&mut self, signatures: &[u32], width: usize, ranks: &mut Vec<u32>) {
        let count = signatures.len() / width;
        let signature = |i: usize| &signatures[i * width..(i + 1) * width];
        self.order.clear();
        self.order.extend(0..count);
        self.order
            .sort_unstable_by(|a, b| signature(*a).cmp(signature(*b)));
        ranks.clear();
        ranks.resize(count, 0);
        let mut rank = 0;
        for w in 1..count {
            let (previous, current) = (self.order[w - 1], self.order[w]);
            if signature(previous) != signature(current) {
                rank += 1;
            }
            ranks[current] = rank;
        }
    }
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

    fn dfa(pattern: &str) -> FastAutomaton {
        minimal_dfa(
            &RegularExpression::new(pattern)
                .unwrap()
                .to_automaton()
                .unwrap(),
        )
        .unwrap()
    }

    fn canonical_with(pattern: &str, rest_is_base: bool) -> CanonicalDfa {
        canonicalize(&dfa(pattern), rest_is_base)
    }

    fn canonical(pattern: &str) -> CanonicalDfa {
        canonical_with(pattern, true)
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
    fn without_the_rest() {
        // Bases: `a`, `[bc]`, `d`; the transitions are those of v1 without
        // the rest's column.
        let (v1, v2) = (canonical("a(b|c)*d"), canonical_with("a(b|c)*d", false));
        assert_eq!((v2.n, v2.k), (3, 3));
        assert_eq!(v2.accept, v1.accept);
        assert!(v1.delta.chunks(4).all(|row| row[0] == DEAD));
        let without_rest: Vec<u32> = v1
            .delta
            .chunks(4)
            .flat_map(|row| &row[1..])
            .copied()
            .collect();
        assert_eq!(v2.delta, without_rest);
        // `a*` and `.*` are the same input; with the rest, `a*` has two bases.
        let (a, dot) = (canonical_with("a*", false), canonical_with(".*", false));
        assert_eq!((a.k, a.accept, a.delta), (1, dot.accept, dot.delta));
        assert_eq!(canonical("a*").k, 2);
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

    /// The straightforward implementation the optimized one must match
    /// exactly: a fresh relabeling per base order, compared in full, and
    /// `Vec` signatures.
    mod reference {
        use super::super::*;

        pub(super) fn canonicalize(automaton: &FastAutomaton, rest_is_base: bool) -> CanonicalDfa {
            let (table, bases) = Table::new(automaton, rest_is_base);
            let (best, order) = canonical_encoding(&table);
            table.into_canonical(best, &order, &bases)
        }

        struct Candidate {
            accepts: Vec<u32>,
            delta: Vec<u32>,
        }

        fn target(table: &Table, s: usize, b: usize) -> Option<usize> {
            let t = table.delta[s * table.k + b];
            (t != DEAD).then_some(t as usize)
        }

        pub(super) fn canonical_encoding(table: &Table) -> (Relabeled, Vec<usize>) {
            let base_colors = refine(table);

            let mut order: Vec<usize> = (0..table.k).collect();
            order.sort_by_key(|b| (base_colors[*b], *b));
            let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
            let mut i = 0;
            while i < table.k {
                let mut j = i + 1;
                while j < table.k && base_colors[order[j]] == base_colors[order[i]] {
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
                return (into_relabeled(table, relabel(table, &order)), order);
            }

            let mut best = relabel(table, &order);
            let mut best_order = order.clone();
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
                let candidate = relabel(table, &order);
                if (&candidate.accepts, &candidate.delta) < (&best.accepts, &best.delta) {
                    best = candidate;
                    best_order.clone_from(&order);
                }
            }
            (into_relabeled(table, best), best_order)
        }

        fn into_relabeled(table: &Table, candidate: Candidate) -> Relabeled {
            let mut accept = vec![false; table.n];
            for s in candidate.accepts {
                accept[s as usize] = true;
            }
            Relabeled {
                accept,
                delta: candidate.delta,
            }
        }

        fn relabel(table: &Table, order: &[usize]) -> Candidate {
            let (n, k) = (table.n, table.k);
            let mut numbering = vec![DEAD; n];
            let mut queue = Vec::with_capacity(n);
            numbering[table.start] = 0;
            queue.push(table.start);
            let mut head = 0;
            while head < queue.len() {
                let s = queue[head];
                head += 1;
                for &b in order {
                    if let Some(t) = target(table, s, b)
                        && numbering[t] == DEAD
                    {
                        numbering[t] = queue.len() as u32;
                        queue.push(t);
                    }
                }
            }
            for (s, number) in numbering.iter_mut().enumerate() {
                if *number == DEAD {
                    *number = queue.len() as u32;
                    queue.push(s);
                }
            }

            let mut delta = vec![DEAD; n * k];
            for (new_s, &s) in queue.iter().enumerate() {
                for (j, &b) in order.iter().enumerate() {
                    if let Some(t) = target(table, s, b) {
                        delta[new_s * k + j] = numbering[t];
                    }
                }
            }
            let mut accepts: Vec<u32> = (0..n)
                .filter(|s| table.accept[*s])
                .map(|s| numbering[s])
                .collect();
            accepts.sort_unstable();
            Candidate { accepts, delta }
        }

        pub(super) fn refine(table: &Table) -> Vec<u32> {
            let (n, k) = (table.n, table.k);
            let mut state_colors = ranks(
                &(0..n)
                    .map(|s| {
                        let out = (0..k).filter(|b| target(table, s, *b).is_some()).count();
                        vec![
                            (s == table.start) as u32,
                            table.accept[s] as u32,
                            out as u32,
                        ]
                    })
                    .collect::<Vec<_>>(),
            );
            let mut base_colors = ranks(
                &(0..k)
                    .map(|b| {
                        let (mut defined, mut loops, mut into_accept) = (0, 0, 0);
                        for s in 0..n {
                            if let Some(t) = target(table, s, b) {
                                defined += 1;
                                loops += (t == s) as u32;
                                into_accept += table.accept[t] as u32;
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
                                .map(|b| {
                                    (base_colors[b], color_of(target(table, s, b), &state_colors))
                                })
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
                                .map(|s| {
                                    (
                                        state_colors[s],
                                        color_of(target(table, s, b), &state_colors),
                                    )
                                })
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

        fn ranks<T: Ord + Clone>(signatures: &[T]) -> Vec<u32> {
            let mut sorted = signatures.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            signatures
                .iter()
                .map(|s| sorted.binary_search(s).unwrap() as u32)
                .collect()
        }
    }

    /// Patterns with many bases tied by symmetry, up to the permutation cap
    /// and beyond it.
    const SYMMETRIC: &[&str] = &[
        "(a|b|c|d|e|f|g)*h",
        "[a-g]{3}",
        "(aa|bb|cc|dd|ee|ff|gg)*",
        "(ab|cd|ef|gh)*",
        "(abc|def|ghi)*x",
        "(a|b|c|d|e|f)(a|b|c|d|e|f)*",
        "(ax|bx|cx|dx|ex|fx|gx)+",
        "a(b|c|d|e|f|g|h)*|(b|c|d|e|f|g|h)a*",
        "(ab|cd|ef|gh|ij|kl|mn)*",
    ];

    fn assert_same(pattern: &str, rest_is_base: bool) {
        let automaton = dfa(pattern);
        let (new, old) = (
            canonicalize(&automaton, rest_is_base),
            reference::canonicalize(&automaton, rest_is_base),
        );
        assert_eq!(
            (new.n, new.k, &new.accept, &new.delta, &new.bases),
            (old.n, old.k, &old.accept, &old.delta, &old.bases),
            "{pattern} (rest_is_base: {rest_is_base})"
        );
    }

    #[test]
    fn matches_reference_on_patterns() {
        let patterns = [
            "",
            "a",
            "a*",
            ".*",
            "ab",
            "ab*",
            "a*b",
            "a(b|c)*d",
            "(ab|cd)+e",
            "a(ba)*",
            "x(y|z)*w",
            "[0-9]+(\\.[0-9]+)?",
            "[a-z]+@[a-z]+\\.(com|org|net)",
            "(a|b)*abb",
            "((a|b)(a|b))*",
            "(a*b*c*)*d",
            "a{2,5}b{0,3}",
            "(foo|bar|baz)+",
            "[^a]*a[^a]*",
            "(ab|ba|aa|bb){2,4}",
            "q(r|s)*t",
            "(x|y|z){1,3}(x|y|z)*",
            "abc|abd|acd|bcd",
        ];
        for pattern in patterns.iter().chain(SYMMETRIC) {
            assert_same(pattern, true);
            assert_same(pattern, false);
        }
    }

    /// Random tables, not necessarily minimal nor fully reachable, many of
    /// them with tied bases.
    #[test]
    fn matches_reference_on_random_tables() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = |bound: usize| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 33) % bound as u64) as usize
        };
        for _ in 0..3000 {
            let (n, k) = (1 + next(7), next(8));
            // Few distinct targets, so that bases often look alike.
            let targets = 1 + next(n);
            let delta = (0..n * k)
                .map(|_| {
                    let t = next(targets + 2);
                    if t >= targets { DEAD } else { t as u32 }
                })
                .collect();
            let table = Table {
                n,
                k,
                delta,
                start: next(n),
                accept: (0..n).map(|_| next(2) == 0).collect(),
            };
            assert_eq!(table.refine(), reference::refine(&table));
            let (new, new_order) = table.canonical_encoding();
            let (old, old_order) = reference::canonical_encoding(&table);
            assert_eq!(
                (new.accept, new.delta, new_order),
                (old.accept, old.delta, old_order)
            );
        }
    }

    /// Before/after timing:
    /// `cargo test --release --features neural-synthesis canonical::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing() {
        use std::time::Instant;
        for pattern in SYMMETRIC {
            let automaton = dfa(pattern);
            let c = canonicalize(&automaton, true);
            let iterations = 50;
            let time = |f: &dyn Fn() -> CanonicalDfa| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(f());
                }
                start.elapsed() / iterations
            };
            let old = time(&|| reference::canonicalize(&automaton, true));
            let new = time(&|| canonicalize(&automaton, true));
            println!(
                "{pattern:40} n={:2} k={:2}  before {old:>10.2?}  after {new:>10.2?}  x{:.1}",
                c.n,
                c.k,
                old.as_secs_f64() / new.as_secs_f64()
            );
        }
    }
}
