//! Property-based tests built on `proptest` strategies that generate random
//! DFAs, NFAs and regular expressions.
//!
//! # Coverage guarantee
//!
//! The strategies are parameterized by a fixed finite alphabet ([`ALPHABET`])
//! and a maximum number of states ([`MAX_STATES`]). Within those bounds every
//! structure has a strictly positive probability of being generated:
//!
//! * [`arb_dfa`] — every deterministic automaton over the alphabet with
//!   `1..=MAX_STATES` states (start state fixed to `0`, any accepting subset,
//!   any total transition function) can be produced. A transition function maps
//!   each `(state, letter)` to at most one target, which is exactly the
//!   definition of a DFA, so the whole DFA space is covered.
//! * [`arb_nfa`] — every nondeterministic automaton is reachable: each ordered
//!   `(from, to)` pair may carry any subset of the alphabet letters as its
//!   label, plus optional epsilon transitions. Since labels may overlap, this
//!   spans all NFAs (and, as a subset, all DFAs).
//! * [`arb_regex`] — every regular expression built from the four
//!   [`RegularExpression`] variants up to the configured recursion depth and
//!   bound sizes is reachable, including the empty language (`[]`) and `.`.
//!
//! The alphabet is small on purpose so the spaces are finite and the
//! determinization / set operations under test stay cheap.
//!
//! While *coverage* is uniform-in-support, the *distribution* is deliberately
//! shaped — the `inspect::stats` test measures the result and asserts floors:
//!
//! * Per-automaton edge/epsilon/accept densities are sampled, with the accept
//!   density centered on ½ (where accepting/rejecting states are hardest to
//!   merge, keeping minimal DFAs — and therefore the work done by minimize /
//!   equivalence / state elimination — large).
//! * An optional "anchor" state is forced accepting so the empty language is
//!   an occasional edge case instead of a fifth of the sample.
//! * A per-automaton acyclic mode (≈⅓ of cases) generates DAGs, whose finite
//!   languages exercise the topological-sort paths of the cardinality and
//!   max-length analyses that cyclic automata never reach.
//! * The character-class strategy favors single letters so that a `[]`
//!   (empty-language) leaf does not collapse most expressions.
//!
//! All weights stay strictly inside (0, 1), preserving the
//! non-null-probability guarantee.

use proptest::prelude::*;
use regex_charclass::char::Char;
use regexsolver::CharRange;
use regexsolver::cardinality::Cardinality;
use regexsolver::error::EngineError;
use regexsolver::execution_profile::ExecutionProfileBuilder;
use regexsolver::fast_automaton::FastAutomaton;
use regexsolver::regex::RegularExpression;

/// Fixed alphabet the strategies draw transition labels from.
pub const ALPHABET: &[char] = &['a', 'b'];

/// Maximum number of states a generated automaton can have.
pub const MAX_STATES: usize = 5;

/// The single-character range for the `i`-th letter of `alphabet`.
fn letter_over(alphabet: &[char], i: usize) -> CharRange {
    let c = Char::new(alphabet[i]);
    CharRange::new_from_range(c..=c)
}

/// The single-character range for the `i`-th [`ALPHABET`] letter.
fn letter(i: usize) -> CharRange {
    letter_over(ALPHABET, i)
}

/// Number of transition-label letters (one per [`ALPHABET`] letter).
fn num_bases() -> usize {
    ALPHABET.len()
}

/// Number of states, biased toward larger automata (the maximum of two uniform
/// draws). Tiny automata are still generated, but the n = 1 space is almost
/// entirely degenerate, so uniform sampling would waste a quarter of all cases
/// on it.
fn arb_num_states() -> impl Strategy<Value = usize> {
    (1usize..=MAX_STATES, 1usize..=MAX_STATES).prop_map(|(a, b)| a.max(b))
}

/// Builds an automaton over `alphabet` from a structural description using
/// only the public API.
///
/// `accepts[s]` marks state `s` accepting; each `(from, to, mask)` adds a
/// transition whose label is the union of the letters selected by `mask`
/// (`add_transition_from_range` grows the automaton's spanning set as
/// needed); each `(from, to)` in `eps` adds an epsilon transition. The start
/// state is `0`.
fn build_over(
    alphabet: &[char],
    n: usize,
    accepts: &[bool],
    char_edges: &[(usize, usize, Vec<bool>)],
    eps: &[(usize, usize)],
) -> FastAutomaton {
    let mut a = FastAutomaton::new_empty();
    for _ in 1..n {
        a.new_state();
    }

    for (s, &acc) in accepts.iter().enumerate() {
        if acc {
            a.accept(s);
        }
    }

    for (from, to, mask) in char_edges {
        let mut range = CharRange::empty();
        for (i, &on) in mask.iter().enumerate() {
            if on {
                range = range.union(&letter_over(alphabet, i));
            }
        }
        a.add_transition_from_range(*from, *to, &range)
            .expect("adding a union of alphabet letters never fails");
    }

    // Epsilon transitions are added last: `add_epsilon_transition` eagerly
    // folds the target's current transitions into the source.
    for (from, to) in eps {
        a.add_epsilon_transition(*from, *to);
    }

    a
}

/// [`build_over`] with the default [`ALPHABET`].
fn build(
    n: usize,
    accepts: &[bool],
    char_edges: &[(usize, usize, Vec<bool>)],
    eps: &[(usize, usize)],
) -> FastAutomaton {
    build_over(ALPHABET, n, accepts, char_edges, eps)
}

/// Strategy producing every DFA over the alphabet with `1..=MAX_STATES` states.
///
/// Determinism is structural: for each state and each letter we choose at most
/// one target, so transitions leaving a state always carry disjoint labels.
///
/// An edge density and an accept density are sampled per automaton (both
/// bounded away from 0 and 1, so every DFA keeps a positive probability).
/// Mostly-total transition functions keep the states connected, which makes
/// degenerate (∅ / {""}) languages the exception rather than the rule. The
/// accept density is centered on ½ because that is where accepting/rejecting
/// states are hardest to merge, i.e. where minimal DFAs stay large.
///
/// A per-automaton `acyclic` flag (≈⅓ of cases) remaps every chosen target
/// into the forward range `from+1..n`, producing a DAG and therefore a
/// **finite** language. Without it nearly every random automaton contains a
/// cycle, and the finite-language paths of the cardinality and max-length
/// analyses go untested. Cyclic mode still reaches every DFA, so coverage is
/// preserved.
pub fn arb_dfa() -> impl Strategy<Value = FastAutomaton> {
    (
        arb_num_states(),
        0.6f64..0.97,
        0.25f64..0.75,
        prop::bool::weighted(0.35),
    )
        .prop_flat_map(|(n, edge_density, accept_density, acyclic)| {
            let accepts = prop::collection::vec(prop::bool::weighted(accept_density), n);
            // An "anchor" state forced accepting most of the time: without it
            // the whole accept vector samples all-false often enough that the
            // empty language eats a fifth of the sample. Non-start states are
            // preferred — anchoring the start only inflates the {""} corner.
            // The `None` branch keeps every accept subset (incl. all-false)
            // reachable.
            let anchor = prop::option::weighted(0.85, 1usize.min(n - 1)..n);
            // transition function: tf[state][base] = optional target state
            let tf = prop::collection::vec(
                prop::collection::vec(prop::option::weighted(edge_density, 0usize..n), num_bases()),
                n,
            );
            (Just(n), accepts, anchor, tf, Just(acyclic))
        })
        .prop_map(|(n, mut accepts, anchor, tf, acyclic)| {
            if let Some(k) = anchor {
                accepts[k] = true;
            }
            let nb = num_bases();
            let mut edges: Vec<(usize, usize, Vec<bool>)> = Vec::new();
            for (from, row) in tf.iter().enumerate() {
                // Group the bases by their chosen target so each (from, to)
                // edge gets a single, disjoint-from-its-siblings label.
                let mut by_target: std::collections::BTreeMap<usize, Vec<bool>> =
                    std::collections::BTreeMap::new();
                for (base, target) in row.iter().enumerate() {
                    if let Some(t) = target {
                        let t = if acyclic {
                            if from + 1 >= n {
                                // the last state of a DAG has no outgoing edge
                                continue;
                            }
                            // remap into the forward range; every forward
                            // target keeps a positive probability
                            from + 1 + (*t % (n - from - 1))
                        } else {
                            *t
                        };
                        by_target.entry(t).or_insert_with(|| vec![false; nb])[base] = true;
                    }
                }
                for (to, mask) in by_target {
                    edges.push((from, to, mask));
                }
            }
            build(n, &accepts, &edges, &[])
        })
}

/// Strategy producing every NFA over the alphabet with `1..=MAX_STATES` states
/// (overlapping labels allowed, plus optional epsilon transitions).
///
/// Instead of a fixed per-bit probability (which blobs large automata and
/// starves small ones), a per-state branching target is sampled and converted
/// into a bit density of `target / (n · |Σ|)`, so the *local* structure is
/// comparable across sizes. Epsilon and accept densities are sampled too. All
/// densities stay strictly inside (0, 1), so every NFA keeps a positive
/// probability.
///
/// As in [`arb_dfa`], the accept density is centered on ½ and a per-automaton
/// `acyclic` flag (≈⅓ of cases) keeps only forward (`from < to`) edges,
/// producing finite languages; the density is rescaled to the smaller target
/// pool so the out-degree stays comparable. Epsilon transitions are kept rare
/// because [`FastAutomaton::add_epsilon_transition`] eagerly folds the target
/// state into the source, which merges languages and shrinks minimal DFAs.
pub fn arb_nfa() -> impl Strategy<Value = FastAutomaton> {
    arb_nfa_over(ALPHABET)
}

/// [`arb_nfa`] generalized to an arbitrary alphabet, so two operands of a
/// binary operation can be generated over *different* alphabets — the only
/// way to exercise `SpanningSet::merge` and the `ConditionConverter`
/// re-projection (same-alphabet operands share an identical spanning set and
/// the conversion is the identity).
pub fn arb_nfa_over(alphabet: &'static [char]) -> impl Strategy<Value = FastAutomaton> {
    (
        arb_num_states(),
        1.0f64..2.8,
        0.02f64..0.12,
        0.25f64..0.75,
        prop::bool::weighted(0.35),
    )
        .prop_flat_map(
            move |(n, target_out_degree, eps_density, accept_density, acyclic)| {
                // In acyclic mode only the upper triangle of the matrix survives,
                // so the average target pool is half as big.
                let effective_targets = if acyclic {
                    (n as f64 / 2.0).max(1.0)
                } else {
                    n as f64
                };
                let label_density = (target_out_degree
                    / (effective_targets * alphabet.len() as f64))
                    .clamp(0.02, 0.95);
                let accepts = prop::collection::vec(prop::bool::weighted(accept_density), n);
                // see arb_dfa: keeps the empty language an edge case, not a fifth
                // of the sample
                let anchor = prop::option::weighted(0.85, 1usize.min(n - 1)..n);
                // labels[from][to] = mask over the alphabet letters
                let labels = prop::collection::vec(
                    prop::collection::vec(
                        prop::collection::vec(prop::bool::weighted(label_density), alphabet.len()),
                        n,
                    ),
                    n,
                );
                // eps[from][to] = whether an epsilon transition is present
                let eps = prop::collection::vec(
                    prop::collection::vec(prop::bool::weighted(eps_density), n),
                    n,
                );
                (Just(n), accepts, anchor, labels, eps, Just(acyclic))
            },
        )
        .prop_map(move |(n, mut accepts, anchor, labels, eps, acyclic)| {
            if let Some(k) = anchor {
                accepts[k] = true;
            }
            let mut char_edges = Vec::new();
            for (from, row) in labels.iter().enumerate() {
                for (to, mask) in row.iter().enumerate() {
                    if acyclic && to <= from {
                        continue;
                    }
                    if mask.iter().any(|&b| b) {
                        char_edges.push((from, to, mask.clone()));
                    }
                }
            }
            let mut eps_edges = Vec::new();
            for (from, row) in eps.iter().enumerate() {
                for (to, &on) in row.iter().enumerate() {
                    let backward = acyclic && to <= from;
                    if on && from != to && !backward {
                        eps_edges.push((from, to));
                    }
                }
            }
            build_over(alphabet, n, &accepts, &char_edges, &eps_edges)
        })
}

/// Strategy for a character class: any subset of the alphabet (including the
/// empty language) and, occasionally, the total range `.`.
///
/// Single letters dominate: the unbiased subset mask would produce the empty
/// class `[]` a quarter of the time, and a single `[]` anywhere in a
/// concatenation collapses the whole expression to the empty language. The
/// mask branch keeps every subset (including `[]`) at a positive probability.
fn arb_charrange() -> impl Strategy<Value = CharRange> {
    prop_oneof![
        8 => (0..ALPHABET.len()).prop_map(letter),
        3 => prop::collection::vec(any::<bool>(), ALPHABET.len()).prop_map(|mask| {
            let mut r = CharRange::empty();
            for (i, &on) in mask.iter().enumerate() {
                if on {
                    r = r.union(&letter(i));
                }
            }
            r
        }),
        1 => Just(CharRange::total()),
    ]
}

/// Strategy producing regular expressions over the four [`RegularExpression`]
/// variants up to a bounded recursion depth.
///
/// Repetition gets the heaviest weight: it is the variant that feeds the
/// `{n,m}` expansion, the simplifier and the loop handling of state
/// elimination, and stacking it (`(a*){2}`-style nesting) is where those
/// paths historically break. Concat and alternation still keep substantial
/// weight so all shapes appear.
pub fn arb_regex() -> impl Strategy<Value = RegularExpression> {
    let leaf = arb_charrange().prop_map(RegularExpression::Character);
    leaf.prop_recursive(4, 48, 3, |inner| {
        prop_oneof![
            3 => (inner.clone(), 0u32..=2, 0u32..=2, any::<bool>()).prop_map(
                |(r, min, extra, has_max)| {
                    // max is min + extra, so the bounds are always valid
                    let max = if has_max { Some(min + extra) } else { None };
                    RegularExpression::Repetition(Box::new(r), min, max)
                }
            ),
            2 => prop::collection::vec(inner.clone(), 1..=3)
                .prop_map(|v| RegularExpression::Concat(v.into())),
            2 => prop::collection::vec(inner, 1..=3).prop_map(RegularExpression::Alternation),
        ]
    })
}

/// Runs `f` under a bounded execution profile, returning `None` when the
/// operation legitimately exceeds the state/time budget (which is not a bug).
fn bounded<T, F: FnOnce() -> Result<T, EngineError>>(f: F) -> Option<T> {
    ExecutionProfileBuilder::new()
        .max_number_of_states(8192)
        .execution_timeout(3000)
        .build()
        .run(|| match f() {
            Ok(v) => Some(v),
            Err(EngineError::AutomatonHasTooManyStates)
            | Err(EngineError::OperationTimeOutError) => None,
            Err(e) => panic!("unexpected engine error: {e:?}"),
        })
}

fn determinized(a: &FastAutomaton) -> Option<FastAutomaton> {
    bounded(|| a.determinize().map(|c| c.into_owned()))
}

fn complemented(a: &FastAutomaton) -> Option<FastAutomaton> {
    bounded(|| {
        let mut c = a.clone();
        c.complement()?;
        Ok(c)
    })
}

/// All strings up to length `max_len` over `alphabet` (plus the empty
/// string).
fn probes_over(alphabet: &[char], max_len: usize) -> Vec<String> {
    let mut all = vec![String::new()];
    let mut frontier = vec![String::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for w in &frontier {
            for &c in alphabet {
                let mut s = w.clone();
                s.push(c);
                next.push(s);
            }
        }
        all.extend(next.iter().cloned());
        frontier = next;
    }
    all
}

/// All strings up to length 4 over [`ALPHABET`] (plus the empty string).
fn probes() -> Vec<String> {
    probes_over(ALPHABET, 4)
}

/// Asserts that intersection / union / difference of `a` and `b` agree with
/// the boolean combination of the operands on every probe string.
fn assert_set_ops_membership(
    a: &FastAutomaton,
    b: &FastAutomaton,
    probes: &[String],
) -> Result<(), TestCaseError> {
    if let Some(inter) = bounded(|| a.intersection(b)) {
        for s in probes {
            prop_assert_eq!(
                inter.is_match(s),
                a.is_match(s) && b.is_match(s),
                "intersection membership for {:?}",
                s
            );
        }
    }
    if let Some(union) = bounded(|| a.union(b)) {
        for s in probes {
            prop_assert_eq!(
                union.is_match(s),
                a.is_match(s) || b.is_match(s),
                "union membership for {:?}",
                s
            );
        }
    }
    // `difference` determinizes the subtrahend itself.
    if let Some(diff) = bounded(|| a.difference(b)) {
        for s in probes {
            prop_assert_eq!(
                diff.is_match(s),
                a.is_match(s) && !b.is_match(s),
                "difference membership for {:?}",
                s
            );
        }
    }
    Ok(())
}

/// Decomposition oracle for repetition: `s` is in L(a){min,max} iff `s`
/// splits into k pieces, each in L(a), for some valid k. Piece counts
/// saturate at `min` once they can only grow (relevant for unbounded
/// maxima and for "" ∈ L(a), which allows padding with empty pieces).
fn repeat_decomposition_oracle(a: &FastAutomaton, s: &str, min: u32, max: Option<u32>) -> bool {
    let min = min as usize;
    let cap = max.map(|m| m as usize).unwrap_or(min).max(min);
    let accepts_empty = a.is_match("");
    let len = s.len();

    // reach[i][k]: the prefix of length i splits into exactly k pieces
    // (k saturated at cap + 1 to keep the table finite).
    let k_slots = cap + 2;
    let mut reach = vec![vec![false; k_slots]; len + 1];
    reach[0][0] = true;
    for i in 0..=len {
        for k in 0..k_slots {
            if !reach[i][k] {
                continue;
            }
            let next_k = (k + 1).min(cap + 1);
            // Pad with an empty piece.
            if accepts_empty {
                reach[i][next_k] = true;
            }
            // Consume a non-empty piece.
            for j in i + 1..=len {
                if a.is_match(&s[i..j]) {
                    reach[j][next_k] = true;
                }
            }
        }
    }

    let k_ok = |k: usize| {
        k >= min
            && match max {
                Some(m) => k <= m as usize,
                None => true,
            }
    };
    (0..k_slots).any(|k| reach[len][k] && k_ok(k))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    /// The DFA strategy really does produce deterministic automata.
    #[test]
    fn dfa_strategy_is_deterministic(a in arb_dfa()) {
        prop_assert!(a.is_deterministic(), "arb_dfa produced a non-deterministic automaton");
    }

    /// `a` and `determinize(a)` accept the same language.
    #[test]
    fn determinize_preserves_language(a in arb_nfa()) {
        if let Some(d) = determinized(&a) {
            prop_assert!(d.is_deterministic());
            if let Some(eq) = bounded(|| a.equivalent(&d)) {
                prop_assert!(eq, "determinize changed the language");
            }
        }
    }

    /// Minimizing a DFA preserves its language.
    #[test]
    fn minimize_preserves_language(a in arb_nfa()) {
        if let Some(d) = determinized(&a) {
            let mut m = d.clone();
            if bounded(|| m.minimize()).is_some()
                && let Some(eq) = bounded(|| d.equivalent(&m))
            {
                prop_assert!(eq, "minimize changed the language");
            }
        }
    }

    /// Complement laws: membership flips, `a ∩ ¬a = ∅`, `a ∪ ¬a = Σ*`.
    #[test]
    fn complement_laws(a in arb_dfa()) {
        let d = match determinized(&a) { Some(d) => d, None => return Ok(()) };
        let c = match complemented(&d) { Some(c) => c, None => return Ok(()) };

        for s in probes() {
            prop_assert_eq!(d.is_match(&s), !c.is_match(&s), "complement membership for {:?}", s);
        }

        if let Some(inter) = bounded(|| d.intersection(&c))
            && let Some(empty) = bounded(|| inter.equivalent(&FastAutomaton::new_empty()))
        {
            prop_assert!(empty, "a ∩ ¬a is not empty");
        }
        if let Some(union) = bounded(|| d.union(&c))
            && let Some(total) = bounded(|| union.equivalent(&FastAutomaton::new_total()))
        {
            prop_assert!(total, "a ∪ ¬a is not total");
        }
    }

    /// Membership of intersection / union / difference matches the boolean
    /// combination of the operands on every probe string.
    #[test]
    fn set_ops_membership(a in arb_nfa(), b in arb_nfa()) {
        assert_set_ops_membership(&a, &b, &probes())?;
    }

    /// Set operations across operands built over *overlapping but different*
    /// alphabets ({a,b} vs {b,c}): the operands carry different spanning
    /// sets, so `SpanningSet::merge` and the `ConditionConverter`
    /// re-projection do real work (same-alphabet pairs convert via the
    /// identity). The shared letter `b` keeps the intersections non-trivial.
    #[test]
    fn set_ops_membership_overlapping_alphabets(
        a in arb_nfa_over(&['a', 'b']),
        b in arb_nfa_over(&['b', 'c']),
    ) {
        assert_set_ops_membership(&a, &b, &probes_over(&['a', 'b', 'c'], 4))?;
    }

    /// Set operations across operands built over *disjoint* alphabets
    /// ({a,b} vs {c,d}): the merged spanning set shares no base with either
    /// source, the most extreme re-projection. The intersection collapses to
    /// at most {""} — itself a worthwhile edge case.
    #[test]
    fn set_ops_membership_disjoint_alphabets(
        a in arb_nfa_over(&['a', 'b']),
        b in arb_nfa_over(&['c', 'd']),
    ) {
        assert_set_ops_membership(&a, &b, &probes_over(&['a', 'b', 'c', 'd'], 3))?;
    }

    /// `get_length` and `get_cardinality` agree with brute-force enumeration.
    /// The probes cover *every* string up to length 4, so they are exactly
    /// the language whenever the maximum length is ≤ 4, and a complete
    /// census of its short strings otherwise.
    #[test]
    fn length_cardinality_match_brute_force(a in arb_nfa()) {
        let (min, max) = a.get_length();
        let matched_lengths: Vec<u32> = probes()
            .iter()
            .filter(|s| a.is_match(s))
            .map(|s| s.chars().count() as u32)
            .collect();

        // Minimum: any string of length ≤ 4 is a probe, so a language with
        // min ≤ 4 has a matched probe of exactly that length.
        match (min, matched_lengths.iter().min()) {
            (Some(min_len), Some(&shortest)) => {
                prop_assert_eq!(min_len, shortest, "min length disagrees with enumeration");
            }
            (Some(min_len), None) => {
                prop_assert!(min_len > 4, "min ≤ 4 but no probe matched");
            }
            (None, Some(_)) => prop_assert!(false, "empty language matched a probe"),
            (None, None) => {}
        }

        if let Some(max_len) = max
            && max_len <= 4
        {
            // The probes enumerate the whole language.
            prop_assert_eq!(
                Some(max_len),
                matched_lengths.iter().max().copied(),
                "max length disagrees with enumeration"
            );
            if let Some(cardinality) = bounded(|| a.get_cardinality()) {
                prop_assert_eq!(
                    cardinality,
                    Cardinality::Integer(matched_lengths.len() as u32),
                    "cardinality disagrees with enumeration"
                );
            }
        } else if max.is_none()
            && min.is_some()
            && let Some(cardinality) = bounded(|| a.get_cardinality())
        {
            // A cycle on an accepting path means infinitely many strings.
            prop_assert_eq!(
                cardinality,
                Cardinality::Infinite,
                "infinite language with non-infinite cardinality"
            );
        }
    }

    /// `FastAutomaton::concat` agrees with the split-membership oracle:
    /// s ∈ L(a)·L(b) iff some split s = u·v has u ∈ L(a) and v ∈ L(b).
    #[test]
    fn automaton_concat_matches_split_oracle(a in arb_nfa(), b in arb_nfa()) {
        if let Some(concat) = bounded(|| a.concat(&b)) {
            for s in probes() {
                let expected =
                    (0..=s.len()).any(|i| a.is_match(&s[..i]) && b.is_match(&s[i..]));
                prop_assert_eq!(
                    concat.is_match(&s), expected,
                    "concat membership for {:?}", s
                );
            }
        }
    }

    /// `FastAutomaton::repeat` agrees with a decomposition oracle computed by
    /// dynamic programming over (position, piece-count) — independent of the
    /// engine's own repeat construction (which the regex route would reuse).
    #[test]
    fn automaton_repeat_matches_decomposition_oracle(
        a in arb_nfa(),
        min in 0u32..3,
        extra in 0u32..2,
        unbounded in any::<bool>(),
    ) {
        let max = if unbounded { None } else { Some(min + extra) };
        if let Some(repeated) = bounded(|| a.repeat(min, max)) {
            for s in probes() {
                let expected = repeat_decomposition_oracle(&a, &s, min, max);
                prop_assert_eq!(
                    repeated.is_match(&s), expected,
                    "repeat({}, {:?}) membership for {:?}", min, max, s
                );
            }
        }
    }

    /// `Term::union` / `Term::intersection` over more than 3 operands (the
    /// parallel dispatch path when the `parallel` feature is on) agree with
    /// sequential pairwise folds.
    #[test]
    fn many_operand_term_ops_match_pairwise_folds(
        a in arb_nfa(), b in arb_nfa(), c in arb_nfa(), d in arb_nfa(), e in arb_nfa(),
    ) {
        use regexsolver::Term;

        let operands: Vec<Term> = [&b, &c, &d, &e]
            .into_iter()
            .map(|x| Term::from_automaton(x.clone()))
            .collect();
        let first = Term::from_automaton(a.clone());

        if let Some(many) = bounded(|| {
            Ok(first.union(&operands)?.to_automaton()?.into_owned())
        }) && let Some(pairwise) = bounded(|| {
            let mut acc = a.clone();
            for x in [&b, &c, &d, &e] {
                acc = acc.union(x)?;
            }
            Ok(acc)
        }) && let Some(eq) = bounded(|| many.equivalent(&pairwise)) {
            prop_assert!(eq, "5-operand union disagrees with pairwise folds");
        }

        if let Some(many) = bounded(|| {
            Ok(first.intersection(&operands)?.to_automaton()?.into_owned())
        }) && let Some(pairwise) = bounded(|| {
            let mut acc = a.clone();
            for x in [&b, &c, &d, &e] {
                acc = acc.intersection(x)?;
            }
            Ok(acc)
        }) && let Some(eq) = bounded(|| many.equivalent(&pairwise)) {
            prop_assert!(eq, "5-operand intersection disagrees with pairwise folds");
        }
    }

    /// `subset` and `equivalent` agree: mutual subset iff equivalent; both are
    /// reflexive.
    #[test]
    fn subset_equivalent_consistency(a in arb_nfa(), b in arb_nfa()) {
        if let Some(refl) = bounded(|| a.equivalent(&a)) {
            prop_assert!(refl, "equivalent is not reflexive");
        }
        if let Some(refl) = bounded(|| a.subset(&a)) {
            prop_assert!(refl, "subset is not reflexive");
        }
        if let (Some(ab), Some(ba), Some(eq)) = (
            bounded(|| a.subset(&b)),
            bounded(|| b.subset(&a)),
            bounded(|| a.equivalent(&b)),
        ) {
            prop_assert_eq!(ab && ba, eq, "mutual subset disagrees with equivalent");
        }
    }

    /// `a -> regex -> a` round-trips: the regular expression extracted from an
    /// automaton compiles back to an equivalent automaton.
    #[test]
    fn automaton_to_regex_roundtrip(a in arb_nfa()) {
        let r = a.to_regex();
        if let Some(a2) = bounded(|| r.to_automaton())
            && let Some(eq) = bounded(|| a.equivalent(&a2))
        {
            prop_assert!(eq, "automaton -> regex -> automaton changed the language: {}", r);
        }
    }

    /// Regular expressions round-trip through an automaton and agree with the
    /// reference `regex` crate on every probe string.
    #[test]
    fn regex_roundtrip_and_oracle(r in arb_regex()) {
        let a = match bounded(|| r.to_automaton()) { Some(a) => a, None => return Ok(()) };

        // regex -> automaton -> regex -> automaton preserves the language.
        let r2 = a.to_regex();
        if let Some(a2) = bounded(|| r2.to_automaton())
            && let Some(eq) = bounded(|| a.equivalent(&a2))
        {
            prop_assert!(eq, "regex round-trip changed the language: {} -> {}", r, r2);
        }

        // Cross-check membership against the standard regex engine (anchored,
        // dot-matches-newline). Patterns denoting the empty language ("[]") are
        // rejected by the `regex` crate, so we only compare when it accepts the
        // pattern.
        let pattern = r.to_string();
        if let Ok(re) = regex::Regex::new(&format!("(?s)^(?:{})$", pattern)) {
            for s in probes() {
                prop_assert_eq!(
                    a.is_match(&s), re.is_match(&s),
                    "pattern {:?} disagrees with reference engine on {:?}", pattern, s
                );
            }
        }
    }
}

#[cfg(test)]
mod inspect {
    use super::*;
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::TestRunner;
    use regex_charclass::CharacterClass;

    fn samples<S: Strategy>(strat: S, n: usize) -> Vec<S::Value> {
        let mut runner = TestRunner::deterministic();
        (0..n)
            .map(|_| strat.new_tree(&mut runner).unwrap().current())
            .collect()
    }

    #[test]
    fn show_regex() {
        for (i, r) in samples(arb_regex(), 30).into_iter().enumerate() {
            println!("regex[{i:02}] = {r}");
        }
    }

    #[test]
    fn show_dfa() {
        for (i, a) in samples(arb_dfa(), 30).into_iter().enumerate() {
            println!("dfa[{i:02}] det={} Graphviz={}", a.is_deterministic(), a);
        }
    }

    #[test]
    fn show_nfa() {
        for (i, a) in samples(arb_nfa(), 30).into_iter().enumerate() {
            println!("nfa[{i:02}] det={} Graphviz={}", a.is_deterministic(), a);
        }
    }

    /// The language class of a generated entity, ordered from degenerate to
    /// rich.
    ///
    /// `Empty`, `EmptyString` and `Total` are the corners of the language
    /// lattice: useful as occasional edge cases (they hit the `is_empty` /
    /// complement / difference fast paths) but they exercise nothing else.
    /// `Finite` languages take the topological-sort path of the cardinality
    /// and max-length analyses; `Infinite` ones take the cycle paths of state
    /// elimination and repeat synthesis. A quality sample needs both in bulk.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    enum LangClass {
        Empty,
        EmptyString,
        Total,
        Finite,
        Infinite,
    }

    /// Reduces an automaton to its canonical minimal DFA.
    fn minimal_dfa(a: &FastAutomaton) -> FastAutomaton {
        let mut m = determinized(a).expect("generated automata always determinize in budget");
        bounded(|| m.minimize()).expect("generated automata always minimize in budget");
        m
    }

    /// Classifies the language of a **minimal DFA**.
    ///
    /// Unlike the trivial-vs-"interesting" split this used to be, the
    /// non-trivial bulk is split into finite and infinite languages, which
    /// exercise disjoint code paths (see [`LangClass`]).
    fn classify(m: &FastAutomaton) -> LangClass {
        if m.is_empty() {
            LangClass::Empty
        } else if m.is_empty_string() {
            LangClass::EmptyString
        } else if m.is_total() {
            // exact on a DFA
            LangClass::Total
        } else if m.get_length().1.is_some() {
            LangClass::Finite
        } else {
            LangClass::Infinite
        }
    }

    /// Canonical fingerprint of a language: the minimal DFA, renumbered in
    /// BFS order with the outgoing transitions of each state sorted by label.
    /// Minimal DFAs are unique up to isomorphism, so two automata share a key
    /// iff they accept the same language — this is what lets the stats count
    /// *distinct* languages instead of distinct syntax trees.
    fn language_key(m: &FastAutomaton) -> String {
        use std::fmt::Write;
        let ss = m.get_spanning_set();
        let mut order = vec![m.get_start_state()];
        let mut ids = std::collections::HashMap::new();
        ids.insert(m.get_start_state(), 0usize);
        let mut key = String::new();
        let mut i = 0;
        while i < order.len() {
            let s = order[i];
            i += 1;
            // In a DFA the labels leaving a state are disjoint, hence unique,
            // so sorting by label gives a deterministic traversal order.
            let mut out: Vec<(String, usize)> = m
                .transitions_from_vec(s)
                .into_iter()
                .map(|(c, t)| {
                    (
                        c.to_range(ss)
                            .expect("condition always converts to a range")
                            .to_regex(),
                        t,
                    )
                })
                .collect();
            out.sort();
            write!(key, "{}", if m.is_accepted(s) { 'A' } else { 'r' }).unwrap();
            for (label, t) in out {
                let id = match ids.get(&t) {
                    Some(&id) => id,
                    None => {
                        let id = order.len();
                        ids.insert(t, id);
                        order.push(t);
                        id
                    }
                };
                write!(key, " {label}>{id}").unwrap();
            }
            key.push(';');
        }
        key
    }

    /// Everything we measure about one generated automaton.
    struct Measure {
        // Structure of the generated entity itself.
        edges: usize,
        multi_base_edges: usize,
        deterministic: bool,
        // Properties of its *language*, computed on the minimal DFA.
        class: LangClass,
        minimal_states: usize,
        accepts_empty_string: bool,
        key: String,
    }

    fn measure(a: &FastAutomaton) -> Measure {
        let mut edges = 0;
        let mut multi_base_edges = 0;
        for s in a.states_vec() {
            for (cond, _) in a.transitions_from_vec(s) {
                edges += 1;
                if cond
                    .get_binary_representation()
                    .iter()
                    .filter(|&&b| b)
                    .count()
                    > 1
                {
                    multi_base_edges += 1;
                }
            }
        }
        let m = minimal_dfa(a);
        Measure {
            edges,
            multi_base_edges,
            deterministic: a.is_deterministic(),
            class: classify(&m),
            minimal_states: m.get_number_of_states(),
            accepts_empty_string: m.is_match(""),
            key: language_key(&m),
        }
    }

    /// Aggregated quality metrics over a sample; what the strategies are
    /// evaluated (and asserted) on.
    struct Quality {
        /// Share of `Empty` + `EmptyString` + `Total` languages. Wanted as a
        /// small minority: present (they are real edge cases) but not eating
        /// the sample.
        degenerate_pct: f64,
        finite_pct: f64,
        infinite_pct: f64,
        /// Distinct languages (by canonical minimal DFA) over sample size.
        /// Duplicates re-test the same language and are wasted cases.
        distinct_pct: f64,
        /// Average minimal-DFA size: the number of Myhill-Nerode classes is
        /// what minimize / equivalence / state elimination actually scale
        /// with, so this — not the raw state count — is language complexity.
        avg_minimal_states: f64,
        /// Share of languages needing a minimal DFA of ≥ 3 states.
        rich_pct: f64,
        accepts_empty_string_pct: f64,
        /// Share of transitions whose condition spans more than one base of
        /// the spanning set (exercises the bitvector paths beyond single
        /// bits).
        multi_base_edge_pct: f64,
        /// Share of genuinely nondeterministic automata (only meaningful for
        /// the NFA strategy: a "NFA" that is already deterministic never
        /// exercises subset construction).
        nondeterministic_pct: f64,
    }

    fn quality(name: &str, measures: &[Measure]) -> Quality {
        let n = measures.len() as f64;
        let count = |f: &dyn Fn(&Measure) -> bool| {
            100.0 * measures.iter().filter(|m| f(m)).count() as f64 / n
        };

        let distinct: std::collections::HashSet<&str> =
            measures.iter().map(|m| m.key.as_str()).collect();
        let edges: usize = measures.iter().map(|m| m.edges).sum();
        let multi: usize = measures.iter().map(|m| m.multi_base_edges).sum();

        let mut histogram = std::collections::BTreeMap::new();
        for m in measures {
            *histogram.entry(m.minimal_states).or_insert(0usize) += 1;
        }

        let q = Quality {
            degenerate_pct: count(&|m| {
                matches!(
                    m.class,
                    LangClass::Empty | LangClass::EmptyString | LangClass::Total
                )
            }),
            finite_pct: count(&|m| m.class == LangClass::Finite),
            infinite_pct: count(&|m| m.class == LangClass::Infinite),
            distinct_pct: 100.0 * distinct.len() as f64 / n,
            avg_minimal_states: measures.iter().map(|m| m.minimal_states).sum::<usize>() as f64 / n,
            rich_pct: count(&|m| m.minimal_states >= 3),
            accepts_empty_string_pct: count(&|m| m.accepts_empty_string),
            multi_base_edge_pct: 100.0 * multi as f64 / edges.max(1) as f64,
            nondeterministic_pct: count(&|m| !m.deterministic),
        };

        println!(
            "{name}: degenerate {:>4.1}% (∅ {:.1}% | {{\"\"}} {:.1}% | Σ* {:.1}%) | finite {:>4.1}% | infinite {:>4.1}%",
            q.degenerate_pct,
            count(&|m| m.class == LangClass::Empty),
            count(&|m| m.class == LangClass::EmptyString),
            count(&|m| m.class == LangClass::Total),
            q.finite_pct,
            q.infinite_pct,
        );
        println!(
            "{name}: distinct languages {:>4.1}% | accepts \"\" {:>4.1}% | nondet {:>4.1}% | multi-base edges {:>4.1}%",
            q.distinct_pct,
            q.accepts_empty_string_pct,
            q.nondeterministic_pct,
            q.multi_base_edge_pct,
        );
        println!(
            "{name}: minimal-DFA states avg {:.2}, ≥3 {:>4.1}%, histogram {:?}",
            q.avg_minimal_states, q.rich_pct, histogram,
        );
        q
    }

    /// Operator coverage of a generated regular expression; the round-trip
    /// (state elimination) and simplification code paths are keyed on these
    /// shapes.
    #[derive(Default)]
    struct RegexFacets {
        unbounded_repetition: bool,
        bounded_repetition: bool,
        nested_repetition: bool,
        alternation: bool,
        multi_char_class: bool,
    }

    fn regex_facets(r: &RegularExpression, inside_repetition: bool, f: &mut RegexFacets) {
        match r {
            RegularExpression::Character(range) => {
                let letters = (0..ALPHABET.len())
                    .filter(|&i| !range.intersection(&letter(i)).is_empty())
                    .count();
                if letters > 1 || range.is_total() {
                    f.multi_char_class = true;
                }
            }
            RegularExpression::Repetition(inner, _, max) => {
                if max.is_some() {
                    f.bounded_repetition = true;
                } else {
                    f.unbounded_repetition = true;
                }
                if inside_repetition {
                    f.nested_repetition = true;
                }
                regex_facets(inner, true, f);
            }
            RegularExpression::Concat(parts) => {
                for p in parts {
                    regex_facets(p, inside_repetition, f);
                }
            }
            RegularExpression::Alternation(parts) => {
                f.alternation = true;
                for p in parts {
                    regex_facets(p, inside_repetition, f);
                }
            }
        }
    }

    /// Quantitative quality summary over a larger sample, with floors the
    /// strategies must keep. The sample runner is deterministic, so the
    /// numbers — and therefore the assertions — are reproducible.
    #[test]
    fn stats() {
        const N: usize = 300;

        let regexes = samples(arb_regex(), N);
        let regex_measures: Vec<Measure> = regexes
            .iter()
            .map(|r| measure(&r.to_automaton().expect("small regexes always convert")))
            .collect();
        let regex_q = quality("regex", &regex_measures);

        let avg_len = regexes.iter().map(|r| r.to_string().len()).sum::<usize>() as f64 / N as f64;
        let mut facet_counts = [0usize; 5];
        for r in &regexes {
            let mut f = RegexFacets::default();
            regex_facets(r, false, &mut f);
            for (i, hit) in [
                f.unbounded_repetition,
                f.bounded_repetition,
                f.nested_repetition,
                f.alternation,
                f.multi_char_class,
            ]
            .into_iter()
            .enumerate()
            {
                facet_counts[i] += hit as usize;
            }
        }
        let fpct = |i: usize| 100.0 * facet_counts[i] as f64 / N as f64;
        println!(
            "regex: avg pattern length {avg_len:.1} | unbounded-rep {:.1}% | bounded-rep {:.1}% | nested-rep {:.1}% | alternation {:.1}% | multi-char class {:.1}%",
            fpct(0),
            fpct(1),
            fpct(2),
            fpct(3),
            fpct(4),
        );

        let dfa_q = quality(
            "dfa  ",
            &samples(arb_dfa(), N)
                .iter()
                .map(measure)
                .collect::<Vec<_>>(),
        );
        let nfa_q = quality(
            "nfa  ",
            &samples(arb_nfa(), N)
                .iter()
                .map(measure)
                .collect::<Vec<_>>(),
        );

        for (name, q) in [("regex", &regex_q), ("dfa", &dfa_q), ("nfa", &nfa_q)] {
            // The trivial corner languages should be present but a minority.
            assert!(
                q.degenerate_pct < 25.0,
                "{name}: too many degenerate languages"
            );
            // Both bulk classes must be well represented.
            assert!(
                q.finite_pct >= 15.0,
                "{name}: finite languages under-represented"
            );
            assert!(
                q.infinite_pct >= 15.0,
                "{name}: infinite languages under-represented"
            );
            // The sample must not keep re-testing the same languages.
            assert!(
                q.distinct_pct >= 45.0,
                "{name}: not enough distinct languages"
            );
            // Language complexity: minimal DFAs must not collapse to 1-2 states.
            assert!(q.rich_pct >= 35.0, "{name}: minimal DFAs too small");
            // Both "" ∈ L and "" ∉ L need bulk representation.
            assert!(
                (20.0..=80.0).contains(&q.accepts_empty_string_pct),
                "{name}: empty-string acceptance unbalanced"
            );
            // Conditions spanning several bases must show up regularly.
            assert!(
                q.multi_base_edge_pct >= 10.0,
                "{name}: multi-base conditions too rare"
            );
        }
        // An NFA strategy that mostly produces DFAs never exercises subset
        // construction.
        assert!(
            nfa_q.nondeterministic_pct >= 50.0,
            "nfa: mostly deterministic"
        );
    }
}
