//! Benchmarks covering the main operation families of the library.
//!
//! Inputs come in named sizes so numbers stay comparable across versions:
//!
//! * `small` / `medium` / `large`: realistic patterns of increasing size.
//! * `blowup_N`: the classic `(a|b)*a(a|b){N}` family whose minimal DFA has
//!   2^N states: the worst case of subset construction.
//!
//! Mutating operations (`minimize`, `complement`) are measured with
//! `iter_batched` on a fresh clone per iteration, so flag short-circuits
//! (e.g. `minimize` early-returning on an already-minimal automaton) don't
//! skew the numbers.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use regex_charclass::char::Char;
use regexsolver::fast_automaton::{CharacterOrder, FastAutomaton, GenerationOptions, PathOrder};
use regexsolver::regex::RegularExpression;
use regexsolver::{CharRange, Term};
use std::hint::black_box;

const SMALL: (&str, &str) = ("small", "(abc|de){2}");
const MEDIUM: (&str, &str) = ("medium", "a(bcfe|bcdg|mkv)*(abc){2,3}(abc){2}");
const LARGE: (&str, &str) = (
    "large",
    "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
);

fn automaton(pattern: &str) -> FastAutomaton {
    RegularExpression::new(pattern)
        .unwrap()
        .to_automaton()
        .unwrap()
}

fn dfa(pattern: &str) -> FastAutomaton {
    automaton(pattern).determinize().unwrap().into_owned()
}

/// `(a|b)*a(a|b){n}`: an n+2-state NFA whose minimal DFA has 2^(n+1) states.
fn blowup_pattern(n: usize) -> String {
    format!("(a|b)*a(a|b){{{n}}}")
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");
    for (name, pattern) in [SMALL, MEDIUM, LARGE] {
        group.bench_with_input(BenchmarkId::from_parameter(name), pattern, |b, pattern| {
            b.iter(|| RegularExpression::new(black_box(pattern)).unwrap())
        });
    }
    group.finish();
}

fn bench_to_automaton(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_automaton");
    for (name, pattern) in [SMALL, MEDIUM, LARGE] {
        let regex = RegularExpression::new(pattern).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(name), &regex, |b, regex| {
            b.iter(|| black_box(regex).to_automaton().unwrap())
        });
    }
    group.finish();
}

fn bench_determinize(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinize");
    for n in [5, 10] {
        let nfa = automaton(&blowup_pattern(n));
        group.bench_with_input(BenchmarkId::new("blowup", n), &nfa, |b, nfa| {
            b.iter(|| black_box(nfa).determinize().unwrap().into_owned())
        });
    }
    let nfa = automaton(LARGE.1);
    group.bench_with_input(BenchmarkId::from_parameter("large"), &nfa, |b, nfa| {
        b.iter(|| black_box(nfa).determinize().unwrap().into_owned())
    });
    group.finish();
}

fn bench_minimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimize");
    for n in [5, 10] {
        let blowup_dfa = dfa(&blowup_pattern(n));
        group.bench_with_input(BenchmarkId::new("blowup", n), &blowup_dfa, |b, dfa| {
            b.iter_batched(
                || dfa.clone(),
                |mut automaton| {
                    automaton.minimize().unwrap();
                    automaton
                },
                BatchSize::SmallInput,
            )
        });
    }
    let large_dfa = dfa(LARGE.1);
    group.bench_with_input(
        BenchmarkId::from_parameter("large"),
        &large_dfa,
        |b, dfa| {
            b.iter_batched(
                || dfa.clone(),
                |mut automaton| {
                    automaton.minimize().unwrap();
                    automaton
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");

    let a = automaton("a(bcfe|bcdg|mkv)*(abc){1,3}");
    let b_op = automaton("a(bcfe|mkv|opr)*(abc){2,4}");
    group.bench_function("intersection", |b| {
        b.iter(|| black_box(&a).intersection(black_box(&b_op)).unwrap())
    });
    group.bench_function("union", |b| {
        b.iter(|| black_box(&a).union(black_box(&b_op)).unwrap())
    });

    let minuend = automaton(".*abc.*");
    let subtrahend = automaton(".*def.*");
    group.bench_function("difference", |b| {
        b.iter(|| {
            black_box(&minuend)
                .difference(black_box(&subtrahend))
                .unwrap()
        })
    });

    let complement_input = dfa(".*abc.*");
    group.bench_function("complement", |b| {
        b.iter_batched(
            || complement_input.clone(),
            |mut automaton| {
                automaton.complement().unwrap();
                automaton
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_decision(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision");

    // Same language, structurally different automata: the `self == other`
    // shortcut cannot fire, forcing the full check in both directions.
    let left_form = automaton("(a|b)*abc(a|b)*");
    let right_form = automaton("(a*b*)*abc(b*a*)*");
    assert_ne!(left_form, right_form);
    assert!(left_form.equivalent(&right_form).unwrap());
    group.bench_function("equivalent", |b| {
        b.iter(|| {
            black_box(&left_form)
                .equivalent(black_box(&right_form))
                .unwrap()
        })
    });

    let smaller = automaton("abc(de|fg){1,3}");
    let bigger = automaton("abc.*");
    group.bench_function("subset", |b| {
        b.iter(|| black_box(&smaller).subset(black_box(&bigger)).unwrap())
    });

    let left = automaton(".*abc.*");
    let right = automaton(".*cba.*");
    group.bench_function("has_intersection", |b| {
        b.iter(|| {
            black_box(&left)
                .has_intersection(black_box(&right))
                .unwrap()
        })
    });

    group.finish();
}

fn bench_analyze(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze");

    let finite = dfa("[a-z]{1,6}");
    group.bench_function("length/finite", |b| b.iter(|| black_box(&finite).length()));
    group.bench_function("cardinality/finite", |b| {
        b.iter(|| black_box(&finite).cardinality().unwrap())
    });

    let infinite = automaton(LARGE.1);
    group.bench_function("length/large", |b| b.iter(|| black_box(&infinite).length()));

    group.finish();
}

fn bench_to_regex(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_regex");

    let nfa = automaton(MEDIUM.1);
    group.bench_function("nfa", |b| b.iter(|| black_box(&nfa).to_regex()));

    let medium_dfa = dfa(MEDIUM.1);
    group.bench_function("dfa", |b| b.iter(|| black_box(&medium_dfa).to_regex()));

    group.finish();
}

fn bench_generate_strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_strings");

    let automaton = dfa("[a-z]{1,4}");
    group.bench_function("first_2000", |b| {
        b.iter(|| {
            black_box(&automaton)
                .generate_strings(2000, 0, PathOrder::Sweep)
                .unwrap()
        })
    });

    // The offset fast-skips whole subtrees by counting paths.
    let deep = dfa("[a-z]{1,10}");
    group.bench_function("deep_offset", |b| {
        b.iter(|| {
            black_box(&deep)
                .generate_strings(100, 1_000_000, PathOrder::Sweep)
                .unwrap()
        })
    });

    // Interleaving walks the automaton once per pass instead of settling on
    // one path, so it pays for the paths it spreads over.
    group.bench_function("interleave_2000", |b| {
        b.iter(|| {
            black_box(&automaton)
                .generate_strings(2000, 0, PathOrder::Interleave)
                .unwrap()
        })
    });

    // Shuffling adds a Feistel permutation per string and a seeded tie-break
    // per queued path on top of that.
    group.bench_function("shuffled_2000", |b| {
        b.iter(|| {
            black_box(&automaton)
                .generate_strings(2000, 0, (PathOrder::Shuffled, CharacterOrder::Shuffled))
                .unwrap()
        })
    });

    // A charset costs one intersection per transition condition, up front.
    let printable = CharRange::new_from_range(Char::new(' ')..=Char::new('~'));
    let options = GenerationOptions::from(PathOrder::Sweep).with_charset(printable);
    group.bench_function("charset_2000", |b| {
        b.iter(|| {
            black_box(&automaton)
                .generate_strings(2000, 0, options.clone())
                .unwrap()
        })
    });

    group.finish();
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    // A 64-transition chain over a growing alphabet: every few transitions
    // extend the spanning set and re-project the existing conditions.
    group.bench_function("add_transition_from_range/chain_64", |b| {
        b.iter(|| {
            let mut automaton = FastAutomaton::new_empty();
            let mut previous = 0;
            for i in 0..64u8 {
                let next = automaton.new_state();
                let character = Char::new(char::from(b'a' + (i % 26)));
                let range = CharRange::new_from_range(character..=character);
                automaton
                    .add_transition_from_range(previous, next, &range)
                    .unwrap();
                previous = next;
            }
            automaton.accept(previous);
            automaton
        })
    });

    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    // The front-page scenario: parse two patterns, intersect, print back.
    group.bench_function("intersection_to_pattern", |b| {
        b.iter(|| {
            let a = Term::from_pattern(black_box("(ab|xy){2}")).unwrap();
            let b_term = Term::from_pattern(black_box(".*xy")).unwrap();
            a.intersection(&[b_term]).unwrap().to_pattern()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parse,
    bench_to_automaton,
    bench_determinize,
    bench_minimize,
    bench_set_operations,
    bench_decision,
    bench_analyze,
    bench_to_regex,
    bench_generate_strings,
    bench_construction,
    bench_end_to_end,
);
criterion_main!(benches);
