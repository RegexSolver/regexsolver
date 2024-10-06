use ahash::AHashSet;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use regexsolver::{fast_automaton::FastAutomaton, regex::RegularExpression};

fn parse_regex(regex: &str) -> RegularExpression {
    RegularExpression::new(regex).unwrap()
}

fn to_regex(automaton: &FastAutomaton) -> RegularExpression {
    automaton.to_regex().unwrap()
}

fn determinize(automaton: &FastAutomaton) -> FastAutomaton {
    automaton.determinize().unwrap()
}

fn intersection(automaton_1: &FastAutomaton, automaton_2: &FastAutomaton) -> FastAutomaton {
    automaton_1.intersection(automaton_2).unwrap()
}

fn generate_strings(automaton: &FastAutomaton) -> AHashSet<String> {
    automaton.generate_strings(2000).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    {
        c.bench_function("parse_regex", |b| {
            b.iter(|| parse_regex(black_box("a(bcfe|bcdg|mkv)*(abc){2,3}(abc){2}")))
        });
    }

    {
        let input_regex = RegularExpression::new("a(bcfe|bcdg|mkv)*(abc){2,3}").unwrap();
        let input_automaton = input_regex.to_automaton().unwrap();

        c.bench_function("to_regex", |b| {
            b.iter(|| to_regex(black_box(&input_automaton)))
        });
    }

    {
        let input_regex = RegularExpression::new(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
        )
        .unwrap();
        let input_automaton = input_regex.to_automaton().unwrap();

        c.bench_function("determinize", |b| {
            b.iter(|| determinize(black_box(&input_automaton)))
        });
    }

    /*{
        let input_regex = RegularExpression::new("((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,5}").unwrap();
        let input_automaton = input_regex.to_automaton().unwrap();

        c.bench_function("test_determinize", |b| {
            b.iter(|| determinize(black_box(&input_automaton)))
        });
    }*/

    {
        let automaton1 = RegularExpression::new("a(bcfe|bcdg|mkv)*(abc){1,3}")
            .unwrap()
            .to_automaton().unwrap();
        let automaton2 = RegularExpression::new("a(bcfe|mkv|opr)*(abc){2,4}")
            .unwrap()
            .to_automaton().unwrap();

        c.bench_function("intersection", |b| {
            b.iter(|| intersection(black_box(&automaton1), black_box(&automaton2)))
        });
    }

    {
        let automaton = RegularExpression::new("a(bcfe|bcdg|mkv)*(abc){1,3}")
            .unwrap()
            .to_automaton().unwrap();

        c.bench_function("generate_strings", |b| {
            b.iter(|| generate_strings(black_box(&automaton)))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
