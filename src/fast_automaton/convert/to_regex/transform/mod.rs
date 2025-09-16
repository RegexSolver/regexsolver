use crate::fast_automaton::{
    FastAutomaton, convert::to_regex::transform::shape::dotstar::dot_star,
};

mod shape;

const TRANSFORM_FUNCTION: &[fn(&FastAutomaton) -> FastAutomaton] = &[dot_star];

pub fn transform(automaton: &FastAutomaton) -> FastAutomaton {
    let mut automaton = automaton.clone();
    for transform in TRANSFORM_FUNCTION {
        automaton = transform(&automaton);
    }

    automaton
}
