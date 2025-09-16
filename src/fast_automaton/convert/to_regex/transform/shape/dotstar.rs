use nohash_hasher::IntSet;

use crate::fast_automaton::{FastAutomaton, State, condition::Condition};

pub(crate) fn dot_star(automaton: &FastAutomaton) -> FastAutomaton {
    let components = identify_and_apply_components(automaton);

    let mut automaton = automaton.clone();
    for component in components {
        dot_star_component(&mut automaton, &component);
    }

    automaton
}

fn dot_star_component(automaton: &mut FastAutomaton, component: &IntSet<State>) {
    let mut start_state = if component.contains(&automaton.start_state) {
        Some(automaton.start_state)
    } else {
        None
    };
    for &state in component {
        for (from_state, _) in automaton.transitions_to_vec(state) {
            if !component.contains(&from_state) {
                if start_state.is_none() {
                    start_state = Some(state);
                } else {
                    // Only one start state possible
                    return;
                }
            }
        }
    }

    if start_state.is_none() {
        // Only one start state possible
        return;
    }
    let start_state = start_state.unwrap();

    let mut first_hop = automaton
        .direct_states(&start_state)
        .filter(|&s| s != start_state)
        .collect::<IntSet<_>>();
    let mut states_to_remove = vec![];

    for state in &first_hop {
        let transitions = automaton.transitions_to_vec(*state);
        if !transitions.iter().all(|(_, c)| *c == transitions[0].1) {
            // Some condition(s) to a given first hop state are not the same.
            return;
        }

        if transitions.len() != component.len() {
            states_to_remove.push(*state);
        }
    }

    states_to_remove.iter().for_each(|s| {
        first_hop.remove(s);
    });

    let mut out_condition = None;
    for &state in component {
        let mut has_transition_to_start_state = false;

        let mut this_condition = Condition::empty(automaton.get_spanning_set());
        for (condition, &to_state) in automaton.transitions_from(state) {
            if to_state == start_state {
                has_transition_to_start_state = true;
            }

            this_condition = this_condition.union(condition);
        }
        if !has_transition_to_start_state {
            // Some state(s) do not have transition to the start state.
            return;
        }

        if let Some(condition) = &out_condition {
            if &this_condition != condition {
                // The union of outcoming condition for some states are not identical
                return;
            }
        } else {
            out_condition = Some(this_condition);
        }
    }

    automaton.add_transition(start_state, start_state, &out_condition.unwrap());
    for &state in component {
        for to_state in automaton.direct_states_vec(&state) {
            if !component.contains(&to_state) {
                continue;
            }

            if state != start_state && (to_state == start_state || first_hop.contains(&to_state)) {
                automaton.remove_transition(state, to_state);
            }
        }
    }
    for state in states_to_remove {
        automaton.remove_state(state);
    }
}

pub fn identify_and_apply_components(automaton: &FastAutomaton) -> Vec<IntSet<State>> {
    let mut index = 0;
    let mut stack = Vec::new();
    let mut indices = vec![-1; automaton.transitions.len()];
    let mut lowlink = vec![-1; automaton.transitions.len()];
    let mut on_stack = vec![false; automaton.transitions.len()];
    let mut scc = Vec::new();

    for state in automaton.states() {
        if indices[state] == -1 {
            strongconnect(
                automaton,
                state,
                &mut index,
                &mut stack,
                &mut indices,
                &mut lowlink,
                &mut on_stack,
                &mut scc,
            );
        }
    }

    scc.into_iter()
        .filter(|states| states.len() != 1)
        .collect::<Vec<_>>()
}

#[allow(clippy::too_many_arguments)]
fn strongconnect(
    automaton: &FastAutomaton,
    v: usize,
    index: &mut usize,
    stack: &mut Vec<usize>,
    indices: &mut Vec<i32>,
    lowlink: &mut Vec<i32>,
    on_stack: &mut Vec<bool>,
    scc: &mut Vec<IntSet<usize>>,
) {
    indices[v] = *index as i32;
    lowlink[v] = *index as i32;
    *index += 1;
    stack.push(v);
    on_stack[v] = true;

    for w in automaton.direct_states(&v) {
        if indices[w] == -1 {
            strongconnect(automaton, w, index, stack, indices, lowlink, on_stack, scc);
            lowlink[v] = lowlink[v].min(lowlink[w]);
        } else if on_stack[w] {
            lowlink[v] = lowlink[v].min(indices[w]);
        }
    }

    if lowlink[v] == indices[v] {
        let mut component = IntSet::default();
        while let Some(w) = stack.pop() {
            on_stack[w] = false;
            component.insert(w);
            if w == v {
                break;
            }
        }
        scc.push(component);
    }
}
