//! The pieces of a split: sub-automata of a minimal DFA, each minimized.

use super::*;

/// The strings leading from `state` back to it, without visiting it in
/// between.
pub(super) fn first_return(
    dfa: &FastAutomaton,
    state: State,
) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    let start_state = copy_outgoing(&mut piece, state);
    remove_outgoing(&mut piece, state);
    accept_only(&mut piece, state);
    start_at(&mut piece, start_state);
    minimal_dfa(&piece)
}

/// The strings accepted from `state`, the start, without returning to it.
pub(super) fn without_return(
    dfa: &FastAutomaton,
    state: State,
) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    for (from_state, _) in dfa.transitions_to_vec(state) {
        piece.remove_transition(from_state, state);
    }
    minimal_dfa(&piece)
}

/// The strings leading from the start to its first visit of `state`.
pub(super) fn up_to(dfa: &FastAutomaton, state: State) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    remove_outgoing(&mut piece, state);
    accept_only(&mut piece, state);
    minimal_dfa(&piece)
}

/// The strings accepted from `state`.
pub(super) fn language_of(dfa: &FastAutomaton, state: State) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    start_at(&mut piece, state);
    minimal_dfa(&piece)
}

/// `dfa` with only the transitions from the start into `component`, and the
/// start not accepting.
pub(super) fn start_into(
    dfa: &FastAutomaton,
    component: &IntSet<State>,
) -> Result<FastAutomaton, EngineError> {
    let start_state = dfa.start_state();
    let mut piece = dfa.clone();
    for to_state in dfa.direct_states(start_state) {
        if !component.contains(&to_state) {
            piece.remove_transition(start_state, to_state);
        }
    }
    piece.unaccept(start_state);
    minimal_dfa(&piece)
}

/// The strings accepted without entering `region`.
pub(super) fn outside(
    dfa: &FastAutomaton,
    region: &IntSet<State>,
) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    remove_entries(dfa, &mut piece, region, None);
    minimal_dfa(&piece)
}

/// The strings leading from the start into `region`, entering it first at
/// `entry`.
pub(super) fn entering_at(
    dfa: &FastAutomaton,
    region: &IntSet<State>,
    entry: State,
) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    remove_entries(dfa, &mut piece, region, Some(entry));
    remove_outgoing(&mut piece, entry);
    accept_only(&mut piece, entry);
    minimal_dfa(&piece)
}

/// The paths from `source` to `target` (acceptance if `None`) that visit no
/// hub after leaving `source`, except `target` at the end.
pub(super) fn between_hubs(
    dfa: &FastAutomaton,
    source: State,
    hubs: &[State],
    target: Option<State>,
) -> Result<FastAutomaton, EngineError> {
    let mut piece = dfa.clone();
    let start_state = copy_outgoing(&mut piece, source);
    for &hub in hubs {
        remove_outgoing(&mut piece, hub);
    }
    match target {
        Some(target) => accept_only(&mut piece, target),
        None => {
            for &hub in hubs {
                piece.unaccept(hub);
            }
            if dfa.is_accepted(source) {
                piece.accept(start_state);
            }
        }
    }
    start_at(&mut piece, start_state);
    minimal_dfa(&piece)
}

/// A new state with the same outgoing transitions as `state`.
fn copy_outgoing(automaton: &mut FastAutomaton, state: State) -> State {
    let copy = automaton.new_state();
    for (condition, to_state) in automaton.transitions_from_vec(state) {
        automaton.add_transition(copy, to_state, &condition);
    }
    copy
}

/// Removes the transitions of `piece` (a copy of `dfa`) entering `region`
/// from outside it, but those to `kept`.
fn remove_entries(
    dfa: &FastAutomaton,
    piece: &mut FastAutomaton,
    region: &IntSet<State>,
    kept: Option<State>,
) {
    for from_state in dfa.states().filter(|state| !region.contains(state)) {
        for to_state in dfa.direct_states(from_state) {
            if region.contains(&to_state) && Some(to_state) != kept {
                piece.remove_transition(from_state, to_state);
            }
        }
    }
}

fn remove_outgoing(automaton: &mut FastAutomaton, state: State) {
    for to_state in automaton.direct_states_vec(state) {
        automaton.remove_transition(state, to_state);
    }
}

fn accept_only(automaton: &mut FastAutomaton, state: State) {
    for accept_state in automaton.accept_states().clone() {
        automaton.unaccept(accept_state);
    }
    automaton.accept(state);
}

fn start_at(automaton: &mut FastAutomaton, state: State) {
    automaton.start_state = state;
    automaton.minimal = false;
}
