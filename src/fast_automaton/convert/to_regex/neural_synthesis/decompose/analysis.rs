//! Where to split a trimmed minimal DFA.

use std::collections::VecDeque;

use super::*;
use crate::IntMap;

/// Most entry states of a frontier split's region.
const MAX_ENTRIES: usize = 4;
/// Most hubs of a hub split.
const MAX_HUBS: usize = 3;
/// Hubs are added until no strongly connected component is larger.
const HUB_TARGET: usize = 8;

/// The state other than the start, closest to it, that every accepting path
/// visits; `dfa`'s start must not accept.
pub(super) fn closest_dominator(dfa: &FastAutomaton) -> Option<State> {
    let start_state = dfa.start_state();
    let distances = distances_from(dfa, start_state, None);
    let mut candidates: Vec<State> = dfa.states().filter(|s| *s != start_state).collect();
    candidates.sort_by_key(|state| distances.get(state).copied().unwrap_or(usize::MAX));
    candidates.into_iter().find(|&state| {
        distances_from(dfa, start_state, Some(state))
            .keys()
            .all(|reached| !dfa.is_accepted(*reached))
    })
}

/// The weakly connected components of `dfa` without `removed`.
pub(super) fn components_without(dfa: &FastAutomaton, removed: State) -> Vec<IntSet<State>> {
    let mut neighbors: IntMap<State, Vec<State>> = IntMap::default();
    for from_state in dfa.states().filter(|s| *s != removed) {
        for to_state in dfa.direct_states(from_state).filter(|s| *s != removed) {
            neighbors.entry(from_state).or_default().push(to_state);
            neighbors.entry(to_state).or_default().push(from_state);
        }
    }
    let mut seen = IntSet::default();
    let mut components = vec![];
    for root in dfa.states().filter(|s| *s != removed) {
        if !seen.insert(root) {
            continue;
        }
        let mut component = IntSet::default();
        component.insert(root);
        let mut stack = vec![root];
        while let Some(state) = stack.pop() {
            for &neighbor in neighbors.get(&state).into_iter().flatten() {
                if seen.insert(neighbor) {
                    component.insert(neighbor);
                    stack.push(neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

/// Where a frontier split cuts: a region closed under successors, without
/// the start, and the states it is entered at.
pub(super) struct Cut {
    pub(super) region: IntSet<State>,
    pub(super) entries: Vec<State>,
}

/// The frontier split whose largest piece is the smallest, among the regions
/// of the states reachable from one state.
pub(super) fn best_cut(dfa: &FastAutomaton) -> Option<Cut> {
    let start_state = dfa.start_state();
    let number_of_states = dfa.number_of_states();
    let mut seen: Vec<IntSet<State>> = vec![];
    let mut best: Option<((usize, usize), Cut)> = None;
    for root in dfa.states() {
        let region: IntSet<State> = distances_from(dfa, root, None).into_keys().collect();
        if region.contains(&start_state) || seen.contains(&region) {
            continue;
        }
        seen.push(region.clone());
        let mut entries: Vec<State> = dfa
            .states()
            .filter(|state| !region.contains(state))
            .flat_map(|from_state| dfa.direct_states(from_state))
            .filter(|to_state| region.contains(to_state))
            .collect();
        entries.sort_unstable();
        entries.dedup();
        let largest_piece = (number_of_states - region.len() + 1).max(region.len());
        let score = (largest_piece, entries.len());
        if entries.len() <= MAX_ENTRIES
            && largest_piece < number_of_states
            && best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, Cut { region, entries }));
        }
    }
    best.map(|(_, cut)| cut)
}

/// `Σ |C|²` over the strongly connected components `C` with a cycle: what
/// every piece of a hub split must lower.
pub(super) fn cyclic_weight(dfa: &FastAutomaton) -> usize {
    strongly_connected_components(dfa, &IntSet::default())
        .iter()
        .filter(|component| component.len() > 1 || dfa.has_transition(component[0], component[0]))
        .map(|component| component.len() * component.len())
        .sum()
}

/// A few states whose removal shatters the largest strongly connected
/// component, chosen greedily; none if it is small or does not shatter.
pub(super) fn choose_hubs(dfa: &FastAutomaton) -> Option<Vec<State>> {
    let largest = |removed: &IntSet<State>| {
        strongly_connected_components(dfa, removed)
            .into_iter()
            .max_by_key(Vec::len)
            .unwrap_or_default()
    };
    let mut removed = IntSet::default();
    let mut component = largest(&removed);
    let initial = component.len();
    if initial < 3 {
        return None;
    }
    let mut hubs = vec![];
    while hubs.is_empty() || component.len() > HUB_TARGET && hubs.len() < MAX_HUBS {
        // The smallest remaining component, then the busiest state.
        let hub = component.iter().copied().min_by_key(|&state| {
            let mut without = removed.clone();
            without.insert(state);
            let degree = dfa.in_degree(state) * dfa.out_degree(state);
            (largest(&without).len(), usize::MAX - degree, state)
        })?;
        removed.insert(hub);
        hubs.push(hub);
        component = largest(&removed);
    }
    // Worth it only if the component shrinks beyond the hubs removed.
    (component.len() + hubs.len() < initial).then_some(hubs)
}

/// BFS distances from `from_state`, never entering `avoided`.
fn distances_from(
    dfa: &FastAutomaton,
    from_state: State,
    avoided: Option<State>,
) -> IntMap<State, usize> {
    let mut distances = IntMap::default();
    distances.insert(from_state, 0);
    let mut queue = VecDeque::from([from_state]);
    while let Some(state) = queue.pop_front() {
        let distance = distances[&state] + 1;
        for to_state in dfa.direct_states(state) {
            if Some(to_state) != avoided && !distances.contains_key(&to_state) {
                distances.insert(to_state, distance);
                queue.push_back(to_state);
            }
        }
    }
    distances
}

/// The strongly connected components of `dfa` without `removed`
/// (Kosaraju's algorithm).
fn strongly_connected_components(dfa: &FastAutomaton, removed: &IntSet<State>) -> Vec<Vec<State>> {
    let states: Vec<State> = dfa.states().filter(|s| !removed.contains(s)).collect();
    let successors = |state: State| -> Vec<State> {
        dfa.direct_states(state)
            .filter(|s| !removed.contains(s))
            .collect()
    };

    // States in post-order of a DFS of the transitions.
    let mut finished = Vec::with_capacity(states.len());
    let mut seen = IntSet::default();
    for &root in &states {
        if !seen.insert(root) {
            continue;
        }
        let mut stack = vec![(root, successors(root), 0)];
        while let Some((state, next_states, next)) = stack.last_mut() {
            if let Some(&to_state) = next_states.get(*next) {
                *next += 1;
                if seen.insert(to_state) {
                    stack.push((to_state, successors(to_state), 0));
                }
            } else {
                finished.push(*state);
                stack.pop();
            }
        }
    }

    // Then the reversed transitions, in reverse post-order.
    let mut predecessors: IntMap<State, Vec<State>> = IntMap::default();
    for &from_state in &states {
        for to_state in successors(from_state) {
            predecessors.entry(to_state).or_default().push(from_state);
        }
    }
    let mut assigned = IntSet::default();
    let mut components = vec![];
    for &root in finished.iter().rev() {
        if !assigned.insert(root) {
            continue;
        }
        let mut component = vec![root];
        let mut stack = vec![root];
        while let Some(state) = stack.pop() {
            for &from_state in predecessors.get(&state).into_iter().flatten() {
                if assigned.insert(from_state) {
                    component.push(from_state);
                    stack.push(from_state);
                }
            }
        }
        components.push(component);
    }
    components
}
