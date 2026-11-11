//! Where to split a trimmed minimal DFA.
//!
//! Every analysis runs on a [`Graph`]: the states numbered densely in
//! `dfa.states()` order, which is also the order ties are broken in.

use bit_set::BitSet;

use super::*;

/// Most entry states of a frontier split's region.
const MAX_ENTRIES: usize = 4;
/// Most hubs of a hub split.
const MAX_HUBS: usize = 3;
/// Hubs are added until no strongly connected component is larger.
const HUB_TARGET: usize = 8;

/// No index: an unvisited state, an unreachable one, a missing dominator.
const NONE: u32 = u32::MAX;

/// The state other than the start, closest to it, that every accepting path
/// visits; `dfa`'s start must not accept.
pub(super) fn closest_dominator(dfa: &FastAutomaton) -> Option<State> {
    let graph = Graph::new(dfa);
    let start = graph.index_of(dfa.start_state())?;

    // The states reachable from the start, in reverse post-order: every one
    // after its parent in the depth-first search.
    let mut post_order = Vec::with_capacity(graph.len());
    let mut seen = BitSet::with_capacity(graph.len());
    seen.insert(start as usize);
    let mut calls = vec![(start, 0u32)];
    while let Some((state, next)) = calls.last_mut() {
        let state = *state;
        if let Some(&to_state) = graph.successors[state as usize].get(*next as usize) {
            *next += 1;
            if seen.insert(to_state as usize) {
                calls.push((to_state, 0));
            }
        } else {
            post_order.push(state);
            calls.pop();
        }
    }
    let order: Vec<u32> = post_order.into_iter().rev().collect();
    let mut position = vec![NONE; graph.len()];
    for (i, &state) in order.iter().enumerate() {
        position[state as usize] = i as u32;
    }
    let mut predecessors: Vec<Vec<u32>> = vec![vec![]; order.len()];
    for (i, &state) in order.iter().enumerate() {
        for &to_state in &graph.successors[state as usize] {
            predecessors[position[to_state as usize] as usize].push(i as u32);
        }
    }

    // Immediate dominators by position, iterated to the fixpoint
    // (Cooper, Harvey and Kennedy); the start, at position 0, is its own.
    let mut dominator = vec![NONE; order.len()];
    dominator[0] = 0;
    let intersect = |dominator: &[u32], mut a: u32, mut b: u32| {
        while a != b {
            while a > b {
                a = dominator[a as usize];
            }
            while b > a {
                b = dominator[b as usize];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..order.len() {
            let mut new_dominator = NONE;
            for &from in &predecessors[i] {
                if dominator[from as usize] == NONE {
                    continue;
                }
                new_dominator = if new_dominator == NONE {
                    from
                } else {
                    intersect(&dominator, from, new_dominator)
                };
            }
            if dominator[i] != new_dominator {
                dominator[i] = new_dominator;
                changed = true;
            }
        }
    }

    // The states every accepting path visits are the dominators of the
    // nearest common dominator of the reachable accepting states: a chain
    // from the start, ever farther from it.
    let common = (0..order.len() as u32)
        .filter(|&i| dfa.is_accepted(graph.states[order[i as usize] as usize]))
        .reduce(|a, b| intersect(&dominator, a, b));
    match common {
        Some(0) => None,
        Some(mut i) => {
            while dominator[i as usize] != 0 {
                i = dominator[i as usize];
            }
            Some(graph.states[order[i as usize] as usize])
        }
        // Nothing accepts: every state qualifies, the closest is chosen.
        None => {
            let distances = graph.distances_from(start);
            (0..graph.len())
                .filter(|&i| i != start as usize)
                .min_by_key(|&i| (distances[i], i))
                .map(|i| graph.states[i])
        }
    }
}

/// The weakly connected components of `dfa` without `removed`.
pub(super) fn components_without(dfa: &FastAutomaton, removed: State) -> Vec<IntSet<State>> {
    let graph = Graph::new(dfa);
    let predecessors = graph.predecessors();
    let mut seen = BitSet::with_capacity(graph.len());
    if let Some(removed) = graph.index_of(removed) {
        seen.insert(removed as usize);
    }
    let mut components = vec![];
    let mut stack = vec![];
    for root in 0..graph.len() {
        if !seen.insert(root) {
            continue;
        }
        let mut component = IntSet::default();
        component.insert(graph.states[root]);
        stack.push(root as u32);
        while let Some(state) = stack.pop() {
            let state = state as usize;
            for &neighbor in graph.successors[state].iter().chain(&predecessors[state]) {
                if seen.insert(neighbor as usize) {
                    component.insert(graph.states[neighbor as usize]);
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
    let graph = Graph::new(dfa);
    let number_of_states = graph.len();
    let start = graph.index_of(dfa.start_state())? as usize;

    // The states reachable from a state are those reachable from its strongly
    // connected component, and two components never reach the same states:
    // one region per component. Tarjan's algorithm finds a component after
    // every component it reaches.
    let mut component_of = vec![NONE; number_of_states];
    let mut regions: Vec<BitSet> = vec![];
    Tarjan::new(number_of_states).run(&graph, &graph.all(), |component| {
        let id = regions.len() as u32;
        let mut region = BitSet::with_capacity(number_of_states);
        for &state in component {
            component_of[state as usize] = id;
            region.insert(state as usize);
        }
        for &state in component {
            for &to_state in &graph.successors[state as usize] {
                let to_id = component_of[to_state as usize];
                if to_id != id {
                    region.union_with(&regions[to_id as usize]);
                }
            }
        }
        regions.push(region);
    });

    let predecessors = graph.predecessors();
    let mut seen = BitSet::with_capacity(regions.len());
    let mut best: Option<((usize, usize), usize, Vec<State>)> = None;
    // From each state in order, as the first state of its component.
    for &id in &component_of {
        let id = id as usize;
        if !seen.insert(id) {
            continue;
        }
        let region = &regions[id];
        if region.contains(start) {
            continue;
        }
        let size = region.count();
        let largest_piece = (number_of_states - size + 1).max(size);
        if largest_piece >= number_of_states
            || best
                .as_ref()
                .is_some_and(|((best_piece, _), ..)| largest_piece > *best_piece)
        {
            continue;
        }
        let mut entries = vec![];
        for state in region.iter() {
            if predecessors[state]
                .iter()
                .any(|&from_state| !region.contains(from_state as usize))
            {
                entries.push(graph.states[state]);
                if entries.len() > MAX_ENTRIES {
                    break;
                }
            }
        }
        let score = (largest_piece, entries.len());
        if entries.len() <= MAX_ENTRIES
            && best
                .as_ref()
                .is_none_or(|(best_score, ..)| score < *best_score)
        {
            best = Some((score, id, entries));
        }
    }
    best.map(|(_, id, entries)| Cut {
        region: regions[id]
            .iter()
            .map(|state| graph.states[state])
            .collect(),
        entries,
    })
}

/// `Σ |C|²` over the strongly connected components `C` with a cycle: what
/// every piece of a hub split must lower.
pub(super) fn cyclic_weight(dfa: &FastAutomaton) -> usize {
    let graph = Graph::new(dfa);
    let mut weight = 0;
    Tarjan::new(graph.len()).run(&graph, &graph.all(), |component| {
        let first = component[0];
        if component.len() > 1 || graph.successors[first as usize].contains(&first) {
            weight += component.len() * component.len();
        }
    });
    weight
}

/// A few states whose removal shatters the largest strongly connected
/// component, chosen greedily; none if it is small or does not shatter.
pub(super) fn choose_hubs(dfa: &FastAutomaton) -> Option<Vec<State>> {
    let graph = Graph::new(dfa);
    let mut tarjan = Tarjan::new(graph.len());
    let mut remaining = graph.all();
    let (mut component, mut others) = largest_component(&graph, &mut tarjan, &remaining);
    let initial = component.len();
    if initial < 3 {
        return None;
    }
    let mut hubs = vec![];
    let mut inside = BitSet::with_capacity(graph.len());
    while hubs.is_empty() || component.len() > HUB_TARGET && hubs.len() < MAX_HUBS {
        // The smallest remaining component, then the busiest state. Removing
        // a state splits only its own component: the others stay as they are.
        inside.make_empty();
        inside.extend(component.iter().map(|&state| state as usize));
        let (.., hub) = component
            .iter()
            .map(|&index| {
                inside.remove(index as usize);
                let mut largest = others;
                tarjan.run(&graph, &inside, |split| largest = largest.max(split.len()));
                inside.insert(index as usize);
                let state = graph.states[index as usize];
                let degree = dfa.in_degree(state) * dfa.out_degree(state);
                (largest, usize::MAX - degree, state, index)
            })
            .min()?;
        remaining.remove(hub as usize);
        hubs.push(graph.states[hub as usize]);
        (component, others) = largest_component(&graph, &mut tarjan, &remaining);
    }
    // Worth it only if the component shrinks beyond the hubs removed.
    (component.len() + hubs.len() < initial).then_some(hubs)
}

/// The largest strongly connected component of the states in `allowed`, the
/// first found on ties, and the size of the largest other one.
fn largest_component(graph: &Graph, tarjan: &mut Tarjan, allowed: &BitSet) -> (Vec<u32>, usize) {
    let mut largest: Vec<u32> = vec![];
    let mut others = 0;
    tarjan.run(graph, allowed, |component| {
        if component.len() > largest.len() {
            others = others.max(largest.len());
            largest = component.to_vec();
        } else {
            others = others.max(component.len());
        }
    });
    (largest, others)
}

/// The transitions of a [`FastAutomaton`] between dense indices: the
/// positions of its states in `dfa.states()`.
struct Graph {
    /// The state at each index, ascending.
    states: Vec<State>,
    /// In `direct_states` order.
    successors: Vec<Vec<u32>>,
}

impl Graph {
    fn new(dfa: &FastAutomaton) -> Self {
        let states: Vec<State> = dfa.states().collect();
        let mut index = vec![NONE; states.last().map_or(0, |state| state + 1)];
        for (i, &state) in states.iter().enumerate() {
            index[state] = i as u32;
        }
        let successors = states
            .iter()
            .map(|&state| {
                dfa.direct_states(state)
                    .map(|to_state| index[to_state])
                    .collect()
            })
            .collect();
        Graph { states, successors }
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn index_of(&self, state: State) -> Option<u32> {
        self.states
            .binary_search(&state)
            .ok()
            .map(|index| index as u32)
    }

    fn all(&self) -> BitSet {
        (0..self.len()).collect()
    }

    fn predecessors(&self) -> Vec<Vec<u32>> {
        let mut predecessors = vec![vec![]; self.len()];
        for (from_state, successors) in self.successors.iter().enumerate() {
            for &to_state in successors {
                predecessors[to_state as usize].push(from_state as u32);
            }
        }
        predecessors
    }

    /// BFS distances from `from_state`, [`NONE`] where it does not reach.
    fn distances_from(&self, from_state: u32) -> Vec<u32> {
        let mut distances = vec![NONE; self.len()];
        distances[from_state as usize] = 0;
        let mut queue = vec![from_state];
        let mut head = 0;
        while let Some(&state) = queue.get(head) {
            head += 1;
            let distance = distances[state as usize] + 1;
            for &to_state in &self.successors[state as usize] {
                if distances[to_state as usize] == NONE {
                    distances[to_state as usize] = distance;
                    queue.push(to_state);
                }
            }
        }
        distances
    }
}

/// Tarjan's strongly connected components, with buffers reused across runs.
struct Tarjan {
    /// When each state was first visited, [`NONE`] before.
    visited: Vec<u32>,
    low: Vec<u32>,
    on_stack: BitSet,
    stack: Vec<u32>,
    /// The depth-first search: each state and its next successor.
    calls: Vec<(u32, u32)>,
}

impl Tarjan {
    fn new(number_of_states: usize) -> Self {
        Tarjan {
            visited: vec![NONE; number_of_states],
            low: vec![0; number_of_states],
            on_stack: BitSet::with_capacity(number_of_states),
            stack: Vec::with_capacity(number_of_states),
            calls: Vec::with_capacity(number_of_states),
        }
    }

    /// Calls `emit` on each strongly connected component of the states in
    /// `allowed`, searched from roots in index order. A component comes after
    /// every component it reaches: the reverse of the order Kosaraju's
    /// algorithm finds them in on the same search.
    fn run(&mut self, graph: &Graph, allowed: &BitSet, mut emit: impl FnMut(&[u32])) {
        self.visited.fill(NONE);
        let mut time = 0;
        for root in allowed.iter() {
            if self.visited[root] != NONE {
                continue;
            }
            self.visit(root as u32, &mut time);
            while let Some((state, next)) = self.calls.last_mut() {
                let state = *state as usize;
                if let Some(&to_state) = graph.successors[state].get(*next as usize) {
                    *next += 1;
                    let to = to_state as usize;
                    if !allowed.contains(to) {
                        continue;
                    }
                    if self.visited[to] == NONE {
                        self.visit(to_state, &mut time);
                    } else if self.on_stack.contains(to) {
                        self.low[state] = self.low[state].min(self.visited[to]);
                    }
                } else {
                    self.calls.pop();
                    if let Some(&(parent, _)) = self.calls.last() {
                        let parent = parent as usize;
                        self.low[parent] = self.low[parent].min(self.low[state]);
                    }
                    if self.low[state] == self.visited[state] {
                        let first = self
                            .stack
                            .iter()
                            .rposition(|&s| s as usize == state)
                            .unwrap_or_default();
                        emit(&self.stack[first..]);
                        for &s in &self.stack[first..] {
                            self.on_stack.remove(s as usize);
                        }
                        self.stack.truncate(first);
                    }
                }
            }
        }
    }

    fn visit(&mut self, state: u32, time: &mut u32) {
        self.visited[state as usize] = *time;
        self.low[state as usize] = *time;
        *time += 1;
        self.on_stack.insert(state as usize);
        self.stack.push(state);
        self.calls.push((state, 0));
    }
}

/// The analyses as first written, which the fast ones must match exactly.
#[cfg(test)]
mod reference {
    use std::collections::VecDeque;

    use super::*;
    use crate::IntMap;

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

    pub(super) fn cyclic_weight(dfa: &FastAutomaton) -> usize {
        strongly_connected_components(dfa, &IntSet::default())
            .iter()
            .filter(|component| {
                component.len() > 1 || dfa.has_transition(component[0], component[0])
            })
            .map(|component| component.len() * component.len())
            .sum()
    }

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
        (component.len() + hubs.len() < initial).then_some(hubs)
    }

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

    fn strongly_connected_components(
        dfa: &FastAutomaton,
        removed: &IntSet<State>,
    ) -> Vec<Vec<State>> {
        let states: Vec<State> = dfa.states().filter(|s| !removed.contains(s)).collect();
        let successors = |state: State| -> Vec<State> {
            dfa.direct_states(state)
                .filter(|s| !removed.contains(s))
                .collect()
        };

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
}

#[cfg(test)]
mod tests {
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };

    use proptest::prelude::*;
    use regex_charclass::char::Char;

    use super::*;
    use crate::{CharRange, execution_profile::ExecutionProfileBuilder};

    fn assert_matches_reference(dfa: &FastAutomaton) {
        assert_eq!(
            closest_dominator(dfa),
            reference::closest_dominator(dfa),
            "{dfa}"
        );
        for state in dfa.states().chain([dfa.states().last().unwrap_or(0) + 1]) {
            assert_eq!(
                components_without(dfa, state),
                reference::components_without(dfa, state),
                "{dfa}"
            );
        }
        let cut = best_cut(dfa).map(|cut| (cut.region, cut.entries));
        let expected = reference::best_cut(dfa).map(|cut| (cut.region, cut.entries));
        assert_eq!(cut, expected, "{dfa}");
        assert_eq!(cyclic_weight(dfa), reference::cyclic_weight(dfa), "{dfa}");
        assert_eq!(choose_hubs(dfa), reference::choose_hubs(dfa), "{dfa}");
    }

    fn palette(index: usize) -> CharRange {
        let bounds = [('a', 'a'), ('b', 'b'), ('a', 'c'), ('x', 'z')];
        let (low, high) = bounds[index % bounds.len()];
        CharRange::new_from_range(Char::new(low)..=Char::new(high))
    }

    /// Up to 24 states, some of them removed afterwards so the states have
    /// gaps, with sparse to dense transitions.
    fn arb_automaton() -> impl Strategy<Value = FastAutomaton> {
        (2usize..25)
            .prop_flat_map(|number_of_states| {
                (
                    Just(number_of_states),
                    proptest::collection::vec(
                        (0..number_of_states, 0..number_of_states, 0usize..4),
                        1..number_of_states * 3,
                    ),
                    any::<u32>(),
                    any::<u32>(),
                )
            })
            .prop_map(|(number_of_states, edges, accept_mask, remove_mask)| {
                let mut automaton = FastAutomaton::new_empty();
                for _ in 1..number_of_states {
                    automaton.new_state();
                }
                for (from_state, to_state, range) in edges {
                    automaton
                        .add_transition_from_range(from_state, to_state, &palette(range))
                        .unwrap();
                }
                for state in 0..number_of_states {
                    if accept_mask & (1 << state) != 0 {
                        automaton.accept(state);
                    }
                }
                // One in four, never the start.
                for state in (1..number_of_states).rev() {
                    if remove_mask >> (state % 16 * 2) & 3 == 0 {
                        automaton.remove_state(state);
                    }
                }
                automaton
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn matches_the_reference_on_random_automata(automaton in arb_automaton()) {
            assert_matches_reference(&automaton);
            if let Ok(dfa) = minimal_dfa(&automaton) {
                assert_matches_reference(&dfa);
            }
        }
    }

    #[test]
    fn matches_the_reference_on_intersections() {
        for (a, b) in [
            ("(a|b)*a(a|b){3}", ".*"),
            ("((ab|ba)*c)*", ".*"),
            (".*abc.*", ".*def.*"),
            ("[a-e]*", ".*a.*b.*c.*"),
            ("(xa*b*)*", ".*ab.*ba.*"),
            ("((ab)*c(de)*f)*g", ".*"),
        ] {
            let a = RegularExpression::new(a).unwrap().to_automaton().unwrap();
            let b = RegularExpression::new(b).unwrap().to_automaton().unwrap();
            assert_matches_reference(&minimal_dfa(&a.intersection(&b).unwrap()).unwrap());
        }
    }

    /// Times the analyses against the reference ones on the DFAs
    /// `bench::large_dfas` builds (a copy, which needs no model weights):
    /// `cargo test --release --features neural-synthesis decompose::analysis::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn bench_analysis() {
        let dfas: Vec<FastAutomaton> = large_dfas(&crate::neural_synthesis::test_patterns(), 16)
            .into_iter()
            .flat_map(|(_, dfas)| dfas)
            .collect();
        for dfa in &dfas {
            assert_matches_reference(dfa);
        }
        let states: usize = dfas.iter().map(FastAutomaton::number_of_states).sum();
        println!("{} DFAs, {states} states", dfas.len());

        const ROUNDS: u32 = 20;
        let time = |analysis: &dyn Fn(&FastAutomaton)| {
            let start = Instant::now();
            for _ in 0..ROUNDS {
                for dfa in &dfas {
                    analysis(dfa);
                }
            }
            start.elapsed() / (ROUNDS * dfas.len() as u32)
        };
        let report = |name: &str, old: Duration, new: Duration| {
            println!(
                "{name:<18} {old:>10.2?} -> {new:>10.2?} per DFA ({:.1}x)",
                old.as_secs_f64() / new.as_secs_f64()
            );
        };
        report(
            "closest_dominator",
            time(&|dfa| {
                black_box(reference::closest_dominator(dfa));
            }),
            time(&|dfa| {
                black_box(closest_dominator(dfa));
            }),
        );
        report(
            "components_without",
            time(&|dfa| {
                black_box(reference::components_without(dfa, dfa.start_state()));
            }),
            time(&|dfa| {
                black_box(components_without(dfa, dfa.start_state()));
            }),
        );
        report(
            "best_cut",
            time(&|dfa| {
                black_box(reference::best_cut(dfa));
            }),
            time(&|dfa| {
                black_box(best_cut(dfa));
            }),
        );
        report(
            "cyclic_weight",
            time(&|dfa| {
                black_box(reference::cyclic_weight(dfa));
            }),
            time(&|dfa| {
                black_box(cyclic_weight(dfa));
            }),
        );
        report(
            "choose_hubs",
            time(&|dfa| {
                black_box(reference::choose_hubs(dfa));
            }),
            time(&|dfa| {
                black_box(choose_hubs(dfa));
            }),
        );
    }

    /// A set operation on three regexes.
    type Operation = fn(&[&RegularExpression; 3]) -> Result<FastAutomaton, EngineError>;

    /// As in `bench`.
    const OPERATIONS: [(&str, Operation); 5] = [
        ("concatenation", |[a, b, c]| {
            a.concat(b, true).concat(c, true).to_automaton()
        }),
        ("star", |[a, b, c]| {
            a.union(b).repeat(0, None).concat(c, true).to_automaton()
        }),
        ("difference", |[a, b, c]| {
            a.concat(b, true)
                .to_automaton()?
                .difference(&containing(c)?)
        }),
        ("intersection", |[a, b, c]| {
            a.union(b)
                .repeat(0, None)
                .to_automaton()?
                .intersection(&containing(c)?)
        }),
        ("union", |[a, b, c]| a.union(b).union(c).to_automaton()),
    ];

    fn containing(regex: &RegularExpression) -> Result<FastAutomaton, EngineError> {
        let total = RegularExpression::new_total();
        total
            .concat(regex, true)
            .concat(&total, true)
            .to_automaton()
    }

    /// As in `bench`.
    fn large_dfas(
        atoms: &[RegularExpression],
        max_states: usize,
    ) -> Vec<(&'static str, Vec<FastAutomaton>)> {
        const PER_OPERATION: usize = 10;
        let profile = ExecutionProfileBuilder::new()
            .execution_timeout(300)
            .max_number_of_states(5000)
            .build();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move |bound: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound as u64) as usize
        };
        OPERATIONS
            .iter()
            .map(|(name, operation)| {
                let mut dfas = vec![];
                for _ in 0..1000 {
                    if dfas.len() == PER_OPERATION {
                        break;
                    }
                    let operands = [0, 1, 2].map(|_| &atoms[random(atoms.len())]);
                    let dfa = profile.run(|| {
                        let dfa = minimal_dfa(&operation(&operands)?)?;
                        let _ = dfa.to_regex()?;
                        Ok::<_, EngineError>(dfa)
                    });
                    if let Ok(dfa) = dfa
                        && (max_states + 1..=80).contains(&dfa.number_of_states())
                    {
                        dfas.push(dfa);
                    }
                }
                (*name, dfas)
            })
            .collect()
    }
}
