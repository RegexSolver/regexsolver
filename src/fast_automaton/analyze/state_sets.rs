//! What [`equivalent`](FastAutomaton::equivalent),
//! [`subset`](FastAutomaton::subset),
//! [`cardinality`](FastAutomaton::cardinality) and
//! [`difference`](FastAutomaton::difference) share: each walks the *sets* of
//! states a word can reach, the way subset construction would, without
//! determinizing an operand first.
//!
//! * [`Operand`] is one automaton prepared for that walk.
//! * [`SetTrie`] stores the sets met so far, and answers "which stored sets
//!   are contained in this one", the inner loop of the two comparisons.
//! * [`Marks`] is a set with membership in constant time, which such a
//!   question is asked about.

use condition::converter::{ConditionConverter, SharedSpanningSet};

use crate::error::EngineError;

use super::*;

/// A state of an operand. Sets of states are sorted slices of ids, so their
/// cost follows the size of the set rather than the size of the automaton.
pub(crate) type Id = u32;

/// One operand of a walk: its transitions restricted to its live states, and
/// projected on the spanning set shared by both operands when there are two.
///
/// Dropping the states that cannot reach an accept state makes a set denote
/// the empty language iff it is empty, so "one side is stuck while the other
/// is not" is an immediate counterexample instead of a sub-search.
pub(crate) struct Operand {
    /// State `s` has the id `offset + s`, which lets two operands share one
    /// id space.
    offset: usize,
    /// Outgoing transitions by state, targets as ids.
    transitions: Vec<Vec<(Condition, Id)>>,
    accept: Vec<bool>,
    /// The start state, unless it is not live (the language is then empty).
    pub(crate) start: Option<Id>,
}

impl Operand {
    /// Prepares both operands over their merged spanning set, and returns
    /// them with its number of bases (the letters of the walk).
    pub(crate) fn pair(
        left: &FastAutomaton,
        left_offset: usize,
        right: &FastAutomaton,
        right_offset: usize,
    ) -> Result<(Operand, Operand, usize), EngineError> {
        let shared = SharedSpanningSet::new(&left.spanning_set, &right.spanning_set);
        let (left_converter, right_converter) = shared.converters()?;
        Ok((
            Operand::new(left, left_offset, left_converter.as_ref())?,
            Operand::new(right, right_offset, right_converter.as_ref())?,
            shared.spanning_set().spanning_ranges_with_rest_len(),
        ))
    }

    /// Prepares one operand; `converter` projects its conditions on another
    /// spanning set than its own.
    pub(crate) fn new(
        automaton: &FastAutomaton,
        offset: usize,
        converter: Option<&ConditionConverter>,
    ) -> Result<Self, EngineError> {
        let states = automaton.transitions.len();
        if Id::try_from(offset + states).is_err() {
            return Err(EngineError::AutomatonHasTooManyStates);
        }
        let mut operand = Operand {
            offset,
            transitions: vec![Vec::new(); states],
            accept: vec![false; states],
            start: None,
        };
        let live = automaton.live_states();
        for &state in live.iter().filter(|&&state| automaton.has_state(state)) {
            operand.accept[state] = automaton.is_accepted(state);
            for (condition, to_state) in automaton.transitions_from(state) {
                if condition.is_empty() || !live.contains(to_state) {
                    continue;
                }
                let condition = match converter {
                    Some(converter) => converter.convert(condition)?,
                    None => condition.clone(),
                };
                operand.transitions[state].push((condition, (offset + to_state) as Id));
            }
        }
        if live.contains(&automaton.start_state) {
            operand.start = Some((offset + automaton.start_state) as Id);
        }
        Ok(operand)
    }

    #[inline]
    pub(crate) fn transitions_from(&self, id: Id) -> &[(Condition, Id)] {
        &self.transitions[id as usize - self.offset]
    }

    #[inline]
    pub(crate) fn accepts(&self, id: Id) -> bool {
        self.accept[id as usize - self.offset]
    }

    #[inline]
    pub(crate) fn is_accepting(&self, set: &[Id]) -> bool {
        set.iter().any(|&id| self.accepts(id))
    }

    /// Writes the successors of `set` under each base into `out`, one sorted
    /// set per base.
    pub(crate) fn successors(&self, set: &[Id], out: &mut [Vec<Id>]) {
        out.iter_mut().for_each(Vec::clear);
        for &id in set {
            for (condition, to) in self.transitions_from(id) {
                for base in condition.iter_set_bits() {
                    out[base].push(*to);
                }
            }
        }
        for successors in out.iter_mut().filter(|successors| successors.len() > 1) {
            successors.sort_unstable();
            successors.dedup();
        }
    }
}

/// Sorted sets stored as paths, each with values attached. The stored sets
/// contained in a queried set are found by descending only into the children
/// whose state the query holds, instead of scanning every stored set.
///
/// One trie can hold several independent collections, each below its own
/// root node.
pub(super) struct SetTrie<T> {
    nodes: Vec<Node<T>>,
}

pub(super) struct Node<T> {
    /// `(state, child)`, in insertion order.
    children: Vec<(Id, usize)>,
    /// `children` by state, once there are more than [`WIDE`] of them.
    index: Option<AHashMap<Id, usize>>,
    /// The values of the sets ending at this node.
    pub(super) values: Vec<T>,
}

/// The number of children past which a node is looked up by state instead of
/// scanned. A deterministic operand only yields singletons, which all hang
/// off the same node: scanning it would make every lookup linear in the
/// number of stored sets.
pub(super) const WIDE: usize = 16;

impl<T> Node<T> {
    /// The children, when they are few enough to be scanned; a wide node has
    /// to be queried with [`child`](Self::child).
    #[inline]
    pub(super) fn few_children(&self) -> Option<&[(Id, usize)]> {
        self.index.is_none().then_some(&self.children[..])
    }

    #[inline]
    pub(super) fn child(&self, id: Id) -> Option<usize> {
        match &self.index {
            Some(index) => index.get(&id).copied(),
            None => self
                .children
                .iter()
                .find(|(state, _)| *state == id)
                .map(|(_, child)| *child),
        }
    }
}

impl<T> SetTrie<T> {
    pub(super) fn new() -> Self {
        SetTrie { nodes: Vec::new() }
    }

    /// Adds a node, to be used as the root of a collection.
    pub(super) fn new_root(&mut self) -> usize {
        self.nodes.push(Node {
            children: Vec::new(),
            index: None,
            values: Vec::new(),
        });
        self.nodes.len() - 1
    }

    #[inline]
    pub(super) fn node(&self, node: usize) -> &Node<T> {
        &self.nodes[node]
    }

    /// Stores `set` below `root` and attaches `value` to it.
    pub(super) fn insert(&mut self, root: usize, set: &[Id], value: T) {
        let mut node = root;
        for &id in set {
            node = match self.nodes[node].child(id) {
                Some(child) => child,
                None => {
                    let child = self.new_root();
                    let parent = &mut self.nodes[node];
                    parent.children.push((id, child));
                    match &mut parent.index {
                        Some(index) => {
                            index.insert(id, child);
                        }
                        None if parent.children.len() > WIDE => {
                            parent.index = Some(parent.children.iter().copied().collect());
                        }
                        None => {}
                    }
                    child
                }
            };
        }
        self.nodes[node].values.push(value);
    }

    /// The values attached to exactly `set` below `root`.
    pub(super) fn get(&self, root: usize, set: &[Id]) -> &[T] {
        let mut node = &self.nodes[root];
        for &id in set {
            match node.child(id) {
                Some(child) => node = &self.nodes[child],
                None => return &[],
            }
        }
        &node.values
    }

    /// Returns `true` if a set stored below `root` is contained in `set`.
    pub(super) fn has_subset_of(&self, root: usize, set: &Marks) -> bool {
        let node = &self.nodes[root];
        if !node.values.is_empty() {
            return true;
        }
        match node.few_children() {
            Some(children) => children
                .iter()
                .any(|&(id, child)| set.contains(id) && self.has_subset_of(child, set)),
            None => set.members().iter().any(|&id| {
                node.child(id)
                    .is_some_and(|child| self.has_subset_of(child, set))
            }),
        }
    }
}

/// A set of states with membership in constant time, and reset in time
/// proportional to its size rather than to the number of states.
pub(super) struct Marks {
    marked: Vec<bool>,
    members: Vec<Id>,
}

impl Marks {
    pub(super) fn new(states: usize) -> Self {
        Marks {
            marked: vec![false; states],
            members: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn contains(&self, id: Id) -> bool {
        self.marked[id as usize]
    }

    /// The states of the set, in insertion order.
    #[inline]
    pub(super) fn members(&self) -> &[Id] {
        &self.members
    }

    /// Returns `true` if `id` was not in the set yet.
    #[inline]
    pub(super) fn insert(&mut self, id: Id) -> bool {
        let marked = &mut self.marked[id as usize];
        if *marked {
            return false;
        }
        *marked = true;
        self.members.push(id);
        true
    }

    pub(super) fn clear(&mut self) {
        for id in self.members.drain(..) {
            self.marked[id as usize] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_subset_of_stays_below_its_root() {
        let wide = 4 * WIDE as Id;
        let mut trie = SetTrie::new();
        let (first, second) = (trie.new_root(), trie.new_root());
        let mut marks = Marks::new(2 * wide as usize);
        let mut has_subset_of = |trie: &SetTrie<()>, root: usize, set: &[Id]| {
            set.iter().for_each(|&id| assert!(marks.insert(id)));
            let found = trie.has_subset_of(root, &marks);
            marks.clear();
            found
        };

        trie.insert(first, &[1, 3], ());
        assert!(has_subset_of(&trie, first, &[1, 3]));
        assert!(has_subset_of(&trie, first, &[1, 2, 3, 4]));
        assert!(!has_subset_of(&trie, first, &[1, 2]));
        assert!(!has_subset_of(&trie, first, &[3]));
        assert!(!has_subset_of(&trie, second, &[1, 3]));

        // Enough children for the root to be looked up instead of scanned.
        for id in 0..wide {
            trie.insert(second, &[id, wide + id], ());
        }
        assert!(trie.node(second).few_children().is_none());
        assert!(has_subset_of(&trie, second, &[5, wide + 5]));
        assert!(has_subset_of(&trie, second, &[2, 5, wide + 5, wide + 6]));
        assert!(!has_subset_of(&trie, second, &[5, wide + 6]));
        assert!(!has_subset_of(&trie, first, &[5, wide + 5]));
    }

    #[test]
    fn get_returns_the_values_of_the_exact_set() {
        let mut trie = SetTrie::new();
        let root = trie.new_root();
        trie.insert(root, &[1, 2], 'a');
        trie.insert(root, &[1, 2], 'b');
        trie.insert(root, &[1], 'c');
        assert_eq!(trie.get(root, &[1, 2]), ['a', 'b']);
        assert_eq!(trie.get(root, &[1]), ['c']);
        assert!(trie.get(root, &[2]).is_empty());
        assert!(trie.get(root, &[1, 2, 3]).is_empty());
    }
}
