//! The model's input: a canonical DFA as a graph (`config.json`, `inputs`).

use super::canonical::{CanonicalDfa, DEAD};

const NODE_STATE: u32 = 0;
const NODE_START: u32 = 1;
const NODE_SINK: u32 = 2;

/// The complete DFA as a graph: a sink is appended (last node) when some
/// transition is missing; one node per state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Graph {
    pub nodes: usize,
    /// 0 state, 1 start (node 0), 2 sink.
    pub node_type: Vec<u32>,
    pub is_accept: Vec<u32>,
    /// Shortest word length from the start, clipped to `max_depth`.
    pub fwd_depth: Vec<u32>,
    /// Shortest word length to an accepting state, clipped to `max_depth`;
    /// `max_depth + 1` if none.
    pub bwd_depth: Vec<u32>,
    /// Shortest word length to a rejecting state, clipped to `max_depth`;
    /// `max_depth + 1` if none.
    pub to_reject: Vec<u32>,
    /// One edge per state and base, then the same edges reversed.
    pub edge_src: Vec<u32>,
    pub edge_dst: Vec<u32>,
    /// `base` forward, `max_bases + base` reversed.
    pub edge_type: Vec<u32>,
}

impl Graph {
    pub(super) fn new(dfa: &CanonicalDfa, max_bases: usize, max_depth: usize) -> Self {
        let k = dfa.k;
        let sink = dfa.delta.contains(&DEAD);
        let n = dfa.n + sink as usize;
        // `delta[s * k + b]`, every transition defined.
        let delta: Vec<usize> = (0..n * k)
            .map(|i| match dfa.delta.get(i) {
                Some(&t) if t != DEAD => t as usize,
                _ => dfa.n,
            })
            .collect();
        let accept: Vec<bool> = (0..n).map(|s| s < dfa.n && dfa.accept[s]).collect();

        let mut node_type = vec![NODE_STATE; n];
        node_type[0] = NODE_START;
        if sink {
            node_type[n - 1] = NODE_SINK;
        }
        let clip = |d: Vec<usize>| -> Vec<u32> {
            d.into_iter()
                .map(|d| if d >= n { max_depth + 1 } else { d.min(max_depth) } as u32)
                .collect()
        };
        let not_accept: Vec<bool> = accept.iter().map(|a| !a).collect();

        let mut edge_src = Vec::with_capacity(2 * n * k);
        let mut edge_dst = Vec::with_capacity(2 * n * k);
        let mut edge_type = Vec::with_capacity(2 * n * k);
        for (reverse, offset) in [(false, 0), (true, max_bases)] {
            for s in 0..n {
                for b in 0..k {
                    let (from, to) = (s as u32, delta[s * k + b] as u32);
                    let (from, to) = if reverse { (to, from) } else { (from, to) };
                    edge_src.push(from);
                    edge_dst.push(to);
                    edge_type.push((offset + b) as u32);
                }
            }
        }

        Graph {
            nodes: n,
            node_type,
            is_accept: accept.iter().map(|a| *a as u32).collect(),
            // Every state is reachable.
            fwd_depth: clip(distance_from_start(&delta, n, k))
                .into_iter()
                .map(|d| d.min(max_depth as u32))
                .collect(),
            bwd_depth: clip(distance_to(&delta, n, k, &accept)),
            to_reject: clip(distance_to(&delta, n, k, &not_accept)),
            edge_src,
            edge_dst,
            edge_type,
        }
    }
}

/// Shortest word length from state 0 to each state; `n` if unreachable.
fn distance_from_start(delta: &[usize], n: usize, k: usize) -> Vec<usize> {
    let mut d = vec![n; n];
    d[0] = 0;
    let mut queue = std::collections::VecDeque::from([0]);
    while let Some(s) = queue.pop_front() {
        for &t in &delta[s * k..(s + 1) * k] {
            if d[t] == n {
                d[t] = d[s] + 1;
                queue.push_back(t);
            }
        }
    }
    d
}

/// Shortest word length from each state into `targets`; `n` if none.
fn distance_to(delta: &[usize], n: usize, k: usize, targets: &[bool]) -> Vec<usize> {
    let mut d: Vec<usize> = targets.iter().map(|t| if *t { 0 } else { n }).collect();
    loop {
        let mut changed = false;
        for s in 0..n {
            let via = delta[s * k..(s + 1) * k]
                .iter()
                .map(|t| d[*t] + 1)
                .min()
                .unwrap_or(n);
            if via < d[s] {
                d[s] = via;
                changed = true;
            }
        }
        if !changed {
            return d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        neural_synthesis::{kleene::canonical::canonicalize, minimal_dfa},
        regex::RegularExpression,
    };

    fn graph(pattern: &str) -> Graph {
        let dfa = minimal_dfa(
            &RegularExpression::new(pattern)
                .unwrap()
                .to_automaton()
                .unwrap(),
        )
        .unwrap();
        Graph::new(&canonicalize(&dfa), 16, 64)
    }

    #[test]
    fn appends_a_sink_when_incomplete() {
        // States: start, after `a`, plus the sink; bases `a` and `b`... `ab`.
        let g = graph("ab");
        assert_eq!(g.nodes, 4);
        assert_eq!(g.node_type, vec![1, 0, 0, 2]);
        assert_eq!(g.is_accept, vec![0, 0, 1, 0]);
        assert_eq!(g.fwd_depth, vec![0, 1, 2, 1]);
        assert_eq!(g.bwd_depth, vec![2, 1, 0, 65]);
        assert_eq!(g.to_reject, vec![0, 0, 1, 0]);
        assert_eq!(g.edge_src.len(), 2 * 4 * 3);
        assert!(g.edge_type[..12].iter().all(|t| *t < 3));
        assert!(g.edge_type[12..].iter().all(|t| (16..19).contains(t)));
    }

    #[test]
    fn complete_dfa_has_no_sink() {
        let g = graph(".*");
        assert_eq!(g.nodes, 1);
        assert_eq!(g.node_type, vec![1]);
        assert_eq!(g.to_reject, vec![65]);
    }
}
