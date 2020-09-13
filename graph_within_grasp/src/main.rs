extern crate alloc;

use alloc::collections::{BTreeSet, VecDeque};

/// Maximum number of nodes for a graph
const MAX_NODES: usize = 64;

/// Maximum input size in bytes
const MAX_INPUT_SIZE: usize = 4;

/// A strongly typed node ID 
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Debug)]
pub struct Graph {
    /// Next available node ID
    next_node: NodeId,

    /// A bitmap indicating which nodes are actively leaves. Once a node has
    /// a `branch` added, it is no longer a leaf node
    leaves: [u64; MAX_NODES / 64],

    /// All nodes in the graph
    nodes: [Option<Node>; MAX_NODES],
}

impl Graph {
    /// Create a new empty graph with only a root node
    pub fn new() -> Self {
        // Ensure the constants are sane
        assert!(MAX_NODES > 0 && (MAX_NODES % 64) == 0,
            "Max nodes must be modulo 64 and non-zero");
        assert!(MAX_INPUT_SIZE > 0, "Maximum input size must be non-zero");

        // Create the root node
        let mut nodes = [None; MAX_NODES];
        nodes[0] = Some(Node {
            avail_bits:     [0xff; MAX_INPUT_SIZE],
            branch:         None,
            reachable_from: [0; MAX_NODES / 64],
            reachable_to:   [0; MAX_NODES / 64],
            probability:    0.,
        });
        
        // Create the graph
        let graph = Graph {
            next_node: NodeId(1),
            leaves:    [0u64; MAX_NODES / 64],
            nodes:     nodes,
        };

        graph
    }

    /// Dump the current graph to `foo.svg` in the current folder
    pub fn dump_svg(&self) {
        // Queue up exploring the root node
        let mut queue   = VecDeque::new();
        let mut visited = BTreeSet::new();

        // Create the DOT file
        let mut dotfile = String::new();

        // Queue up exploring from the root
        queue.push_back(NodeId(0));

        dotfile += "digraph {\n";
        while !queue.is_empty() {
            // Get the node we want to explore
            let node_id = queue.pop_front().unwrap();

            // Skip nodes we've visited
            if !visited.insert(node_id) {
                continue;
            }
        
            // Get access to the node
            let node = self.nodes[node_id.0].as_ref().unwrap();

            let mut reachable_to = BTreeSet::new();
            for (idx, &qword) in node.reachable_to.iter().enumerate() {
                if qword == 0 { continue; }
                for bit in 0..64 {
                    if (qword >> bit) & 1 == 0 { continue; }
                    reachable_to.insert(idx * 64 + bit);
                }
            }

            let mut reachable_from = BTreeSet::new();
            for (idx, &qword) in node.reachable_from.iter().enumerate() {
                if qword == 0 { continue; }
                for bit in 0..64 {
                    if (qword >> bit) & 1 == 0 { continue; }
                    reachable_from.insert(idx * 64 + bit);
                }
            }

            // Create the node label
            dotfile += &format!("  \"{node_id:?}\" \
                    [label=\"{node_id:?}\n\
                    probability = {probability:.6}%\n\
                    to = {reachable_to:?}\n\
                    from = {reachable_from:?}\"]\n",
                node_id = node_id,
                reachable_to = reachable_to,
                reachable_from = reachable_from,
                probability = node.probability * 100.);

            if let Some((edge, ttgt, ftgt)) = node.branch {
                queue.push_back(ttgt);
                queue.push_back(ftgt);

                let tprob = edge.probability();
                let fprob = 1. - tprob;
                dotfile += &format!("  \"{:?}\" -> \"{:?}\" \
                                    [label=\"{:.6} %\"];\n",
                    node_id, ttgt, tprob * 100.);
                dotfile += &format!("  \"{:?}\" -> \"{:?}\" \
                                    [label=\"{:.6} %\"];\n",
                    node_id, ftgt, fprob * 100.);
            } else {
                dotfile += &format!("  \"{:?}\" -> \"return\";\n", node_id);
            }
        }
        dotfile += "}\n";

        // Write the dotfile to disk
        std::fs::write("foo.dot", &dotfile).expect("Failed to write dotfile");

        // Invoke graphviz to convert the dotfile to an SVG
        assert!(std::process::Command::new("dot")
            .args(&["-Tsvg", "-o", "foo.svg", "foo.dot"])
            .status().expect("Failed to run `dot`, is graphviz installed?")
            .success(), "Graphviz returned with error");
    }

    /// Allocate a new node ID
    fn alloc_node(&mut self) -> NodeId {
        // Bounds check the node allocation
        assert!(self.next_node < NodeId(MAX_NODES), "Out of nodes");

        // Allocate a node ID
        let node_id = self.next_node;
        self.next_node.0 += 1;

        // Return the node ID
        node_id
    }

    /// Gets a mutable reference to `NodeId`
    fn get_node_mut(&mut self, node_id: NodeId) -> Option<NodeRefMut> {
        if self.nodes.get(node_id.0).is_some() {
            // Create the reference to the node
            Some(NodeRefMut {
                node_id: node_id,
                graph:   self,
            })
        } else {
            None
        }
    }

    fn calculate_probabilities(&mut self) {
        /// Current probability
        let mut cur_prob = 1.;

        /// Current stack of traversed nodes
        let mut queue = Vec::new();
        let mut stack = Vec::new();

        // Store the node ID of the root node
        queue.push((cur_prob, 0, NodeId(0)));

        let mut last_depth = 0;
        while !queue.is_empty() {
            let (node_prob, node_depth, node_id) = queue.pop().unwrap();
            
            for _ in node_depth..=last_depth {
                stack.pop();
            }
            last_depth = node_depth;

            if stack.contains(&node_id) {
                last_depth = node_depth - 1;
                continue;
            }

            stack.push(node_id);

            let node = self.nodes[node_id.0].as_mut().unwrap();
            node.probability += node_prob;

            if let Some((edge, ttgt, ftgt)) = node.branch {
                let tprob = edge.probability();
                let fprob = 1. - tprob;
                queue.push((node_prob * tprob, node_depth + 1, ttgt));
                queue.push((node_prob * fprob, node_depth + 1, ftgt));
            }
        }

    }
}

struct NodeRefMut<'a> {
    /// The ID of the node that we're working with
    node_id: NodeId,

    /// A mutable reference to the graph this node belongs to
    graph: &'a mut Graph,
}

impl<'a> NodeRefMut<'a> {
    /// Set the branch for this node
    fn set_branch(&mut self, edge: Edge, ttgt: NodeType, ftgt: NodeType) {
        // Make sure the `self` doesn't already have a branch established
        assert!(self.graph.nodes[self.node_id.0].as_ref().unwrap()
                .branch.is_none(), "Cannot change the target of an edge");

        // Check the input space for this node can handle the edge
        match edge {
            Edge::Input { start, size, .. } => {
                // Get access to the node we're going to modify
                let node = self.graph.nodes[self.node_id.0].as_ref().unwrap();

                // Go through each bit in the input and make sure it is
                // available
                for ii in start..start + size as usize {
                    let idx = ii / 8;
                    let bit = ii % 8;
                    assert!(node.avail_bits.get(idx).map(|x| {
                        (x & (1 << bit)) != 0
                    }) == Some(true), "Input bit is not available, {}", ii);
                }
            }
        }

        // Resolve existing nodes and create new nodes if needed
        let mut node_ids = [NodeId(!0), NodeId(!0)];
        for (tgt_node, &target) in
                node_ids.iter_mut().zip([ttgt, ftgt].iter()) { 
            *tgt_node = match target {
                NodeType::New => {
                    // Copy the node we're branching from
                    let mut new_node =
                        self.graph.nodes[self.node_id.0].unwrap();

                    // Reduce the input space according to the edge constraints
                    match edge {
                        Edge::Input { start, size, .. } => {
                            for ii in start..start + size as usize {
                                let idx = ii / 8;
                                let bit = ii % 8;
                                new_node.avail_bits[idx] &= !(1 << bit);
                            }
                        }
                    }

                    // Set the probability to zero for the new node, we'll add
                    // in the probability generically below
                    new_node.probability = 0.;

                    // Allocate a node and update it with the new node
                    let node_id = self.graph.alloc_node();
                    self.graph.nodes[node_id.0] = Some(new_node);

                    node_id
                }
                NodeType::Existing(node_id) => node_id,
            };
        }

        // Compute tgt0_to | tgt0 | tgt1_to | tgt1
        let mut tos = self.graph.nodes[node_ids[0].0].unwrap().reachable_to;
        for (toset, &tgt) in tos.iter_mut().zip(
                &self.graph.nodes[node_ids[1].0].unwrap().reachable_to) {
            *toset |= tgt;
        }
        tos[node_ids[0].0 / 64] |= 1 << (node_ids[0].0 % 64);
        tos[node_ids[1].0 / 64] |= 1 << (node_ids[1].0 % 64);

        // Update current node to targets
        let cur_node = self.graph.nodes[self.node_id.0].as_mut().unwrap();
        for (toset, &tgt) in cur_node.reachable_to.iter_mut().zip(&tos) {
            *toset |= tgt;
        }

        // Get the current node from and to state
        let cur_from = cur_node.reachable_from;
        let cur_to   = cur_node.reachable_to;
       
        // Update the target froms to include the current node froms as well
        // as the current node itself
        for &target in &node_ids {
            // Or the from locations for the current node into the targets as
            // they're not reachable through the same paths
            let target = self.graph.nodes[target.0].as_mut().unwrap();
            for (toset, &tgt) in
                    target.reachable_from.iter_mut().zip(&cur_from) {
                *toset |= tgt;
            }
        
            // Or in the current node
            target.reachable_from[self.node_id.0 / 64] |=
                1 << (self.node_id.0 % 64);
        }
        
        // Go through everything which can reach the current node and update
        // their to paths to include the current to path
        for (idx, &qword) in cur_from.iter().enumerate() {
            if qword == 0 { continue; }
            for bit in 0..64 {
                if (qword >> bit) & 1 == 0 { continue; }
                
                // Get the node which is reachable from us
                let node_id = NodeId(idx * 64 + bit);
                let target = self.graph.nodes[node_id.0].as_mut().unwrap();
                for (toset, &tgt) in
                        target.reachable_to.iter_mut().zip(&cur_to) {
                    *toset |= tgt;
                }
            }
        }

        // Create a set containing all of the possible to targets for both
        // branch targets
        let mut tos = self.graph.nodes[node_ids[0].0].unwrap().reachable_to;
        for (toset, &tgt) in tos.iter_mut().zip(
                &self.graph.nodes[node_ids[1].0].unwrap().reachable_to) {
            *toset |= tgt;
        }
        
        // Go through each reachable target from the current node and update
        // from states
        for (idx, &qword) in tos.iter().enumerate() {
            if qword == 0 { continue; }
            for bit in 0..64 {
                if (qword >> bit) & 1 == 0 { continue; }
                let node_id = NodeId(idx * 64 + bit);

                let target = self.graph.nodes[node_id.0].as_mut().unwrap();
                for (toset, &tgt) in
                        target.reachable_from.iter_mut().zip(&cur_from) {
                    *toset |= tgt;
                }
                
                // Or in the current node
                target.reachable_from[self.node_id.0 / 64] |=
                    1 << (self.node_id.0 % 64);
            }
        }
        
        // Set the branch target
        let cur_node = self.graph.nodes[self.node_id.0].as_mut().unwrap();
        cur_node.branch = Some((edge, node_ids[0], node_ids[1]));
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    /// A conditional branch based on `Edge`, which will go to the first node
    /// if the condition is `true`, otherwise will go to the second node
    /// If `None`, this node is a leaf
    branch: Option<(Edge, NodeId, NodeId)>,

    /// A bitmap of available bits in the input at this node
    avail_bits: [u8; MAX_INPUT_SIZE],

    /// Nodes which can reach this node
    reachable_from: [u64; MAX_NODES / 64],
    
    /// Nodes which can be reached from this node
    reachable_to: [u64; MAX_NODES / 64],

    /// Probability of hitting this node (regardless of path)
    probability: f64,
}

/// A conditional edge on the graph
#[derive(Debug, Clone, Copy)]
enum Edge {
    /// Condition based on a value taken from the user-controlled input
    Input {
        /// Bit offset in the input where the data is sourced
        start: usize,

        /// Size of the value taken from the input (in bits)
        size: u8,

        /// Conditional to use for the branch
        cond: Condition,
    },
}

impl Edge {
    /// Returns the probability of a condition being true based on uniform
    /// random.
    fn probability(&self) -> f64 {
        match self {
            Edge::Input { size, cond, .. } => {
                // Compute the number of bit combinations for `size`
                let size = 2f64.powf(*size as f64);

                // Compute the probability of the condition
                match cond {
                    Condition::Equal(u64) => {
                        // The value must exact match something, thus the
                        // probability is one in `size`
                        1. / size
                    }
                }
            }
        }
    }
}

/// An enum used where it's possible to pass in a new node or an existing node
#[derive(Clone, Copy, Debug)]
enum NodeType {
    /// Create a new node
    New,

    /// Use an existing node ID
    Existing(NodeId),
}

/// A conditional operation
#[derive(Debug, Clone, Copy)]
enum Condition {
    Equal(u64),
}

fn main() {
    let mut graph = Graph::new();

    let edge = Edge::Input { start: 0, size: 7, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(0)).unwrap();
    node.set_branch(edge, NodeType::New, NodeType::New);
   
    let edge = Edge::Input { start: 8, size: 8, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(1)).unwrap();
    node.set_branch(edge, NodeType::New, NodeType::New);
    
    let edge = Edge::Input { start: 26, size: 1, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(3)).unwrap();
    node.set_branch(edge, NodeType::New, NodeType::New);
    
    let edge = Edge::Input { start: 16, size: 1, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(2)).unwrap();
    node.set_branch(edge, NodeType::Existing(NodeId(3)), NodeType::New);
  
    let edge = Edge::Input { start: 24, size: 1, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(5)).unwrap();
    node.set_branch(edge, NodeType::Existing(NodeId(2)), NodeType::Existing(NodeId(1)));
    
    let edge = Edge::Input { start: 25, size: 1, cond: Condition::Equal(5) };
    let mut node = graph.get_node_mut(NodeId(4)).unwrap();
    node.set_branch(edge, NodeType::Existing(NodeId(2)), NodeType::Existing(NodeId(4)));

    graph.calculate_probabilities();

    graph.dump_svg();
}

