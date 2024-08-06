from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
import random
from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt

from auto_circuit.types import SrcNode, DestNode, Edge, Node, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm


@dataclass(frozen=True)
class SeqNode(Node):
    seq_idx: int | None = None

    def __repr__(self) -> str:
        return self.name + "_" + str(self.seq_idx)

    def __str__(self) -> str:
        return self.name + "_" + str(self.seq_idx)


Graph = dict[SeqNode, list[SeqNode]]
PathCounts = dict[SeqNode, int]
NodeIdx = Tuple[int, int] # layer, head



def get_node_idx(node: DestNode) -> NodeIdx:
    return (node.layer, node.head_idx)


def node_to_seq_node(node: Node, seq_idx: int | None) -> SeqNode:
    return SeqNode(
        name=node.name,
        module_name=node.module_name,
        layer=node.layer,
        head_idx=node.head_idx,
        head_dim=node.head_dim,
        seq_idx=seq_idx,
    )

def valid_parent(edge: Edge, seq_idx: int):
    if edge.dest.name.endswith(("K", "V")):
        return True
    return edge.seq_idx == seq_idx

def valid_node(node: SeqNode, max_layer: int, last_seq_idx: int, attn_only: bool=False) -> bool:
    before_last = ((node.layer < max_layer - 2) or (attn_only and node.layer < max_layer - 1)) 
    if before_last: 
        return True  
    if node.seq_idx == last_seq_idx: # last sequence index
        return True 
    if node.name.endswith(("K", "V")): # is k or value
        return True
    return False

# construt node Graph class
class NodeGraph():

    def __init__(
        self, 
        edges: list[Edge],
        token: bool = True,
        attn_only: bool = False
    ):
        self.edges = edges
        self.token = token
        self.attn_only = attn_only
        self.last_seq_idx = max([edge.seq_idx for edge in edges]) if token else None
        self.max_layer = max([edge.dest.layer for edge in edges])

        # helper data structures
        # dest and src idx to edges
        self.edges_by_dest_idx: dict[NodeIdx, list[Edge]] = defaultdict(list)
        self.edges_by_src_idx: dict[NodeIdx, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            self.edges_by_dest_idx[get_node_idx(edge.dest)].append(edge) # used for upstream propogation
            self.edges_by_src_idx[get_node_idx(edge.src)].append(edge) # used for downstream propogation
        # convert sequence dest nodes to edges 
        self.dest_pairs_to_edges: dict[Tuple[int, int|None, str, int|None, int]: Edge] = {}
        for edge in self.edges:
            self.dest_pairs_to_edges[(*get_node_idx(edge.src), edge.dest.name, edge.dest.head_idx, edge.seq_idx)] = edge
        # nodes to dictionary of sequence nodes
        self.nodes_to_seq_nodes: dict[Node, dict[int, SeqNode]] = defaultdict(dict[int, SeqNode])
        for edge in self.edges:
            if edge.src.name == "Resid Start":
                self.nodes_to_seq_nodes[edge.src][edge.seq_idx] = node_to_seq_node(edge.src, edge.seq_idx)
            self.nodes_to_seq_nodes[edge.dest][edge.seq_idx] = node_to_seq_node(edge.dest, edge.seq_idx)
        self.seq_nodes = [
            seq_node for seq_node_dict in self.nodes_to_seq_nodes.values() 
            for seq_node in seq_node_dict.values() 
            if valid_node(seq_node, self.max_layer, self.last_seq_idx, attn_only=self.attn_only)
        ]
        
        # core data structures
        self.children: Graph = {}
        self.parents: Graph = defaultdict(list[SeqNode])
        self.path_counts: PathCounts = defaultdict(int)
        self.reachable: dict[SeqNode, bool] = {}
    
    def build_graph(self):
        self.build_children()
        self.build_parents()

    def build_children(self):
        # construct dest graph from resid end of last token, layer by layer, tracking path counts at each node
        for seq_node in tqdm(sorted(self.seq_nodes, key=lambda node: node.layer, reverse=True)):
            # if dest is resid end (leaf) add to graph and set path count to 1, and skip
            if seq_node.name == "Resid End":
                self.children[seq_node] = []
                self.path_counts[seq_node] = 1
                continue 
                # if dest is K, V 
            # get downstream edges from dest
            child_edges = self.edges_by_src_idx[get_node_idx(seq_node)]
            if seq_node.name.endswith(("K", "V")) or not self.token:
                # if dest.layer >= max_layer - 2 (last layer or two layers depending on whether attention and mlp are counted together)
                if seq_node.layer >= self.max_layer - 2 and self.token:
                    # convert edge dests to seq nodes with seq_idx = last_seq_idx
                    child_dests = [
                        self.nodes_to_seq_nodes[c_edge.dest][self.last_seq_idx] 
                        for c_edge in child_edges if c_edge.seq_idx == self.last_seq_idx
                    ]
                else:
                    # convert edge dests to all seq nodes
                    child_dests = [self.nodes_to_seq_nodes[c_edge.dest][c_edge.seq_idx] for c_edge in child_edges]
            # else (q or mlp)
            else: 
                # already checked if valid, but checking again
                assert valid_node(seq_node, self.max_layer, self.last_seq_idx, attn_only=self.attn_only)
                # convet edge dests to all seq nodes with same seq_idx
                child_dests = [
                    self.nodes_to_seq_nodes[c_edge.dest][seq_node.seq_idx] 
                    for c_edge in child_edges if c_edge.seq_idx == seq_node.seq_idx # must be aligned b/c not k or v
                ]
            # filter for valid child nodes
            self.children[seq_node] = [child_dest for child_dest in child_dests if valid_node(child_dest, self.max_layer, self.last_seq_idx, attn_only=self.attn_only)]
            self.path_counts[seq_node] = sum([self.path_counts[child_dest] for child_dest in child_dests])

        # test graph is sorted by layer
        self.children = {
            seq_node: self.children[seq_node] 
            for seq_node in sorted(self.children.keys(), key=lambda node: node.layer, reverse=True)}
        seq_nodes = self.get_nodes()
        assert all([seq_nodes[i].layer >= seq_nodes[i + 1].layer for i in range(len(seq_nodes) - 1)])

    
    def build_parents(self):
        # iterate over sequnce nodes starting from layer 0
        for seq_node in tqdm(sorted(self.seq_nodes, key=lambda node: node.layer, reverse=False)):
            if seq_node not in self.children:
                raise ValueError(f"Node {seq_node} not in children")
            # add parent to each child
            for child in self.children[seq_node]:
                self.parents[child].append(seq_node)
            # set reachable
            if seq_node.name == "Resid Start":
                self.reachable[seq_node] = True
                continue
            self.reachable[seq_node] = any([self.reachable[parent] for parent in self.parents[seq_node]])
    
    def edge_in_path(self, edge: Edge) -> bool:
        seq_node = self.nodes_to_seq_nodes[edge.dest][edge.seq_idx]
        in_path = self.path_counts[seq_node] > 0 
        if not in_path:
            return False
        if edge.src.name == "Resid Start":
            return True
        # look up edges by dest with src idx
        p_edges = self.edges_by_dest_idx[get_node_idx(edge.src)]
        p_edges = [p_edge for p_edge in p_edges if valid_parent(p_edge, edge.seq_idx)]
        # ok so here we want to check if the dest is valid (must be in the same sequence)
        p_seq_nodes = [self.nodes_to_seq_nodes[p_edge.dest][p_edge.seq_idx] for p_edge in p_edges]
        reachable = any([self.reachable[p_seq_node] for p_seq_node in p_seq_nodes])
        return reachable


    def get_nodes(self) -> list[SeqNode]:
        return [k for k in self.children.keys()]

class SampleType(Enum):
    WALK = 0
    UNIFORM = 1

# sample path
def sample_path(node_graph: NodeGraph, sample_type: SampleType=SampleType.WALK) -> list[SeqNode]:
    path = []
    start_slice = slice(-node_graph.last_seq_idx if node_graph.token else -1, None)
    current = random.choice(node_graph.get_nodes()[start_slice]) # resid starts #TODO: fix
    assert current.name == "Resid Start"
    path.append(current)

    if sample_type == SampleType.UNIFORM:
        while current.name != "Resid End":
            neighbors = node_graph.children[current]
            probs = [node_graph.path_counts[neighbor] / node_graph.path_counts[current] for neighbor in neighbors]
            current = random.choices(node_graph.children[current], weights=probs)[0] 
            path.append(current)
    elif sample_type == SampleType.WALK:
        while current.name != "Resid End":
            neighbors = node_graph.children[current]
            current = random.choice(neighbors)
            path.append(current)
    return path 


def dest_path_to_edge_path(dest_path: list[SeqNode], node_graph:NodeGraph) -> list[Edge]:
    edge_path = []
    for i in range(len(dest_path) - 1):
        dest_src = dest_path[i]
        dest_dest = dest_path[i + 1]
        edge = node_graph.dest_pairs_to_edges[(*get_node_idx(dest_src), dest_dest.name, dest_dest.head_idx, dest_dest.seq_idx)]
        edge_path.append(edge)
    return edge_path


def sample_paths(
        node_graph: NodeGraph, 
        n_paths: int, 
        sample_type: SampleType, 
        tested_edges: list[Edge], 
) -> list[list[Edge]]:
    filtered_paths = []
    for _ in tqdm(range(n_paths)):
        dest_path = sample_path(node_graph, sample_type=sample_type)
        path = dest_path_to_edge_path(dest_path, node_graph)
        while not any((edge not in tested_edges for edge in path)):
            dest_path = sample_path(node_graph, sample_type=sample_type)
            path = dest_path_to_edge_path(dest_path, node_graph)
        filtered_paths.append(path)
    return filtered_paths


def visualize_graph(graph:Graph , path_counts: PathCounts, sort_by_head: bool=True):
    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for source, targets in graph.items():
        G.add_node(str(source), layer=source.layer, seq_idx=source.seq_idx, head_idx=source.head_idx)
        for target in targets:
            G.add_edge(str(source), str(target))

    # Set up the plot
    plt.figure(figsize=(24, 16))
    
    # Create a custom layout for the graph
    pos = {}
    seq_idx_set = sorted(set(data['seq_idx'] for _, data in G.nodes(data=True)))
    layer_set = sorted(set(data['layer'] for _, data in G.nodes(data=True)))  # No longer reversed
    
    # Group nodes by layer and seq_idx
    grouped_nodes = defaultdict(list)
    for node, data in G.nodes(data=True):
        grouped_nodes[(data['layer'], data['seq_idx'])].append((node, data))

    # Calculate layout
    column_width = 1.5  # Adjust this value to increase horizontal spacing
    row_height = 5  # Adjust this value to increase vertical spacing
    max_nodes_in_group = max(len(nodes) for nodes in grouped_nodes.values())
    
    for (layer, seq_idx), nodes in grouped_nodes.items():
        x = seq_idx_set.index(seq_idx) * column_width
        y = (len(layer_set) - 1 - layer_set.index(layer)) * row_height  # Invert y-axis
        
        # Sort nodes by head_idx (if available) or by node name
        if sort_by_head:
            sorted_nodes = sorted(nodes, key=lambda n: (n[1]['head_idx'] if n[1]['head_idx'] is not None else float('inf'), n[0]))
        else: # sort by Q, K, V, MLP
            sorted_nodes = sorted(nodes, key=lambda n: (n[0].split('_')[0].split('.')[-1]))
        
        # Position nodes in a vertical line within their layer and seq_idx group
        for i, (node, data) in enumerate(sorted_nodes):
            node_y = y - i * (row_height / (max_nodes_in_group + 1))  # Distribute nodes evenly within the row
            pos[node] = (x, node_y)

    # Draw the nodes
    node_size = 100  # Adjust as needed
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=0.5, arrowsize=10)

    # Add labels to the nodes
    labels = {node: f"{node.split('_')[0]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    # Add path counts as labels on the nodes (uncomment if needed)
    # path_count_labels = {str(node): f"Paths: {count}" for node, count in path_counts.items()}
    # nx.draw_networkx_labels(G, pos, path_count_labels, font_size=4, font_color='red')

    plt.title("Graph Visualization with Corrected Layer Spacing")
    plt.axis('off')
    plt.tight_layout()
    plt.show()