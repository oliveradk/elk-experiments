from collections import defaultdict, namedtuple
from typing import Tuple, Optional 
import random
from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt

from auto_circuit.types import SrcNode, DestNode, Edge, Node
from auto_circuit.utils.custom_tqdm import tqdm



NodeIdx = Tuple[int, int] # layer, head
def get_node_idx(node: DestNode) -> NodeIdx:
    return (node.layer, node.head_idx)

def valid_node(node: DestNode, seq_idx: int, max_layer: int, last_seq_idx: int, attn_only: bool=False) -> bool:
    before_last = ((node.layer < max_layer - 2) or (attn_only and node.layer < max_layer - 1)) 
    if before_last: 
        return True  
    if seq_idx == last_seq_idx: # last sequence index
        return True 
    if node.name.endswith(("K", "V")): # is k or value
        return True
    return False

def filter_edges(
    edges: list[Edge], 
    max_layer: int, 
    last_seq_idx: int, 
    attn_only: bool=False
) -> list[Edge]:
    return [
        edge for edge in edges 
        if valid_node(edge.dest, edge.seq_idx, max_layer, last_seq_idx, attn_only=attn_only)
    ]


# EdgeGraph types
Graph = dict[Edge, list[Edge]]
PathCounts = dict[Edge, int]

class EdgeGraph():
    """
    "Edges" from AutoCircuit are "Nodes", connected to parent and child "Edges" 
    with children (dict[Edge, list[Edge]]) and parents (dict[Edge, list[Edge])

    Automatically removed edges which cannot reach output at last_seq_idx
    (TODO: generalize to arbitary output nodes - required for tracr)

    """

    def __init__(
        self, 
        edges: list[Edge],
        token: bool = True,
        attn_only: bool = False
    ):  
        
        # used for filtering edges
        self.last_seq_idx = max([edge.seq_idx for edge in edges]) if token else None
        self.max_layer = max([edge.dest.layer for edge in edges])
        self.token = token
        self.attn_only = attn_only
        
        # filter and sort edges
        edges = filter_edges(edges, self.max_layer, self.last_seq_idx, attn_only=attn_only)
        self.edges = sorted(edges, key=lambda edge: (edge.dest.layer, edge.seq_idx))

        # the core data structures are children and parents, which are dictinaries 
        # from edges to lists of edges 
        self.children: Graph = {}
        self.parents: Graph = defaultdict(list[Edge])
        # we also map nod
        self.path_counts: PathCounts = defaultdict(int)
        self.reachable: dict[Edge, bool] = {}

        # to construt this, we'll need 
        # a "forward" map, which connects destination nodes to source nodes - edges_by_src_idx 
        # a "backward" map, which connects source nodes to destination nodes - edges_by_dest_idx
        # we use these to get candidate children and parents for each edge, and then filter 
        # for valid edges based seq_idx, layer, and cross sequence connections (K, V)
        self.edges_by_dest_idx: dict[NodeIdx, list[Edge]] = defaultdict(list)
        self.edges_by_src_idx: dict[NodeIdx, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            self.edges_by_dest_idx[get_node_idx(edge.dest)].append(edge) # used for upstream propogation
            self.edges_by_src_idx[get_node_idx(edge.src)].append(edge) # used for downstream propogation
    
    def get_edges(self, by_src: bool=True) -> list[Edge]:
        sort_key = lambda edge: (edge.src.layer if by_src else edge.dest.layer, edge.seq_idx)
        return sorted(self.edges, key=sort_key)

    
    def build_graph(self):
        self.build_children()
        self.build_parents()

    def build_children(self):
        # construct dest graph from resid end of last token, layer by layer, tracking path counts at each node
        for edge in tqdm(reversed(self.get_edges(by_src=False))):
            # if dest is resid end (leaf) add to graph and set path count to 1, and skip
            if edge.dest.name == "Resid End":
                self.children[edge] = []
                self.path_counts[edge] = 1
                continue 
            # get downstream edges from edge
            child_edges = self.edges_by_src_idx[get_node_idx(edge.dest)]
            if edge.dest.name.endswith(("K", "V")) or not self.token:
                # if dest.layer >= max_layer - 2 (last layer or two layers depending on whether attention and mlp are counted together)
                if edge.dest.layer >= self.max_layer - 2 and self.token:
                    # convert edge dests to seq nodes with seq_idx = last_seq_idx
                    child_edges = [
                        c_edge for c_edge in child_edges 
                        if c_edge.seq_idx == self.last_seq_idx
                    ]
                # else don't filter edges
            else: # (q or mlp)
                # filter for edges in the same sequence position
                child_edges = [
                    c_edge for c_edge in child_edges 
                    if c_edge.seq_idx == edge.seq_idx # must be aligned b/c not k or v
                ]
            self.children[edge] = child_edges
            self.path_counts[edge] = sum([self.path_counts[c_edge] for c_edge in child_edges])
        self.path_counts = dict(self.path_counts) # convert to dict

    def build_parents(self):
        # iterate over sequnce nodes starting from layer 0
        for edge in tqdm(self.get_edges(by_src=True)):
            # add parent to each child
            for child in self.children[edge]:
                self.parents[child].append(edge) # add 
            # set reachable
            if edge.src.name == "Resid Start":
                self.reachable[edge] = True
                self.parents[edge] = [] # no parents for start
                continue
            # for queries, want to check if k or v is reachable in a different sequence positin
            self.reachable[edge] = any([self.reachable[parent] for parent in self.parents[edge]])
        self.reachable = dict(self.reachable) # convert to dict


def edge_in_path(edge: Edge, edge_graph: EdgeGraph, in_path_req=True, reach_req=True) -> bool:
    assert (in_path_req or reach_req)
    if in_path_req:
        if edge not in edge_graph.path_counts or edge_graph.path_counts[edge] == 0:
            return False
    if not reach_req:
        return True
    return edge_graph.reachable[edge]

class SampleType(Enum):
    WALK = 0
    UNIFORM = 1

# goal is to return equivalent of path counts, but only count paths that have at least one edge in provided edges 
def get_edge_path_counts(
    edges: list[Edge],
    edge_graph: EdgeGraph,
) -> PathCounts:
    max_layer = max([edge.src.layer for edge in edges])
    
    # initialize with edges to 1 
    edge_path_counts: PathCounts = defaultdict(int)
    for edge in edges: 
        edge_path_counts[edge] = 1
    # in revesred order from max layer 
    # set to max of edge_path_counts of children or edge_path counts of self
    edges_to_process = [edge for edge in edge_graph.get_edges(by_src=False) if edge.dest.layer <= max_layer]
    for edge in tqdm(reversed(edges_to_process)):
        child_path_counts = sum([edge_path_counts[child] for child in edge_graph.children[edge]])
        edge_path_counts[edge] = max(child_path_counts, edge_path_counts[edge])
    return edge_path_counts


# sample path
def sample_path_uniform(
    edge_graph: EdgeGraph, 
    edge_path_counts: Optional[PathCounts]=None, 
    edges: Optional[set[Edge]]=None
) -> list[Edge]:
    assert (edge_path_counts is None) == (edges is None)
    # sample first node in path
    path = []
    start_edges = [edge for edge in edge_graph.get_edges(by_src=True) if edge.src.name == "Resid Start"]
    if edge_path_counts is not None:
        total_edge_paths = sum([edge_path_counts[node] for node in start_edges])
        start_probs = [edge_path_counts[node] / total_edge_paths for node in start_edges]
    else: 
        start_probs = None
    current = random.choices(start_edges, weights=start_probs)[0]
    edge_added = current in edges
    path.append(current)

    # sample path to include edge (if edge with path)
    if edge_path_counts is not None: 
        while not edge_added:
            neighbors = edge_graph.children[current]
            probs = [edge_path_counts[neighbor] / edge_path_counts[current] for neighbor in neighbors]
            current = random.choices(neighbors, weights=probs)[0]
            path.append(current)
            edge_added = current in edges
    
    # sample rest of path
    while current.dest.name != "Resid End":
        neighbors = edge_graph.children[current]
        probs = [edge_graph.path_counts[neighbor] / edge_graph.path_counts[current] for neighbor in neighbors]
        current = random.choices(edge_graph.children[current], weights=probs)[0] 
        path.append(current)
    return path


def sample_path_random_walk(
    edge_graph: EdgeGraph,
    tested_edges: Optional[list[Edge]],
) -> list[Edge]:
    while True:
        path = []
        start_slice = slice(None, edge_graph.last_seq_idx if edge_graph.token else 1)
        current = random.choice(edge_graph.edges[start_slice])
        while current.dest.name != "Resid End":
            neighbors = edge_graph.children[current]
            current = random.choice(neighbors)
            path.append(current)
        if tested_edges == None:
            return path
        if any((edge not in tested_edges for edge in path)): # must be an edge not in tested edges 
            return path
    

def sample_paths(
    edge_graph: EdgeGraph, 
    n_paths: int, 
    sample_type: SampleType, 
    tested_edges: list[Edge], 
) -> list[list[Edge]]:
    if sample_type == SampleType.UNIFORM:
        compelement_edges = set(edge_graph.edges) - set(tested_edges)
        edge_path_counts = get_edge_path_counts(compelement_edges, edge_graph)
        return [sample_path_uniform(edge_graph, edge_path_counts, compelement_edges) for _ in tqdm(range(n_paths))]
    elif sample_type == SampleType.WALK:
        return [sample_path_random_walk(edge_graph, tested_edges) for _ in tqdm(range((n_paths)))]
    else:
        raise ValueError(f"Invalid sample type {sample_type}")


# def visualize_graph(graph:Graph, sort_by_head: bool=True):
#     # Create a new directed graph
#     G = nx.DiGraph()

#     SeqNode = namedtuple('SeqNode', ['destnode', 'seq_idx'])

#     # Add nodes and edges to the graph
#     for source, targets in graph.items():
#         source_seq = SeqNode(destnode=source.dest, seq_idx=source.seq_idx)
#         G.add_node(str(source.dest), layer=source.dest.layer, seq_idx=source.seq_idx, head_idx=source.dest.head_idx)
#         for target in targets:
#             G.add_edge(str(source), str(target))

#     # Set up the plot
#     plt.figure(figsize=(24, 16))
    
#     # Create a custom layout for the graph
#     pos = {}
#     seq_idx_set = sorted(set(data['seq_idx'] for _, data in G.nodes(data=True)))
#     layer_set = sorted(set(data['layer'] for _, data in G.nodes(data=True)))  # No longer reversed
    
#     # Group nodes by layer and seq_idx
#     grouped_nodes = defaultdict(list)
#     for node, data in G.nodes(data=True):
#         grouped_nodes[(data['layer'], data['seq_idx'])].append((node, data))

#     # Calculate layout
#     column_width = 1.5  # Adjust this value to increase horizontal spacing
#     row_height = 5  # Adjust this value to increase vertical spacing
#     max_nodes_in_group = max(len(nodes) for nodes in grouped_nodes.values())
    
#     for (layer, seq_idx), nodes in grouped_nodes.items():
#         x = seq_idx_set.index(seq_idx) * column_width
#         y = (len(layer_set) - 1 - layer_set.index(layer)) * row_height  # Invert y-axis
        
#         # Sort nodes by head_idx (if available) or by node name
#         if sort_by_head:
#             sorted_nodes = sorted(nodes, key=lambda n: (n[1]['head_idx'] if n[1]['head_idx'] is not None else float('inf'), n[0]))
#         else: # sort by Q, K, V, MLP
#             sorted_nodes = sorted(nodes, key=lambda n: (n[0].split('_')[0].split('.')[-1]))
        
#         # Position nodes in a vertical line within their layer and seq_idx group
#         for i, (node, data) in enumerate(sorted_nodes):
#             node_y = y - i * (row_height / (max_nodes_in_group + 1))  # Distribute nodes evenly within the row
#             pos[node] = (x, node_y)

#     # Draw the nodes
#     node_size = 100  # Adjust as needed
#     nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')

#     # Draw the edges
#     nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=0.5, arrowsize=10)

#     # Add labels to the nodes
#     labels = {node: f"{node.split('_')[0]}" for node in G.nodes()}
#     nx.draw_networkx_labels(G, pos, labels, font_size=6)

#     # Add path counts as labels on the nodes (uncomment if needed)
#     # path_count_labels = {str(node): f"Paths: {count}" for node, count in path_counts.items()}
#     # nx.draw_networkx_labels(G, pos, path_count_labels, font_size=4, font_color='red')

#     plt.title("Graph Visualization with Corrected Layer Spacing")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()