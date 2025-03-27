import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import argparse

class GraphVisualizer:
    def __init__(self, graph_path):
        # Load graph data
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # Extract model name from graph
        self.model_name = self.G.graph['model_structure']['model_name'] if 'model_structure' in self.G.graph else 'model'
        
        # Organize nodes by layer
        self.layer_nodes = defaultdict(list)
        for node, data in self.G.nodes(data=True):
            layer = data['layer']
            self.layer_nodes[layer].append(node)
        
        # Get unique layers (ordered by typical network flow)
        self.layers = list(self.layer_nodes.keys())
        
        # Sort layers in typical network order: input -> conv -> fc
        self.layers.sort(key=lambda x: 0 if 'input' in x else 
                          1 if 'conv' in x.lower() else 
                          2 if 'fc' in x.lower() or 'linear' in x.lower() else 3)
    
    def visualize(self, output_dir, layout='hierarchical', show_all_edges=False, 
                 edge_threshold=0.5, node_size=100, edge_scale=1.0):
        """Generate visualization with specified parameters"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the appropriate layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'layered':
            pos = self._layered_layout()
        elif layout == 'spring':
            pos = self._spring_layout() 
        elif layout == 'community':
            pos = self._community_layout()
        else:
            pos = self._hierarchical_layout()  # default
            
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Draw edges with weights
        if show_all_edges:
            # Show all edges (may be dense)
            self._draw_weighted_edges(pos, edge_threshold, edge_scale)
        else:
            # Show only top edges by weight
            self._draw_top_edges(pos, edge_threshold, edge_scale)
        
        # Draw nodes grouped by layer
        self._draw_nodes_by_layer(pos, node_size)
        
        # Add title and legend
        plt.title(f'{self.model_name} Graph Structure', fontsize=14)
        plt.legend(loc='upper right', prop={'size': 8})
        plt.axis('off')
        
        # Save visualization
        output_path = os.path.join(output_dir, f'{self.model_name}_{layout}_graph.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()
        
        # Also create a simplified layer-focused representation
        self._create_layer_summary_visualization(output_dir)
    
    def _hierarchical_layout(self):
        """Create a hierarchical layout with layers stacked vertically"""
        pos = {}
        layer_heights = {}
        
        # Define vertical spacing based on number of layers
        total_height = 10
        layer_height = total_height / len(self.layers)
        
        # Assign y-coordinates to layers
        for i, layer in enumerate(self.layers):
            layer_heights[layer] = (len(self.layers) - 1 - i) * layer_height
        
        # Layout nodes within each layer
        for layer in self.layers:
            nodes = self.layer_nodes[layer]
            n_nodes = len(nodes)
            
            # Space nodes horizontally within layer
            width = max(10, n_nodes * 0.5)  # Adjust for visibility
            
            # Use different arrangements based on node count
            if n_nodes <= 10:
                # Simple horizontal line for small layers
                for i, node in enumerate(nodes):
                    pos[node] = ((i - n_nodes/2) * (width/n_nodes) + width/n_nodes/2, layer_heights[layer])
            else:
                # Circular arrangement for larger layers
                radius = width / 4
                for i, node in enumerate(nodes):
                    angle = 2 * np.pi * i / n_nodes
                    pos[node] = (radius * np.cos(angle), layer_heights[layer] + radius * np.sin(angle) * 0.5)
        
        return pos
    
    def _layered_layout(self):
        """Create a simple layered layout"""
        pos = {}
        for i, layer in enumerate(self.layers):
            nodes = self.layer_nodes[layer]
            layer_width = max(len(nodes) * 0.3, 5)  # Scale width based on nodes
            
            for j, node in enumerate(nodes):
                x_pos = j * layer_width / max(len(nodes), 1) - layer_width / 2
                pos[node] = (x_pos, -i * 2)  # Each layer 2 units down
                
        return pos
    
    def _spring_layout(self):
        """Use force-directed layout, but constrain y-coordinates by layer"""
        # First, compute a basic spring layout
        basic_pos = nx.spring_layout(self.G, k=1/np.sqrt(len(self.G.nodes())), iterations=50)
        
        # Then constrain y-coordinates by layer
        pos = {}
        for i, layer in enumerate(self.layers):
            nodes = self.layer_nodes[layer]
            for node in nodes:
                if node in basic_pos:
                    # Keep x from spring layout, set y by layer
                    pos[node] = (basic_pos[node][0], -i * 2)
        
        return pos
    
    def _community_layout(self):
        """Create layout based on community detection"""
        # Get communities
        from community import community_louvain
        communities = community_louvain.best_partition(self.G)
        
        # First organize by community
        community_nodes = defaultdict(list)
        for node, community_id in communities.items():
            community_nodes[community_id].append(node)
        
        # Create positioning
        pos = {}
        community_count = len(community_nodes)
        
        # Position each community
        for i, (community_id, nodes) in enumerate(community_nodes.items()):
            # Place communities in a circular arrangement
            community_angle = 2 * np.pi * i / community_count
            community_r = 5  # Radius of community centers
            community_x = community_r * np.cos(community_angle)
            community_y = community_r * np.sin(community_angle)
            
            # Within each community, arrange nodes in a mini-circle
            node_count = len(nodes)
            for j, node in enumerate(nodes):
                node_angle = 2 * np.pi * j / node_count
                node_r = 1  # Radius within community
                pos[node] = (
                    community_x + node_r * np.cos(node_angle),
                    community_y + node_r * np.sin(node_angle)
                )
        
        return pos
    
    def _draw_weighted_edges(self, pos, weight_threshold=0.3, scale=1.0):
        """Draw edges with colors and widths based on weights"""
        # Get all edge weights
        edge_weights = [data.get('weight', 0.1) for _, _, data in self.G.edges(data=True)]
        if not edge_weights:
            return
            
        # Normalize weights to 0-1 range for consistent visualization
        vmin = np.percentile(edge_weights, weight_threshold * 100)
        vmax = max(edge_weights)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Color mapping
        cmap = plt.cm.Blues
        
        # Draw edges with weight-based appearance
        for u, v, data in self.G.edges(data=True):
            weight = data.get('weight', 0.1)
            if weight >= vmin:  # Filter by threshold
                # Scale weight for width
                width = scale * 5 * norm(weight) ** 2  # Apply non-linear scaling for better visibility
                
                # Get color from colormap
                color = cmap(norm(weight))
                
                # Draw the edge
                nx.draw_networkx_edges(
                    self.G, pos, 
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=[color],
                    alpha=0.7
                )
    
    def _draw_top_edges(self, pos, percentile=0.5, scale=1.0):
        """Draw only the top edges by weight"""
        # Get all edge weights
        edge_weights = [(u, v, data.get('weight', 0)) for u, v, data in self.G.edges(data=True)]
        if not edge_weights:
            return
            
        # Get threshold value
        threshold = np.percentile([w for _, _, w in edge_weights], percentile * 100)
        
        # Filter and sort edges by weight
        filtered_edges = [(u, v, w) for u, v, w in edge_weights if w >= threshold]
        filtered_edges.sort(key=lambda x: x[2])  # Sort by weight
        
        # Draw edges with varying width based on weight
        edge_list = [(u, v) for u, v, _ in filtered_edges]
        weights = [w for _, _, w in filtered_edges]
        
        # Normalize weights
        if weights:
            max_weight = max(weights)
            normalized_weights = [w/max_weight for w in weights]
            
            # Map weights to widths (using non-linear scaling for better visibility)
            widths = [scale * 3 * (w**2) + 0.5 for w in normalized_weights]
            
            # Draw edges
            nx.draw_networkx_edges(
                self.G, pos,
                edgelist=edge_list,
                width=widths,
                alpha=0.7,
                edge_color='lightblue'
            )
    
    def _draw_nodes_by_layer(self, pos, node_size=100):
        """Draw nodes grouped and colored by layer"""
        colors = cm.tab20.colors  # 20 distinct colors
        
        for i, layer in enumerate(self.layers):
            nodes = self.layer_nodes[layer]
            
            # Create color with alpha depending on layer importance
            color = colors[i % len(colors)]
            
            # Label the first node in each layer for legend
            if nodes:
                # Draw all nodes in this layer
                nx.draw_networkx_nodes(
                    self.G, pos,
                    nodelist=nodes,
                    node_color=[color] * len(nodes),
                    node_size=node_size,
                    label=layer,
                    alpha=0.9
                )
                
                # Add labels to a subset of nodes to avoid clutter
                if len(nodes) <= 5:  # Only label small layers
                    labels = {node: str(node) for node in nodes}
                    nx.draw_networkx_labels(
                        self.G, pos,
                        labels=labels,
                        font_size=8,
                        font_color='black'
                    )
    
    def _create_layer_summary_visualization(self, output_dir):
        """Create a simplified visualization showing only connections between layers"""
        # Create a layer-level graph
        layer_graph = nx.DiGraph()
        
        # Add nodes for each layer
        for layer in self.layers:
            layer_graph.add_node(layer, size=len(self.layer_nodes[layer]))
        
        # Add edges between layers
        layer_connections = defaultdict(float)
        
        for u, v, data in self.G.edges(data=True):
            u_layer = self.G.nodes[u]['layer']
            v_layer = self.G.nodes[v]['layer']
            
            if u_layer != v_layer:  # Only consider connections between different layers
                weight = data.get('weight', 1.0)
                layer_connections[(u_layer, v_layer)] += weight
        
        # Add the accumulated connections to the layer graph
        for (source, target), weight in layer_connections.items():
            layer_graph.add_edge(source, target, weight=weight)
        
        # Draw the layer-level graph
        plt.figure(figsize=(10, 8))
        
        # Position nodes in a vertical hierarchy
        pos = {layer: (0, -i) for i, layer in enumerate(self.layers)}
        
        # Get edge weights for scaling
        edge_weights = [data['weight'] for _, _, data in layer_graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1.0
        
        # Draw edges with varying width based on total connection strength
        for u, v, data in layer_graph.edges(data=True):
            width = 3.0 * data['weight'] / max_weight
            nx.draw_networkx_edges(
                layer_graph, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.7,
                edge_color='blue'
            )
        
        # Draw nodes with size proportional to number of neurons/feature maps
        node_sizes = [200 * np.log1p(layer_graph.nodes[layer]['size']) for layer in layer_graph.nodes]
        
        nx.draw_networkx_nodes(
            layer_graph, pos,
            node_size=node_sizes,
            node_color=cm.tab10.colors[:len(layer_graph.nodes)],
            alpha=0.8
        )
        
        # Add labels
        labels = {layer: f"{layer}\n({layer_graph.nodes[layer]['size']} nodes)" for layer in layer_graph.nodes}
        nx.draw_networkx_labels(
            layer_graph, pos,
            labels=labels,
            font_size=8,
            font_color='black'
        )
        
        plt.title(f'{self.model_name} Layer Connectivity', fontsize=14)
        plt.axis('off')
        
        # Save visualization
        output_path = os.path.join(output_dir, f'{self.model_name}_layer_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Layer summary visualization saved to {output_path}")
        plt.close()


def visualize_graph(graph_path, output_dir, layout='hierarchical', 
                   show_all_edges=False, edge_threshold=0.5, 
                   node_size=100, edge_scale=1.0):
    """Generate visualization with the specified parameters"""
    visualizer = GraphVisualizer(graph_path)
    visualizer.visualize(
        output_dir, 
        layout=layout,
        show_all_edges=show_all_edges,
        edge_threshold=edge_threshold,
        node_size=node_size,
        edge_scale=edge_scale
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize neural network graph structure')
    parser.add_argument('--graph_path', default='../save_graphs/alexnet_graph.pkl', 
                        help='Path to the graph pickle file')
    parser.add_argument('--output_dir', default='../visuals/', 
                        help='Directory to save visualizations')
    parser.add_argument('--layout', default='hierarchical', 
                        choices=['hierarchical', 'layered', 'spring', 'community'],
                        help='Layout algorithm to use')
    parser.add_argument('--show_all_edges', action='store_true',
                        help='Show all edges instead of just the top ones')
    parser.add_argument('--edge_threshold', type=float, default=0.5,
                        help='Threshold for edge filtering (0-1)')
    parser.add_argument('--node_size', type=int, default=100,
                        help='Size of nodes in visualization')
    parser.add_argument('--edge_scale', type=float, default=1.0,
                        help='Scaling factor for edge widths')
                        
    args = parser.parse_args()
    
    # Generate visualizations
    visualize_graph(
        args.graph_path, 
        args.output_dir, 
        layout=args.layout,
        show_all_edges=args.show_all_edges,
        edge_threshold=args.edge_threshold,
        node_size=args.node_size,
        edge_scale=args.edge_scale
    )
    
    print(f"To generate different visualizations, try different layouts:")
    print("python graphVisual.py --layout hierarchical")
    print("python graphVisual.py --layout community")