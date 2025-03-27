import pickle
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


class GraphMetricsCalculator:
    def __init__(self, graph_path):
        """Initialize the calculator with path to a graph pickle file"""
        print(f"Loading graph from {graph_path}")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # Extract model name
        self.model_name = self.G.graph['model_structure']['model_name'] if 'model_structure' in self.G.graph else 'model'
        print(f"Loaded graph for {self.model_name} with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        # Create directory to save results if it doesn't exist
        self.output_dir = "../graph_metrics/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Organize nodes by layer
        self.layer_nodes = defaultdict(list)
        for node, data in self.G.nodes(data=True):
            if 'layer' in data:
                layer = data['layer']
                self.layer_nodes[layer].append(node)
        
        # Get unique layers
        self.layers = list(self.layer_nodes.keys())
        self.layers.sort(key=lambda x: 0 if 'input' in x else 
                         1 if 'conv' in x.lower() else 
                         2 if 'fc' in x.lower() or 'linear' in x.lower() else 3)
    
    def calculate_all_metrics(self):
        """Calculate and save all graph metrics"""
        print("Calculating graph metrics...")
        
        # Create a dictionary to store all metrics
        all_metrics = {}
        
        # Basic graph statistics
        all_metrics['basic_stats'] = self.calculate_basic_stats()
        
        # Degree distribution metrics
        all_metrics['degree_metrics'] = self.calculate_degree_metrics()
        
        # Centrality metrics
        all_metrics['centrality_metrics'] = self.calculate_centrality_metrics()
        
        # Path metrics (can be slow for large graphs)
        if self.G.number_of_nodes() <= 10000:
            try:
                all_metrics['path_metrics'] = self.calculate_path_metrics()
            except:
                print("WARNING: Path metrics calculation failed, possibly due to graph size or structure")
                all_metrics['path_metrics'] = {"error": "Calculation failed"}
        else:
            print("WARNING: Graph too large for path metrics calculation")
            all_metrics['path_metrics'] = {"skipped": "Graph too large"}
        
        # Layer-wise metrics
        all_metrics['layer_metrics'] = self.calculate_layer_metrics()
        
        # Save metrics to file
        output_path = os.path.join(self.output_dir, f"{self.model_name}_metrics.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(all_metrics, f)
        
        # Also save as text file for easy reading
        self.save_metrics_as_text(all_metrics)
        
        # Create visualizations of key metrics
        self.visualize_metrics(all_metrics)
        
        return all_metrics
    
    def calculate_basic_stats(self):
        """Calculate basic graph statistics"""
        metrics = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'edge_to_node_ratio': self.G.number_of_edges() / max(1, self.G.number_of_nodes()),
            'is_directed': self.G.is_directed(),
            'density': nx.density(self.G),
        }
        
        # Calculate number of connected components (use undirected version if graph is directed)
        if self.G.is_directed():
            undirected_G = self.G.to_undirected()
            metrics['num_connected_components'] = nx.number_connected_components(undirected_G)
        else:
            metrics['num_connected_components'] = nx.number_connected_components(self.G)
        
        # Number of isolated nodes
        metrics['num_isolated_nodes'] = len(list(nx.isolates(self.G)))
        
        return metrics
    
    def calculate_degree_metrics(self):
        """Calculate degree-related metrics"""
        # Get degrees
        if self.G.is_directed():
            in_degrees = dict(self.G.in_degree())
            out_degrees = dict(self.G.out_degree())
            in_degree_values = list(in_degrees.values())
            out_degree_values = list(out_degrees.values())
            total_degrees = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) 
                            for node in set(in_degrees) | set(out_degrees)}
            degree_values = list(total_degrees.values())
        else:
            degrees = dict(self.G.degree())
            degree_values = list(degrees.values())
            in_degree_values = degree_values
            out_degree_values = degree_values
        
        # Calculate metrics
        metrics = {
            'max_degree': max(degree_values) if degree_values else 0,
            'min_degree': min(degree_values) if degree_values else 0,
            'avg_degree': np.mean(degree_values) if degree_values else 0,
            'median_degree': np.median(degree_values) if degree_values else 0,
            'std_degree': np.std(degree_values) if degree_values else 0,
        }
        
        # Add in/out degree metrics for directed graphs
        if self.G.is_directed():
            metrics.update({
                'max_in_degree': max(in_degree_values) if in_degree_values else 0,
                'min_in_degree': min(in_degree_values) if in_degree_values else 0,
                'avg_in_degree': np.mean(in_degree_values) if in_degree_values else 0,
                'max_out_degree': max(out_degree_values) if out_degree_values else 0,
                'min_out_degree': min(out_degree_values) if out_degree_values else 0,
                'avg_out_degree': np.mean(out_degree_values) if out_degree_values else 0,
            })
        
        # Degree histogram
        hist, bin_edges = np.histogram(degree_values, bins=min(50, max(degree_values) + 1 if degree_values else 1))
        metrics['degree_histogram'] = list(hist)
        metrics['degree_bins'] = list(bin_edges)
        
        return metrics
    
    def calculate_centrality_metrics(self):
        """Calculate various centrality measures"""
        metrics = {}
        
        # If graph is very large, skip some computationally expensive metrics
        is_large = self.G.number_of_nodes() > 5000
        
        # Calculate degree centrality (normalized)
        if self.G.is_directed():
            in_degree_centrality = nx.in_degree_centrality(self.G)
            out_degree_centrality = nx.out_degree_centrality(self.G)
            metrics['max_in_degree_centrality'] = max(in_degree_centrality.values()) if in_degree_centrality else 0
            metrics['max_out_degree_centrality'] = max(out_degree_centrality.values()) if out_degree_centrality else 0
            metrics['avg_in_degree_centrality'] = np.mean(list(in_degree_centrality.values())) if in_degree_centrality else 0
            metrics['avg_out_degree_centrality'] = np.mean(list(out_degree_centrality.values())) if out_degree_centrality else 0
        else:
            degree_centrality = nx.degree_centrality(self.G)
            metrics['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values())) if degree_centrality else 0
        
        # Calculate betweenness centrality (can be very slow for large graphs)
        if not is_large:
            try:
                betweenness = nx.betweenness_centrality(self.G, k=min(100, self.G.number_of_nodes()))
                metrics['max_betweenness'] = max(betweenness.values()) if betweenness else 0
                metrics['avg_betweenness'] = np.mean(list(betweenness.values())) if betweenness else 0
            except:
                print("WARNING: Betweenness centrality calculation failed")
                metrics['max_betweenness'] = "calculation_failed"
                metrics['avg_betweenness'] = "calculation_failed"
        else:
            print("Skipping betweenness centrality calculation due to graph size")
            metrics['max_betweenness'] = "skipped_large_graph"
            metrics['avg_betweenness'] = "skipped_large_graph"
        
        # Calculate closeness centrality (can be slow)
        if not is_large:
            try:
                closeness = nx.closeness_centrality(self.G)
                metrics['max_closeness'] = max(closeness.values()) if closeness else 0
                metrics['avg_closeness'] = np.mean(list(closeness.values())) if closeness else 0
            except:
                print("WARNING: Closeness centrality calculation failed")
                metrics['max_closeness'] = "calculation_failed"
                metrics['avg_closeness'] = "calculation_failed"
        else:
            print("Skipping closeness centrality calculation due to graph size")
            metrics['max_closeness'] = "skipped_large_graph"
            metrics['avg_closeness'] = "skipped_large_graph"
        
        return metrics
    
    def calculate_path_metrics(self):
        """Calculate path-related metrics"""
        metrics = {}
        
        # If graph is very large, only calculate on a sample
        is_large = self.G.number_of_nodes() > 1000
        
        # If the graph is directed, consider using the undirected version for some metrics
        if self.G.is_directed():
            G_undirected = self.G.to_undirected()
        else:
            G_undirected = self.G
        
        # Find largest connected component for path calculations
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        largest_cc_subgraph = G_undirected.subgraph(largest_cc).copy()
        
        # Log the component sizes
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_ratio'] = len(largest_cc) / self.G.number_of_nodes()
        
        # Calculate average shortest path length in largest component
        if not is_large:
            try:
                avg_path = nx.average_shortest_path_length(largest_cc_subgraph)
                metrics['avg_shortest_path_length'] = avg_path
            except:
                print("WARNING: Average shortest path calculation failed")
                metrics['avg_shortest_path_length'] = "calculation_failed"
        else:
            # Sample nodes for approximation
            sample_size = min(100, len(largest_cc))
            sampled_nodes = np.random.choice(list(largest_cc), size=sample_size, replace=False)
            path_lengths = []
            
            for source in sampled_nodes:
                for target in sampled_nodes:
                    if source != target:
                        try:
                            path_lengths.append(nx.shortest_path_length(largest_cc_subgraph, source, target))
                        except:
                            pass
            
            if path_lengths:
                metrics['avg_shortest_path_length_sampled'] = np.mean(path_lengths)
                metrics['max_shortest_path_length_sampled'] = max(path_lengths)
            else:
                metrics['avg_shortest_path_length_sampled'] = "calculation_failed"
        
        # Calculate diameter (longest shortest path) - can be very slow for large graphs
        if len(largest_cc) < 500:
            try:
                diameter = nx.diameter(largest_cc_subgraph)
                metrics['diameter'] = diameter
            except:
                print("WARNING: Diameter calculation failed")
                metrics['diameter'] = "calculation_failed"
        else:
            metrics['diameter'] = "skipped_large_graph"
            print("Skipping exact diameter calculation due to graph size")
        
        return metrics
    
    def calculate_layer_metrics(self):
        """Calculate metrics for each layer"""
        layer_metrics = {}
        
        for layer in self.layers:
            nodes = self.layer_nodes[layer]
            if not nodes:
                continue
                
            # Calculate subgraph of this layer
            layer_subgraph = self.G.subgraph(nodes).copy()
            
            # Basic metrics for this layer
            layer_metrics[layer] = {
                'num_nodes': len(nodes),
                'num_internal_edges': layer_subgraph.number_of_edges(),
                'internal_density': nx.density(layer_subgraph),
            }
            
            # Count connections to other layers
            outgoing_connections = defaultdict(int)
            incoming_connections = defaultdict(int)
            
            for node in nodes:
                # Check outgoing connections
                for successor in self.G.successors(node):
                    successor_layer = self.G.nodes[successor]['layer']
                    if successor_layer != layer:
                        outgoing_connections[successor_layer] += 1
                
                # Check incoming connections
                for predecessor in self.G.predecessors(node):
                    predecessor_layer = self.G.nodes[predecessor]['layer']
                    if predecessor_layer != layer:
                        incoming_connections[predecessor_layer] += 1
            
            layer_metrics[layer]['outgoing_connections'] = dict(outgoing_connections)
            layer_metrics[layer]['incoming_connections'] = dict(incoming_connections)
            layer_metrics[layer]['total_outgoing'] = sum(outgoing_connections.values())
            layer_metrics[layer]['total_incoming'] = sum(incoming_connections.values())
        
        return layer_metrics
    
    def save_metrics_as_text(self, metrics):
        """Save metrics in a human-readable text format"""
        output_path = os.path.join(self.output_dir, f"{self.model_name}_metrics.txt")
        
        with open(output_path, 'w') as f:
            f.write(f"Graph Metrics for {self.model_name}\n")
            f.write("="*50 + "\n\n")
            
            # Basic stats
            f.write("Basic Statistics\n")
            f.write("-"*50 + "\n")
            for k, v in metrics['basic_stats'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            # Degree metrics
            f.write("Degree Metrics\n")
            f.write("-"*50 + "\n")
            for k, v in metrics['degree_metrics'].items():
                if k not in ['degree_histogram', 'degree_bins']:
                    f.write(f"{k}: {v}\n")
            f.write("\n")
            
            # Centrality metrics
            f.write("Centrality Metrics\n")
            f.write("-"*50 + "\n")
            for k, v in metrics['centrality_metrics'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            # Path metrics
            f.write("Path Metrics\n")
            f.write("-"*50 + "\n")
            for k, v in metrics['path_metrics'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            # Layer metrics
            f.write("Layer Metrics\n")
            f.write("-"*50 + "\n")
            for layer, layer_data in metrics['layer_metrics'].items():
                f.write(f"\n  {layer}:\n")
                for k, v in layer_data.items():
                    if k not in ['outgoing_connections', 'incoming_connections']:
                        f.write(f"    {k}: {v}\n")
                
                f.write("    Outgoing connections to:\n")
                for target_layer, count in layer_data.get('outgoing_connections', {}).items():
                    f.write(f"      {target_layer}: {count}\n")
                
                f.write("    Incoming connections from:\n")
                for source_layer, count in layer_data.get('incoming_connections', {}).items():
                    f.write(f"      {source_layer}: {count}\n")
            
        print(f"Metrics saved to {output_path}")
    
    def visualize_metrics(self, metrics):
        """Create visualizations for key metrics"""
        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Degree distribution
        self._plot_degree_distribution(metrics['degree_metrics'], vis_dir)
        
        # Layer connectivity
        self._plot_layer_connectivity(metrics['layer_metrics'], vis_dir)
        
        # Node importance
        self._plot_node_importance(vis_dir)
    
    def _plot_degree_distribution(self, degree_metrics, output_dir):
        """Plot degree distribution"""
        plt.figure(figsize=(10, 6))
        
        if 'degree_histogram' in degree_metrics and 'degree_bins' in degree_metrics:
            hist = degree_metrics['degree_histogram']
            bins = degree_metrics['degree_bins']
            
            # Plot histogram
            plt.bar(bins[:-1], hist, width=max(1, (bins[1]-bins[0])*0.8), alpha=0.7)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('Degree (log scale)')
            plt.ylabel('Count (log scale)')
            plt.title(f'Degree Distribution for {self.model_name}')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.model_name}_degree_distribution.png"), dpi=300)
            plt.close()
    
    def _plot_layer_connectivity(self, layer_metrics, output_dir):
        """Plot layer connectivity as a heatmap"""
        # Create matrices for layer connectivity
        n_layers = len(self.layers)
        layer_to_idx = {layer: i for i, layer in enumerate(self.layers)}
        
        # Initialize connectivity matrix
        connectivity = np.zeros((n_layers, n_layers))
        
        # Fill connectivity matrix
        for layer, metrics in layer_metrics.items():
            if layer in layer_to_idx:
                src_idx = layer_to_idx[layer]
                
                # Add outgoing connections
                for target_layer, count in metrics.get('outgoing_connections', {}).items():
                    if target_layer in layer_to_idx:
                        tgt_idx = layer_to_idx[target_layer]
                        connectivity[src_idx, tgt_idx] = count
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(connectivity, cmap='viridis')
        plt.colorbar(label='Number of connections')
        plt.title(f'Layer Connectivity for {self.model_name}')
        plt.xlabel('Target Layer')
        plt.ylabel('Source Layer')
        
        # Add layer labels
        layer_labels = [l[:15] + '...' if len(l) > 15 else l for l in self.layers]
        plt.xticks(range(n_layers), layer_labels, rotation=90)
        plt.yticks(range(n_layers), layer_labels)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_layer_connectivity.png"), dpi=300)
        plt.close()
    
    def _plot_node_importance(self, output_dir):
        """Plot node importance metrics"""
        # Calculate degree centrality for all nodes
        if self.G.is_directed():
            centrality = nx.in_degree_centrality(self.G)
        else:
            centrality = nx.degree_centrality(self.G)
        
        # Group by layer
        layer_centrality = defaultdict(list)
        for node, cent in centrality.items():
            if node in self.G.nodes and 'layer' in self.G.nodes[node]:
                layer = self.G.nodes[node]['layer']
                layer_centrality[layer].append(cent)
        
        # Calculate statistics for each layer
        layer_stats = {}
        for layer, values in layer_centrality.items():
            if values:
                layer_stats[layer] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'max': max(values),
                    'min': min(values),
                    'std': np.std(values)
                }
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Sort layers based on network flow
        layers = sorted(layer_stats.keys(), 
                        key=lambda x: 0 if 'input' in x else 
                                     1 if 'conv' in x.lower() else 
                                     2 if 'fc' in x.lower() or 'linear' in x.lower() else 3)
        
        # Prepare data for plotting
        means = [layer_stats[l]['mean'] for l in layers]
        stds = [layer_stats[l]['std'] for l in layers]
        
        # Plot with error bars
        plt.errorbar(range(len(layers)), means, yerr=stds, fmt='o-', capsize=5, 
                    markersize=8, linewidth=2, elinewidth=1)
        
        plt.xlabel('Layer')
        plt.ylabel('Average Degree Centrality')
        plt.title(f'Node Importance by Layer for {self.model_name}')
        plt.xticks(range(len(layers)), [l[:15] + '...' if len(l) > 15 else l for l in layers], rotation=90)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_node_importance.png"), dpi=300)
        plt.close()


def calculate_graph_metrics(graph_path):
    """Calculate metrics for the specified graph file"""
    calculator = GraphMetricsCalculator(graph_path)
    metrics = calculator.calculate_all_metrics()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics for a neural network graph')
    parser.add_argument('--graph_path', type=str, required=True,
                        help='Path to the graph pickle file')
    args = parser.parse_args()
    
    # Calculate metrics
    metrics = calculate_graph_metrics(args.graph_path)
    
    print("Graph metrics calculation complete!")
    print(f"Results saved to ../graph_metrics/ directory") 