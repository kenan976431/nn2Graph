import torch.nn as nn
import numpy as np
import networkx as nx
from collections import OrderedDict
import os
import pickle


class GraphBuilder(nn.Module):
    def __init__(self, model, model_name, edge_threshold=0.75, ignore_activations=True, ignore_pooling=True):
        super(GraphBuilder, self).__init__()
        self.model = model
        self.model_name = model_name
        self.graph = nx.DiGraph()  # Using directed graph for better representation
        self.node_counter = 0
        self.layer_mapping = OrderedDict()
        self.edge_threshold = edge_threshold  # Only keep top (1-threshold) edges
        self.ignore_activations = ignore_activations
        self.ignore_pooling = ignore_pooling
        self.target_layers = None  # Will be set if user selects specific layers
        
        # register hooks to capture activations
        self.activations = OrderedDict()
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = {
                    'input': input[0].detach(),
                    'output': output.detach()
                }
            return hook
        
        # Store all layer info before filtering
        self.all_layers = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d, nn.Dropout)):
                module.register_forward_hook(get_activation(name))
                self.all_layers[name] = {
                    'type': type(module).__name__,
                    'module': module
                }

    def print_layer_info(self):
        """Print information about available conv and fc layers to select from"""
        print("\nAvailable layers for graph construction:")
        print("----------------------------------------")
        print("IDX  | TYPE        | NAME")
        print("----------------------------------------")
        
        conv_fc_layers = []
        for idx, (name, info) in enumerate(self.all_layers.items()):
            if isinstance(info['module'], (nn.Conv2d, nn.Linear)):
                layer_type = info['type']
                conv_fc_layers.append((idx, layer_type, name))
                print(f"{idx:<4} | {layer_type:<11} | {name}")
        
        return conv_fc_layers

    def set_target_layers(self, layer_indices=None):
        """Set specific layers to include in the graph by their indices"""
        if layer_indices is None:
            # If no indices provided, include all Conv and Linear layers
            self.target_layers = [name for name, info in self.all_layers.items() 
                                if isinstance(info['module'], (nn.Conv2d, nn.Linear))]
            # Always include input layer
            self.target_layers.insert(0, 'input')
        else:
            # Get layer names from the provided indices
            layers = list(self.all_layers.keys())
            self.target_layers = ['input']  # Always include input layer
            for idx in layer_indices:
                if 0 <= idx < len(layers):
                    self.target_layers.append(layers[idx])
        
        print(f"Selected layers for graph construction: {self.target_layers}")
        
        # Update layer_mapping with only the target layers
        self.layer_mapping = OrderedDict()
        for name in self.target_layers:
            if name == 'input':
                self.layer_mapping[name] = {'type': 'InputLayer'}
            elif name in self.all_layers:
                self.layer_mapping[name] = self.all_layers[name]

    def build_graph(self, input_tensor):
        # If no target layers specified, select all conv and fc layers
        if self.target_layers is None:
            self.set_target_layers()
        
        # Forward pass to get activations
        _ = self.model(input_tensor)
        
        # Manually record input layer activation data
        self.activations['input'] = {
            'output': input_tensor.detach()
        }
        
        # Initialize model structure metadata
        self.graph.graph['model_structure'] = {
            'model_name': self.model_name,
            'input_shape': input_tensor.shape[1:],
            'layers': OrderedDict()
        }
        
        # Add input layer nodes - one per channel
        input_shape = input_tensor.shape[1:]  # C x H x W
        self.layer_mapping['input'] = {
            'type': 'InputLayer',
            'nodes': self._add_feature_map_nodes('input', input_shape[0])
        }
        
        # Process each layer to build graph structure
        prev_layer = 'input'
        for name in self.target_layers:
            if name == 'input':  # Skip input layer we manually added
                continue
                
            layer_info = self.layer_mapping[name]
            layer = layer_info['module']
            
            # Skip activation and pooling layers if specified
            if (self.ignore_activations and isinstance(layer, (nn.ReLU, nn.Dropout))) or \
               (self.ignore_pooling and isinstance(layer, nn.MaxPool2d)):
                continue
                
            # Process layer based on type
            if isinstance(layer, nn.Conv2d):
                self._process_conv_layer(name, layer, prev_layer)
                prev_layer = name
            elif isinstance(layer, nn.Linear):
                self._process_fc_layer(name, layer, prev_layer)
                prev_layer = name
            elif not self.ignore_activations and isinstance(layer, (nn.ReLU, nn.Dropout)):
                self._process_activation_layer(name, layer, prev_layer)
                # Don't update prev_layer for these types
            elif not self.ignore_pooling and isinstance(layer, nn.MaxPool2d):
                self._process_activation_layer(name, layer, prev_layer)
                # Don't update prev_layer for these types
            
            # Record structure metadata
            if name in self.layer_mapping:
                layer_metadata = {
                    'type': layer_info['type'],
                    'nodes': self.layer_mapping[name].get('nodes', [])
                }
                
                # Add layer specific parameters
                if isinstance(layer, nn.Conv2d):
                    layer_metadata.update({
                        'in_channels': layer.in_channels,
                        'out_channels': layer.out_channels,
                        'kernel_size': layer.kernel_size
                    })
                elif isinstance(layer, nn.Linear):
                    layer_metadata.update({
                        'in_features': layer.in_features,
                        'out_features': layer.out_features
                    })
                    
                self.graph.graph['model_structure']['layers'][name] = layer_metadata
        
        # Prune edges to keep only the top n% by weight
        self._prune_edges()

    def _find_previous_significant_layer(self, current_idx):
        """Find the most recent layer that creates nodes (conv/linear)"""
        layer_names = list(self.all_layers.keys())
        current_pos = layer_names.index(current_idx)
        
        # Search backwards for the most recent convolutional or linear layer
        for i in range(current_pos-1, -1, -1):
            layer_name = layer_names[i]
            if layer_name in self.layer_mapping and 'nodes' in self.layer_mapping[layer_name]:
                return layer_name
        
        # If none found, return input layer
        return 'input'

    def _prune_edges(self):
        """Remove edges with weights below the threshold (keeping only top n%)"""
        print("Pruning edges to reduce graph size...")
        
        # Get all edge weights
        edge_weights = []
        for u, v, data in self.graph.edges(data=True):
            if 'weight' in data:
                edge_weights.append(abs(data['weight']))
        
        if not edge_weights:
            print("No weighted edges found to prune.")
            return
            
        # Calculate threshold value (75th percentile)
        threshold = np.percentile(edge_weights, self.edge_threshold * 100)
        print(f"Edge weight threshold: {threshold} (keeping top {(1-self.edge_threshold)*100:.1f}%)")
        
        # Identify edges to remove
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data or abs(data['weight']) < threshold:
                edges_to_remove.append((u, v))
        
        # Remove edges
        self.graph.remove_edges_from(edges_to_remove)
        print(f"Removed {len(edges_to_remove)} edges.")
        print(f"Graph now has {self.graph.number_of_edges()} edges.")

    def _add_feature_map_nodes(self, layer_name, num_channels):
        """Create nodes for feature maps/channels instead of individual neurons"""
        nodes = []
        for c in range(num_channels):
            node_id = self.node_counter
            self.graph.add_node(node_id, 
                              layer=layer_name,
                              channel=c,
                              type='feature_map')
            nodes.append(node_id)
            self.node_counter += 1
        return nodes

    def _add_neuron_nodes(self, layer_name, num_neurons):
        """Create nodes for individual neurons (for FC layers)"""
        nodes = []
        for i in range(num_neurons):
            node_id = self.node_counter
            self.graph.add_node(node_id, 
                              layer=layer_name,
                              neuron_idx=i,
                              type='neuron')
            nodes.append(node_id)
            self.node_counter += 1
        return nodes

    def _process_conv_layer(self, layer_name, layer, prev_layer):
        """Process convolutional layer at feature map level"""
        # Get output shape to determine number of feature maps
        output_shape = self.activations[layer_name]['output'].shape[1:]  # [C, H, W]
        out_channels = output_shape[0]
        
        # Create nodes for feature maps in this layer
        curr_nodes = self._add_feature_map_nodes(layer_name, out_channels)
        self.layer_mapping[layer_name]['nodes'] = curr_nodes
        
        # Get previous layer nodes
        prev_nodes = self.layer_mapping[prev_layer]['nodes']
        
        # For each output channel, connect to all input channels with aggregated weights
        weight = layer.weight.detach().cpu().numpy()  # shape: [out_ch, in_ch, k_h, k_w]
        
        for out_c in range(out_channels):
            out_node = curr_nodes[out_c]
            for in_c in range(layer.in_channels):
                if in_c < len(prev_nodes):  # Ensure we have a corresponding input node
                    in_node = prev_nodes[in_c]
                    # Calculate aggregated weight (sum of absolute filter weights)
                    agg_weight = np.sum(np.abs(weight[out_c, in_c]))
                    if agg_weight > 0:  # Only add significant connections
                        self.graph.add_edge(in_node, out_node, weight=float(agg_weight))
        
        # Add layer info as node attributes
        for node in curr_nodes:
            self.graph.nodes[node].update({
                'layer_type': 'Conv2d',
                'kernel_size': layer.kernel_size,
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels
            })

    def _process_fc_layer(self, layer_name, layer, prev_layer):
        """Process fully connected layer"""
        # Create nodes for neurons in this layer
        num_out_neurons = layer.out_features
        curr_nodes = self._add_neuron_nodes(layer_name, num_out_neurons)
        self.layer_mapping[layer_name]['nodes'] = curr_nodes
        
        # Get previous layer nodes
        prev_nodes = self.layer_mapping[prev_layer]['nodes']
        prev_layer_type = self.layer_mapping[prev_layer]['type']
        
        # Get weight matrix
        weight = layer.weight.detach().cpu().numpy()  # shape: [out_features, in_features]
        
        # Handle transition from conv to FC (need to reshape/adapt)
        if prev_layer_type in ['Conv2d', 'MaxPool2d']:
            # If coming from a conv/pool layer to FC, we need special handling
            # The previous nodes represent feature maps, but FC expects flattened neurons
            prev_activation = self.activations[prev_layer]['output']
            expected_flattened_size = layer.in_features
            
            # Get the number of elements in each feature map
            if len(prev_activation.shape) == 4:  # [B, C, H, W]
                elements_per_channel = prev_activation.shape[2] * prev_activation.shape[3]
            else:
                elements_per_channel = 1
            
            # Connect each output neuron to each input feature map with aggregated weights
            for out_idx, out_node in enumerate(curr_nodes):
                for in_idx, in_node in enumerate(prev_nodes):
                    # Calculate range of flattened elements for this feature map
                    start_idx = in_idx * elements_per_channel
                    end_idx = min(start_idx + elements_per_channel, expected_flattened_size)
                    
                    if start_idx < expected_flattened_size:
                        # Sum weights for all neurons from this feature map
                        agg_weight = 0
                        for i in range(start_idx, end_idx):
                            if i < weight.shape[1]:  # Ensure we don't exceed weight matrix bounds
                                agg_weight += abs(weight[out_idx, i])
                        
                        if agg_weight > 0:
                            self.graph.add_edge(in_node, out_node, weight=float(agg_weight))
        else:
            # Normal FC to FC connection
            for out_idx, out_node in enumerate(curr_nodes):
                for in_idx, in_node in enumerate(prev_nodes):
                    if in_idx < weight.shape[1]:  # Ensure we don't exceed weight matrix bounds
                        weight_value = abs(weight[out_idx, in_idx])
                        if weight_value > 0:
                            self.graph.add_edge(in_node, out_node, weight=float(weight_value))

    def _process_activation_layer(self, layer_name, layer, prev_layer):
        """Process activation layers (ReLU, BatchNorm, etc.)"""
        # Add activation metadata to previous layer nodes
        prev_nodes = self.layer_mapping[prev_layer]['nodes']
        
        # Store layer type
        activation_type = None
        if isinstance(layer, nn.ReLU):
            activation_type = 'ReLU'
        elif isinstance(layer, nn.MaxPool2d):
            activation_type = 'MaxPool2d'
        elif isinstance(layer, nn.BatchNorm2d):
            activation_type = 'BatchNorm2d'
        elif isinstance(layer, nn.Dropout):
            activation_type = 'Dropout'
        
        if activation_type:
            # Add the activation info to all previous nodes
            for node in prev_nodes:
                self.graph.nodes[node][activation_type.lower()] = True
    
    def save_graph(self, save_dir="../save_graphs"):
        """Save the graph to a pickle file"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save graph to pickle file
        with open(f"{save_dir}/{self.model_name}.pkl", "wb") as f:
            pickle.dump(self.graph, f)
        
        # Print some statistics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        print(f"Graph saved with {num_nodes} nodes and {num_edges} edges")