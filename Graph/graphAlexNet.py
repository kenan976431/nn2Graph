import torch
import sys
import argparse
import os
import torchvision
import torchvision.transforms as transforms
sys.path.append('../')
from Model.AlexNet import AlexNet
from Graph.graphBuilder import GraphBuilder
from Backdoor.BadNets import add_trigger


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create graph representation of AlexNet')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--edge_threshold', type=float, default=0.95,
                        help='Threshold for edge pruning (0-1)')
    parser.add_argument('--output_dir', type=str, default='../save_graphs',
                        help='Directory to save the graph')
    parser.add_argument('--trigger_type', type=str, default='block',
                        choices=['block', 'cross', 'center_square', 'noise'],
                        help='Type of backdoor trigger to apply')
    parser.add_argument('--trigger_color', type=str, default='white',
                        choices=['red', 'green', 'blue', 'white', 'black'],
                        help='Color of backdoor trigger')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for backdoor attacks')
    parser.add_argument('--clean_graph', action='store_true',
                        help='Also create graph with clean input')
    args = parser.parse_args()
    
    # Prepare data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # Get one sample image from each class
    class_samples = {}
    for data, target in testloader:
        if target.item() not in class_samples:
            class_samples[target.item()] = data
        if len(class_samples) == 10:
            break
    
    # Prepare clean and backdoored inputs
    clean_input = class_samples[args.target_class]  # Use target class for both
    backdoor_input = add_trigger(clean_input.clone(), args.trigger_type, args.trigger_color)
    
    # Extract model type and backdoor info from weight path
    model_info = "alexnet"
    is_backdoored = "benign"
    
    if args.weight_path:
        # Extract information from the weight path
        filename = os.path.basename(args.weight_path)
        if "backdoor" in args.weight_path.lower():
            is_backdoored = "backdoored"
        # Extract additional model details if available in filename
        if "_" in filename:
            model_parts = filename.split('_')
            if len(model_parts) >= 2:
                model_info = model_parts[0]  # e.g., alexnet
    
    # Load model
    model = AlexNet()
    
    # Load trained weights if provided
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()  # Set model to evaluation mode
    
    # Create graph builder with edge pruning and filtering options
    graph_model = GraphBuilder(
        model=model, 
        model_name=f"{model_info}_{is_backdoored}", 
        edge_threshold=args.edge_threshold,  
        ignore_activations=True,  
        ignore_pooling=True  
    )
    
    # Print available layers to select from
    available_layers = graph_model.print_layer_info()
    
    # Interactive layer selection
    print("\nEnter the indices of layers to include (comma separated, e.g. '0,2,5')")
    print("Leave empty to include all Conv/FC layers")
    
    selection = input("Layer indices: ").strip()
    selected_layers = "all"
    
    if selection:
        # Parse user selection
        selected_indices = [int(i) for i in selection.split(',')]
        graph_model.set_target_layers(selected_indices)
        selected_layers = selection.replace(',', '-')
    else:
        # Use all Conv/FC layers
        graph_model.set_target_layers()
    
    # Create directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Building and saving backdoor graph
    backdoor_graph_model = GraphBuilder(
        model=model, 
        model_name=f"{model_info}_{is_backdoored}_backdoor_input_thresh{args.edge_threshold}_layers{selected_layers}_{timestamp}", 
        edge_threshold=args.edge_threshold,
        ignore_activations=True,
        ignore_pooling=True
    )
    
    if selection:
        backdoor_graph_model.set_target_layers(selected_indices)
    else:
        backdoor_graph_model.set_target_layers()
    
    print("\nBuilding graph with backdoored input...")
    backdoor_graph_model.build_graph(backdoor_input)
    backdoor_graph_model.save_graph(save_dir=args.output_dir)
    
    # Optionally build and save clean graph
    if args.clean_graph:
        clean_graph_model = GraphBuilder(
            model=model, 
            model_name=f"{model_info}_{is_backdoored}_clean_input_thresh{args.edge_threshold}_layers{selected_layers}_{timestamp}", 
            edge_threshold=args.edge_threshold,
            ignore_activations=True,
            ignore_pooling=True
        )
        
        if selection:
            clean_graph_model.set_target_layers(selected_indices)
        else:
            clean_graph_model.set_target_layers()
        
        print("\nBuilding graph with clean input...")
        clean_graph_model.build_graph(clean_input)
        clean_graph_model.save_graph(save_dir=args.output_dir)
    
    print(f"\nGraph(s) saved to {args.output_dir}")