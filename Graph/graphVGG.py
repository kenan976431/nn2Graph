import torch
import sys
sys.path.append('../')
from Model.VGG import VGG16
from Graph.graphBuilder import GraphBuilder


if __name__ == "__main__":
    # load model
    model = VGG16()
    graph_model = GraphBuilder(model, "vgg16")
    
    # construct graph
    random_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape
    graph_model.build_graph(random_input)
    
    # save graph
    graph_model.save_graph()