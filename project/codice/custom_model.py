from discopy.tensor import Diagram, Dim, Id, Tensor, Ty
import numpy as np
import torch
from lambeq import PytorchModel

class TensorModel(PytorchModel):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: list[Diagram]) -> torch.Tensor:
        max_dim = max(len(diagram) for diagram in x)
        normalized_diagrams = []
        for diagram in x:
            num_added_nodes = max_dim - len(diagram)
            new_dims = []
            new_tensor = torch.Tensor(*new_dims)
            normalized_diagram = diagram @ new_tensor
            normalized_diagrams.append(normalized_diagram)
        
        return super().forward(normalized_diagrams)