import torch
from torch import nn

class Freddie(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

freddie = Freddie()
x = torch.tensor(1.0)
output = freddie(x)
print(output)