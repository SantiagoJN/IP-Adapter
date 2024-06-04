import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchviz import make_dot
import hiddenlayer as hl

transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.

iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
iris.head()

X = torch.tensor(iris.drop("variety", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if vty == "Setosa" else 1 if vty == "Versicolor" else 2 for vty in iris["variety"]], 
    dtype=torch.long
)

print(X[:3])
print()
print(y[:3])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=4, out_features=16)
        self.hidden_1 = nn.Linear(in_features=16, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=3)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        return self.output(x)
    
    
model = Net()
print(model)

y = model(X)
make_dot(y.mean(), params=dict(model.named_parameters())).render("rnn_torchviz", format="png")

yhat = model(X) # Give dummy batch to forward().

# graph = hl.build_graph(model, X, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('rnn_hiddenlayer', format='png')


input_names = ['Sentence']
output_names = ['yhat']
torch.onnx.export(model, X, 'rnn.onnx', input_names=input_names, output_names=output_names)
# This generates a rnn.onnx file that can be opened with the Netron application
