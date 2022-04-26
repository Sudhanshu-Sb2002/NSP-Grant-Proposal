import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        n0 = 6
        n1 = 4
        n2 = 2
        super(Net, self).__init__()  # calling base class
        self.fc1 = nn.Linear(n0, n1)  # input is 11x1
        # self.init_weights(self.fc1)
        nn.init.xavier_uniform_(self.fc1.weight)
        # print(self.fc1.weight)
        self.fc2 = nn.Linear(n1, n2)
        # self.init_weights(self.fc2)
        nn.init.xavier_uniform_(self.fc1.weight)
        # print(self.fc2.weight)
        self.fc3 = nn.Linear(n2, 1)
        # self.init_weights(self.fc3)
        nn.init.xavier_uniform_(self.fc1.weight)
        # print(self.fc3.weight)

    '''def init_weights(self,m):
        if isinstance(m):
            nn.init.xavier_uniform(m.weight)
            
    '''

    def forward(self, x):
        x = f.relu(self.fc1(x))
        # print(x)
        # assert x.size == n1
        x = f.relu(self.fc2(x))
        # print(x)
        x = t.sigmoid(self.fc3(x))  # change t to f if it doesn't work
        # print(x)
        return x

    def main(self, inp, target):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        # print(net)
        for i in range(len(inp)):
            # print(input)
            out = self[inp[i, :]]
            # print(out)
            #target = t.randn(1)
            # print(target)
            target = target.view(1, -1)
            criterion = nn.MSELoss()
            loss = criterion(out, target[i, 1])
            print(loss)
            loss.backward()
            optimizer.step()
