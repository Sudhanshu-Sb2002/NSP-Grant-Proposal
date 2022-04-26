from Input_function import import_data1
import numpy as np
import matplotlib.pyplot as plt
#from NeuralNetNSP import Net
#%%
path="D:\\OneDrive - Indian Institute of Science\\4th Sem\\NSP\\NSP-Grant-Proposal\\MATLAB-COde\\forPython.mat"
[train_attributes,train_labels,test_attributes,test_labels]=import_data1(path)
#jupyter nbextension enable varInspector/main

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr

class Net(nn.Module):

    def __init__(self):
        '''
        n0 = 11
        n1 = 7
        n2 = 4
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
        # print(self.fc3.weight) '''
        n0 = 11
        n1 = 5
        n2=1
        super(Net, self).__init__()  # calling base class
        self.fc1 = nn.Linear(n0, n1,bias=True)  # input is 11x1
        # self.init_weights(self.fc1)
        nn.init.xavier_uniform_(self.fc1.weight)
        # print(self.fc1.weight)
        self.fc2 = nn.Linear(n1, n2,bias=False)
        # self.init_weights(self.fc2)
        nn.init.xavier_uniform_(self.fc1.weight)
        # print(self.fc2.weight)

        # print(self.fc3.weight)

    '''def init_weights(self,m):
        if isinstance(m):
            nn.init.xavier_uniform(m.weight)

    '''
    def forward(self, x):
        '''x = F.relu(self.fc1(x))
        # print(x)
        # assert x.size == n1
        x = F.relu(self.fc2(x))
        # print(x)
        x = t.sigmoid(self.fc3(x))  # change t to f if it doesn't work
        # print(x)'''
        x = F.relu(self.fc1(x))
        # print(x)
        # assert x.size == n1 )
        x = t.sigmoid(self.fc2(x))  # change t to f if it doesn't work
        # print(x)
        return x

net = Net()
print(net)
params = list(net.parameters())

print(type(train_labels[0][0]))

optimizer = optim.SGD(net.parameters(), lr=0.005)
num_epochs = 30000
lambda1 = 0.005
lambda3=0
lambda2 = 0

for i in range(num_epochs):
    optimizer.zero_grad()
    inp=t.tensor(train_attributes).float()
    output =net(inp)
    target=t.tensor(train_labels)
    criterion = nn.MSELoss()

    lin_params = t.cat([x.view(-1) for x in net.fc1.parameters()])
    L1_regularisation = lambda1 * t.norm(lin_params, 1)+lambda3 * t.norm(lin_params, 2)

    lin_params2 = t.cat([x.view(-1) for x in net.fc2.parameters()])
    L1_regularisation1 = lambda2 * t.norm(lin_params, 2)
    loss = criterion(output, target)+L1_regularisation + L1_regularisation1

    loss.backward()
    optimizer.step()
    if(i%500==0):
        print(loss.data)

inp=t.tensor(test_attributes).float()
output =net(inp)
target=t.tensor(test_labels)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss.data)
y=np.array(target.data).flatten()
x=np.array(output.data).flatten()
print("correlation ",pearsonr(x,y))
plt.subplot(2,1,1)
plt.plot(x)
plt.plot(y)
plt.subplot(2,1,2)
plt.scatter(x,y,marker='.',s=0.5)
plt.show()

inp=t.tensor(train_attributes).float()
output =net(inp)
target=t.tensor(train_labels)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss.data)
y=np.array(target.data).flatten()
x=np.array(output.data).flatten()
print("correlation ",pearsonr(x,y))
plt.subplot(2,1,1)
plt.plot(x)
plt.plot(y)
plt.subplot(2,1,2)
plt.scatter(x,y,marker='.',s=0.5)
plt.show()



