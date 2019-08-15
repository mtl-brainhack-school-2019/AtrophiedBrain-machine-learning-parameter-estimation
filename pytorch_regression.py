import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

# create dummy data for training
x_values = [(i*1.0, 2.0*i, i/2.0) for i in range(40)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 3)

noise_scaling_factor = 5
noise_mean = 0.0
x_train[:, 0] += np.random.normal(loc=noise_mean, scale=np.std(x_train[:, 0]), size=len(x_train[:, 0]))/noise_scaling_factor
x_train[:, 1] += np.random.normal(loc=noise_mean, scale=np.std(x_train[:, 1]), size=len(x_train[:, 1]))/noise_scaling_factor
x_train[:, 2] += np.random.normal(loc=noise_mean, scale=np.std(x_train[:, 2]), size=len(x_train[:, 2]))/noise_scaling_factor

y_values = [2.0*i + 3.0*j - 0.7*k + 1.0 for i, j, k in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.5, std=0.5)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        out = self.linear(x)
        return out


class SimpleNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        size_multiplier = 1
        self.linear1 = torch.nn.Linear(input_size, input_size*size_multiplier)
        self.act1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_size*size_multiplier, input_size)
        self.act2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(input_size*size_multiplier, input_size)
        # self.act3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        # x = self.linear3(x)
        # x = self.act3(x)
        x = self.linear4(x)
        return x


inputDim = 3        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 1e-5
epochs = 4000

model = SimpleNet(inputDim, outputDim)

if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate) # torch.optim.Adam(model.parameters())   #

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'ro', label='Noise data', alpha=0.5)
plt.plot(x_values, y_values, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
