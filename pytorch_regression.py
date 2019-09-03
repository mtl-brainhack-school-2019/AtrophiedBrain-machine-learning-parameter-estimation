import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from math import sqrt

# Setting the seed or random_state for reproducibility
torch.manual_seed(143)

# create dummy data for training

# cognition = f(t,u,v,w,x,y,z) = 2.1*t/140 + 3.6*u/70 + 2.8*v/7 + 5*w/80 + 1.4*x/5 - 2.5*y/12 + 4*z/40
#   t = bp_sys = systolic blood pressure (mean=140, std dev=20)
#   u = bp_dia = diastolic blood pressure (mean=70, std dev=10)
#   v = hba1c = glycosylated Hb (mean=7, std dev=2)
#   w = age = age (mean=80, std dev=10)
#   x = ldl = LDL (mean=5, std dev=2)
#   y = edu = years of education (mean=12, std dev=2)
#   z = sed_rate = sedimentation rate (mean=40, std dev=20)

num_subjects = 1000

bp_sys = np.floor(np.random.normal(140.0, 20.0, num_subjects))
bp_dia = np.floor(np.random.normal(70.0, 10.0, num_subjects))
hba1c = np.random.normal(7.0, 2.0, num_subjects)
age = np.floor(np.random.normal(80.0, 10.0, num_subjects))
ldl = np.random.normal(5.0, 2.0, num_subjects)
edu = np.floor(np.random.normal(12.0, 2.0, num_subjects))
sed_rate = np.floor(np.random.normal(40.0, 20.0, num_subjects))

d = {'bp_sys': bp_sys,
     'bp_dia': bp_dia,
     'hba1c': hba1c,
     'age': age,
     'ldl': ldl,
     'edu': edu,
     'sed_rate': sed_rate}

mu, sigma = 0.0, 1.0
bp_sys_noise = bp_sys + np.random.normal(loc=mu, scale=sigma, size=len(bp_sys))
bp_dia_noise = bp_dia + np.random.normal(loc=mu, scale=sigma, size=len(bp_dia))
hba1c_noise = hba1c + np.random.normal(loc=mu, scale=sigma, size=len(hba1c))
age_noise = age + np.random.normal(loc=mu, scale=sigma, size=len(age))
ldl_noise = ldl + np.random.normal(loc=mu, scale=sigma, size=len(ldl))
edu_noise = edu + np.random.normal(loc=mu, scale=sigma, size=len(edu))
sed_rate_noise = sed_rate + np.random.normal(loc=mu, scale=sigma, size=len(sed_rate))

d_noise = {'bp_sys': bp_sys_noise,
           'bp_dia': bp_dia_noise,
           'hba1c': hba1c_noise,
           'age': age_noise,
           'ldl': ldl_noise,
           'edu': edu_noise,
           'sed_rate': sed_rate_noise}

df = pd.DataFrame(data=d)
df_n = pd.DataFrame(data=d_noise)

x_values = np.array(df.values)
x_train = np.array(df_n.values, dtype=np.float32)
x_train = x_train.reshape(-1, 7)

coeffs = (2.1/140.0, 3.6/70.0, 2.8/7.0, 5.0/80.0, 1.4/5.0, -2.5/12.0, 4.0/40.0)
y_values = [coeffs[0]*t + coeffs[1]*u + coeffs[2]*v + coeffs[3]*w + coeffs[4]*x + coeffs[5]*y + coeffs[6]*z
            for t, u, v, w, x, y, z in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)

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
        # self.linear1 = torch.nn.Linear(input_size, input_size*size_multiplier)
        # self.act1 = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(input_size*size_multiplier, input_size)
        # self.act2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(input_size*size_multiplier, input_size)
        # self.act3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        # x = self.linear1(x)
        # x = self.act1(x)
        # x = self.linear2(x)
        # x = self.act2(x)
        # # x = self.linear3(x)
        # # x = self.act3(x)
        x = self.linear4(x)
        return x


inputDim = 7        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 1e-5
epochs = 10000

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

rms = sqrt(mean_squared_error(y_train, predicted))
print("RMSE: ", rms)

print(df.head())

[w, b] = model.parameters()

print(coeffs)
print(w, b)

plt.clf()
# plt.plot(x_train[:, 1], y_train, 'ro', label='Noise data', alpha=0.5)
# plt.plot(x_values[:, 1], y_values, 'go', label='True data', alpha=0.5)
# plt.plot(x_train[:, 1], predicted, 'bx', label='Predictions', alpha=0.5)
plt.plot(x_train, y_train, 'ro', label='Noise data', alpha=0.5)
plt.plot(x_values, y_values, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, 'bx', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
