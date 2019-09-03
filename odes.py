import argparse
import os
import numpy as np
import time
import functions as funct
import torch
import torch.nn as nn
import torch.optim as optim
import uuid

REGION_COUNT = 3
TIME_STEP = 0.5
TIMES = np.arange(5, 90, TIME_STEP)

parser = argparse.ArgumentParser('ODE solver')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='rk4')
parser.add_argument('--data_size', type=int, default=len(TIMES))  # analog: number of subjects
parser.add_argument('--batch_time', type=int, default=80)  # width of time window for each batch (in time_step units)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=4000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


uid = str(uuid.uuid4().hex)

a = np.array(funct.random_alpha_negative(REGION_COUNT))
b = np.array(funct.random_beta(REGION_COUNT))
u0 = np.array([3.78, 2.54, 2.87])

estims = funct.estimate(a, b, TIMES, u0, TIME_STEP)
estims_n = np.zeros(estims.shape)

mean = 0.0
std = 0.1
estims_n[:, 0] = estims[:, 0] + np.random.normal(loc=mean, scale=std, size=len(estims_n[:, 0]))
estims_n[:, 1] = estims[:, 1] + np.random.normal(loc=mean, scale=std, size=len(estims_n[:, 1]))
estims_n[:, 2] = estims[:, 2] + np.random.normal(loc=mean, scale=std, size=len(estims_n[:, 2]))

temp = estims  # np.concatenate((estims), axis=1)
true_y = torch.from_numpy(temp)
true_y0 = true_y[0, :]

data_size = len(true_y)
ct_input_size = estims_n.shape[1]
tt = torch.tensor(TIMES)


def get_batch():
    subjects = np.random.choice(np.arange(data_size - args.batch_time, dtype=np.int64),
                                args.batch_size, replace=False)
    subjects.sort()
    s = torch.from_numpy(subjects)
    # s now contains batch_size randomly chosen subjects
    batch_y0 = true_y[s]  # (M, D)
    # batch_y0 now has the starting data for batch_size subjects
    batch_t = tt[:args.batch_time]  # (T)
    # batch_t is the first batch_time points
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # batch_y now has the data for next batch_time years for batch_size # of subjects
    return batch_y0.float(), batch_t, batch_y.float()


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax_traj = fig.add_subplot(111)

    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('CT (cm)')
        ax_traj.plot(tt.numpy(), true_y.numpy(), 'g-')
        ax_traj.plot(tt.numpy(), pred_y.numpy(), 'b--')
        ax_traj.set_xlim(tt.min(), tt.max())

        #ax_traj2.set_ylim(0, 6)
        #ax_traj.legend()

        plt.savefig('png/{:s}_{:03d}'.format(uid, itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.ct_alphas = torch.nn.Parameter(torch.Tensor(ct_input_size).uniform_(-0.0004, -0.0001))   # torch.nn.Parameter(torch.Tensor(ctbes))   #
        self.ct_betas = torch.nn.Parameter(torch.Tensor(ct_input_size).uniform_(-0.00004, 0.00004)) # torch.nn.Parameter(torch.Tensor(ctals))  #

        input_multiplier = 2

        self.net_ct = nn.Sequential(
            nn.Linear(ct_input_size, ct_input_size*input_multiplier),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(ct_input_size*input_multiplier, ct_input_size)
         #   nn.Linear(ct_input_size, ct_input_size)
        )

        for m in self.net_ct.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                #nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
                #m.weight.requires_grad = False
                #m.bias.requires_grad = False

    def forward(self, t, y):
        # return self.net_ct(y).float()
        y_shape = y.shape

        if len(y_shape) > 1:
            sp = torch.split(y, (ct_input_size), -1)

            ctb = self.ct_betas
            cta = self.ct_alphas

            ct_data = sp[0]

            gm_row = ct_data
            at = cta * gm_row
            bt = torch.zeros(gm_row.shape)
            for i in np.arange(0, gm_row.shape[0]):
                #dat = gm_row[i, :]
                #bt[i, :] = ctb[0]*(dat[1] + dat[2]) + ctb[1]*(dat[0]+dat[2]) + ctb[2]*(dat[0]+dat[1])
                m = gm_row[0, :].repeat(REGION_COUNT, 1)
                ind = np.diag_indices(m.shape[0])
                m[ind[0], ind[1]] = torch.zeros(m.shape[0])
                bt[i, :] = torch.sum(ctb * m.sum(-1))

            ct_out = at + bt
            cto = self.net_ct(ct_out).float()
        else:
            sp = torch.split(y, (ct_input_size), -1)

            ctb = self.ct_betas
            cta = self.ct_alphas

            ct_data = sp[0]

            gm_row = ct_data
            at = cta * gm_row

            m = gm_row.repeat(REGION_COUNT, 1)
            ind = np.diag_indices(m.shape[0])
            m[ind[0], ind[1]] = torch.zeros(m.shape[0])
            bt = torch.sum(ctb * m.sum(-1))

            ct_out = at + bt
            cto = self.net_ct(ct_out).float()

        return cto  # torch.cat((cto), -1)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=args.method)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0.float(), tt, method=args.method)
                loss = torch.mean(torch.abs(pred_y - true_y.float()))
                print('Iter {:04d} | Total Loss {:.6f} | Time {:.6f}'.format(itr, loss.item(), time_meter.avg),
                      flush=True)
                visualize(true_y, pred_y, func, ii)
                print(a, func.ct_alphas)
                print(b, func.ct_betas)
                ii += 1
                # path = "func_{:s}_{:04d}.pwf".format(uid, itr)
                # torch.save({
                #     'epoch': itr,
                #     'model_state_dict': func.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss
                # }, path)

        end = time.time()

    last_a = func.ct_alphas.detach().numpy()
    last_b = func.ct_betas.detach().numpy()
    for m in func.net_ct.modules():
        if isinstance(m, nn.Linear):
            print(m.weight)
            print(m.bias)

    last_estims = funct.estimate(last_a, last_b, TIMES, u0, TIME_STEP)

    plt.figure()
    plt.plot(TIMES, estims)
    plt.plot(TIMES, estims_n)
    plt.plot(TIMES, last_estims)
    plt.xlabel('Age')
    plt.ylabel('CT (cm)')
    plt.show()
