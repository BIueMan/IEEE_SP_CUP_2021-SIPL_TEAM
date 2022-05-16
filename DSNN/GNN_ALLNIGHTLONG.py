import numpy as np
import math
import torch

# import matplotlib.pyplot as plt
import scipy.io as sc
from scipy.io import savemat
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def rate(B, N0, K, M, V_r, V_i, t_r, t_i, hd_r, hd_i):
    y_r = V_r @ t_r + hd_r
    y_j = V_i @ t_r + hd_i
    y = y_r ** 2 + y_j ** 2
    return -1 * (B / (K + M - 1)) * torch.sum(torch.log2(1 + y / N0), 0)


def grad(B, N0, K, M, V, theta, hd):
    temp = N0 * np.ones(K, 1) + np.power(np.abs(V.dot(theta) + hd), 2)
    temp2 = 2 * np.divide((V.dot(theta) + hd), temp)
    return -1 * (B / (np.log(2) * (K + M - 1))).dot(np.conjugate(V.T).dot(temp2))


"""#################"""
""" MAIN start here """
"""#################"""

""" set device """
print("torch version: ", torch.__version__)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("cuda version: ", torch.version.cuda)
    device = torch.device('cuda:0')
    print("running on GPU")
else:
    print("no cuda available")
    device = torch.device('cpu')
    print("running on CPU")

""" set vectors on device """
hd_real = torch.from_numpy(
    sc.loadmat('/home/tomerf/Downloads/OneDrive_2021-05-07/h_denoised_real.mat', mat_dtype=True)['hd_real']).float().to(device)
hd_imag = torch.from_numpy(
    sc.loadmat('/home/tomerf/Downloads/OneDrive_2021-05-07/h_denoised_imag.mat', mat_dtype=True)['hd_imag']).float().to(device)
V_real = torch.from_numpy(
    sc.loadmat('/home/tomerf/Downloads/OneDrive_2021-05-07/V_denoised_real.mat', mat_dtype=True)['V_real']).float().to(device)
V_imag = torch.from_numpy(
    sc.loadmat('/home/tomerf/Downloads/OneDrive_2021-05-07/V_denoised_imag.mat', mat_dtype=True)['V_imag']).float().to(device)


SEED_LIST = (566, 129, 3455, 6562, 1287)
USER_LIST = (11-1, 16-1, 30-1, 32-1, 33-1, 35-1, 38-1, 45-1, 9-1, 10-1, 13-1) # USER -1 (Python is shit)

EPOCHS_RAND_START_CONFIGE = 10
EPOCHS_CONFING = 8000
LEARNING_RATE_DNN = 1e-2
LEARNING_RATE_CNN = 5**-2
EPOCHS_CONV = 20000
if __name__ == "__main__":
    B = 10e6
    N0 = 3.077e-13
    K = 500
    M = 20
    N = 4096
    nbrOfUsers = 50

    # loop for every seed
    for seed in SEED_LIST:
        print("start seed: ", seed)

        # reset for the next seed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.manual_seed(seed)

        param_to_save = {}

        # loop for every user in list
        for userToCheck in USER_LIST: # Python is the best
            print("user: ", userToCheck+1)
            best_configs = []
            rates = []
            t_j = torch.zeros((1, 4096)).to(device)

            """ first NN which takes random vector 64*1 and returns 64*1 vector which repeated to 64*64 matrix that reshaped to
             4096*1 configuration"""
            for i in range(EPOCHS_RAND_START_CONFIGE):
                # print("sickle: ", i)
                xr = torch.rand((1, 4096)).float().to(device)
                xj = torch.rand((1, 1)).float().to(device)

                # for net 1
                current_config = (((xr[0].detach() > 0).float() - 0.5) * 2).to(device)
                # for net 2
                # current_config = (((xr[0].detach() > 0).float() - 0.5) * 2).to(device)
                current_rate = (-1 * rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck],
                                          current_config, t_j,hd_real[:, userToCheck],
                                          hd_imag[:, userToCheck])).to(device)

                # Use the nn package to define our model and loss function.
                model_r = torch.nn.Sequential(
                    # """ test net 1 """
                    torch.nn.Linear(4096, 1024, bias=True),
                    torch.nn.Linear(1024, 1024, bias=True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(1024, 4096, bias=True),
                    # torch.nn.Linear(4096, 4096, bias=True),
                    # torch.nn.Linear(64, 64, bias=True),
                    #torch.nn.Tanh(),64

                    # """ test net 2 """
                    # torch.nn.Linear(4096, 4096, bias=True),
                    # torch.nn.Linear(4096, 4096, bias=True),
                ).to(device)
                # loss_fn = torch.nn.MSELoss(reduction='sum')

                # Use the optim package to define an Optimizer that will update the weights of
                # the model for us. Here we will use RMSprop; the optim package contains many other
                # optimization algorithms. The first argument to the RMSprop constructor tells the
                # optimizer which Tensors it should update.
                learning_rate = LEARNING_RATE_DNN
                optimizer_r = torch.optim.Adam(model_r.parameters(), lr=learning_rate, weight_decay=1e-5)
                # optimizer_r = torch.optim.SGD(model_r.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-5)  # lr is min lr
                # scheduler = CosineAnnealingWarmupRestarts(optimizer_r, first_cycle_steps=200, cycle_mult=1.0, max_lr=1e-3,
                #                                          min_lr=1e-5, warmup_steps=50, gamma=0.5)

                for t in range(EPOCHS_CONFING):
                    # Forward pass: compute predicted y by passing x to the model.
                    # for net 1
                    t_r = model_r(xr)[0]
                    # for net 2
                    # t_r = model_r(xr)[0]

                    # Compute and print loss.
                    loss = rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck], t_r, t_j,
                                hd_real[:, userToCheck], hd_imag[:, userToCheck])

                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable
                    # weights of the model). This is because by default, gradients are
                    # accumulated in buffers( i.e, not overwritten) whenever .backward()
                    # is called. Checkout docs of torch.autograd.backward for more details.
                    optimizer_r.zero_grad()

                    # Backward pass: compute gradient of the loss with respect to model
                    # parameters
                    loss.backward()

                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    optimizer_r.step()

                    # print(-1*loss)
                    t_q = ((t_r.detach() > 0).float() - 0.5) * 2

                    loss_q = rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck], t_q, t_j,
                                  hd_real[:, userToCheck], hd_imag[:, userToCheck])
                    if -1 * loss_q > current_rate:
                        current_rate = -1 * loss_q
                        current_config = t_q
                    # print(-1*loss_q)
                best_configs.append(current_config.cpu())
                print(current_rate)
                rates.append(current_rate.cpu())


            #    linear_layer = mode[0]
            #    print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
            # plt.plot(np.sort(rates))
            #plt.show()
            # plt.figure()
            # plt.hist(rates)
            #plt.show()
            # print(np.max(rates))

            conf_max_idx = np.argmax(rates)
            best_configuration = best_configs[conf_max_idx].to(device)
            best_loss = -1*rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck], best_configuration, t_j,
                                  hd_real[:, userToCheck], hd_imag[:, userToCheck])
            # print(best_loss)
            # save params
            name_best_confing = "_".join(["user", str(userToCheck+1), "bestConfing"])
            name_loss = "_".join(["user", str(userToCheck+1), "bestConfingloss"])

            param_to_save[name_best_confing] = best_configuration.cpu().numpy()
            param_to_save[name_loss] = best_loss.cpu().numpy()

            """ set params for conv """
            # list of models conv
            model_r_list = []

            model_r = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=(9, 9), stride=1, padding=(4, 4)),
                torch.nn.Conv2d(32, 64, kernel_size=(9, 9), stride=1, padding=(4, 4)),
                torch.nn.Tanh(),
                torch.nn.Conv2d(64, 32, kernel_size=(9, 9), stride=1, padding=(4, 4)),
                torch.nn.Conv2d(32, 1, kernel_size=(9, 9), stride=1, padding=(4, 4)),
                torch.nn.Tanh()
            ).to(device)




            # set best conf from the conf DSNN
            best_qmat = torch.reshape(best_configuration, (64, 64))

            learning_rate = LEARNING_RATE_CNN
            optimizer_r = torch.optim.Adam(model_r.parameters(), lr=learning_rate, weight_decay=1e-5)

            x = best_qmat.to(device)
            x = x.unsqueeze(0)
            current_CNNconfig = torch.reshape(x, (4096, 1)).squeeze(1)
            current_CNNrate = best_loss
            # print(x[None, ...].size())
            # print(model_r(x[None, ...]).size())
            for t in range(EPOCHS_CONV):
                # Forward pass: compute predicted y by passing x to the model.
                t_r = torch.reshape(model_r(x[None, ...]).squeeze(0)[0], (4096, 1)).squeeze(1)

                # Compute and print loss.-1 *
                loss = rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck], t_r, t_j,
                            hd_real[:, userToCheck], hd_imag[:, userToCheck])

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_r.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_r.step()

                # print(-1*loss)
                t_q = ((t_r.detach() > 0).float() - 0.5) * 2
                t_qmat = torch.reshape(t_q, (64, 64))
                # process_configs[:, t] = t_q.cpu().T
                loss_q = rate(B, N0, K, M, V_real[:, :, userToCheck], V_imag[:, :, userToCheck], t_q, t_j,
                              hd_real[:, userToCheck], hd_imag[:, userToCheck])
                # rates_iter.append(loss_q.cpu().numpy())
                if -1*loss_q > current_CNNrate:
                    current_CNNrate = -1*loss_q
                    current_CNNconfig = t_q

            print("conv_net: ", current_CNNrate)


            # save params for conv part
            print("CNN_LOSS =", current_CNNrate)

            name_best_confing = "_".join(["user", str(userToCheck+1), "bestConv"])
            name_loss = "_".join(["user", str(userToCheck+1), "bestConvloss"])

            param_to_save[name_best_confing] = current_CNNconfig.cpu().numpy()
            param_to_save[name_loss] = current_CNNrate.cpu().numpy()


        """ final save for this seed """
        file_name = "".join(["/home/tomerf/Downloads/OneDrive_2021-05-07/BestConv_seed/bestConfigs_net2_Seed_", str(seed),".mat"])
        savemat(file_name, param_to_save)
        print("finish this seed, saving params")