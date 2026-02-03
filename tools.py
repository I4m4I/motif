import torch
import math
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.utils.data import Dataset, DataLoader

def combination(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

class motifRegular:
    def __init__(self, fre, device=torch.device("cuda"), numOfNeuron=512, amplitude=1000, bias=0.05):
        self.L = torch.ones([1, numOfNeuron]).to(device)
        self.I = torch.zeros([numOfNeuron, numOfNeuron]).to(device)
        self.P= torch.zeros([numOfNeuron, numOfNeuron]).to(device)
        self.obs = torch.zeros([14],requires_grad=False).to(device)
        self.fre=torch.zeros([13]).to(device)
        self.sum=combination(numOfNeuron,3)
        self.recordSum=0
        self.amplitude=amplitude
        self.bias=bias
        self.device=device
        for i in range(13):
            self.fre[i]=fre[i]
        for i in range(numOfNeuron):
            self.I[i][i] = 1
        for i in range(numOfNeuron):
            for j in range(numOfNeuron):
                if i==j:
                    continue
                self.P[i][j]=1
    
    def cal(self, a):
        m=torch.mul
        mm=torch.matmul
        a2=a*a
        w=torch.sigmoid(self.amplitude*(a2-self.bias*self.bias))
        w=w*self.P
        pmw=self.P-w
        w0=pmw*pmw.T
        w1=w*pmw.T
        w2=pmw*w.T
        w3=w*w.T

        q=torch.zeros([14]).to(self.device)
        # q[1] = 1/2*self.L@(w1*(w1@w0))@self.L.T
        # q[2] = 1/2*self.L@(w0*(w1@w2))@self.L.T
        # q[3] = self.L@(w1*(w0@w2))@self.L.T
        # q[7] = 1/2*self.L@(w1*(w1@w2))@self.L.T

        # q[4] = self.L@(w3*(w1@w0))@self.L.T
        # q[5] = self.L@(w3*(w2@w0))@self.L.T
        # q[9] = 1/2*self.L@(w3*(w1@w2))@self.L.T
        # q[10] = 1/3*self.L@(w3*(w2@w1))@self.L.T

        # q[6] = self.L@(w3*(w3@w0))@self.L.T
        # q[8] = 1/2*self.L@(w3*(w1@w2))@self.L.T
        # q[11] = self.L@(w3*(w3@w2))@self.L.T
        # q[12] = self.L@(w3*(w3@w2))@self.L.T
        # q[13] = 1/6*self.L@(w3*(w3@w3))@self.L.T

        q[1] = 1/2*self.L@(w1*(w1@w0))@self.L.T
        q[2] = 1/2*self.L@(w0*(w1@w2))@self.L.T
        q[3] = self.L@(w1*(w0@w2))@self.L.T
        q[4] = 1/2*self.L@(w1*(w1@w2))@self.L.T

        q[5] = self.L@(w3*(w1@w0))@self.L.T
        q[6] = self.L@(w3*(w2@w0))@self.L.T
        q[7] = 1/2*self.L@(w3*(w1@w2))@self.L.T
        q[8] = 1/3*self.L@(w3*(w2@w1))@self.L.T

        q[9] = self.L@(w3*(w3@w0))@self.L.T
        q[10] = 1/2*self.L@(w3*(w1@w2))@self.L.T
        q[11] = self.L@(w3*(w3@w2))@self.L.T
        q[12] = self.L@(w3*(w3@w2))@self.L.T
        q[13] = 1/6*self.L@(w3*(w3@w3))@self.L.T
        r=torch.zeros([2]).to(self.device)
        # self.recordSum=torch.sum(q)
        for i in range(13):
            if self.fre[i]>=0:
                r[0]+=(q[i+1]/self.sum-self.fre[i])**2
        with torch.no_grad():
            for i in range(13):
                self.obs[i+1] = q[i+1] / self.sum
        return r[0]

class NDArrayDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def tidigitsLoader(batchSize=16, ):
    f=open("Dataset/packed_tidigits_nbands_20_nframes_20.pkl","rb")
    data = pickle.load(f)

    train_data = data[0][0]
    train_data=np.reshape(train_data,(data[0][0].shape[0], 20, 20))
    train_labels = data[0][1]

    test_data = data[2][0]
    test_data=np.reshape(test_data,(data[2][0].shape[0], 20, 20))
    test_labels = data[2][1]

    train_dataset = NDArrayDataset(np.float32(train_data), np.int64(train_labels))
    test_dataset = NDArrayDataset(np.float32(test_data), np.int64(test_labels))
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
    
    # check
    for batch_data, batch_labels in train_loader:
        print(f'Batch data shape: {batch_data.shape}')
        print(f'Batch labels shape: {batch_labels.shape}')
        break

    return train_loader, test_loader


def discretize_uniform_fast(x, n=128):
    """
    将 x∈[-1,1] 按值域等分为 n 份并量化到各分段中心，返回与 x 同 dtype、同 device。
    """
    if n <= 0:
        raise ValueError("n 必须是正整数")

    # 分段索引（0..n-1），x==1 时通过 clamp 归到最后一段
    k = torch.floor((x + 1) * (n / 2))
    k = torch.clamp(k, 0, n - 1)

    # 段中心：-1 + (k + 0.5) * (2/n)
    return -1 + (k + 0.5) * (2 / n)

# Actor网络
class Actor(nn.Module):
    def __init__(self, N_S, N_A, hidden_size, device):
        super(Actor, self).__init__()
        self.N_S = N_S
        self.hidden_size = hidden_size
        self.device = device
        self.cell = torch.nn.RNNCell(input_size=N_S, hidden_size=self.hidden_size)
        self.w = self.cell.weight_hh
        self.sigma = nn.Linear(self.hidden_size, N_A)
        self.mu = nn.Linear(self.hidden_size, N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal

    # 初始化网络参数
    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s, print_log=False):
        if (
            torch.isnan(self.cell.weight_ih).any()
            or torch.isnan(self.cell.weight_hh).any()
            or torch.isnan(self.cell.bias_ih).any()
            or torch.isnan(self.cell.bias_hh).any()
        ):
            print("NaN detected in RNNCell weights")
            raise ValueError("Actor cell weights contain NaN")
        if print_log:
            print(s.shape)
            print(s)
        zeroInput = torch.zeros([s.shape[0], self.N_S]).to(self.device)
        zeroHidden = torch.zeros([s.shape[0], self.hidden_size]).to(self.device)
        x = self.cell(s, zeroHidden)
        x = self.cell(zeroInput, x)
        x = self.cell(zeroInput, x)
        if print_log:
            print(x.shape)
            print(x)
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        if print_log:
            print(mu.shape)
            print(mu)
            print(log_sigma.shape)
            print(log_sigma)
        # log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu, sigma
    
    def forward_discrete(self, s):
        zeroInput = torch.zeros([s.shape[0], self.N_S]).to(self.device)
        zeroHidden = torch.zeros([s.shape[0], self.hidden_size]).to(self.device)
        x = self.cell(s, zeroHidden)
        x=discretize_uniform_fast(x)
        x = self.cell(zeroInput, x)
        x=discretize_uniform_fast(x)
        x = self.cell(zeroInput, x)
        x=discretize_uniform_fast(x)

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        # log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def choose_action(self, s):
        mu, sigma = self.forward(s)
        Pi = self.distribution(mu, sigma)
        return Pi.sample().detach().cpu().numpy()
    
    def choose_action_discrete(self, s):
        mu, sigma = self.forward_discrete(s)
        Pi = self.distribution(mu, sigma)
        return Pi.sample().detach().cpu().numpy()


# Critic网洛
class Critic(nn.Module):
    def __init__(self, N_S, hidden_size, device):
        super(Critic, self).__init__()
        self.N_S = N_S
        self.hidden_size= hidden_size
        self.device = device
        self.cell = torch.nn.RNNCell(input_size=N_S, hidden_size=self.hidden_size)
        self.w = self.cell.weight_hh
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        zeroInput = torch.zeros([s.shape[0], self.N_S]).to(self.device)
        zeroHidden = torch.zeros([s.shape[0], self.hidden_size]).to(self.device)
        x = self.cell(s, zeroHidden)
        x = self.cell(zeroInput, x)
        x = self.cell(zeroInput, x)
        values = self.fc3(x)
        return values


class Ppo:
    def __init__(self, N_S, N_A, hidden_size, actor_lr, critic_lr, batch_size, epsilon, gamma, lambd, l2_rate, device, max_grad_norm=1.0):
        self.actor_net = Actor(N_S, N_A, hidden_size, device).to(device)
        self.critic_net = Critic(N_S, hidden_size, device).to(device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=critic_lr, weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()
        self.device = device
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm

    def train(self, memory, print_log=False):
        memory = np.array(memory)
        states = torch.tensor(np.vstack(memory[:, 0]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(memory[:, 1]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(list(memory[:, 2]), dtype=torch.float32).to(self.device)
        masks = torch.tensor(list(memory[:, 3]), dtype=torch.float32).to(self.device)

        values = self.critic_net(states)

        returns, advants = self.get_gae(rewards, masks, values)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)

        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n // self.batch_size):
                b_index = arr[self.batch_size * i:self.batch_size * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std = self.actor_net(b_states, print_log=print_log)
                if torch.isnan(mu).any() or torch.isnan(std).any():
                    print("NaN detected in actor outputs")
                    print("mu:", mu)
                    print("std:", std)
                    raise ValueError("NaN detected in actor outputs")
                
                pi = self.actor_net.distribution(mu, std)
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(new_prob - old_prob)

                surrogate_loss = ratio * b_advants
                values = self.critic_net(b_states)

                critic_loss = self.critic_loss_func(values, b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

                clipped_loss = ratio * b_advants

                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)

                self.actor_optim.step()

    # 计算GAE
    def get_gae(self, rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            # 计算A_t并进行加权求和
            running_returns = rewards[t] + self.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + self.gamma * previous_value * masks[t] - \
                              values.data[t]
            running_advants = running_tderror + self.gamma * self.lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        # advants的归一化
        adv_mean = advants.mean()
        adv_std = advants.std()
        if adv_std.item() > 1e-6:
            advants = (advants - adv_mean) / adv_std
        else:
            advants = advants - adv_mean
        return returns, advants


class Nomalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S,))
        self.stdd = np.zeros((N_S,))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            # 更新样本均值和方差
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
            # 状态归一化
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean
        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x