import gym
import torch
import numpy as np
import argparse
import os

import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import pickle
from tqdm import tqdm
from tools import *
import configparser
import argparse
import sys
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--fre',nargs=13, default=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0.130366,0.449035,-1])
# parser.add_argument('--fre',nargs=13, default=[0.0017,0.0000,0.0068,0.0963,0.0946,0.0777,0.0034,0.0000,0.0253,0.0684,0.0253,0.1867,0.4139])
# parser.add_argument('--fre',nargs=13, default=[0.3,0.0159, 0.0158, 0.0317, 0.0799, 0.0820, 0.0883, 0.0292, 0.0097, 0.0365, 0.0386, 0.0749, 0.1673, 0.2459])
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--numOfNeuron',type=int, default=512)
parser.add_argument('--batchSize',type=int, default=64)
parser.add_argument('--amplitude',type=float, default=1000)
parser.add_argument('--bias',type=float, default=0.05)
parser.add_argument('--epoch1',type=int, default=0)
parser.add_argument('--env',type=str, default="ip")
parser.add_argument('--diagBool', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--diag',type=float, default=-0.0)
parser.add_argument('--cuda',type=int, default=0)
parser.add_argument('--discrete',type=bool, default=True)
parser.add_argument('--prefix',type=str, default="woPrefix")
args = parser.parse_args()
print(args)
cfg_list = configparser.ConfigParser()
cfg_list.read('params.ini', encoding='utf-8')
cfg=cfg_list[args.env]
env = gym.make(cfg["env_name"])
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
seed=args.seed
torch.manual_seed(seed)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

amplitude=args.amplitude
cuda=args.cuda
bias=args.bias
epoch1=args.epoch1
# 超参数
input_size = 28     # 每个时间步的输入大小
hidden_size = args.numOfNeuron   # RNN 隐藏层大小
output_size = 10    # MNIST 的 10 个类别
num_layers = 1      # RNN层数
batch_size = args.batchSize     # 批次大小
learning_rate1 = 0.001
learning_rate2 = 0.01
fre=[]
for i in range(13):
    fre.append(float(args.fre[i]))

# 如果 motif 目标都是负数，则跳过预训练
if all(v < 0 for v in fre):
    epoch1 = 0

# 检查是否有可用的 GPU
device = torch.device(f'cuda:{cuda}')

##parameters
lr_actor = 0.000003
lr_critic = 0.00003
Iter = int(cfg["epoch2"])
MAX_STEP = 10000
gamma = 0.98
lambd = 0.98
epsilon = 0.2
l2_rate = 0.001

# device = torch.device("cpu")

def saveW(a, name):
    w = np.array(a.cpu().detach().numpy())
    w.reshape(-1)
    np.save(name + ".npy", w)

loss2 = motifRegular(
    fre,
    device=device,
    numOfNeuron=hidden_size,
    amplitude=amplitude,
    bias=bias,
)

ppo = Ppo(
    N_S,
    N_A,
    hidden_size=hidden_size,
    actor_lr=lr_actor,
    critic_lr=lr_critic,
    batch_size=batch_size,
    epsilon=epsilon,
    gamma=gamma,
    lambd=lambd,
    l2_rate=l2_rate,
    device=device,
)
nomalize = Nomalize(N_S)
episodes = 0
eva_episodes = 0
avg_rewards = []
show_episodes = []
learning_rate1 = 0.01  # 设置学习率
optimizer1 = torch.optim.SGD(params=ppo.actor_net.parameters(), lr=learning_rate1, momentum=0.5)
p = True

for i in range(epoch1):
    a = loss2.cal(ppo.actor_net.w)
    motif_fre = loss2.obs
    print("motifloss: " + str(a))

    lossa = 1e5 * a

    optimizer1.zero_grad()
    lossa.backward()
    # torch.nn.utils.clip_grad_norm_(ppo.actor_net.parameters(), 1.0)
    optimizer1.step()

    if i % 20 == 0:
        print(i)
        print(motif_fre)
lll = np.zeros(Iter)
lll_dis = np.zeros(Iter)
for iter in range(Iter):
    memory = deque()
    steps = 0
    while steps < 2048:  # Horizen
        episodes += 1
        s = nomalize(env.reset())
        score = 0
        for _ in range(MAX_STEP):
            steps += 1
            # 选择行为
            act = ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).to(device).unsqueeze(0))[0]
            '''
            if episodes % 50 == 0:
                #env.render()
                print(1)
            '''
            s_, r, done, info = env.step(act)
            s_ = nomalize(s_)

            mask = (1 - done) * 1
            memory.append([s, act, r, mask])

            score += r
            s = s_
            if done:
                break
    
    if args.discrete:
        steps = 0
        while steps < 2048:  # Horizen
            s = nomalize(env.reset())
            score_dis = 0
            for _ in range(MAX_STEP):
                steps += 1
                # 选择行为
                act = ppo.actor_net.choose_action_discrete(torch.from_numpy(np.array(s).astype(np.float32)).to(device).unsqueeze(0))[0]
                '''
                if episodes % 50 == 0:
                    #env.render()
                    print(1)
                '''
                s_, r, done, info = env.step(act)
                s_ = nomalize(s_)
                score_dis += r
                s = s_
                if done:
                    break
        
    print("{}: {}".format(iter, np.mean(score)))
    if args.discrete:
        print("discrete {}: {}".format(iter, np.mean(score_dis)))
        lll_dis[iter] = np.mean(score_dis)
    lll[iter] = np.mean(score)
    ppo.train(memory)
    # if iter>100:
    #     ppo.train(memory, print_log=True)
    # else:
    #     ppo.train(memory)

output_dir = os.path.join("./output", cfg['output_name'])
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, f"{args.prefix}_{str(args.seed)}.npy"), lll)
if args.discrete:
    np.save(os.path.join(output_dir, f"{args.prefix}_{str(args.seed)}_discrete.npy"), lll_dis)