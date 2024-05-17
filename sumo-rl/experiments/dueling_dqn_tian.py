import os, sys, sumo_rl, traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import pandas as pd, numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.trainer import OffpolicyTrainer

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger, WandbLogger


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            *[
                nn.Linear(np.prod(state_shape), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
            ]
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state



out_csv = "outputs/mini/dqn_train"
total_step = 80000


env = sumo_rl.SumoEnvironment(
    net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    out_csv_name=out_csv,
    use_gui=True,
    single_agent=True,
    num_seconds=int(total_step),
    sumo_seed=3,
)

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)


policy = DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=0.99,  # gamma = 0.99
    action_space=env.action_space,
    is_double=False,
)
policy.set_eps(0.7)  # epsilon = 0.7




env = DummyVectorEnv([lambda: env])
buffer = ReplayBuffer(size=int(1e5))
collector = Collector(policy, env, buffer)


result = OffpolicyTrainer(
    policy,
    train_collector=collector,
    test_collector=None,
    max_epoch=0,
    max_epoch = 0,
    step_per_epoch = int(total_step*3.216),  
    step_per_collect = 1,
    episode_per_test = 0,
    update_per_step = 0.1,
    batch_size = 64,
).run()
print(result)