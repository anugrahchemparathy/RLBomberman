from pathlib import Path
from typing import *
from collections import namedtuple
from dataclasses import dataclass
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from copy import deepcopy
from itertools import count
from tqdm import tqdm
from DAgger_utils import *
from master import ExpertAgent
from new_env import BombermanEnv

import torch
import torch.nn as nn

from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd

from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env
from easyrl.utils.torch_util import freeze_model
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np

def set_configs(exp_name='bc'):
    set_config('ppo')
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 500
    cfg.alg.max_steps = 600000
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = 'Bomberman-v1'
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += f'/{exp_name}'
    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))

    print(f'====================================')
    print(f'      Device:{cfg.alg.device}')
    print(f'====================================')

@dataclass
class BasicAgent:
    actor: nn.Module

    def __post_init__(self):
        move_to([self.actor],
                device=cfg.alg.device)

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        t_ob = torch_float(ob, device=cfg.alg.device)
        # the policy returns a multi-variate gaussian distribution
        act_dist, _ = self.actor(t_ob)
        # sample from the distribution
        action = action_from_dist(act_dist,
                                  sample=sample)
        # get the log-probability of the sampled actions
        log_prob = action_log_prob(action, act_dist)
        # get the entropy of the action distribution
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
        )
        return torch_to_np(action), action_info

@dataclass
class DaggerAgent():
    actor: nn.Module
    expert_actor: nn.Module
    lr: float

    def __post_init__(self):
        move_to([self.actor],
                device=cfg.alg.device)
        #freeze_model(self.expert_actor)
        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=self.lr)
    
    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        t_ob = torch_float(ob, device=cfg.alg.device)
        # the policy returns a multi-variate gaussian distribution
        act_dist, _ = self.actor(t_ob)
        # sample from the distribution
        action = action_from_dist(act_dist,
                                  sample=sample)
        # get the log-probability of the sampled actions
        log_prob = action_log_prob(action, act_dist)
        # get the entropy of the action distribution
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
        )
        # get the expert action from the expert policy
        exp_act_dist, _ = self.expert_actor.get_action(t_ob)
        action_info['exp_act'] = exp_act_dist
        return torch_to_np(action), action_info

    def optimize(self, data, **kwargs):
        for key, val in data.items():
            data[key] = torch_float(val, device=cfg.alg.device)
        ob = data['state']
        exp_act = data['action']
        act_dist, _ = self.actor(x=ob)
        loss = -torch.mean(action_log_prob(exp_act, act_dist))
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = clip_grad(self.actor.parameters(),
                              cfg.alg.max_grad_norm)
        self.optimizer.step()

        optim_info = dict(
            loss=loss.item(),
            grad_norm=grad_norm,
        )
        return optim_info
    
    def save_model(self, is_best=False, step=None):
        data_to_save = {
            'actor_state_dict': self.actor.state_dict()
        }
        save_model(data_to_save, cfg.alg, is_best=is_best, step=step)

@dataclass
class DaggerEngine:
    agent: Any
    runner: Any
    env: Any
    trajs: Any

    def __post_init__(self):
        self.dataset = TrajDataset(self.trajs)
        
    def train(self):
        success_rates = []
        dataset_sizes = []
        optim_infos_list = []
        self.cur_step = 0
        for iter_t in count():
            if iter_t % cfg.alg.eval_interval == 0:
                success_rate, ret_mean, ret_std, rets, successes = eval_agent(self.agent, 
                                                                              self.env, 
                                                                              20,
                                                                              disable_tqdm=True)
                success_rates.append(success_rate)
                dataset_sizes.append(len(self.dataset))
            # rollout the current policy and get a trajectory
            traj = self.runner(sample=True, get_last_val=False, time_steps=cfg.alg.episode_steps)
            # optimize the policy
            optim_infos = self.train_once(traj)
            optim_infos_list.append(optim_infos)
            if self.cur_step > cfg.alg.max_steps:
                break
        return optim_infos_list, dataset_sizes, success_rates

    def train_once(self, traj):
        self.cur_step += traj.total_steps
        print('got here', traj.total_steps)

        action_infos = traj.action_infos
        #print(action_infos)
        exp_act = [ainfo['exp_act'] for ainfo in action_infos]
        #exp_act = torch.stack([ainfo['exp_act'] for ainfo in action_infos])

        self.dataset.add_traj(states=traj.obs,
                              actions=exp_act)
        rollout_dataloader = DataLoader(self.dataset,
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True,
                                       )
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize(batch_data)
                optim_infos.append(optim_info)  
        return optim_infos

def train_dagger(expert_actor, trajs, actor=None):
    expert_actor = deepcopy(expert_actor)
    actor = deepcopy(actor)
    set_configs('dagger')
    cfg.alg.episode_steps = 500
    cfg.alg.max_steps = 600000
    cfg.alg.eval_interval = 1
    cfg.alg.log_interval = 1
    cfg.alg.batch_size = 256
    cfg.alg.opt_epochs = 500
    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
    env.reset()
    if actor is None:
        actor = create_actor(env=env)
    dagger_agent = DaggerAgent(actor=actor, expert_actor=expert_actor, lr=0.001)
    runner = EpisodicRunner(agent=dagger_agent, env=env)
    engine = DaggerEngine(agent=dagger_agent,
                          env=env,
                          runner=runner,
                          trajs=trajs)
    optim_infos_list, dataset_sizes, success_rates = engine.train()
    return optim_infos_list, dagger_agent, dataset_sizes, success_rates

if __name__ == '__main__':
    set_configs()
    env = make_vec_env(cfg.alg.env_name,
                      cfg.alg.num_envs,
                      seed=cfg.alg.seed)
    env.reset()
    actor = create_actor(env)
    expert_actor = ExpertAgent()
    agent = BasicAgent(actor)
    num_trajs = 500
    expert_trajs = generate_demonstration_data(expert_agent=expert_actor,
                                          env=env,
                                          num_trials=100)
    bc_num_trajs = 100
    trained_agent, _, size = train_bc_agent(agent, trajs=expert_trajs[:bc_num_trajs], disable_tqdm=False)
    print('trained')
    success_rate_bc, ret_mean_bc, ret_std_bc, rets_bc, successes_bc = eval_agent(trained_agent, env, num_trials=200)
    # expert_trajs = [np.array([])]