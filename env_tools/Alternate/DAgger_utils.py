from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from torch.utils.data import DataLoader
from easyrl.models.mlp import MLP
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np
from easyrl.utils.torch_util import move_to
from tqdm import tqdm
from easyrl.utils.common import save_traj
from easyrl.configs import cfg
from dataclasses import dataclass
from models import ActorModel



def create_actor(env):
    #print(env.observation_space.shape, env.action_space.n)
    actor_body = ActorModel(env.action_space.n)
    #ob_dim = env.observation_space.shape[0]
    #print(env.observation_space.shape, action_dim)
    # actor_body = MLP(input_size=ob_dim,
    #                 hidden_sizes=[64],
    #                 output_size=64,
    #                 hidden_act=nn.Tanh,
    #                 output_act=nn.Tanh)
    actor = CategoricalPolicy(actor_body,
                                in_features=64,
                                action_dim=env.action_space.n)
    # actor = DiagGaussianPolicy(actor_body,
    #                            in_features=64,
    #                            action_dim=env.action_space.n)
    return actor

def generate_demonstration_data(expert_agent, env, num_trials):
    return run_inference(expert_agent, env, num_trials, return_on_done=True)

def run_inference(agent, env, num_trials, return_on_done=False, sample=True, disable_tqdm=False, render=False):
    runner = EpisodicRunner(agent=agent, env=env)
    trajs = []
    for trial_id in tqdm(range(num_trials), desc='Run', disable=disable_tqdm):
        env.reset()
        traj = runner(time_steps=cfg.alg.episode_steps,
                      sample=sample,
                      return_on_done=return_on_done,
                      evaluation=True,
                      render_image=render)
        trajs.append(traj)
    return trajs

def eval_agent(agent, env, num_trials, disable_tqdm=False, render=False):
    trajs = run_inference(agent, env, num_trials, return_on_done=True, 
                          disable_tqdm=disable_tqdm, render=render)
    tsps = []
    successes = []
    rets = []
    for traj in trajs:
        tsps = traj.steps_til_done.copy().tolist()
        rewards = traj.raw_rewards
        infos = traj.infos
        #print(infos)
        for ej in range(rewards.shape[1]):
            ret = np.sum(rewards[:tsps[ej], ej])
            rets.append(ret)
            successes.append(infos[tsps[ej] - 1][ej]['success'])
        if render:
            save_traj(traj, 'tmp')
    ret_mean = np.mean(rets)
    ret_std = np.std(rets)
    success_rate = np.mean(successes)
    return success_rate, ret_mean, ret_std, rets, successes

class TrajDataset(Dataset):
    def __init__(self, trajs):
        states = []
        actions = []
        for traj in trajs:
            states.append(traj.obs)
            actions.append(traj.actions)
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        sample['state'] = self.states[idx]
        sample['action'] = self.actions[idx]
        return sample
    
    def add_traj(self, traj=None, states=None, actions=None):
        if traj is not None:
            self.states = np.concatenate((self.states, traj.obs), axis=0)
            self.actions = np.concatenate((self.actions, traj.actions), axis=0)
        else:
            self.states = np.concatenate((self.states, states), axis=0)
            self.actions = np.concatenate((self.actions, actions), axis=0)

def train_bc_agent(agent, eval, trajs, max_epochs=50, batch_size=256, lr=0.0005, disable_tqdm=True):
    dataset = TrajDataset(trajs)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True)
    optimizer = optim.Adam(agent.actor.parameters(),
                           lr=lr)
    pbar = tqdm(range(max_epochs), desc='Epoch', disable=disable_tqdm)
    logs = dict(loss=[], epoch=[], coins=[], returns=[])
    for iter in pbar:
        avg_loss = []
        for batch_idx, sample in enumerate(dataloader):
            states = sample['state'].float().to(cfg.alg.device)
            expert_actions = sample['action'].float().to(cfg.alg.device)
            optimizer.zero_grad()
            act_dist, _ = agent.actor(states)
            loss = -torch.mean(action_log_prob(expert_actions, act_dist))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            avg_loss.append(loss.item())
        returns, coins = eval(agent)
        
        logs['coins'].append(coins)
        logs['returns'].append(returns)

        logs['loss'].append(np.mean(avg_loss))
        logs['epoch'].append(iter)
    return agent, logs, len(dataset)