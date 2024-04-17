#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: sunqian
Date: 2022-11-22 18:52:47
LastEditTime: 2022-11-28 23:07:03
LastEditors: sunqian13@baidu.com
Description: 
FilePath: /sunqian/RESCO-main/agents/navTL.py
"""

from typing import Any, Sequence
import numpy as np
import random
from collections import deque, defaultdict
from functools import partial
import sumolib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

np.random.seed(0)

class WReplayBuffer(object):
    """DQN‘s replay memory
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, goal, emb):
        self.buffer.append((state, action, reward, next_state, done, mask, goal, emb))

    def sample(self, batch_size):
        state, action, reward, next_state, done, mask, goal, emb = zip(
            *random.sample(self.buffer, batch_size))
        return [torch.stack(state), torch.tensor(action), torch.tensor(reward), torch.stack(next_state), \
        torch.tensor(done), torch.stack(mask), torch.stack(goal), torch.stack(emb)]

    def len(self):
        return len(self.buffer)

class MReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, intention):
        self.buffer.append((state, action, reward, next_state, done, intention))

    def sample(self, batch_size):
        state, action, reward, next_state, done, intention = zip(
            *random.sample(self.buffer, batch_size))
        return [torch.stack(state), torch.stack(action), torch.stack(reward), torch.stack(next_state), \
        torch.tensor(done).to(torch.int), torch.stack(intention)]

class GConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None) 

    def forward(self, input, adj):
        support = torch.einsum('mik,kj->mij', input, self.weight)
        output = torch.einsum('ki,mij->mkj', adj, support)
        if self.bias is not None:	
            return output + self.bias

class GCN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, dropout):
        super(GCN, self).__init__()

        self.gcn1 = GConv(num_input, num_hidden)
        self.gcn2 = GConv(num_hidden, num_output)
        self.dropout = dropout
        self.training = True

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        return x

class MDQNAgent(nn.Module):
    def __init__(self, num_inputs, num_hidden, manager_action, worker_state, device, dropout):
        super(MDQNAgent, self).__init__()
        self.device = device
        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs[0], 128), 
            nn.Linear(128, 64), 
            nn.Linear(64, num_hidden),
        )
        self.action_layer = GCN(num_hidden, 64, num_hidden, self.dropout)
        self.out_layer = nn.Sequential(
            nn.Linear(num_hidden, num_hidden), 
            nn.ReLU(),
            nn.Linear(num_hidden, manager_action)
        )
        self.goal_layer = nn.Sequential(
            nn.Linear(num_hidden, worker_state, bias=False)
        )


    def forward(self, x, adj):
        x = x.to(self.device)
        latent = self.encoder(x)
        graph = self.action_layer(latent, adj)
        hidden = graph + latent
        q = self.out_layer(hidden)
        goal = self.goal_layer(latent)
        return q, goal, latent

class WDQNAgent(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_action, device):
        super(WDQNAgent, self).__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(num_inputs[0], num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_action),
        )

    def forward(self, x, mask):
        emb = self.layers(x)
        mask = torch.tensor(mask).to(self.device)
        return emb

    def act(self, state, epsilon, num_actions, mask):
        if np.random.uniform() > epsilon:
            state = state.to(self.device)
            action_value = self.forward(state, mask).clone().detach()
            action = action_value.argmax(dim=-1).item()
        else:
            action = random.randint(0, num_actions - 1)
            while mask[action] == 0:
                action = random.randint(0, num_actions - 1)
        return action

class NavTL():

    def __init__(self, config, net, obs_act, device, signal_ids, action_mask, adj, train=True, worker_ckpt='', manager_ckpt=''):
        super(NavTL, self).__init__()
        self.device = device
        self.config = config
        self.lr = self.config['LR']
        self.gamma = self.config['GAMMA']
        self.loss_func = nn.MSELoss(reduction='mean')
        self.signal_ids = signal_ids
        self.action_mask = action_mask
        self.obs_space = obs_act['cav'][0]
        self.act_space = obs_act['cav'][1]
        self.obs_space_ = obs_act['signal'][0]
        self.act_space_ = obs_act['signal'][1]
        self.net = net
        self.hidden = self.config['HIDDEN']
        self.dropout = self.config['DROPOUT']
        self.adj = adj
        self.manager = MDQNAgent(torch.Size([self.obs_space_[0] + self.obs_space]), self.hidden, self.act_space_, self.obs_space, device, self.dropout).to(self.device)
        self.manager_target = MDQNAgent(torch.Size([self.obs_space_[0] + self.obs_space]), self.hidden, self.act_space_, \
        self.obs_space, device, self.dropout).to(self.device)
        self.manager_target.load_state_dict(self.manager.state_dict())
        self.manager_target.eval()
        self.manager_optimizer = optim.RMSprop(self.manager.parameters(), lr=self.lr,
                                       alpha=0.9, centered=False, eps=1e-7)
        self.manager_buffer = MReplayBuffer(5000)
        
        self.worker = WDQNAgent(torch.Size([self.obs_space + self.hidden]), self.hidden, \
        self.act_space, device).to(self.device)
        self.worker_target = WDQNAgent(torch.Size([self.obs_space + self.hidden]), \
        self.hidden, self.act_space, device).to(self.device)

        self.worker_target.load_state_dict(self.worker.state_dict())
        self.worker_target.eval()
        self.worker_optimizer = optim.RMSprop(self.worker.parameters(), lr=self.lr,
                                        alpha=0.9, centered=False, eps=1e-7)
        self.worker_buffer = WReplayBuffer(5000)

        self.train = train
        if self.train is False:
            if worker_ckpt == '' or manager_ckpt == '':
                print("No model checkpoint to load")
                exit()
            else:
                manager_ckpt = torch.load(manager_ckpt)
                self.manager.load_state_dict(manager_ckpt['manager_state_dict'])
                self.manager_target.load_state_dict(manager_ckpt['manager_state_dict'])
                self.manager_optimizer.load_state_dict(manager_ckpt['manager_optimizer_state_dict'])
                worker_ckpt = torch.load(worker_ckpt)
                self.worker.load_state_dict(worker_ckpt['worker_state_dict'])
                self.worker_target.load_state_dict(worker_ckpt['worker_state_dict'])
                self.worker_optimizer.load_state_dict(worker_ckpt['worker_optimizer_state_dict'])

    def learn_intention(self, veh_obs, obs_act, cav_signals):
        intention = defaultdict(list)
        for cav in veh_obs:
            signal = cav_signals[cav]
            if signal in self.signal_ids:
                intention[signal].append(veh_obs[cav])
        for signal in self.signal_ids:
            if signal not in intention:
                intention[signal] = torch.tensor([0] * obs_act['cav'][0]).to(self.device)
            elif len(intention[signal]) == 0:
                intention[signal] = torch.tensor([0] * obs_act['cav'][0]).to(self.device)
            else:
                if len(intention[signal]) == 1:
                    intention[signal] = intention[signal][0]
                else:
                    intention[signal] = torch.mean(torch.stack(intention[signal], 0), 0).to(self.device)
        return intention

    def act(self, state, epsilon, cav_signals, signal_states, cav_edges, intention):
        net = sumolib.net.readNet(self.net, withInternal=True)
        # manager act
        signal_act, action_value, signal_goals, signal_emb = dict(), dict(), dict(), dict()

        global_state, global_intention = [], []
        for signal in self.signal_ids:
            global_state.append(signal_states[signal])
            global_intention.append(intention[signal])
        global_state, global_intention = torch.stack(global_state), torch.stack(global_intention)
        with torch.no_grad():
            global_action_value, global_signal_goals, global_signal_emb = self.manager(torch.cat((global_state, global_intention), dim=-1).unsqueeze(0), self.adj)
        
        for ind, signal in enumerate(self.signal_ids):
            action_value[signal], signal_goals[signal], signal_emb[signal] = global_action_value[0, ind], global_signal_goals[0, ind], global_signal_emb[0, ind]
            if np.random.uniform() > epsilon:
                action = action_value[signal].argmax(dim=-1).item()
            else:
                action = random.randint(0, self.act_space_ - 1)
            signal_act[signal] = action

        action, mask, goals, latent_emb = dict(), dict(), dict(), dict()
        # worker act
        for cav in state:
            signal = cav_signals[cav]
            if signal in self.signal_ids:
                goals[cav] = signal_goals[signal]
                latent_emb[cav] = signal_emb[signal]
                edge, position = cav_edges[cav][0], cav_edges[cav][1]
                outgoing = net.getEdge(edge).getOutgoing().keys()
                outgoing = [e.getID() for e in outgoing]
                direction_mask = [1] * len(outgoing)
                if len(outgoing) < self.act_space:
                    direction_mask.extend([0] * (self.act_space - len(outgoing)))
                mask[cav] = torch.tensor(direction_mask).to(self.device)
                action[cav] = self.worker.act(torch.cat((state[cav].to(torch.float32).to(self.device), \
                    latent_emb[cav]), 0), epsilon, self.act_space, mask[cav])
        return action, signal_act, mask, goals, latent_emb
        return

    def update_target(self, label):
        if label == 'worker':
            self.worker_target.load_state_dict(self.worker.state_dict())
        elif label == 'manager':
            self.manager_target.load_state_dict(self.manager.state_dict())

    def worker_push_memory(self, obs, act, rew, new_obs, done, mask, goals, emb):
        for key in obs:
            if key in act and key in rew and key in new_obs:
                self.worker_buffer.push(obs[key], act[key], rew[key], new_obs[key], int(done[key]), \
                mask[key], goals[key], emb[key])
        
    def manager_push_memory(self, obs, act, rew, new_obs, done, intention):
        rews, acts = [], []
        for ind, key in enumerate(self.signal_ids): 
            if ind == 0:
                global_state = obs[key].unsqueeze(0)
                global_state_ = new_obs[key].unsqueeze(0)
                global_intention = intention[key].unsqueeze(0)
            else:
                global_state = torch.cat((global_state, obs[key].unsqueeze(0)), 0)
                global_state_ = torch.cat((global_state_, new_obs[key].unsqueeze(0)), 0)
                global_intention = torch.cat((global_intention, intention[key].unsqueeze(0)), 0)
            rews.append(rew[key])
            acts.append(act[key])
        rews, acts = torch.tensor(rews), torch.tensor(acts)

        self.manager_buffer.push(global_state, acts, rews, global_state_, done, global_intention)

    def manager_optimize_loss(self, batch_size):
        losses = []
        manager_sample = self.manager_buffer.sample(batch_size)
        state, action, reward, next_state, done, intention = manager_sample[0], manager_sample[1], manager_sample[2], \
        manager_sample[3], manager_sample[4], manager_sample[5]
        state = state.to(torch.float32).to(self.device)
        next_state = next_state.to(torch.float32).to(self.device)
        action = action.to(self.device)
        reward = reward.to(torch.float32).to(self.device)
        done = done.to(self.device)
        intention = intention.to(torch.float32).to(self.device)

        with torch.no_grad():
            global_target, _, _ = self.manager_target(torch.cat((next_state, intention), dim=-1), self.adj)
        global_current, global_goal, global_latent = self.manager(torch.cat((state, intention), dim=-1), self.adj)

        for ind in range(len(self.signal_ids)):
            q_target = reward[range(batch_size), ind] + self.gamma * (1 - done) \
            * global_target[range(batch_size), ind].max(dim=-1)[0]
            sig_act = action[range(batch_size), ind]
            q_current = global_current[range(batch_size), ind].gather(-1, sig_act.unsqueeze(1)).squeeze()
            td_err = q_current - q_target
            loss = (td_err ** 2).mean()
            losses.append(loss)
        
        loss = torch.stack(losses).mean()
        self.manager_optimizer.zero_grad()
        loss.backward()
        self.manager_optimizer.step()
        return losses


    def worker_optimize_loss(self, batch_size):
        losses = []
        worker_sample = self.worker_buffer.sample(batch_size)
        state, action, reward, next_state, done, mask, goal, emb = worker_sample[0], worker_sample[1], worker_sample[2], \
        worker_sample[3], worker_sample[4], worker_sample[5], worker_sample[6], worker_sample[7]
        state = state.to(torch.float32).to(self.device)
        next_state = next_state.to(torch.float32).to(self.device)
        action = action.to(self.device)
        reward = reward.to(torch.float32).to(self.device)
        done = done.to(self.device)
        goal = goal.to(self.device)
        emb = emb.to(self.device)

        cos_sim = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            reward = reward - cos_sim(next_state - state, goal)
            q = self.worker_target(torch.cat((next_state, emb), 1), mask)
            q_target = reward + self.gamma * (1 - done) * q.max(dim=-1)[0]
        q = self.worker(torch.cat((state, emb), 1), mask)
        q_current = q.gather(-1, action.unsqueeze(1)).squeeze()
        td_err = q_current - q_target
        loss = (td_err ** 2).mean()
        self.worker_optimizer.zero_grad()
        loss.backward()
        self.worker_optimizer.step()
        losses.append(td_err ** 2)
        return losses

