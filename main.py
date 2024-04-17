#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################

import pathlib
import os
import multiprocessing as mp
from typing_extensions import Self
import numpy as np
import argparse
import torch
import scipy
import time
from multi_signal import MultiSignal
from agent_config import agent_configs
from map_config import map_configs
from signal_config import signal_configs
from mdp_config import mdp_configs
from itertools import cycle
from collections import defaultdict
import sumolib

os.environ["LIBSUMO_AS_TRACI"] = "1"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default='navTL',
                    choices=['MAXPRESSURE', 'DQNnavigation', 'HDQNembedding', 'DQNsignal', 'DQNsignalgraph', 
                             'A2Cnavigation', 'HA2Cembedding', 'A2Csignal', 'HDQNgraph', 'HA2Cgraph', 
                             'A2Csignalgraph', 'HA2Cintention', 'HDQNintention', 'jointA2C', 'jointDQN', 
                             'colight', 'fma2c', 'dijkstra', 'xrouting', 'navTL', 'DQNConcat', 'ColiNav'])
    ap.add_argument("--eps", type=int, default=50)
    ap.add_argument("--map", type=str, default='hangzhou',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                             'cologne1', 'cologne3', 'cologne8', 'baoding', '30min', '15min', '4x4', 'hangzhou'
                             ])
    ap.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute()) + os.sep)
    ap.add_argument("--log_dir", type=str, default=str(pathlib.Path().absolute()) + os.sep + 'logs' + os.sep)
    ap.add_argument("--ckpt_dir", type=str, default=str(pathlib.Path().absolute()) + os.sep + 'ckpt' + os.sep)
    ap.add_argument("--libsumo", type=bool, default=True)
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--device", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--train", type=bool, default=True)
    ap.add_argument("--worker_ckpt", type=str, default='')
    ap.add_argument("--manager_ckpt", type=str, default='')
    args = ap.parse_args()

    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    run(args)


def get_graph_adj(map):
    signal_config = signal_configs.get(map)
    ts_ids = [ts for ts in list(signal_config.keys()) if ts not in ['phase_pairs', 'valid_acts', 'direction_encoding', 'margins']]
    matrix = np.asarray([[0.0 for c in range(len(ts_ids))] for r in range(len(ts_ids))])
    degree = np.asarray([[0.0 for c in range(len(ts_ids))] for r in range(len(ts_ids))])
    for i, ts in enumerate(ts_ids):
        neighbors_dict = signal_config[ts]['downstream']
        neighbors = []
        for key in neighbors_dict:
            if neighbors_dict[key] is not None:
                neighbors.append(neighbors_dict[key])
        degree[i][i] = len(neighbors) + 1
        for j, tl in enumerate(ts_ids):
            if tl in neighbors or tl == ts:
                matrix[i][j] = 1.0
                matrix[j][i] = 1.0

    rowsum = np.array(matrix.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = np.diag(d_inv)

    norm_adj = d_mat.dot(matrix)
    norm_adj = norm_adj.dot(d_mat)

    return torch.from_numpy(norm_adj).to(torch.float32)


def get_graph_edge(adj):
    return adj.to_sparse()


def run(args):
    agt_config = agent_configs[args.agent]
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']
    map_config = map_configs[args.map]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None: route = args.pwd + route

    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:  # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors
    
    signal_config = signal_configs.get(args.map)
    env = MultiSignal(alg.__name__,
                      args.map,
                      args.pwd + map_config['net'],
                      agt_config['state'],
                      agt_config['reward'],
                      route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'], \
                      end_time=map_config['end_time'], max_distance=agt_config['max_distance'], \
                      lights=map_config['lights'], gui=args.gui, log_dir=args.log_dir, libsumo=args.libsumo, \
                      warmup=map_config['warmup'], signal_config=signal_config)
    agt_config['steps'] = args.eps * num_steps_eps
    agt_config['log_dir'] = args.log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)

    obs_act = dict()
    obs_act['signal'] = [torch.Size([0]), 0]
    for key in env.signal_obs_shape:
        obs_act[key] = [env.signal_obs_shape[key], len(env.green_phases[key]) if key in env.green_phases else None]
        if env.signal_obs_shape[key] > obs_act['signal'][0]:
            obs_act['signal'][0] = env.signal_obs_shape[key]
        if len(env.green_phases[key]) > obs_act['signal'][1]:
            obs_act['signal'][1] = len(env.green_phases[key])
    # obs_act['cav'] = [85, 3] # navigation only 4x4
    # obs_act['cav'] = [41, 3] # HRL 4x4
    # obs_act['cav'] = [63, 3] # baoding-19 dqnnav
    # obs_act['cav'] = [47, 3] # baoding-19 HRL
    # obs_act['cav'] = [20, 3]  # baoding-19 xrouting
    # obs_act['cav'] = [25, 3] # koh 2020
    # obs_act['cav'] = [40, 3] # xrouting
    # obs_act['cav'] = [61, 3] # dqnnav
    obs_act['cav'] = [25, 3] # dqnnav hangzhou
    # obs_act['cav'] = [20, 3] # xrouting hangzhou

    if torch.cuda.is_available():
        device = "cuda:" + str(args.device)
    else:
        device = "cpu"
    device = torch.device(device)

    action_mask = dict()
    valid_acts = signal_configs.get(args.map)['valid_acts']
    for key in valid_acts:
        mask = [0] * obs_act['signal'][1]
        for i in valid_acts[key]:
            mask[i] = 1
        action_mask[key] = mask
    direction_encoding = signal_configs.get(args.map)['direction_encoding']
    env.direction_encoding = direction_encoding

    adj = get_graph_adj(args.map).to(device)
    sparse = get_graph_edge(adj)

    if args.train:
        ckpt_path = args.ckpt_dir + env.connection_name + os.sep
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    if args.agent == 'navTL':
        signal_ids = list(env.signals.keys())
        signal_phases = {signal: cycle(valid_acts[signal]) for signal in signal_ids}
        signal_act = {signal: next(signal_phases[signal]) for signal in signal_ids}
        agent = alg(agt_config, env.net, obs_act, device, signal_ids, action_mask, adj, args.train, args.worker_ckpt, args.manager_ckpt)
        decay = agt_config['EPS_DECAY']
        epsilon = agt_config['EPS_START']
        epsilon_end = agt_config['EPS_END']
        update = agt_config['TARGET_UPDATE']
        T = 0
        for t in range(args.eps):
            start = time.time()
            signal_obs, veh_obs = env.reset()  # reset environment in every episode
            veh_obs = {k: v for k, v in veh_obs.items() if v is not None}
            done = False
            intention_log = defaultdict(list)
            while not done:
                worker_reward = []
                log_veh_reward = defaultdict(list)
                prev_signal_obs = signal_obs
                for signal in env.signals:
                    env.signals[signal].prep_phase(signal_act[signal])
                # vehicle act and step
                signal_intention = defaultdict(list)
                for step in range(env.yellow_length):
                    env.update_cav_signals()
                    for signal in env.signal_ids:
                        env.signals[signal].observe(env.step_length, env.max_distance)
                    signal_obs = env.state_fn(env.signals)
                    env.update_veh_routes()
                    intention = agent.learn_intention(veh_obs, obs_act, env.cav_signals)
                    veh_act, _, mask, goals, emb = agent.act(veh_obs, epsilon, \
                    env.cav_signals, signal_obs, env.cav_edges, intention)
                    env.veh_step(veh_act)
                    env.step_sim()
                    veh_rew = {cav: env.cavs[cav].get_reward() for cav in env.cavs}
                    new_veh_obs = {cav: env.cavs[cav].get_state() for cav in env.cavs}
                    new_veh_obs = {k: v for k, v in new_veh_obs.items() if v is not None}
                    veh_rew.update({cav: 1 for cav in env.completed_cavs})
                    new_veh_obs.update({cav: env.completed_cavs[cav].get_state() for cav in env.completed_cavs})
                    veh_done = {cav: False for cav in env.cavs}
                    veh_done.update({cav: True for cav in env.completed_cavs})
                    if args.train:
                        agent.worker_push_memory(veh_obs, veh_act, veh_rew, new_veh_obs, veh_done, mask, goals, emb)
                    veh_obs = new_veh_obs
                    worker_reward.extend(list(veh_rew.values()))
                    for signal in intention:
                        signal_intention[signal].append(intention[signal])
                    for cav in veh_rew:
                        if cav in env.cav_signals:
                            signal = env.cav_signals[cav]
                            if signal in env.signal_ids:
                                log_veh_reward[signal].append(veh_rew[cav])
                for signal in env.signal_ids:
                    env.signals[signal].set_phase()
                for step in range(env.step_length - env.yellow_length):
                    env.update_cav_signals()
                    for signal in env.signal_ids:
                        env.signals[signal].observe(env.step_length, env.max_distance)
                    signal_obs = env.state_fn(env.signals)
                    env.update_veh_routes()
                    intention = agent.learn_intention(veh_obs, obs_act, env.cav_signals)
                    veh_act, _, mask, goals, emb = agent.act(veh_obs, epsilon, \
                    env.cav_signals, signal_obs, env.cav_edges, intention)
                    env.veh_step(veh_act)
                    env.step_sim()
                    veh_rew = {cav: env.cavs[cav].get_reward() for cav in env.cavs}
                    new_veh_obs = {cav: env.cavs[cav].get_state() for cav in env.cavs}
                    new_veh_obs = {k: v for k, v in new_veh_obs.items() if v is not None}
                    veh_rew.update({cav: 1 for cav in env.completed_cavs})
                    new_veh_obs.update({cav: env.completed_cavs[cav].get_state() for cav in env.completed_cavs})
                    veh_done = {cav: False for cav in env.cavs}
                    veh_done.update({cav: True for cav in env.completed_cavs})
                    if args.train:
                        agent.worker_push_memory(veh_obs, veh_act, veh_rew, new_veh_obs, veh_done, mask, goals, emb)
                    veh_obs = new_veh_obs
                    worker_reward.extend(list(veh_rew.values()))
                    for signal in intention:
                        signal_intention[signal].append(intention[signal])
                    for cav in veh_rew:
                        if cav in env.cav_signals:
                            signal = env.cav_signals[cav]
                            if signal in env.signal_ids:
                                log_veh_reward[signal].append(veh_rew[cav])
                signal_rewards = env.reward_fn(env.signals)
                for signal in signal_rewards:
                    signal_rewards[signal] += 0.001 * sum(log_veh_reward[signal])
                for signal in signal_intention:
                    signal_intention[signal] = torch.mean(torch.stack(signal_intention[signal], 0).to(torch.float32), 0)
                if args.train:
                    agent.manager_push_memory(prev_signal_obs, signal_act, signal_rewards, signal_obs, done, signal_intention)
                if len(worker_reward) > 0:
                    worker_reward = sum(worker_reward) / len(worker_reward)
                else:
                    worker_reward = 0
                manager_reward = sum(signal_rewards.values()) / len(signal_rewards)
                env.calc_metrics(worker_reward, manager_reward)
                # update policy
                if args.train:
                    if T > 100:
                        manager_losses = agent.manager_optimize_loss(agt_config['BATCH_SIZE'])
                    if agent.worker_buffer.len() > agt_config['BATCH_SIZE']:
                        worker_losses = agent.worker_optimize_loss(agt_config['BATCH_SIZE'])
                    if T % update == 0:
                        agent.update_target(label='worker')
                        agent.update_target(label='manager')
                T += 1
                signal_act = agent.act(dict(), epsilon, env.cav_signals, signal_obs, env.cav_edges, signal_intention)[1]
                epsilon = max(epsilon * decay, epsilon_end)
                done = env.sumo.simulation.getTime() >= env.end_time
            end = time.time()
            if args.train:
                print("training epoch: ", t, "time: ", end - start)
                if t % 10 == 0:
                    torch.save({
                    'worker_state_dict': agent.worker.state_dict(),
                    'worker_optimizer_state_dict': agent.worker_optimizer.state_dict(),
                    }, ckpt_path + 'worker_' + str(t) + '.pt')
                    torch.save({
                    'manager_state_dict': agent.manager.state_dict(),
                    'manager_optimizer_state_dict': agent.manager_optimizer.state_dict(),
                    }, ckpt_path + 'manager_' + str(t) + '.pt')
            else:
                print("testing epoch: ", t, "time: ", end - start)
        env.close()


if __name__ == '__main__':
    main()
