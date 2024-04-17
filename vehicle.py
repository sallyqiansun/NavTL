#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: sunqian
Date: 2022-10-31 12:25:20
LastEditTime: 2022-12-02 17:15:34
LastEditors: sunqian13@baidu.com
Description: 
FilePath: /sunqian/RESCO-main/vehicle.py
"""

import traci
import sumolib
import torch
from collections import defaultdict
import numpy as np
from signal_config import signal_configs

class Vehicle:
    def __init__(self, cav_id, sumo, net, config, signal_encoding, signals, max_v=15.0, max_acc=3.0, delta_time=1, begin_time=1):
        self.id = cav_id
        self.delta_time = delta_time
        self.sumo = sumo
        self.max_v = max_v
        self.max_acc = max_acc
        self.collided = []
        self.arrived = []
        self.last_reward = None
        self.next_action_time = begin_time
        self.net = net
        self.config = config
        self.signal_encoding = signal_encoding
        self.signal_ids = list(signal_encoding.keys())
        self.travel_time = 0
        self.cur_road = self.sumo.vehicle.getRoadID(self.id)
        self.destination = self.sumo.vehicle.getRoute(self.id)[-1]
        self.signals = signals
        net = sumolib.net.readNet(self.net, withInternal=True)
        self.end_pos = list(self.sumo.junction.getPosition(net.getEdge(self.destination).getToNode().getID()))
        self.current_zone = None
    
    def get_state(self):
        # NavTL HRL state -> signal encoding + direction encoding
        # net = sumolib.net.readNet(self.net, withInternal=True)
        # current_edge = net.getEdge(self.sumo.vehicle.getRoadID(self.id))
        # cur_signal = current_edge.getToNode().getID()
        # if cur_signal in self.signal_encoding:
        #     cur_signal_encoding = self.signal_encoding[cur_signal]
        # else:
        #     return None
        # cur_from_node = current_edge.getFromNode().getID()
        # if cur_signal == cur_from_node:
        #     direction = ':' # internal edge
        # else:
        #     if cur_from_node in self.config and cur_signal in self.config:
        #         direction = [k for k, v in self.config[cur_from_node]['downstream'].items() if v == cur_signal][0]
        #     else:
        #         direction = self.config['margins'][cur_from_node][cur_signal]
        # if direction == 'N':
        #     onehot = [1, 0, 0, 0]
        # elif direction == 'E':
        #     onehot = [0, 1, 0, 0]
        # elif direction == 'S':
        #     onehot = [0, 0, 1, 0]
        # elif direction == 'W':
        #     onehot = [0, 0, 0, 1]
        # else:
        #     onehot = [0, 0, 0, 0]
        # state = [self.sumo.vehicle.getSpeed(self.id) / self.max_v] # speed
        # state.extend(cur_signal_encoding)
        # state.extend(onehot)
        
        # dest_edge = net.getEdge(self.destination)
        # dest_signal = dest_edge.getFromNode().getID()
        # dest_to_node = dest_edge.getToNode().getID()
        # if dest_signal in self.signal_encoding:
        #     dest_signal_encoding = self.signal_encoding[dest_signal]
        # elif dest_to_node in self.signal_encoding:
        #     dest_signal_encoding = self.signal_encoding[dest_to_node]
        # else:
        #     return None
        
        # if dest_to_node in self.config and dest_signal in self.config:
        #     dest_direction = [k for k, v in self.config[dest_signal]['downstream'].items() if v == dest_to_node][0]
        # else:
        #     dest_direction = self.config['margins'][dest_signal][dest_to_node]
        # if dest_direction == 'N':
        #     dest_onehot = [1, 0, 0, 0]
        # elif dest_direction == 'E':
        #     dest_onehot = [0, 1, 0, 0]
        # elif dest_direction == 'S':
        #     dest_onehot = [0, 0, 1, 0]
        # elif dest_direction == 'W':
        #     dest_onehot = [0, 0, 0, 1]
        # state.extend(dest_signal_encoding)
        # state.extend(dest_onehot)

        # # # # 只做navigation的时候需要把信号灯的state加进来
        # if cur_signal in self.signals:
        #     signal = self.signals[cur_signal]
        # elif cur_from_node in self.signals:
        #     signal = self.signals[cur_from_node]
        # else:
        #     # state.extend([0] * (36 + 8)) # last int represents num of phases 
        #     return None
        # obs = []
        # enc = signal.phase_encoding[signal.phase]
        # obs.extend(enc)
        # north = {'veh_count': 0, 'total_speed': 0, 'pressure': 0}
        # south, east, west = north.copy(), north.copy(), north.copy()
        # for direction in signal.lane_sets:
        #     queue_length = 0
        #     veh_count = 0
        #     total_speed, avg_speed = 0, 0
        #     for lane in signal.lane_sets[direction]:
        #         queue_length += signal.full_observation[lane]['queue']
        #         vehicles = signal.full_observation[lane]['vehicles']
        #         veh_count += len(vehicles)
        #         for vehicle in vehicles:
        #             total_speed += vehicle['speed']
        #         length = signal.full_observation[lane]['length']
        #     for lane in signal.lane_sets_outbound[direction]:
        #         dwn_signal = signal.out_lane_to_signalid[lane]
        #         if dwn_signal in signal.signals:
        #             queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']

        #     if '_N' in direction:
        #         north['veh_count'] += veh_count
        #         north['total_speed'] += total_speed
        #         north['pressure'] += queue_length
        #     elif '_S' in direction:
        #         south['veh_count'] += veh_count
        #         south['total_speed'] += total_speed
        #         south['pressure'] += queue_length
        #     elif '_E' in direction:
        #         east['veh_count'] += veh_count
        #         east['total_speed'] += total_speed
        #         east['pressure'] += queue_length
        #     elif '_W' in direction:
        #         west['veh_count'] += veh_count
        #         west['total_speed'] += total_speed
        #         west['pressure'] += queue_length
        # if north['veh_count'] > 0:
        #     obs.extend([north['veh_count'], north['total_speed'] / north['veh_count'], north['pressure']])
        # else:
        #     obs.extend([north['veh_count'], 0, north['pressure']])
        # if east['veh_count'] > 0:
        #     obs.extend([east['veh_count'], east['total_speed'] / east['veh_count'], east['pressure']])
        # else:
        #     obs.extend([east['veh_count'], 0, east['pressure']])
        # if south['veh_count'] > 0:
        #     obs.extend([south['veh_count'], south['total_speed'] / south['veh_count'], south['pressure']])
        # else:
        #     obs.extend([south['veh_count'], 0, south['pressure']])
        # if west['veh_count'] > 0:
        #     obs.extend([west['veh_count'], west['total_speed'] / west['veh_count'], west['pressure']])
        # else:
        #     obs.extend([west['veh_count'], 0, west['pressure']])
        # state.extend(obs)

        # #     obs.append(queue_length / 10.0)
        # #     obs.append(veh_count / 10.0)
        # #     if veh_count > 0:
        # #         avg_speed = total_speed / veh_count
        # #     obs.append(avg_speed / 10.0)
        # # state.extend(obs)
        # return torch.tensor(state)


    # dqn navigation state
        net = sumolib.net.readNet(self.net, withInternal=True)
        speed = [self.sumo.vehicle.getSpeed(self.id)]
        location = list(self.sumo.vehicle.getPosition(self.id))
        self.route = self.sumo.vehicle.getRoute(self.id)
        self.destination = net.getEdge(self.route[-1]).getToNode().getID()
        destination = list(self.sumo.junction.getPosition(self.destination))
        state = [speed, location, destination]
        state = [item for sublist in state for item in sublist]
        current_edge = net.getEdge(self.sumo.vehicle.getRoadID(self.id))
        cur_signal = current_edge.getToNode().getID()
        cur_from_node = current_edge.getFromNode().getID()
        if cur_signal in self.signals:
            signal = self.signals[cur_signal]
        elif cur_from_node in self.signals:
            signal = self.signals[cur_from_node]
        else:
            return torch.tensor([0] * 25)
        enc = signal.phase_encoding[signal.phase]

        obs = []
        obs.extend(enc)
        north = {'veh_count': 0, 'total_speed': 0, 'pressure': 0}
        south, east, west = north.copy(), north.copy(), north.copy()
        for direction in signal.lane_sets:
            queue_length = 0
            veh_count = 0
            total_speed, avg_speed = 0, 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                vehicles = signal.full_observation[lane]['vehicles']
                veh_count += len(vehicles)
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                length = signal.full_observation[lane]['length']
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']

            if '_N' in direction:
                north['veh_count'] += veh_count
                north['total_speed'] += total_speed
                north['pressure'] += queue_length
            elif '_S' in direction:
                south['veh_count'] += veh_count
                south['total_speed'] += total_speed
                south['pressure'] += queue_length
            elif '_E' in direction:
                east['veh_count'] += veh_count
                east['total_speed'] += total_speed
                east['pressure'] += queue_length
            elif '_W' in direction:
                west['veh_count'] += veh_count
                west['total_speed'] += total_speed
                west['pressure'] += queue_length

        if north['veh_count'] > 0:
            obs.extend([north['veh_count'], north['total_speed'] / north['veh_count'], north['pressure']])
        else:
            obs.extend([north['veh_count'], 0, north['pressure']])
        if east['veh_count'] > 0:
            obs.extend([east['veh_count'], east['total_speed'] / east['veh_count'], east['pressure']])
        else:
            obs.extend([east['veh_count'], 0, east['pressure']])
        if south['veh_count'] > 0:
            obs.extend([south['veh_count'], south['total_speed'] / south['veh_count'], south['pressure']])
        else:
            obs.extend([south['veh_count'], 0, south['pressure']])
        if west['veh_count'] > 0:
            obs.extend([west['veh_count'], west['total_speed'] / west['veh_count'], west['pressure']])
        else:
            obs.extend([west['veh_count'], 0, west['pressure']])
        state.extend(obs)
        state = torch.tensor(state)

        return state


    def get_reward(self):
        # return - self.sumo.vehicle.getAccumulatedWaitingTime(self.id)
        # return - self.sumo.vehicle.getWaitingTime(self.id)

        road = self.sumo.vehicle.getRoadID(self.id)
        if road == self.cur_road:
            self.travel_time += 1
        else:
            self.cur_road = road
            new_from_node = road.split('_')[0]
            if new_from_node in self.signal_ids:
                self.travel_time = 0
            else:
                self.travel_time += 1
        return - self.travel_time

        # road = self.sumo.vehicle.getRoadID(self.id)
        # if road == self.cur_road:
        #     self.travel_time += 1
        # else:
        #     self.travel_time = 0
        #     self.cur_road = road

        # self.travel_time += 1
        # location = list(self.sumo.vehicle.getPosition(self.id))
        # distance = ((location[0] - self.end_pos[0]) ** 2 + (location[1] - self.end_pos[1]) ** 2) ** 0.5
        # return - self.travel_time / distance