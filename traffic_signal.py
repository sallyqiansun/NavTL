#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

import traci
import copy
import re
from signal_config import signal_configs
from collections import defaultdict


class Signal:
    """
    traffic signal class
    """
    def __init__(self, map_name, sumo, id, yellow_length, all_phases, green_phases):
        self.sumo = sumo
        self.id = id
        self.yellow_time = yellow_length
        self.next_phase = 0

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        lanes = []
        self.lanes = []
        self.outbound_lanes = []

        reversed_directions = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

        myconfig = signal_configs[map_name]
        self.lane_sets = myconfig[self.id]['lane_sets']
        self.lane_sets_outbound = self.lane_sets.copy()
        for key in self.lane_sets_outbound:
            self.lane_sets_outbound[key] = []
        self.downstream = myconfig[self.id]['downstream']

        self.inbounds_fr_direction = dict()
        for direction in self.lane_sets:
            for lane in self.lane_sets[direction]:
                inbound_to_direction = direction.split('-')[0]
                inbound_fr_direction = reversed_directions[inbound_to_direction]
                if inbound_fr_direction in self.inbounds_fr_direction:
                    dir_lanes = self.inbounds_fr_direction[inbound_fr_direction]
                    if lane not in dir_lanes:
                        self.inbounds_fr_direction[inbound_fr_direction].append(lane)
                else:
                    self.inbounds_fr_direction[inbound_fr_direction] = [lane]
                if lane not in self.lanes: self.lanes.append(lane)

        self.out_lane_to_signalid = dict()
        for direction in self.downstream:
            dwn_signal = self.downstream[direction]
            if dwn_signal is not None:
                dwn_lane_sets = myconfig[dwn_signal]['lane_sets']
                for key in dwn_lane_sets:
                    if key.split('-')[0] == direction:
                        dwn_lane_set = dwn_lane_sets[key]
                        if dwn_lane_set is None: raise Exception('Invalid signal config')
                        for lane in dwn_lane_set:
                            if lane not in self.outbound_lanes: self.outbound_lanes.append(lane)
                            self.out_lane_to_signalid[lane] = dwn_signal
                            for selfkey in self.lane_sets:
                                if selfkey.split('-')[1] == key.split('-')[0]:
                                    self.lane_sets_outbound[selfkey] += dwn_lane_set
        for key in self.lane_sets_outbound:
            self.lane_sets_outbound[key] = list(set(self.lane_sets_outbound[key]))

        self.waiting_times = dict()

        # logic = self.sumo.trafficlight.Logic(id, 0, 0, phases=self.phases) # not compatible with libsumo
        # programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        # logic = programs[0]
        # logic.type = 0
        # logic.phases = self.phases
        # self.sumo.trafficlight.setProgramLogic(self.id, logic)

        self.signals = None
        self.full_observation = None
        self.last_step_vehicles = None
        self.phases = all_phases
        self.phase = self.next_phase
        self.green_phases = green_phases

        # generate a signal encoding
        self.phase_encoding = defaultdict(list)
        for ph_id in list(self.phases.keys()):
            phase_onehot = [0] * len(self.green_phases)
            if ph_id in self.green_phases:
                phase_onehot[self.green_phases.index(ph_id)] = 1
            else:
                phase_onehot = self.phase_encoding[ph_id - 1]
            self.phase_encoding[ph_id] = phase_onehot

    def get_phase(self):
        return self.sumo.trafficlight.getPhase(self.id)

    def prep_phase(self, new_phase):
        green_phase = self.green_phases[new_phase]
        self.next_phase = green_phase
        self.phase = self.sumo.trafficlight.getPhase(self.id)
        if self.phase != green_phase: # 需要一个transition
            yel_idx = self.phase + 1
            self.sumo.trafficlight.setPhase(self.id, yel_idx)  # turns yellow

    def set_phase(self):
        self.sumo.trafficlight.setPhase(self.id, int(self.next_phase))
        self.phase = self.sumo.trafficlight.getPhase(self.id)

    def observe(self, step_length, distance=150):
        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'queue': 0, 'approach': 0, 'total_wait': 0, 'max_wait': 0, 'length': 0}
            lane_vehicles = self.get_vehicles(lane, distance)
            for vehicle in lane_vehicles:
                all_vehicles.add(vehicle)
                # Update waiting time
                if vehicle in self.waiting_times:
                    self.waiting_times[vehicle] += step_length
                elif self.sumo.vehicle.getWaitingTime(vehicle) > 0:
                    self.waiting_times[vehicle] = self.sumo.vehicle.getWaitingTime(vehicle)

                vehicle_measures = dict()
                vehicle_measures['id'] = vehicle
                vehicle_measures['wait'] = self.waiting_times[vehicle] if vehicle in self.waiting_times else 0
                vehicle_measures['speed'] = self.sumo.vehicle.getSpeed(vehicle)
                vehicle_measures['acceleration'] = self.sumo.vehicle.getAcceleration(vehicle)
                vehicle_measures['position'] = self.sumo.vehicle.getLanePosition(vehicle)
                vehicle_measures['type'] = self.sumo.vehicle.getTypeID(vehicle)
                vehicles.append(vehicle_measures)
                if vehicle_measures['wait'] > 0:
                    lane_measures['total_wait'] = lane_measures['total_wait'] + vehicle_measures['wait']
                    lane_measures['queue'] = lane_measures['queue'] + 1
                    if vehicle_measures['wait'] > lane_measures['max_wait']:
                        lane_measures['max_wait'] = vehicle_measures['wait']
                else:
                    lane_measures['approach'] = lane_measures['approach'] + 1
            lane_measures['vehicles'] = vehicles
            lane_measures['length'] = self.sumo.lane.getLength(lane)
            full_observation[lane] = lane_measures

        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs

            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)

        self.last_step_vehicles = all_vehicles
        self.full_observation = full_observation


    def get_vehicles(self, lane, max_distance):
        detectable = []
        for vehicle in self.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:
                    detectable.append(vehicle)
        return detectable
