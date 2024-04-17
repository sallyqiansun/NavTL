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
    def __init__(self, map_name, sumo, id, yellow_length, all_phases, green_phases):
        self.sumo = sumo
        self.id = id
        self.yellow_time = yellow_length
        self.next_phase = 0

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        lanes = []
        # Unique lanes
        self.lanes = []
        self.outbound_lanes = []

        reversed_directions = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

        # Group of lanes constituting a direction of traffic
        myconfig = signal_configs[map_name]
        self.lane_sets = myconfig[self.id]['lane_sets']
        self.lane_sets_outbound = self.lane_sets.copy()
        for key in self.lane_sets_outbound:     # Remove values from copy
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

        # Populate outbound lane information
        self.out_lane_to_signalid = dict()
        for direction in self.downstream:
            dwn_signal = self.downstream[direction] # downstream intersection
            if dwn_signal is not None:  # A downstream intersection exists
                dwn_lane_sets = myconfig[dwn_signal]['lane_sets']    # downstream signal's lanes
                for key in dwn_lane_sets:   # Find all inbound lanes from upstream, direction e.g. 'S-S'
                    if key.split('-')[0] == direction:    # Downstream direction matches
                        dwn_lane_set = dwn_lane_sets[key]
                        if dwn_lane_set is None: raise Exception('Invalid signal config')
                        for lane in dwn_lane_set:
                            if lane not in self.outbound_lanes: self.outbound_lanes.append(lane)
                            self.out_lane_to_signalid[lane] = dwn_signal
                            for selfkey in self.lane_sets:
                                if selfkey.split('-')[1] == key.split('-')[0]:    # Out dir. matches dwnstrm in dir.
                                    self.lane_sets_outbound[selfkey] += dwn_lane_set
        for key in self.lane_sets_outbound:  # Remove duplicates
            self.lane_sets_outbound[key] = list(set(self.lane_sets_outbound[key]))

        self.waiting_times = dict()     # SUMO's WaitingTime and AccumulatedWaiting are both wrong for multiple signals

        # logic = self.sumo.trafficlight.Logic(id, 0, 0, phases=self.phases) # not compatible with libsumo
        # programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        # logic = programs[0]
        # logic.type = 0
        # logic.phases = self.phases
        # self.sumo.trafficlight.setProgramLogic(self.id, logic)

        self.signals = None     # Used to allow signal sharing
        self.full_observation = None
        self.last_step_vehicles = None
        self.phases = all_phases
        self.phase = self.next_phase
        self.green_phases = green_phases

        # generate a signal encoding
        self.phase_encoding = defaultdict(list)
        for ph_id in list(self.phases.keys()):
            phase_onehot = [0] * len(self.green_phases)
            if ph_id in self.green_phases: # if green, onehot is directly its own onehot
                phase_onehot[self.green_phases.index(ph_id)] = 1
            else: # if yellow, onehot is its prev green's onehot
                phase_onehot = self.phase_encoding[ph_id - 1]
            self.phase_encoding[ph_id] = phase_onehot


    def generate_config(self):
        print('GENERATING CONFIG')
        # TODO raise Exception('Invalid signal config')
        index_to_movement = {0: 'S-W', 1: 'S-S', 2: 'S-E', 3: 'W-N', 4: 'W-W', 5: 'W-S', 6: 'N-E',
                             7: 'N-N', 8: 'N-W', 9: 'E-S', 10: 'E-E', 11: 'E-N'}
        self.lane_sets = {}
        for idx, movement in index_to_movement.items():
            self.lane_sets[movement] = []
        self.lane_sets_outbound = {}
        self.downstream = {'N': None, 'E': None, 'S': None, 'W': None}

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        #print(self.id, links)
        for i, link in enumerate(links):
            link = link[0]  # unpack so link[0] is inbound, link[1] outbound
            if link[0] not in self.lanes: self.lanes.append(link[0])
            # Group of lanes constituting a direction of traffic
            if i % 3 == 0:
                index = int(i / 3)
                self.lane_sets[index_to_movement[index]].append(link[0])
        #print(self.id, self.lane_sets)
        """split = self.lane_sets['S-W'][0].split('_')[0]
        if 'np' not in split: self.downstream['N'] = split
        split = self.lane_sets['W-N'][0].split('_')[0]
        if 'np' not in split: self.downstream['E'] = split
        split = self.lane_sets['N-E'][0].split('_')[0]
        if 'np' not in split: self.downstream['S'] = split
        split = self.lane_sets['E-S'][0].split('_')[0]
        if 'np' not in split: self.downstream['W'] = split"""
        lane = self.lane_sets['S-S'][0]
        fr_sig = re.findall('[a-zA-Z]+[0-9]+', lane)[0]
        fringes, isfringe = ['top', 'right', 'left', 'bottom'], False
        for fringe in fringes:
            if fringe in fr_sig: isfringe = True
        if not isfringe: self.downstream['N'] = fr_sig

        lane = self.lane_sets['N-N'][0]
        fr_sig = re.findall('[a-zA-Z]+[0-9]+', lane)[0]
        fringes, isfringe = ['top', 'right', 'left', 'bottom'], False
        for fringe in fringes:
            if fringe in fr_sig: isfringe = True
        if not isfringe: self.downstream['S'] = fr_sig

        lane = self.lane_sets['W-W'][0]
        fr_sig = re.findall('[a-zA-Z]+[0-9]+', lane)[0]
        fringes, isfringe = ['top', 'right', 'left', 'bottom'], False
        for fringe in fringes:
            if fringe in fr_sig: isfringe = True
        if not isfringe: self.downstream['E'] = fr_sig

        lane = self.lane_sets['E-E'][0]
        fr_sig = re.findall('[a-zA-Z]+[0-9]+', lane)[0]
        fringes, isfringe = ['top', 'right', 'left', 'bottom'], False
        for fringe in fringes:
            if fringe in fr_sig: isfringe = True
        if not isfringe: self.downstream['W'] = fr_sig
        print("'" + self.id + "'" + ": {")
        print("'lane_sets':" + str(self.lane_sets) + ',')
        print("'downstream':" + str(self.downstream) + '},')

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

    def observe(self, step_length, distance=200):
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
                elif self.sumo.vehicle.getWaitingTime(vehicle) > 0:  # Vehicle stopped here, add it
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
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)

        self.last_step_vehicles = all_vehicles
        self.full_observation = full_observation

    # Remove undetectable vehicles from lane
    def get_vehicles(self, lane, max_distance):
        detectable = []
        for vehicle in self.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable
