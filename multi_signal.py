
import os
import sys
import torch

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME")
import traci
import sumolib
import gym
from traffic_signal import Signal
from vehicle import Vehicle
import math
import random
from collections import defaultdict

class MultiSignal(gym.Env):
    def __init__(self, run_name, map_name, net, state_fn, reward_fn, signal_config, route=None, \
    gui=False, end_time=3600, step_length=10, yellow_length=4, max_distance=200, lights='', \
    log_dir='/', libsumo=False, warmup=0):
        print(map_name, net, state_fn.__name__, reward_fn.__name__)
        self.libsumo = libsumo
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup
        self.map_name = map_name
        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.connection_name = run_name + '-' + map_name
        self.signal_config = signal_config
        # Run some steps in the simulation with default light configurations to detect phases
        if self.route is not None:
            if self.gui:
                sumo_cmd = [sumolib.checkBinary('sumo-gui'), '-n', net, '-r', \
                            self.route + '.rou.xml', '--no-warnings', 'True', '--waiting-time-memory', '20']
            else:
                sumo_cmd = [sumolib.checkBinary('sumo'), '-n', net, '-r', \
                            self.route + '.rou.xml', '--no-warnings', 'True', '--waiting-time-memory', '20']
        else:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True', '--waiting-time-memory', '20']
        
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.connection_name)  # start sumo connection
            self.sumo = traci.getConnection(self.connection_name)
        self.signal_ids = self.sumo.trafficlight.getIDList()

        valid_phases = defaultdict(dict)
        self.green_phases = defaultdict(list)
        for ts in self.signal_ids:
            for ind, phase in enumerate(self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(ts)[0].phases):
                valid_phases[ts][ind] = phase.state
                if phase.minDur > 5.0: # green
                    self.green_phases[ts].append(ind)
        self.phases = valid_phases
        self.signals = dict()

        self.all_ts_ids = lights if len(lights) > 0 else self.sumo.trafficlight.getIDList()
        self.signal_ids = self.all_ts_ids

        # Pull signal observation shapes
        self.signal_obs_shape = dict()
        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts], self.green_phases[ts])
        for ts in self.all_ts_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)
        observations = self.state_fn(self.signals)
        for ts in observations:
            self.signal_obs_shape[ts] = observations[ts].shape
            # self.signal_obs_shape[ts] = torch.Size([1]) # for maxpressure this does not play a role

        self.run = 0
        self.metrics = []
        self.wait_metric = dict()

        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.connection_name = run_name + '-' + map_name + '-' + str(len(lights)) \
                               + '-' + state_fn.__name__ + '-' + reward_fn.__name__
        if not os.path.exists(log_dir + self.connection_name):
            os.makedirs(log_dir + self.connection_name)
        self.sumo_cmd = None
        print('Connection ID', self.connection_name)

        self.cav_ids = []
        self.penetration_rate = 0.3
        self.cavs = dict()
        self.total_cav = 0
        self.throughput = 0
        self.cav_signals = dict()
        self.signal_cavs = defaultdict(list)
        self.veh_destinations = dict()
        self.existing_routes = []

    def step_sim(self):
        self.add_cav(self.sumo.simulation.getDepartedNumber())
        self.sumo.simulationStep()
        self.remove_finished_cav()   

    def add_cav(self, veh_num):
        cav_num = self.penetration_rate * (veh_num)
        existing_routes = self.sumo.route.getIDList()
        existing_routes = [r for r in existing_routes if 'cav' not in r]
        cav_num = round(cav_num)
        for i in range(self.total_cav + 1, self.total_cav + cav_num + 1):
            route = random.sample(existing_routes, 1)[0]
            veh_id = "cav_" + str(i)
            self.sumo.vehicle.add(veh_id, routeID=route, departPos="free", departSpeed='10')
            self.sumo.vehicle.setColor(veh_id, color=(0, 255, 0, 238))
            self.sumo.vehicle.setSpeedMode(veh_id, 31)
            self.cav_ids.append(veh_id)
            self.cavs.update({veh_id: Vehicle(veh_id, self.sumo, self.net, self.signal_config, self.signal_encoding, self.signals)})
        self.total_cav += cav_num

    def update_cav_signals(self):
        net = sumolib.net.readNet(self.net, withInternal=True)
        self.signal_cavs = defaultdict(list)
        for signal in self.signal_ids:
            lanes = self.signals[signal].lanes
            for lane in lanes:
                edge = self.sumo.lane.getEdgeID(lane)
                vehicles = self.signals[signal].get_vehicles(lane, self.max_distance)
                cavs = [v for v in vehicles if 'cav' in v]
                self.signal_cavs[signal].extend(cavs)
                for cav in cavs:
                    self.cav_edges[cav] = [edge, self.sumo.vehicle.getNextTLS(cav)[0][2]]
        self.cav_signals = dict.fromkeys(self.cav_ids)
        for signal, cav_ls in self.signal_cavs.items():
            for cav in cav_ls:
                if cav in self.cav_signals:
                    self.cav_signals[cav] = signal


    def remove_finished_cav(self):
        self.completed_cavs = dict()
        for cav_id in self.cav_ids[:]:
            if cav_id not in self.sumo.vehicle.getIDList():
                self.cav_ids.remove(cav_id)
                self.cavs.pop(cav_id)
                self.throughput += 1
            elif self.sumo.vehicle.getRoadID(cav_id) == self.sumo.vehicle.getRoute(cav_id)[-1]:
                self.completed_cavs.update({cav_id: self.cavs[cav_id]})

    def reset(self):
        if self.run != 0:
            if not self.libsumo: traci.switch(self.connection_name)
            traci.close()
            self.save_metrics()
        self.metrics = []
        self.run += 1

        # Start a new simulation
        self.sumo_cmd = []
        if self.gui:
            self.sumo_cmd.append(sumolib.checkBinary('sumo-gui'))
            self.sumo_cmd.append('--start')
        else:
            self.sumo_cmd.append(sumolib.checkBinary('sumo'))
        if self.route is not None:
            # self.sumo_cmd += ['-n', self.net, '-r', self.route + '_' + str(self.run) + '.rou.xml']
            self.sumo_cmd += ['-n', self.net, '-r', self.route + '.rou.xml']
        else:
            self.sumo_cmd += ['-c', self.net]
        self.sumo_cmd += ['--random', '--time-to-teleport', '-1', '--tripinfo-output',
                          self.log_dir + self.connection_name + os.sep + 'tripinfo_' + str(self.run) + '.xml',
                          '--tripinfo-output.write-unfinished',
                          '--no-step-log', 'True',
                          '--no-warnings', 'True']
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        for _ in range(self.warmup):
            self.sumo.simulationStep()

        self.signal_ids = self.all_ts_ids

        for ts in self.signal_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts], self.green_phases[ts])
            self.wait_metric[ts] = 0.0
        for ts in self.signal_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)

        self.cav_ids = []
        self.cavs = dict()
        self.total_cav = 0
        self.throughput = 0
        self.cav_signals = dict()
        self.cav_edges = dict()
        self.signal_cavs = defaultdict(list)

        net = sumolib.net.readNet(self.net, withInternal=True)
        signal_encoding = defaultdict(list)
        for ind, signal in enumerate(self.signal_ids):
            signal_onehot = [0] * len(self.signal_ids)
            signal_onehot[ind] = 1
            signal_encoding[signal].extend(signal_onehot)
        self.signal_encoding = signal_encoding
        return self.state_fn(self.signals), {cav: self.cavs[cav].get_state() for cav in self.cavs}

    def step(self, signal_act, veh_act):
        # signals take actions
        for signal in self.signals:
            self.signals[signal].prep_phase(signal_act[signal])

        for step in range(self.yellow_length):
            self.veh_step(veh_act)
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        self.remove_finished_cav()
        for step in range(self.step_length - self.yellow_length):
            self.veh_step(veh_act)
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)
        # signals observe new state and reward
        signal_rewards = self.reward_fn(self.signals)
        signal_observations = self.state_fn(self.signals)

        # vehicles observe new state and reward
        veh_rewards = {cav: self.cavs[cav].get_reward() for cav in self.cavs}
        veh_observations = {cav: self.cavs[cav].get_state() for cav in self.cavs}

        if len(veh_rewards) > 0:
            metric_reward = sum(veh_rewards.values()) / len(veh_rewards)
        else:
            metric_reward = 0
        # self.calc_metrics(signal_rewards)
        # self.calc_metrics(metric_reward)
        done = self.sumo.simulation.getTime() >= self.end_time
        return [signal_observations, signal_rewards, done, veh_observations, veh_rewards]


    def veh_step(self, veh_act):
        net = sumolib.net.readNet(self.net, withInternal=True)
        for cav in veh_act:
            if cav in self.cav_ids:
                route = self.sumo.vehicle.getRoute(cav)
                edge = self.sumo.vehicle.getRoadID(cav)
                if ':' in edge:  # internal edge
                    continue
                new_route = [edge]
                if edge in self.direction_encoding:
                    outgoing_idx = self.direction_encoding[edge][0]
                    if outgoing_idx[veh_act[cav]] == 0:
                        continue
                    outgoing = self.direction_encoding[edge][1]
                    edge = outgoing[veh_act[cav]]
                    new_route.extend(list(self.sumo.simulation.findRoute(edge, route[-1]).edges))
                    self.sumo.vehicle.setRoute(cav, new_route)

    def calc_metrics(self, worker_rewards, manager_rewards):
        queue_lengths, waiting_time = dict(), dict()
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_length, wait = 0, 0
            for lane in signal.lanes:
                queue_length += signal.full_observation[lane]['queue']
                wait += signal.full_observation[lane]['total_wait']
            queue_lengths[signal_id] = queue_length
            waiting_time[signal_id] = wait
        queue = sum(queue_lengths.values()) / len(queue_lengths)
        wait = sum(waiting_time.values()) / len(waiting_time)
        self.metrics.append({
            'step': self.sumo.simulation.getTime(),
            'worker_reward': worker_rewards,
            'manager_reward': manager_rewards, 
            'queue': queue, 
            'wait': wait
        })


    def save_metrics(self):
        log = self.log_dir + self.connection_name + os.sep + 'metrics_' + str(self.run) + '.csv'
        print('saving', self.connection_name, self.run, ' to ', self.log_dir)
        with open(log, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ''
                for metric in ['step', 'worker_reward', 'manager_reward']:
                    csv_line = csv_line + str(line[metric]) + ', '
                output_file.write(csv_line + '\n')

    def close(self):
        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.save_metrics()


    def update_veh_routes(self):
        net = sumolib.net.readNet(self.net, withInternal=True)
        for cav in self.cav_ids:
            if cav not in self.veh_destinations:
                route = self.sumo.vehicle.getRoute(cav)
                destination = net.getEdge(route[-1]).getID()
                self.veh_destinations[cav] = destination
