
import numpy as np
import torch
import torch.nn as nn
from mdp_config import mdp_configs
def signal_only(signals):
    # observations = dict()
    # for signal_id in signals:
    #     signal = signals[signal_id]
    #     obs = [signal.phase]
    #     for direction in signal.lane_sets:
    #         # Add inbound
    #         queue_length = 0
    #         for lane in signal.lane_sets[direction]:
    #             queue_length += signal.full_observation[lane]['queue']
    #         # Subtract downstream
    #         for lane in signal.lane_sets_outbound[direction]:
    #             dwn_signal = signal.out_lane_to_signalid[lane]
    #             if dwn_signal in signal.signals:
    #                 queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
    #         obs.append(queue_length)
    #     observations[signal_id] = torch.tensor(obs).to(torch.float32)
    # return observations
    observations = dict()
    for signal_id in signals:
        obs = []
        signal = signals[signal_id]
        enc = signal.phase_encoding[signal.phase]
        obs.extend(enc)
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
            obs.append(queue_length / 10.0)
            obs.append(veh_count / 10.0)
            if veh_count > 0:
                avg_speed = total_speed / veh_count
            obs.append(avg_speed / 10.0)
        observations[signal_id] = torch.tensor(obs).to(torch.float32)
    return observations

def navTL(signals):
    # 方向级
    # observations = dict()
    # for signal_id in signals:
    #     signal = signals[signal_id]
    #     obs = [signal.phase]
    #     north = {'veh_count': 0, 'total_speed': 0, 'pressure': 0}
    #     south, east, west = north.copy(), north.copy(), north.copy()
    #     for direction in signal.lane_sets:
    #         queue_length = 0
    #         veh_count = 0
    #         total_speed, avg_speed = 0, 0
    #         for lane in signal.lane_sets[direction]:
    #             queue_length += signal.full_observation[lane]['queue']
    #             vehicles = signal.full_observation[lane]['vehicles']
    #             veh_count += len(vehicles)
    #             for vehicle in vehicles:
    #                 total_speed += vehicle['speed']
    #             length = signal.full_observation[lane]['length']
    #         for lane in signal.lane_sets_outbound[direction]:
    #             dwn_signal = signal.out_lane_to_signalid[lane]
    #             if dwn_signal in signal.signals:
    #                 queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
    #         if '_N' in direction:
    #             north['veh_count'] += veh_count
    #             north['total_speed'] += total_speed
    #             north['pressure'] += queue_length
    #         elif '_S' in direction:
    #             south['veh_count'] += veh_count
    #             south['total_speed'] += total_speed
    #             south['pressure'] += queue_length
    #         elif '_E' in direction:
    #             east['veh_count'] += veh_count
    #             east['total_speed'] += total_speed
    #             east['pressure'] += queue_length
    #         elif '_W' in direction:
    #             west['veh_count'] += veh_count
    #             west['total_speed'] += total_speed
    #             west['pressure'] += queue_length
    #     if north['veh_count'] > 0:
    #         obs.extend([north['veh_count'], north['total_speed'] / north['veh_count'], north['pressure']])
    #     else:
    #         obs.extend([north['veh_count'], 0, north['pressure']])
    #     if east['veh_count'] > 0:
    #         obs.extend([east['veh_count'], east['total_speed'] / east['veh_count'], east['pressure']])
    #     else:
    #         obs.extend([east['veh_count'], 0, east['pressure']])
    #     if south['veh_count'] > 0:
    #         obs.extend([south['veh_count'], south['total_speed'] / south['veh_count'], south['pressure']])
    #     else:
    #         obs.extend([south['veh_count'], 0, south['pressure']])
    #     if west['veh_count'] > 0:
    #         obs.extend([west['veh_count'], west['total_speed'] / west['veh_count'], west['pressure']])
    #     else:
    #         obs.extend([west['veh_count'], 0, west['pressure']])
    #     observations[signal_id] = torch.tensor(obs).to(torch.float32)
    # return observations


    # 流向级
    observations = dict()
    for signal_id in signals:
        obs = []
        signal = signals[signal_id]
        enc = signal.phase_encoding[signal.phase]
        obs.extend(enc)
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
            obs.append(queue_length / 10.0)
            obs.append(veh_count / 10.0)
            if veh_count > 0:
                avg_speed = total_speed / veh_count
            obs.append(avg_speed / 10.0)
        observations[signal_id] = torch.tensor(obs).to(torch.float32)
    return observations

def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)  # if phase is active for that lane, then 1
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach']) # int, number of approaching vehicles
            lane_obs.append(signal.full_observation[lane]['total_wait']) # int, total waiting time
            lane_obs.append(signal.full_observation[lane]['queue']) # int, queue length

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += vehicle['speed']
            lane_obs.append(total_speed) # float, total speed of vehicles

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations

def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'] / 28)
            lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
            lane_obs.append(signal.full_observation[lane]['queue'] / 28)

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations

def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'])
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                total_speed = total_speed / len(vehicles)
                tot_approach += (signal.full_observation[lane]['approach'])

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = torch.tensor(obs).to(torch.float32)
    return observations

def wave(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = 0
            for lane in signal.lane_sets[direction]:
                wave_sum += signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations

def shared(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        phase = dict()
        for i, lane in enumerate(signal.lanes):
            if i == act_index:
                phase[lane] = 1  # if phase is active for that lane, then 1
            else:
                phase[lane] = 0

        for direction in signal.lane_sets:
            act = []
            approach = 0
            wait = 0
            queue = 0
            speed = 0.0
            for lane in signal.lane_sets[direction]:
                act.append(phase[lane])
                approach += signal.full_observation[lane]['approach'] # int, number of approaching vehicles
                wait += signal.full_observation[lane]['total_wait'] # int, total waiting time
                queue += signal.full_observation[lane]['queue'] # int, queue length
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                speed += total_speed # float, total speed of vehicles
            if len(signal.lane_sets[direction]) > 0:
                act = max(set(act), key = act.count)
                approach = approach / len(signal.lane_sets[direction])
                wait = wait / len(signal.lane_sets[direction])
                queue = queue / len(signal.lane_sets[direction])
                speed = speed / len(signal.lane_sets[direction])
            else:
                act = 2

            obs.append([act, approach, wait, queue, speed])
        observations[signal_id] = torch.from_numpy(np.expand_dims(np.asarray(obs), axis=0)).to(torch.float32)
    return observations

def mplight(signals):
    # observations = dict()
    # for signal_id in signals:
    #     signal = signals[signal_id]
    #     veh_count = []
    #     for direction in signal.lane_sets:
    #         count = 0
    #         for lane in signal.lane_sets[direction]:
    #             vehicles = signal.full_observation[lane]['vehicles']
    #             count += len(vehicles)
    #         veh_count.append(count)
    #     # phase = nn.functional.one_hot(torch.tensor(signal.phase), 13).tolist()
    #     phase = signal.phase_encoding[signal.phase]
    #     veh_count.extend(phase)
    #     observations[signal_id] = torch.tensor(veh_count).to(torch.float32)
    # return observations

    # observations = dict()
    # for signal_id in signals:
    #     obs = []
    #     signal = signals[signal_id]
    #     for direction in signal.lane_sets:
    #         queue_length = 0
    #         for lane in signal.lane_sets[direction]:
    #             queue_length += signal.full_observation[lane]['queue']
    #         for lane in signal.lane_sets_outbound[direction]:
    #             dwn_signal = signal.out_lane_to_signalid[lane]
    #             if dwn_signal in signal.signals:
    #                 queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
    #         obs.append(queue_length)
    #     observations[signal_id] = torch.tensor(obs).to(torch.float32)
    # return observations

    observations = dict()
    for signal_id in signals:
        obs = dict()
        signal = signals[signal_id]
        for direction in signal.lane_sets:
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs[direction] = queue_length
        observations[signal_id] = obs
    return observations

def veh_count(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        veh_count = []
        for direction in signal.lane_sets:
            count = 0
            for lane in signal.lane_sets[direction]:
                vehicles = signal.full_observation[lane]['vehicles']
                count += len(vehicles)
            veh_count.append(count)
        phase = nn.functional.one_hot(torch.tensor(signal.phase), 13).tolist()
        veh_count.extend(phase)
        observations[signal_id] = torch.tensor(veh_count).to(torch.float32)
    return observations

def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations

def veh_only(signals):
    return veh_count(signals)