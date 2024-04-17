import rewards
import states
from agents.mp import maxpressure
from agents.a2c import A2C as A2Cnavigation
from agents.h_a2c_embedding import HA2C as HA2Cembedding
from agents.dqn import DQN as DQNnavigation
from agents.h_dqn_embedding import HDQN as HDQNembedding
from agents.ppo import PPO as PPOnavigation
from agents.h_dqn_graph import HDQN as HDQNgraph
from agents.h_a2c_graph import HA2C as HA2Cgraph
from agents.h_a2c_intention import HA2C as HA2Cintention
from agents.h_dqn_intention import HDQN as HDQNintention
from agents.joint_dqn import DQN as jointDQN
from agents.colight import colight 
from agents.fma2c import fma2c
from agents.navTL import NavTL
from agents.dqn_concat import DQNConcat
from agents.colight_nav import ColiNav

agent_configs = {
    'MAXPRESSURE': {
        'agent': maxpressure,
        'state': states.mplight,
        'reward': rewards.queue,
        'max_distance': 200
    },
    'A2Cnavigation': {
        'agent': A2Cnavigation,
        'state': states.veh_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    },
    'DQNnavigation': {
        'agent': DQNnavigation,
        'state': states.veh_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    },
    'A2Csignal': {
        'agent': HA2Cembedding,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 32, 
        'DROPOUT': 0.5
    },
    'A2Csignalgraph': {
        'agent': HA2Cgraph,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    },
    'HA2Cembedding': {
        'agent': HA2Cembedding,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    },
    'HA2Cgraph': {
        'agent': HA2Cgraph,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 32, 
        'DROPOUT': 0.5
    },
    'DQNsignal': {
        'agent': HDQNembedding,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'DQNsignalgraph': {
        'agent': HDQNgraph,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 128, 
        'DROPOUT': 0.5
    }, 
    'HDQNembedding': {
        'agent': HDQNembedding,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'HDQNgraph': {
        'agent': HDQNgraph,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'HA2Cintention': {
        'agent': HA2Cintention,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 64,
        'GAMMA': 0.95,
        'LR': 0.001, 
        'HIDDEN': 32, 
        'DROPOUT': 0.5
    },
    'HDQNintention': {
        'agent': HDQNintention,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 128, 
        'DROPOUT': 0.5
    }, 
    'jointDQN': {
        'agent': jointDQN,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'colight': {
        'agent': colight,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'fma2c': {
        'agent': fma2c,
        'state': states.signal_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'dijkstra':{
        'agent': DQNnavigation,
        'state': states.veh_only,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'navTL': {
        'agent': NavTL,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    },
    'xrouting': {
        'agent': PPOnavigation,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64
    },
    'DQNConcat': {
        'agent': DQNConcat,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
    'ColiNav': {
        'agent': ColiNav,
        'state': states.navTL,
        'reward': rewards.queue,
        'max_distance': 200,
        'BATCH_SIZE': 16,
        'GAMMA': 0.95,
        'EPS_START': 0.8,
        'EPS_END': 0.01,
        'EPS_DECAY': 0.9995,
        'TARGET_UPDATE': 10, 
        'LR': 0.001, 
        'HIDDEN': 64, 
        'DROPOUT': 0.5
    }, 
}

