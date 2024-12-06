"""
traffic signal configuration file
"""


signal_configs = {
	'grid4x4': {
		'direction_encoding': {
			'A0A1': [[1, 1, 1], ['A1B1', 'A1A2', 'A1left1']],
			'A0B0': [[1, 1, 1], ['B0bottom1', 'B0C0', 'B0B1']],
			'A1A0': [[1, 1, 1], ['A0left0', 'A0bottom0', 'A0B0']],
			'A1A2': [[1, 1, 1], ['A2B2', 'A2A3', 'A2left2']],
			'A1B1': [[1, 1, 1], ['B1B0', 'B1C1', 'B1B2']],
			'A2A1': [[1, 1, 1], ['A1left1', 'A1A0', 'A1B1']],
			'A2A3': [[1, 1, 1], ['A3B3', 'A3top0', 'A3left3']],
			'A2B2': [[1, 1, 1], ['B2B1', 'B2C2', 'B2B3']],
			'A3A2': [[1, 1, 1], ['A2left2', 'A2A1', 'A2B2']],
			'A3B3': [[1, 1, 1], ['B3B2', 'B3C3', 'B3top1']],
			'B0A0': [[1, 1, 1], ['A0A1', 'A0left0', 'A0bottom0']],
			'B0B1': [[1, 1, 1], ['B1C1', 'B1B2', 'B1A1']],
			'B0C0': [[1, 1, 1], ['C0bottom2', 'C0D0', 'C0C1']],
			'B1A1': [[1, 1, 1], ['A1A2', 'A1left1', 'A1A0']],
			'B1B0': [[1, 1, 1], ['B0A0', 'B0bottom1', 'B0C0']],
			'B1B2': [[1, 1, 1], ['B2C2', 'B2B3', 'B2A2']],
			'B1C1': [[1, 1, 1], ['C1C0', 'C1D1', 'C1C2']],
			'B2A2': [[1, 1, 1], ['A2A3', 'A2left2', 'A2A1']],
			'B2B1': [[1, 1, 1], ['B1A1', 'B1B0', 'B1C1']],
			'B2B3': [[1, 1, 1], ['B3C3', 'B3top1', 'B3A3']],
			'B2C2': [[1, 1, 1], ['C2C1', 'C2D2', 'C2C3']],
			'B3A3': [[1, 1, 1], ['A3top0', 'A3left3', 'A3A2']],
			'B3B2': [[1, 1, 1], ['B2A2', 'B2B1', 'B2C2']],
			'B3C3': [[1, 1, 1], ['C3C2', 'C3D3', 'C3top2']],
			'C0B0': [[1, 1, 1], ['B0B1', 'B0A0', 'B0bottom1']],
			'C0C1': [[1, 1, 1], ['C1D1', 'C1C2', 'C1B1']],
			'C0D0': [[1, 1, 1], ['D0bottom3', 'D0right0', 'D0D1']],
			'C1B1': [[1, 1, 1], ['B1B2', 'B1A1', 'B1B0']],
			'C1C0': [[1, 1, 1], ['C0B0', 'C0bottom2', 'C0D0']],
			'C1C2': [[1, 1, 1], ['C2D2', 'C2C3', 'C2B2']],
			'C1D1': [[1, 1, 1], ['D1D0', 'D1right1', 'D1D2']],
			'C2B2': [[1, 1, 1], ['B2B3', 'B2A2', 'B2B1']],
			'C2C1': [[1, 1, 1], ['C1B1', 'C1C0', 'C1D1']],
			'C2C3': [[1, 1, 1], ['C3D3', 'C3top2', 'C3B3']],
			'C2D2': [[1, 1, 1], ['D2D1', 'D2right2', 'D2D3']],
			'C3B3': [[1, 1, 1], ['B3top1', 'B3A3', 'B3B2']],
			'C3C2': [[1, 1, 1], ['C2B2', 'C2C1', 'C2D2']],
			'C3D3': [[1, 1, 1], ['D3D2', 'D3right3', 'D3top3']],
			'D0C0': [[1, 1, 1], ['C0C1', 'C0B0', 'C0bottom2']],
			'D0D1': [[1, 1, 1], ['D1right1', 'D1D2', 'D1C1']],
			'D1C1': [[1, 1, 1], ['C1C2', 'C1B1', 'C1C0']],
			'D1D0': [[1, 1, 1], ['D0C0', 'D0bottom3', 'D0right0']],
			'D1D2': [[1, 1, 1], ['D2right2', 'D2D3', 'D2C2']],
			'D2C2': [[1, 1, 1], ['C2C3', 'C2B2', 'C2C1']],
			'D2D1': [[1, 1, 1], ['D1C1', 'D1D0', 'D1right1']],
			'D2D3': [[1, 1, 1], ['D3right3', 'D3top3', 'D3C3']],
			'D3C3': [[1, 1, 1], ['C3top2', 'C3B3', 'C3C2']],
			'D3D2': [[1, 1, 1], ['D2C2', 'D2D1', 'D2right2']],
			'bottom0A0': [[1, 1, 1], ['A0B0', 'A0A1', 'A0left0']],
			'bottom1B0': [[1, 1, 1], ['B0C0', 'B0B1', 'B0A0']],
			'bottom2C0': [[1, 1, 1], ['C0D0', 'C0C1', 'C0B0']],
			'bottom3D0': [[1, 1, 1], ['D0right0', 'D0D1', 'D0C0']],
			'left0A0': [[1, 1, 1], ['A0bottom0', 'A0B0', 'A0A1']],
			'left1A1': [[1, 1, 1], ['A1A0', 'A1B1', 'A1A2']],
			'left2A2': [[1, 1, 1], ['A2A1', 'A2B2', 'A2A3']],
			'left3A3': [[1, 1, 1], ['A3A2', 'A3B3', 'A3top0']],
			'right0D0': [[1, 1, 1], ['D0D1', 'D0C0', 'D0bottom3']],
			'right1D1': [[1, 1, 1], ['D1D2', 'D1C1', 'D1D0']],
			'right2D2': [[1, 1, 1], ['D2D3', 'D2C2', 'D2D1']],
			'right3D3': [[1, 1, 1], ['D3top3', 'D3C3', 'D3D2']],
			'top0A3': [[1, 1, 1], ['A3left3', 'A3A2', 'A3B3']],
			'top1B3': [[1, 1, 1], ['B3A3', 'B3B2', 'B3C3']],
			'top2C3': [[1, 1, 1], ['C3B3', 'C3C2', 'C3D3']],
			'top3D3': [[1, 1, 1], ['D3C3', 'D3D2', 'D3right3']]}, 
		'phase_pairs': [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]],
		'valid_acts': {
			'A0': [0, 1, 2, 3, 4, 5, 6, 7], 
			'A1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'A2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'A3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'B0':[0, 1, 2, 3, 4, 5, 6, 7], 
			'B1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'B2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'B3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'C0':[0, 1, 2, 3, 4, 5, 6, 7], 
			'C1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'C2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'C3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'D0':[0, 1, 2, 3, 4, 5, 6, 7], 
			'D1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'D2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'D3':[0, 1, 2, 3, 4, 5, 6, 7]
		},
		'margins':{
			'top0': {'A3': 'S'}, 
			'A3': {'top0': 'N', 'left3':'W'}, 
			'top1': {'B3': 'S'}, 
			'B3': {'top1': 'N'}, 
			'top2': {'C3': 'S'}, 
			'C3': {'top2': 'N'}, 
			'top3': {'D3': 'S'}, 
			'D3': {'top3': 'N', 'right3':'E'}, 
			'right0': {'D0':'W'}, 
			'D0': {'right0':'E', 'bottom3':'S'}, 
			'right1': {'D1':'W'}, 
			'D1': {'right1':'E'}, 
			'right2': {'D2':'W'}, 
			'D2': {'right2':'E'}, 
			'right3': {'D3':'W'}, 
			'bottom0': {'A0':'N'}, 
			'A0': {'bottom0':'S', 'left0':'W'}, 
			'bottom1': {'B0':'N'}, 
			'B0': {'bottom1':'S'}, 
			'bottom2': {'C0':'N'}, 
			'C0': {'bottom2':'S'}, 
			'bottom3': {'D0':'N'}, 
			'left0': {'A0':'E'}, 
			'left1': {'A1':'E'}, 
			'A1': {'left1':'W'}, 
			'left2': {'A2':'E'}, 
			'A2': {'left2':'W'}, 
			'left3': {'A3':'E'}
		}, 
		'A0': {
		'lane_sets':{'S-W': ['A1A0_0'], 'S-S': ['A1A0_1'], 'S-E': ['A1A0_2'], 'W-N': ['B0A0_0'], 'W-W': ['B0A0_1'], 'W-S': ['B0A0_2'], 'N-E': ['bottom0A0_0'], 'N-N': ['bottom0A0_1'], 'N-W': ['bottom0A0_2'], 'E-S': ['left0A0_0'], 'E-E': ['left0A0_1'], 'E-N': ['left0A0_2']},
		'downstream':{'N': 'A1', 'E': 'B0', 'S': None, 'W': None}},

		'A1': {
		'lane_sets':{'S-W': ['A2A1_0'], 'S-S': ['A2A1_1'], 'S-E': ['A2A1_2'], 'W-N': ['B1A1_0'], 'W-W': ['B1A1_1'], 'W-S': ['B1A1_2'], 'N-E': ['A0A1_0'], 'N-N': ['A0A1_1'], 'N-W': ['A0A1_2'], 'E-S': ['left1A1_0'], 'E-E': ['left1A1_1'], 'E-N': ['left1A1_2']},
		'downstream':{'N': 'A2', 'E': 'B1', 'S': 'A0', 'W': None}},

		'A2': {
		'lane_sets':{'S-W': ['A3A2_0'], 'S-S': ['A3A2_1'], 'S-E': ['A3A2_2'], 'W-N': ['B2A2_0'], 'W-W': ['B2A2_1'], 'W-S': ['B2A2_2'], 'N-E': ['A1A2_0'], 'N-N': ['A1A2_1'], 'N-W': ['A1A2_2'], 'E-S': ['left2A2_0'], 'E-E': ['left2A2_1'], 'E-N': ['left2A2_2']},
		'downstream':{'N': 'A3', 'E': 'B2', 'S': 'A1', 'W': None}},

		'A3': {
		'lane_sets':{'S-W': ['top0A3_0'], 'S-S': ['top0A3_1'], 'S-E': ['top0A3_2'], 'W-N': ['B3A3_0'], 'W-W': ['B3A3_1'], 'W-S': ['B3A3_2'], 'N-E': ['A2A3_0'], 'N-N': ['A2A3_1'], 'N-W': ['A2A3_2'], 'E-S': ['left3A3_0'], 'E-E': ['left3A3_1'], 'E-N': ['left3A3_2']},
		'downstream':{'N': None, 'E': 'B3', 'S': 'A2', 'W': None}},

		'B0': {
		'lane_sets':{'S-W': ['B1B0_0'], 'S-S': ['B1B0_1'], 'S-E': ['B1B0_2'], 'W-N': ['C0B0_0'], 'W-W': ['C0B0_1'], 'W-S': ['C0B0_2'], 'N-E': ['bottom1B0_0'], 'N-N': ['bottom1B0_1'], 'N-W': ['bottom1B0_2'], 'E-S': ['A0B0_0'], 'E-E': ['A0B0_1'], 'E-N': ['A0B0_2']},
		'downstream':{'N': 'B1', 'E': 'C0', 'S': None, 'W': 'A0'}},

		'B1': {
		'lane_sets':{'S-W': ['B2B1_0'], 'S-S': ['B2B1_1'], 'S-E': ['B2B1_2'], 'W-N': ['C1B1_0'], 'W-W': ['C1B1_1'], 'W-S': ['C1B1_2'], 'N-E': ['B0B1_0'], 'N-N': ['B0B1_1'], 'N-W': ['B0B1_2'], 'E-S': ['A1B1_0'], 'E-E': ['A1B1_1'], 'E-N': ['A1B1_2']},
		'downstream':{'N': 'B2', 'E': 'C1', 'S': 'B0', 'W': 'A1'}},

		'B2': {
		'lane_sets':{'S-W': ['B3B2_0'], 'S-S': ['B3B2_1'], 'S-E': ['B3B2_2'], 'W-N': ['C2B2_0'], 'W-W': ['C2B2_1'], 'W-S': ['C2B2_2'], 'N-E': ['B1B2_0'], 'N-N': ['B1B2_1'], 'N-W': ['B1B2_2'], 'E-S': ['A2B2_0'], 'E-E': ['A2B2_1'], 'E-N': ['A2B2_2']},
		'downstream':{'N': 'B3', 'E': 'C2', 'S': 'B1', 'W': 'A2'}},

		'B3': {
		'lane_sets':{'S-W': ['top1B3_0'], 'S-S': ['top1B3_1'], 'S-E': ['top1B3_2'], 'W-N': ['C3B3_0'], 'W-W': ['C3B3_1'], 'W-S': ['C3B3_2'], 'N-E': ['B2B3_0'], 'N-N': ['B2B3_1'], 'N-W': ['B2B3_2'], 'E-S': ['A3B3_0'], 'E-E': ['A3B3_1'], 'E-N': ['A3B3_2']},
		'downstream':{'N': None, 'E': 'C3', 'S': 'B2', 'W': 'A3'}},

		'C0': {
		'lane_sets':{'S-W': ['C1C0_0'], 'S-S': ['C1C0_1'], 'S-E': ['C1C0_2'], 'W-N': ['D0C0_0'], 'W-W': ['D0C0_1'], 'W-S': ['D0C0_2'], 'N-E': ['bottom2C0_0'], 'N-N': ['bottom2C0_1'], 'N-W': ['bottom2C0_2'], 'E-S': ['B0C0_0'], 'E-E': ['B0C0_1'], 'E-N': ['B0C0_2']},
		'downstream':{'N': 'C1', 'E': 'D0', 'S': None, 'W': 'B0'}},

		'C1': {
		'lane_sets':{'S-W': ['C2C1_0'], 'S-S': ['C2C1_1'], 'S-E': ['C2C1_2'], 'W-N': ['D1C1_0'], 'W-W': ['D1C1_1'], 'W-S': ['D1C1_2'], 'N-E': ['C0C1_0'], 'N-N': ['C0C1_1'], 'N-W': ['C0C1_2'], 'E-S': ['B1C1_0'], 'E-E': ['B1C1_1'], 'E-N': ['B1C1_2']},
		'downstream':{'N': 'C2', 'E': 'D1', 'S': 'C0', 'W': 'B1'}},

		'C2': {
		'lane_sets':{'S-W': ['C3C2_0'], 'S-S': ['C3C2_1'], 'S-E': ['C3C2_2'], 'W-N': ['D2C2_0'], 'W-W': ['D2C2_1'], 'W-S': ['D2C2_2'], 'N-E': ['C1C2_0'], 'N-N': ['C1C2_1'], 'N-W': ['C1C2_2'], 'E-S': ['B2C2_0'], 'E-E': ['B2C2_1'], 'E-N': ['B2C2_2']},
		'downstream':{'N': 'C3', 'E': 'D2', 'S': 'C1', 'W': 'B2'}},

		'C3': {
		'lane_sets':{'S-W': ['top2C3_0'], 'S-S': ['top2C3_1'], 'S-E': ['top2C3_2'], 'W-N': ['D3C3_0'], 'W-W': ['D3C3_1'], 'W-S': ['D3C3_2'], 'N-E': ['C2C3_0'], 'N-N': ['C2C3_1'], 'N-W': ['C2C3_2'], 'E-S': ['B3C3_0'], 'E-E': ['B3C3_1'], 'E-N': ['B3C3_2']},
		'downstream':{'N': None, 'E': 'D3', 'S': 'C2', 'W': 'B3'}},

		'D0': {
		'lane_sets':{'S-W': ['D1D0_0'], 'S-S': ['D1D0_1'], 'S-E': ['D1D0_2'], 'W-N': ['right0D0_0'], 'W-W': ['right0D0_1'], 'W-S': ['right0D0_2'], 'N-E': ['bottom3D0_0'], 'N-N': ['bottom3D0_1'], 'N-W': ['bottom3D0_2'], 'E-S': ['C0D0_0'], 'E-E': ['C0D0_1'], 'E-N': ['C0D0_2']},
		'downstream':{'N': 'D1', 'E': None, 'S': None, 'W': 'C0'}},

		'D1': {
		'lane_sets':{'S-W': ['D2D1_0'], 'S-S': ['D2D1_1'], 'S-E': ['D2D1_2'], 'W-N': ['right1D1_0'], 'W-W': ['right1D1_1'], 'W-S': ['right1D1_2'], 'N-E': ['D0D1_0'], 'N-N': ['D0D1_1'], 'N-W': ['D0D1_2'], 'E-S': ['C1D1_0'], 'E-E': ['C1D1_1'], 'E-N': ['C1D1_2']},
		'downstream':{'N': 'D2', 'E': None, 'S': 'D0', 'W': 'C1'}},

		'D2': {
		'lane_sets':{'S-W': ['D3D2_0'], 'S-S': ['D3D2_1'], 'S-E': ['D3D2_2'], 'W-N': ['right2D2_0'], 'W-W': ['right2D2_1'], 'W-S': ['right2D2_2'], 'N-E': ['D1D2_0'], 'N-N': ['D1D2_1'], 'N-W': ['D1D2_2'], 'E-S': ['C2D2_0'], 'E-E': ['C2D2_1'], 'E-N': ['C2D2_2']},
		'downstream':{'N': 'D3', 'E': None, 'S': 'D1', 'W': 'C2'}},

		'D3': {
		'lane_sets':{'S-W': ['top3D3_0'], 'S-S': ['top3D3_1'], 'S-E': ['top3D3_2'], 'W-N': ['right3D3_0'], 'W-W': ['right3D3_1'], 'W-S': ['right3D3_2'], 'N-E': ['D2D3_0'], 'N-N': ['D2D3_1'], 'N-W': ['D2D3_2'], 'E-S': ['C3D3_0'], 'E-E': ['C3D3_1'], 'E-N': ['C3D3_2']},
		'downstream':{'N': None, 'E': None, 'S': 'D2', 'W': 'C3'}}
	},
	'hangzhou': {
		'direction_encoding': {
			'road_0_1_0': [[1, 1, 1], ['road_1_1_3', 'road_1_1_0', 'road_1_1_1']],
			'road_0_2_0': [[1, 1, 1], ['road_1_2_3', 'road_1_2_0', 'road_1_2_1']],
			'road_0_3_0': [[1, 1, 1], ['road_1_3_3', 'road_1_3_0', 'road_1_3_1']],
			'road_0_4_0': [[1, 1, 1], ['road_1_4_3', 'road_1_4_0', 'road_1_4_1']],
			'road_1_0_1': [[1, 1, 1], ['road_1_1_0', 'road_1_1_1', 'road_1_1_2']],
			'road_1_1_0': [[1, 1, 1], ['road_2_1_3', 'road_2_1_0', 'road_2_1_1']],
			'road_1_1_1': [[1, 1, 1], ['road_1_2_0', 'road_1_2_1', 'road_1_2_2']],
			'road_1_1_2': [[1, 0, 0], ['road_0_1_0']],
			'road_1_1_3': [[1, 1, 0], ['road_1_0_1']],
			'road_1_2_0': [[1, 1, 1], ['road_2_2_3', 'road_2_2_0', 'road_2_2_1']],
			'road_1_2_1': [[1, 1, 1], ['road_1_3_0', 'road_1_3_1', 'road_1_3_2']],
			'road_1_2_2': [[1, 0, 0], ['road_0_2_0']],
			'road_1_2_3': [[1, 1, 1], ['road_1_1_2', 'road_1_1_3', 'road_1_1_0']],
			'road_1_3_0': [[1, 1, 1], ['road_2_3_3', 'road_2_3_0', 'road_2_3_1']],
			'road_1_3_1': [[1, 1, 1], ['road_1_4_0', 'road_1_4_1', 'road_1_4_2']],
			'road_1_3_2': [[1, 0, 0], ['road_0_3_0']],
			'road_1_3_3': [[1, 1, 1], ['road_1_2_2', 'road_1_2_3', 'road_1_2_0']],
			'road_1_4_0': [[1, 1, 1], ['road_2_4_3', 'road_2_4_0', 'road_2_4_1']],
			'road_1_4_1': [[1, 0, 0], ['road_1_5_3']],
			'road_1_4_2': [[1, 0, 0], ['road_0_4_0']],
			'road_1_4_3': [[1, 1, 1], ['road_1_3_2', 'road_1_3_3', 'road_1_3_0']],
			'road_1_5_3': [[1, 1, 1], ['road_1_4_2', 'road_1_4_3', 'road_1_4_0']],
			'road_2_0_1': [[1, 1, 1], ['road_2_1_0', 'road_2_1_1', 'road_2_1_2']],
			'road_2_1_0': [[1, 1, 1], ['road_3_1_3', 'road_3_1_0', 'road_3_1_1']],
			'road_2_1_1': [[1, 1, 1], ['road_2_2_0', 'road_2_2_1', 'road_2_2_2']],
			'road_2_1_2': [[1, 1, 1], ['road_1_1_1', 'road_1_1_2', 'road_1_1_3']],
			'road_2_1_3': [[1, 0, 0], ['road_2_0_1']],
			'road_2_2_0': [[1, 1, 1], ['road_3_2_3', 'road_3_2_0', 'road_3_2_1']],
			'road_2_2_1': [[1, 1, 1], ['road_2_3_0', 'road_2_3_1', 'road_2_3_2']],
			'road_2_2_2': [[1, 1, 1], ['road_1_2_1', 'road_1_2_2', 'road_1_2_3']],
			'road_2_2_3': [[1, 1, 1], ['road_2_1_2', 'road_2_1_3', 'road_2_1_0']],
			'road_2_3_0': [[1, 1, 1], ['road_3_3_3', 'road_3_3_0', 'road_3_3_1']],
			'road_2_3_1': [[1, 1, 1], ['road_2_4_0', 'road_2_4_1', 'road_2_4_2']],
			'road_2_3_2': [[1, 1, 1], ['road_1_3_1', 'road_1_3_2', 'road_1_3_3']],
			'road_2_3_3': [[1, 1, 1], ['road_2_2_2', 'road_2_2_3', 'road_2_2_0']],
			'road_2_4_0': [[1, 1, 1], ['road_3_4_3', 'road_3_4_0', 'road_3_4_1']],
			'road_2_4_1': [[1, 0, 0], ['road_2_5_3']],
			'road_2_4_2': [[1, 1, 1], ['road_1_4_1', 'road_1_4_2', 'road_1_4_3']],
			'road_2_4_3': [[1, 1, 1], ['road_2_3_2', 'road_2_3_3', 'road_2_3_0']],
			'road_2_5_3': [[1, 1, 1], ['road_2_4_2', 'road_2_4_3', 'road_2_4_0']],
			'road_3_0_1': [[1, 1, 1], ['road_3_1_0', 'road_3_1_1', 'road_3_1_2']],
			'road_3_1_0': [[1, 1, 1], ['road_4_1_3', 'road_4_1_0', 'road_4_1_1']],
			'road_3_1_1': [[1, 1, 1], ['road_3_2_0', 'road_3_2_1', 'road_3_2_2']],
			'road_3_1_2': [[1, 1, 1], ['road_2_1_1', 'road_2_1_2', 'road_2_1_3']],
			'road_3_1_3': [[1, 0, 0], ['road_3_0_1']],
			'road_3_2_0': [[1, 1, 1], ['road_4_2_3', 'road_4_2_0', 'road_4_2_1']],
			'road_3_2_1': [[1, 1, 1], ['road_3_3_0', 'road_3_3_1', 'road_3_3_2']],
			'road_3_2_2': [[1, 1, 1], ['road_2_2_1', 'road_2_2_2', 'road_2_2_3']],
			'road_3_2_3': [[1, 1, 1], ['road_3_1_2', 'road_3_1_3', 'road_3_1_0']],
			'road_3_3_0': [[1, 1, 1], ['road_4_3_3', 'road_4_3_0', 'road_4_3_1']],
			'road_3_3_1': [[1, 1, 1], ['road_3_4_0', 'road_3_4_1', 'road_3_4_2']],
			'road_3_3_2': [[1, 1, 1], ['road_2_3_1', 'road_2_3_2', 'road_2_3_3']],
			'road_3_3_3': [[1, 1, 1], ['road_3_2_2', 'road_3_2_3', 'road_3_2_0']],
			'road_3_4_0': [[1, 1, 1], ['road_4_4_3', 'road_4_4_0', 'road_4_4_1']],
			'road_3_4_1': [[1, 0, 0], ['road_3_5_3']],
			'road_3_4_2': [[1, 1, 1], ['road_2_4_1', 'road_2_4_2', 'road_2_4_3']],
			'road_3_4_3': [[1, 1, 1], ['road_3_3_2', 'road_3_3_3', 'road_3_3_0']],
			'road_3_5_3': [[1, 1, 1], ['road_3_4_2', 'road_3_4_3', 'road_3_4_0']],
			'road_4_0_1': [[1, 1, 1], ['road_4_1_0', 'road_4_1_1', 'road_4_1_2']],
			'road_4_1_0': [[1, 0, 0], ['road_5_1_2']],
			'road_4_1_1': [[1, 1, 1], ['road_4_2_0', 'road_4_2_1', 'road_4_2_2']],
			'road_4_1_2': [[1, 1, 1], ['road_3_1_1', 'road_3_1_2', 'road_3_1_3']],
			'road_4_1_3': [[1, 0, 0], ['road_4_0_1']],
			'road_4_2_0': [[1, 0, 0], ['road_5_2_2']],
			'road_4_2_1': [[1, 1, 1], ['road_4_3_0', 'road_4_3_1', 'road_4_3_2']],
			'road_4_2_2': [[1, 1, 1], ['road_3_2_1', 'road_3_2_2', 'road_3_2_3']],
			'road_4_2_3': [[1, 1, 1], ['road_4_1_2', 'road_4_1_3', 'road_4_1_0']],
			'road_4_3_0': [[1, 0, 0], ['road_5_3_2']],
			'road_4_3_1': [[1, 1, 1], ['road_4_4_0', 'road_4_4_1', 'road_4_4_2']],
			'road_4_3_2': [[1, 1, 1], ['road_3_3_1', 'road_3_3_2', 'road_3_3_3']],
			'road_4_3_3': [[1, 1, 1], ['road_4_2_2', 'road_4_2_3', 'road_4_2_0']],
			'road_4_4_0': [[1, 0, 0], ['road_5_4_2']],
			'road_4_4_1': [[1, 0, 0], ['road_4_5_3']],
			'road_4_4_2': [[1, 1, 1], ['road_3_4_1', 'road_3_4_2', 'road_3_4_3']],
			'road_4_4_3': [[1, 1, 1], ['road_4_3_2', 'road_4_3_3', 'road_4_3_0']],
			'road_4_5_3': [[1, 1, 1], ['road_4_4_2', 'road_4_4_3', 'road_4_4_0']],
			'road_5_1_2': [[1, 1, 1], ['road_4_1_1', 'road_4_1_2', 'road_4_1_3']],
			'road_5_2_2': [[1, 1, 1], ['road_4_2_1', 'road_4_2_2', 'road_4_2_3']],
			'road_5_3_2': [[1, 1, 1], ['road_4_3_1', 'road_4_3_2', 'road_4_3_3']],
			'road_5_4_2': [[1, 1, 1], ['road_4_4_1', 'road_4_4_2', 'road_4_4_3']]}, 
		'phase_pairs': [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]],
		'valid_acts': {
			'intersection_1_1': [0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_2_1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_3_1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_4_1':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_1_2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_2_2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_3_2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_4_2':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_1_3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_2_3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_3_3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_4_3':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_1_4':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_2_4':[0, 1, 2, 3, 4, 5, 6, 7], 
			'intersection_3_4':[0, 1, 2, 3, 4, 5, 6, 7],
			'intersection_4_4':[0, 1, 2, 3, 4, 5, 6, 7]
		},
		'margins':{
			'intersection_1_1': {'intersection_0_1': 'W', 'intersection_1_0': 'S'}, 
			'intersection_1_2': {'intersection_0_2': 'W'},
			'intersection_1_3': {'intersection_0_3': 'W'},
			'intersection_1_4': {'intersection_0_4': 'W', 'intersection_1_5': 'N'},
			'intersection_2_1': {'intersection_2_0': 'S'},
			'intersection_2_4': {'intersection_2_5': 'N'},
			'intersection_3_1': {'intersection_3_0': 'S'},
			'intersection_3_4': {'intersection_3_5': 'N'},
			'intersection_4_1': {'intersection_5_1': 'E', 'intersection_4_0': 'S'},
			'intersection_4_2': {'intersection_5_2': 'E'},
			'intersection_4_3': {'intersection_5_3': 'E'},
			'intersection_4_4': {'intersection_4_5': 'N', 'intersection_5_4': 'E'},

			'intersection_0_1': {'intersection_1_1': 'E'},
			'intersection_1_0': {'intersection_1_1': 'N'},
			'intersection_0_2': {'intersection_1_2': 'E'},
			'intersection_0_3': {'intersection_1_3': 'E'}, 
			'intersection_0_4': {'intersection_1_4': 'E'},
			'intersection_1_5': {'intersection_1_4': 'S'},
			'intersection_2_0': {'intersection_2_1': 'N'}, 
			'intersection_2_5': {'intersection_2_4': 'S'}, 
			'intersection_3_0': {'intersection_3_1': 'N'}, 
			'intersection_3_5': {'intersection_3_4': 'S'}, 
			'intersection_5_1': {'intersection_4_1': 'W'}, 
			'intersection_4_0': {'intersection_4_1': 'N'}, 
			'intersection_5_2': {'intersection_4_2': 'W'}, 
			'intersection_5_3': {'intersection_4_3': 'W'}, 
			'intersection_4_5': {'intersection_4_4': 'S'}, 
			'intersection_5_4': {'intersection_4_4': 'W'}
		}, 
		'intersection_1_1': {
		'lane_sets':{'S-W': ['road_1_2_3_0'], 'S-S': ['road_1_2_3_1'], 'S-E': ['road_1_2_3_2'], 'W-N': ['road_2_1_2_0'], 'W-W': ['road_2_1_2_1'], 'W-S': ['road_2_1_2_2'], 'N-E': ['road_1_0_1_0'], 'N-N': ['road_1_0_1_1'], 'N-W': ['road_1_0_1_2'], 'E-S': ['road_0_1_0_0'], 'E-E': ['road_0_1_0_1'], 'E-N': ['road_0_1_0_2']},
		'downstream':{'N': 'intersection_1_2', 'E': 'intersection_2_1', 'S': None, 'W': None}},

		'intersection_2_1': {
		'lane_sets':{'S-W': ['road_2_2_3_0'], 'S-S': ['road_2_2_3_1'], 'S-E': ['road_2_2_3_2'], 'W-N': ['road_3_1_2_0'], 'W-W': ['road_3_1_2_1'], 'W-S': ['road_3_1_2_2'], 'N-E': ['road_2_0_1_0'], 'N-N': ['road_2_0_1_1'], 'N-W': ['road_2_0_1_2'], 'E-S': ['road_1_1_0_0'], 'E-E': ['road_1_1_0_1'], 'E-N': ['road_1_1_0_2']},
		'downstream':{'N': 'intersection_2_2', 'E': 'intersection_3_1', 'S': None, 'W': 'intersection_1_1'}},

		'intersection_3_1': {
		'lane_sets':{'S-W': ['road_3_2_3_0'], 'S-S': ['road_3_2_3_1'], 'S-E': ['road_3_2_3_2'], 'W-N': ['road_4_1_2_0'], 'W-W': ['road_4_1_2_1'], 'W-S': ['road_4_1_2_2'], 'N-E': ['road_3_0_1_0'], 'N-N': ['road_3_0_1_1'], 'N-W': ['road_3_0_1_2'], 'E-S': ['road_2_1_0_0'], 'E-E': ['road_2_1_0_1'], 'E-N': ['road_2_1_0_2']},
		'downstream':{'N': 'intersection_3_2', 'E': 'intersection_4_1', 'S': None, 'W': 'intersection_2_1'}},

		'intersection_4_1': {
		'lane_sets':{'S-W': ['road_4_2_3_0'], 'S-S': ['road_4_2_3_1'], 'S-E': ['road_4_2_3_2'], 'W-N': ['road_5_1_2_0'], 'W-W': ['road_5_1_2_1'], 'W-S': ['road_5_1_2_2'], 'N-E': ['road_4_0_1_0'], 'N-N': ['road_4_0_1_1'], 'N-W': ['road_4_0_1_2'], 'E-S': ['road_3_1_0_0'], 'E-E': ['road_3_1_0_1'], 'E-N': ['road_3_1_0_2']},
		'downstream':{'N': 'intersection_4_2', 'E': None, 'S': None, 'W': 'intersection_3_1'}},

		'intersection_1_2': {
		'lane_sets':{'S-W': ['road_1_2_2_0'], 'S-S': ['road_1_2_2_1'], 'S-E': ['road_1_2_2_2'], 'W-N': ['road_2_2_2_0'], 'W-W': ['road_2_2_2_1'], 'W-S': ['road_2_2_2_2'], 'N-E': ['road_1_1_1_0'], 'N-N': ['road_1_1_1_1'], 'N-W': ['road_1_1_1_2'], 'E-S': ['road_0_2_0_0'], 'E-E': ['road_0_2_0_1'], 'E-N': ['road_0_2_0_2']},
		'downstream':{'N': 'intersection_1_3', 'E': 'intersection_2_2', 'S': 'intersection_1_1', 'W': None}},

		'intersection_2_2': {
		'lane_sets':{'S-W': ['road_2_3_3_0'], 'S-S': ['road_2_3_3_1'], 'S-E': ['road_2_3_3_2'], 'W-N': ['road_2_2_1_0'], 'W-W': ['road_2_2_1_1'], 'W-S': ['road_2_2_1_2'], 'N-E': ['road_2_1_1_0'], 'N-N': ['road_2_1_1_1'], 'N-W': ['road_2_1_1_2'], 'E-S': ['road_1_2_0_0'], 'E-E': ['road_1_2_0_1'], 'E-N': ['road_1_2_0_2']},
		'downstream':{'N': 'intersection_2_3', 'E': 'intersection_3_2', 'S': 'intersection_2_1', 'W': 'intersection_1_2'}},

		'intersection_3_2': {
		'lane_sets':{'S-W': ['road_3_3_3_0'], 'S-S': ['road_3_3_3_1'], 'S-E': ['road_3_3_3_2'], 'W-N': ['road_4_2_2_0'], 'W-W': ['road_4_2_2_1'], 'W-S': ['road_4_2_2_2'], 'N-E': ['road_3_1_1_0'], 'N-N': ['road_3_1_1_1'], 'N-W': ['road_3_1_1_2'], 'E-S': ['road_2_2_0_0'], 'E-E': ['road_2_2_0_1'], 'E-N': ['road_2_2_0_2']},
		'downstream':{'N': 'intersection_3_3', 'E': 'intersection_4_2', 'S': 'intersection_3_1', 'W': 'intersection_2_2'}},

		'intersection_4_2': {
		'lane_sets':{'S-W': ['road_4_3_3_0'], 'S-S': ['road_4_3_3_1'], 'S-E': ['road_4_3_3_2'], 'W-N': ['road_5_2_2_0'], 'W-W': ['road_5_2_2_1'], 'W-S': ['road_5_2_2_2'], 'N-E': ['road_4_1_1_0'], 'N-N': ['road_4_1_1_1'], 'N-W': ['road_4_1_1_2'], 'E-S': ['road_3_2_0_0'], 'E-E': ['road_3_2_0_1'], 'E-N': ['road_3_2_0_2']},
		'downstream':{'N': 'intersection_4_3', 'E': None, 'S': 'intersection_4_1', 'W': 'intersection_3_2'}},

		'intersection_1_3': {
		'lane_sets':{'S-W': ['road_1_4_3_0'], 'S-S': ['road_1_4_3_1'], 'S-E': ['road_1_4_3_2'], 'W-N': ['road_2_3_2_0'], 'W-W': ['road_2_3_2_1'], 'W-S': ['road_2_3_2_2'], 'N-E': ['road_1_2_1_0'], 'N-N': ['road_1_2_1_1'], 'N-W': ['road_1_2_1_2'], 'E-S': ['road_0_3_0_0'], 'E-E': ['road_0_3_0_1'], 'E-N': ['road_0_3_0_2']},
		'downstream':{'N': 'intersection_1_4', 'E': 'intersection_2_3', 'S': 'intersection_1_2', 'W': None}},

		'intersection_2_3': {
		'lane_sets':{'S-W': ['road_2_4_3_0'], 'S-S': ['road_2_4_3_1'], 'S-E': ['road_2_4_3_2'], 'W-N': ['road_3_3_2_0'], 'W-W': ['road_3_3_2_1'], 'W-S': ['road_3_3_2_2'], 'N-E': ['road_2_2_1_0'], 'N-N': ['road_2_2_1_1'], 'N-W': ['road_2_2_1_2'], 'E-S': ['road_1_3_0_0'], 'E-E': ['road_1_3_0_1'], 'E-N': ['road_1_3_0_2']},
		'downstream':{'N': 'intersection_2_4', 'E': 'intersection_3_3', 'S': 'intersection_2_2', 'W': 'intersection_1_3'}},

		'intersection_3_3': {
		'lane_sets':{'S-W': ['road_3_4_3_0'], 'S-S': ['road_3_4_3_1'], 'S-E': ['road_3_4_3_2'], 'W-N': ['road_4_3_2_0'], 'W-W': ['road_4_3_2_1'], 'W-S': ['road_4_3_2_2'], 'N-E': ['road_3_2_1_0'], 'N-N': ['road_3_2_1_1'], 'N-W': ['road_3_2_1_2'], 'E-S': ['road_2_3_0_0'], 'E-E': ['road_2_3_0_1'], 'E-N': ['road_2_3_0_2']},
		'downstream':{'N': 'intersection_3_4', 'E': 'intersection_4_3', 'S': 'intersection_3_2', 'W': 'intersection_2_3'}},

		'intersection_4_3': {
		'lane_sets':{'S-W': ['road_4_4_3_0'], 'S-S': ['road_4_4_3_1'], 'S-E': ['road_4_4_3_2'], 'W-N': ['road_5_3_2_0'], 'W-W': ['road_5_3_2_1'], 'W-S': ['road_5_3_2_2'], 'N-E': ['road_4_2_1_0'], 'N-N': ['road_4_2_1_1'], 'N-W': ['road_4_2_1_2'], 'E-S': ['road_3_3_0_0'], 'E-E': ['road_3_3_0_1'], 'E-N': ['road_3_3_0_2']},
		'downstream':{'N': 'intersection_4_4', 'E': None, 'S': 'intersection_4_2', 'W': 'intersection_3_3'}},

		'intersection_1_4': {
		'lane_sets':{'S-W': ['road_1_5_3_0'], 'S-S': ['road_1_5_3_1'], 'S-E': ['road_1_5_3_2'], 'W-N': ['road_2_4_2_0'], 'W-W': ['road_2_4_2_1'], 'W-S': ['road_2_4_2_2'], 'N-E': ['road_1_3_1_0'], 'N-N': ['road_1_3_1_1'], 'N-W': ['road_1_3_1_2'], 'E-S': ['road_0_4_0_0'], 'E-E': ['road_0_4_0_1'], 'E-N': ['road_0_4_0_2']},
		'downstream':{'N': None, 'E': 'intersection_2_4', 'S': 'intersection_1_3', 'W': None}},

		'intersection_2_4': {
		'lane_sets':{'S-W': ['road_2_5_3_0'], 'S-S': ['road_2_5_3_1'], 'S-E': ['road_2_5_3_2'], 'W-N': ['road_3_4_2_0'], 'W-W': ['road_3_4_2_1'], 'W-S': ['road_3_4_2_2'], 'N-E': ['road_2_3_1_0'], 'N-N': ['road_2_3_1_1'], 'N-W': ['road_2_3_1_2'], 'E-S': ['road_1_4_0_0'], 'E-E': ['road_1_4_0_1'], 'E-N': ['road_1_4_0_2']},
		'downstream':{'N': None, 'E': 'intersection_3_4', 'S': 'intersection_2_3', 'W': 'intersection_1_4'}},

		'intersection_3_4': {
		'lane_sets':{'S-W': ['road_3_5_3_0'], 'S-S': ['road_3_5_3_1'], 'S-E': ['road_3_5_3_2'], 'W-N': ['road_4_4_2_0'], 'W-W': ['road_4_4_2_1'], 'W-S': ['road_4_4_2_2'], 'N-E': ['road_3_3_1_0'], 'N-N': ['road_3_3_1_1'], 'N-W': ['road_3_3_1_2'], 'E-S': ['road_2_4_0_0'], 'E-E': ['road_2_4_0_1'], 'E-N': ['road_2_4_0_2']},
		'downstream':{'N': None, 'E': 'intersection_4_4', 'S': 'intersection_3_3', 'W': 'intersection_2_4'}},

		'intersection_4_4': {
		'lane_sets':{'S-W': ['road_4_5_3_0'], 'S-S': ['road_4_5_3_1'], 'S-E': ['road_4_5_3_2'], 'W-N': ['road_5_4_2_0'], 'W-W': ['road_5_4_2_1'], 'W-S': ['road_5_4_2_2'], 'N-E': ['road_4_3_1_0'], 'N-N': ['road_4_3_1_1'], 'N-W': ['road_4_3_1_2'], 'E-S': ['road_3_4_0_0'], 'E-E': ['road_3_4_0_1'], 'E-N': ['road_3_4_0_2']},
		'downstream':{'N': None, 'E': None, 'S': 'intersection_4_3', 'W': 'intersection_3_4'}},
	},

}