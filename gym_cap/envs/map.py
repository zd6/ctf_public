from gym import spaces

import numpy as np
import random
import warnings
from .const import *

"""
This module provides Board class with map related methods

gen_random_map:
    It creates random map given dimension size, number of obstacles,
    and number of agents for each team.
    The background can be divided in half, or random box.
custom_map:
    It generates map given a numpy array.
"""

# State space for capture the flag
class Board(spaces.Space):
    """A Board in R^3 used for CtF 

    * Board is stored in padded memory
        - For multi-agent centering
    """
    def __init__(self, shape=None, dtype=np.bool_):
        super(Board, self).__init__(shape, dtype)

        self._shape = shape
        self._dtype = dtype

        #self.map_pool = []
        #self.pool_size = 0

        self.current_map = None
        self.current_map_static = None
        self.current_id = 0

    def __repr__(self):
        return "Board size " + str(self.shape)

    def sample(self):
        map_obj = [NUM_BLUE, NUM_BLUE_UAV, NUM_RED, NUM_RED_UAV, NUM_GRAY]
        state, _, _ = gen_random_map('map',
                self.shape[0], island_zone=False, map_obj=map_obj)
        return state

    def generate_map(self, mode, **kwargs):
        if mode == 'random':
            state, static_state, obj_loc = self.gen_random_map(**kwargs)
            self.current_map = state
            self.current_map_static = static_state
            self.current_id = 0
        else:
            raise NotImplementedError('Only random mode available')

        return obj_loc

    def gen_random_map(self, shape=(20,20), num_flag=(1,1), map_obj=None, island_zone=False, np_random=None):
        """ Generate structured map with random allocating obstacles and agents

        Generate map with given setting

        Parameters
        ----------
        shape : tuple(int,int) or int
            Size of the map (ly, lx)
        num_flag: tuple(int,int)
            Number of flag in each team
        map_obj     : list
            The necessary elements to build the map
            'UAV', 'UGV', 'UGV2', 'UGV3', 'UGV4'
        island_zone  : bool
            True if zones are defined random
        """

        # ASSERTION
        assert map_obj is not None

        # INITIALIZE THE SEED 
        if np_random is None:
            np_random = np.random

        # MAP SHAPE FOR SQUARE
        if type(shape) is int:
            shape = tuple(shape, shape)
        self._shape = (shape[0], shape[1], NUM_CHANNEL)
        dim = np.amin(shape[:2])

        # PRE-COUNT ELEMENT
        blue_split_ind, red_split_ind = [num_flag[0]], [num_flag[1]]
        names = ['flag']
        for key, value in map_obj.items():
            names.append(key)
            blue_split_ind.append(value[0])
            red_split_ind.append(value[1])
        total_blue, total_red = sum(blue_split_ind), sum(red_split_ind)
        blue_split_ind = np.cumsum(blue_split_ind)
        red_split_ind = np.cumsum(red_split_ind)

        # RETURNS
        env = np.zeros(self._shape, dtype=self._dtype)
        env_static = np.zeros(shape, dtype=int)

        warning_counter = 0
        warning_max = 20
        while warning_counter < warning_max: # Continue until all elements are placed
            # CH 1 : ZONE (included in static)
            zone = np.ones(shape, dtype=int)  # 1 for blue, -1 for red, 0 for obstacle
            if island_zone:
                sx, sy = np_random.randint(dim//2, 4*dim//5, [2])
                lx, ly = np_random.randint(0, dim - max(sx,sy)-1, [2])
                zone[lx:lx+sx, ly:ly+sy] = -1
            else:
                zone[:,0:dim//2] = -1
            if 0.5 < np_random.rand():
                zone = -zone  # Reverse for equal expectation

            # CH 3 : OBSTACLE
            num_obst = int(np.sqrt(np.min(shape)))
            for i in range(num_obst):
                lx, ly = np_random.randint(0, dim, [2])
                sx, sy = np_random.randint(0, dim//5, [2]) + 1
                zone[lx-sx:lx+sx, ly-sy:ly+sy] = 0

            ## Coordinate Selection for Red and Blue
            blue_pool = np.argwhere(zone==1)
            if len(blue_pool) < total_blue:
                warning_counter += 1
                warnings.warn("Map is too small to allocate all elements.")
                continue
            #blue_indices = np_random.choice(len(blue_pool), total_blue, replace=False)
            #blue_coord = np.take(blue_pool, blue_indices, axis=0)

            red_pool = np.argwhere(zone==-1)
            if len(red_pool) < total_red:
                warning_counter += 1
                warnings.warn("Map is too small to allocate all elements.")
                continue

            break
        if warning_counter == warning_max:
            raise InterruptedError("Map size is too small. Warning counter reached max")
        env[:,:,CHANNEL[TEAM1_BACKGROUND]] = zone==1
        env[:,:,CHANNEL[TEAM2_BACKGROUND]] = zone==-1
        env[:,:,CHANNEL[OBSTACLE]] = zone==0

        # Elements
        element_locs = {}
        blue_idx = np.random.permutation(np.arange(len(blue_pool)))
        red_idx = np.random.permutation(np.arange(len(red_pool)))
        blue_coords_idx = np.split(blue_idx, blue_split_ind)
        red_coords_idx = np.split(red_idx, red_split_ind)
        for name, blue_idx, red_idx in zip(names, blue_coords_idx, red_coords_idx):
            blue_coord = blue_pool[blue_idx]
            red_coord = red_pool[red_idx]
            if name == 'flag':
                team1_id = TEAM1_FLAG
                team2_id = TEAM2_FLAG
            elif name == 'UAV':
                team1_id = TEAM1_UAV
                team2_id = TEAM2_UAV
            elif name == 'UGV':
                team1_id = TEAM1_UGV
                team2_id = TEAM2_UGV
            elif name == 'UGV2':
                team1_id = TEAM1_UGV2
                team2_id = TEAM2_UGV2
            elif name == 'UGV3':
                team1_id = TEAM1_UGV3
                team2_id = TEAM2_UGV3
            elif name == 'UGV4':
                team1_id = TEAM1_UGV4
                team2_id = TEAM2_UGV4
            else:
                raise NotImplementedError("Element Type is not defined")
            team1_ch = CHANNEL[team1_id]
            team2_ch = CHANNEL[team2_id]
            element_locs[team1_id] = blue_coord.tolist()
            element_locs[team2_id] = red_coord.tolist()
            if team1_ch < NUM_CHANNEL:
                for y, x in blue_coord:
                    env[y,x,team1_ch] = 1
            if team2_ch < NUM_CHANNEL:
                for y, x in red_coord:
                    env[y,x,team2_ch] = 1
        element_locs.pop(TEAM1_FLAG, None) 
        element_locs.pop(TEAM2_FLAG, None) 

        env_static[env[:,:,CHANNEL[TEAM1_BACKGROUND]]] = TEAM1_BACKGROUND
        env_static[env[:,:,CHANNEL[TEAM2_BACKGROUND]]] = TEAM2_BACKGROUND
        env_static[env[:,:,CHANNEL[OBSTACLE]]] = OBSTACLE
        env_static[env[:,:,CHANNEL[TEAM1_FLAG]]] = TEAM1_FLAG
        env_static[env[:,:,CHANNEL[TEAM2_FLAG]]] = TEAM2_FLAG

        return env, env_static, element_locs

    def flush_pool(self):
        self.map_pool.clear()
        self.pool_size = 0

    def load_maps(self, path='map_save'):
        pass

    def save_maps(self, path='map_save'):
        pass

    def custom_map(self, custom_board):
        """ Generate map from predefined array

        Parameters
        ----------
        custom_board : numpy array
            new_map
        The necessary elements:
            ugv_1   : blue UGV
            ugv_2   : red UGV
            uav_2   : red UAV
            gray    : gray units
            
        """
        
        # BUILD OBJECT COUNT ARRAY
        element_count = dict(zip(*np.unique(custom_board, return_counts=True)))
        keys = {TEAM1_BACKGROUND: [TEAM1_UGV, TEAM1_UAV, TEAM1_UGV2, TEAM1_UGV3, TEAM1_UGV4],
                TEAM2_BACKGROUND: [TEAM2_UGV, TEAM2_UAV, TEAM2_UGV2, TEAM2_UGV3, TEAM2_UGV4] }

        # RETURNS
        obj_dict = {TEAM1_BACKGROUND: [],
                    TEAM2_BACKGROUND: []}
        agent_locs = {}

        l, b = new_map.shape
        self._shape = (l, b, NUM_CHANNEL)
        self.current_map_static = np.copy(custom_board)

        # BUILD 3D MAP
        env = np.zeros(self._shape, dtype=self._dtype)
        for element, channel in CHANNEL.items():
            if element in new_map and channel < NUM_CHANNEL:
                env[new_map==element,channel] = 1

        for team, elems in keys.items():
            for e in elems:
                # Count element and append on obj_dict
                count = element_count.get(e, 0)
                obj_dict[team].append(count)

                # Refill agent's team color in static map and env
                loc = new_map==e
                self.current_map_static[loc] = team
                agent_locs[e] = np.argwhere(loc)
                env[loc, CHANNEL[team]] = 1

        return obj_dict, agent_locs

    @property
    def space(self):
        return self._shape
