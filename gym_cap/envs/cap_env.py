import __future__

import io
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
    
import random
import sys
import traceback

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .agent import *
from .create_map import gen_random_map, custom_map
from gym_cap.envs import const

"""
Requires that all units initially exist in home zone.
"""


class CapEnv(gym.Env):
    metadata = {
        "render.modes": ["fast", "human", 'rgb_array'],
        'video.frames_per_second' : 50
    }

    ACTION = ["X", "N", "E", "S", "W"]

    def __init__(self, map_size=20, mode="random", **kwargs):
        """

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        self.seed()
        self.viewer = None
        self._parse_config()

        self.blue_memory = np.empty((map_size, map_size))
        self.red_memory = np.empty((map_size, map_size))

        self._policy_blue = None
        self._policy_red = None

        self._blue_trajectory = []
        self._red_trajectory = []

        self.reset(
                map_size,
                mode=mode,
                policy_blue=kwargs.get('policy_blue', None),
                policy_red=kwargs.get('policy_red', None),
                custom_board=kwargs.get('custom_board', None),
                config_path=kwargs.get('config_path', None),
            )

    def seed(self, seed=None):
        """
        todo docs still

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _parse_config(self, config_path=None):
        """
        Parse Configuration

        If configuration file is explicitly provided, any given settings overide the default parameters.
        Default parameters can be found in const.py module

        To add additional configuration parameter, make sure to add the default value in const.py

        The parameter is accessible as class variable.
        """

        config_param = { # Configurable parameters
                'elements': ['NUM_BLUE', 'NUM_RED', 'NUM_BLUE_UAV', 'NUM_RED_UAV', 'NUM_GRAY',
                        'NUM_BLUE_UGV2', 'NUM_RED_UGV2',
                        'NUM_BLUE_UGV3', 'NUM_RED_UGV3',
                        'NUM_BLUE_UGV4', 'NUM_RED_UGV4',
                        ],
                'control': ['CONTROL_ALL'],
                'communication': ['COM_GROUND', 'COM_AIR', 'COM_DISTANCE', 'COM_FREQUENCY'],
                'memory': ['INDIV_MEMORY', 'TEAM_MEMORY', 'RENDER_INDIV_MEMORY', 'RENDER_TEAM_MEMORY'],
                'settings': ['RL_SUGGESTIONS', 'STOCH_TRANSITIONS', 'STOCH_TRANSITIONS_EPS', 'STOCH_TRANSITIONS_MOD',
                        'STOCH_ATTACK', 'STOCH_ATTACK_BIAS', 'STOCH_ZONES', 'RED_PARTIAL', 'BLUE_PARTIAL']
            }
        config_datatype = {
                'elements': [int, int, int ,int, int,
                        int, int,
                        int, int,
                        int, int],
                'control': [bool],
                'communication': [bool, bool, int, float],
                'memory': [str, str, bool, bool],
                'settings': [bool, bool, float, str,
                        bool, int, bool, bool, bool]
            }

        if config_path is None:
            # Default configuration
            get = lambda key, name: getattr(const, name)
        else:
            # Custom configuration
            config = configparser.ConfigParser()
            config.read(config_path)
            get = lambda key, name: config.get(key, name, fallback=getattr(const, name))

        try:
            # Set environment attributes
            for section in config_param:
                for name, datatype in zip(config_param[section], config_datatype[section]):
                    value = get(section, name)
                    if datatype is bool:
                        if type(value) == str:
                            value = True if value == 'True' else False
                    elif datatype is int or datatype is float:
                        value = datatype(value)
                    setattr(self, name, value)
        except Exception as e:
            print(e)
            raise Exception('Configuration import fails: recheck whether all config variables are included')

    def reset(self, map_size=None, mode="random", policy_blue=None, policy_red=None,
            custom_board=None, config_path=None):
        """ 
        Resets the game

        Parameters
        ----------------

        map_size : [int] 
        mode : [str] 
        policy_blue : [policy] 
        policy_red : [policy] 
        custom_board : [str, numpy.ndarray] 
        config_path : [str] 

        """

        # ASSERTIONS
        assert map_size is None or type(map_size) is int

        # WARNINGS

        # STORE ARGUMENTS
        self.mode = mode

        # LOAD DEFAULT PARAMETERS
        if config_path is not None:
            self._parse_config(config_path)
        if map_size is None:
            map_size = self.map_size[0]

        # INITIALIZE MAP
        if custom_board is None:  # Random Generated Map
            map_obj = {
                    (const.TEAM1_UGV, const.TEAM2_UGV): (self.NUM_BLUE, self.NUM_RED),
                    (const.TEAM1_UAV, const.TEAM2_UAV): (self.NUM_BLUE_UAV, self.NUM_RED_UAV),
                    (const.TEAM1_UGV2, const.TEAM2_UGV2): (self.NUM_BLUE_UGV2, self.NUM_RED_UGV2),
                    (const.TEAM1_UGV3, const.TEAM2_UGV3): (self.NUM_BLUE_UGV3, self.NUM_RED_UGV3),
                    (const.TEAM1_UGV4, const.TEAM2_UGV4): (self.NUM_BLUE_UGV4, self.NUM_RED_UGV4)
                }

            self._env, self._static_map, agent_locs = gen_random_map('map',
                    map_size, rand_zones=self.STOCH_ZONES, np_random=self.np_random, map_obj=map_obj)
        elif type(custom_board) is str:
            custom_map = np.loadtxt(custom_board, dtype = int, delimiter = " ")
            self._env, self._static_map, map_obj, agent_locs = custom_map(custom_map)
            self.NUM_BLUE, self.NUM_BLUE_UAV, self.NUM_BLUE_UGV2, self.NUM_BLUE_UGV3, self.NUM_BLUE_UGV4 = map_obj[TEAM1_BACKGROUND]
            self.NUM_RED, self.NUM_RED_UAV, self.NUM_RED_UGV2, self.NUM_RED_UGV3, self.NUM_RED_UGV4 = map_obj[TEAM2_BACKGROUND]
        elif type(custom_board) is np.ndarray:
            custom_map = custom_board
            self._env, self._static_map, map_obj, agent_locs = custom_map(custom_map)
            self.NUM_BLUE, self.NUM_BLUE_UAV, self.NUM_BLUE_UGV2, self.NUM_BLUE_UGV3, self.NUM_BLUE_UGV4 = map_obj[TEAM1_BACKGROUND]
            self.NUM_RED, self.NUM_RED_UAV, self.NUM_RED_UGV2, self.NUM_RED_UGV3, self.NUM_RED_UGV4 = map_obj[TEAM2_BACKGROUND]
        self.map_size = tuple(self._static_map.shape)

        # INITIALIZE TEAM
        self._team_blue, self._team_red = self._construct_agents(agent_locs, self._static_map)

        self.action_space = spaces.Discrete(len(self.ACTION) ** len(self._team_blue))
        self.observation_space = Board(shape=[self.map_size[0], self.map_size[1], NUM_CHANNEL])
        if len(self._team_red) == 0:
            self.mode = "sandbox"

        # INITIATE POLICY
        if policy_blue is not None:
            self._policy_blue = policy_blue
        if policy_red is not None:
            self._policy_red = policy_red
        if self._policy_blue is not None:
            self._policy_blue.initiate(self._static_map, self._team_blue)
        if self._policy_red is not None:
            self._policy_red.initiate(self._static_map, self._team_red)

        # INITIALIZE MEMORY
        if self.TEAM_MEMORY == "fog":
            self.blue_memory[:] = const.UNKNOWN
            self.red_memory[:] = const.UNKNOWN

        if self.INDIV_MEMORY == "fog":
            for agent in self._team_blue + self._team_red:
                agent.memory[:] = const.UNKNOWN
                agent.memory_mode = "fog"

        # INITIALIZE TRAJECTORY (DEBUG)
        self._blue_trajectory = []
        self._red_trajectory = []

        self._create_observation_mask()

        self.blue_win = False
        self.red_win = False
        self.red_flag_captured = False
        self.blue_flag_captured = False
        self.red_eliminated = False
        self.blue_eliminated = False

        # Necessary for human mode
        self.first = True

        self.run_step = 0  # Number of step of current episode

        return self.get_obs_blue

    def _construct_agents(self, agent_coords, static_map):
        """
        From given coordinates, it generates objects of agents and make them into the list.

        team_blue --> [air1, air2, ... , ground1, ground2, ...]
        team_red  --> [air1, air2, ... , ground1, ground2, ...]

        complete_map    : 2d numpy array
        static_map      : 2d numpy array

        """
        team_blue = []
        team_red = []

        for element, coords in agent_coords.items():
            if coords is None: continue
            for coord in coords:
                if element == TEAM1_UGV:
                    cur_ent = GroundVehicle(coord, static_map, TEAM1_BACKGROUND, TEAM1_UGV)
                    team_blue.append(cur_ent)
                elif element == TEAM1_UGV2:
                    cur_ent = GroundVehicle_Tank(coord, static_map, TEAM1_BACKGROUND, TEAM1_UGV2)
                    team_blue.append(cur_ent)
                elif element == TEAM1_UAV:
                    cur_ent = AerialVehicle(coord, static_map, TEAM1_BACKGROUND, TEAM1_UAV)
                    team_blue.insert(0, cur_ent)
                elif element == TEAM2_UGV:
                    cur_ent = GroundVehicle(coord, static_map, TEAM2_BACKGROUND, TEAM2_UGV)
                    team_red.append(cur_ent)
                elif element == TEAM2_UGV2:
                    cur_ent = GroundVehicle_Tank(coord, static_map, TEAM2_BACKGROUND, TEAM2_UGV2)
                    team_red.append(cur_ent)
                elif element == TEAM2_UAV:
                    cur_ent = AerialVehicle(coord, static_map, TEAM2_BACKGROUND, TEAM2_UAV)
                    team_red.insert(0, cur_ent)

        return team_blue, team_red

    def _create_observation_mask(self):
        """
        Creates the mask 

        Mask is True(1) for the location where it CANNOT see.
        For full observation setting, mask is zero matrix

        Parameters
        ----------
        self    : object
            CapEnv object
        team    : int
            Team to create obs space for
        """

        def create_vision_mask(centers, radii):
            h, w = self._static_map.shape
            mask = np.zeros([h,w], dtype=bool)
            Y, X = np.ogrid[:h, :w]
            for center, radius in zip(centers, radii):
                mask += np.sqrt((X - center[1])**2 + (Y-center[0])**2) <= radius
            return ~mask

        if self.BLUE_PARTIAL:
            centers, radii = [], []
            for agent in self._team_blue:
                if not agent.isAlive: continue
                centers.append(agent.get_loc())
                radii.append(agent.range)
            self._blue_mask = create_vision_mask(centers, radii)
        else:
            self._blue_mask = np.zeros_like(self._static_map, dtype=bool)

        if self.RED_PARTIAL:
            centers, radii = [], []
            for agent in self._team_red:
                if not agent.isAlive: continue
                centers.append(agent.get_loc())
                radii.append(agent.range)
            self._red_mask = create_vision_mask(centers, radii)
        else:
            self._red_mask = np.zeros_like(self._static_map, dtype=bool)


        # TODO need to be added observation for grey team
        # self.observation_space_grey = np.full_like(self._env, -1)


    def step(self, entities_action=None, cur_suggestions=None):
        """
        Takes one step in the capture the flag game

        :param
            entities_action: contains actions for entity 1-n
            cur_suggestions: suggestions from rl to human
        :return:
            state    : object
            CapEnv object
            reward  : float
            float containing the reward for the given action
            isDone  : bool
            decides if the game is over
            info    :
        """

        indiv_action_space = len(self.ACTION)

        if self.CONTROL_ALL:
            assert entities_action is not None, 'Under CONTROL_ALL setting, action must be specified'
            assert (type(entities_action) is list) or (type(entities_action) is np.ndarray), \
                    'CONTROLL_ALL setting requires list (or numpy array) type of action'
            assert len(entities_action) == self.NUM_BLUE+self.NUM_RED+self.NUM_UAV+self.NUM_UAV, \
                    'You entered wrong number of moves.'

            move_list_blue = entities_action[:self.NUM_UAV+self.NUM_BLUE+self.NUM_BLUE_UGV2]
            move_list_red  = entities_action[-self.NUM_UAV-self.NUM_RED-self.NUM_RED_UGV2:]
        else:
            # Get actions from uploaded policies
            move_list_red = []
            if self.mode != "sandbox":
                try:
                    move_list_red = self._policy_red.gen_action(self._team_red, self.get_obs_red)
                except Exception as e:
                    print("No valid policy for red team", e)
                    traceback.print_exc()
                    exit()

            move_list_blue = []
            if entities_action is None:
                try:
                    move_list_blue = self._policy_blue.gen_action(self._team_blue, self.get_obs_blue)
                except Exception as e:
                    print("No valid policy for blue team and no actions provided", e)
                    traceback.print_exc()
                    exit()
            elif type(entities_action) is int:
                if entities_action >= len(self.ACTION) ** (self.NUM_BLUE + self.NUM_UAV + self.NUM_BLUE_UGV2):
                    sys.exit("ERROR: You entered too many moves. There are " + str(self.NUM_BLUE + self.NUM_UAV + self.NUM_BLUE_UGV2) + " entities.")
                while len(move_list_blue) < (self.NUM_BLUE + self.NUM_UAV + self.NUM_BLUE_UGV2):
                    move_list_blue.append(entities_action % indiv_action_space)
                    entities_action = int(entities_action / indiv_action_space)
            else:
                if len(entities_action) != self.NUM_BLUE + self.NUM_UAV + self.NUM_BLUE_UGV2:
                    sys.exit("ERROR: You entered wrong number of moves. There are " + str(self.NUM_BLUE + self.NUM_UAV + self.NUM_BLUE_UGV2) + " entities.")
                move_list_blue = entities_action


        # Move team1
        positions = []
        for idx, act in enumerate(move_list_blue):
            if self.STOCH_TRANSITIONS and self.np_random.rand() < self.STOCH_TRANSITIONS_EPS:
                if self._policy_blue is not None and not self._policy_blue._random_transition_safe:
                    act = 0
                else:
                    act = self._stoch_transition(self._team_blue[idx].get_loc())
            self._team_blue[idx].move(self.ACTION[act], self._env, self._static_map)
            positions.append((self._team_blue[idx].get_loc(), self._team_blue[idx].isAlive))
        self._blue_trajectory.append(positions)

        # Move team2
        positions = []
        for idx, act in enumerate(move_list_red):
            if self.STOCH_TRANSITIONS and self.np_random.rand() < self.STOCH_TRANSITIONS_EPS:
                if self._policy_red is not None and not self._policy_red._random_transition_safe:
                    act = 0
                else:
                    act = self._stoch_transition(self._team_red[idx].get_loc())
            self._team_red[idx].move(self.ACTION[act], self._env, self._static_map)
            positions.append((self._team_red[idx].get_loc(), self._team_red[idx].isAlive))
        self._red_trajectory.append(positions)

        self._create_observation_mask()
        
        # Update individual's memory
        for agent in self._team_blue + self._team_red:
            if agent.memory_mode == "fog":
                agent.update_memory(env=self)
        
        # Update team memory
        if self.TEAM_MEMORY == "fog":
            self._update_global_memory(env=self)


        # Run interaction
        survive_list = []
        for entity in self._team_blue + self._team_red:
            if not entity.isAlive:
                survive_list.append(False)
            else:
                survive_list.append(self._interaction(entity))
        for status, entity in zip(survive_list, self._team_blue+self._team_red):
            entity.isAlive = status

        # Check win and lose conditions
        has_alive_entity = False
        for i in self._team_red:
            if i.isAlive and not i.is_air:
                has_alive_entity = True
                locx, locy = i.get_loc()
                if self._static_map[locx][locy] == TEAM1_FLAG:  # TEAM 1 == BLUE
                    self.red_win = True
                    self.blue_flag_captured = True
                    
        # TODO Change last condition for multi agent model
        if not has_alive_entity and self.mode != "sandbox" and self.mode != "human_blue":
            self.blue_win = True
            self.red_eliminated = True

        has_alive_entity = False
        for i in self._team_blue:
            if i.isAlive and not i.is_air:
                has_alive_entity = True
                locx, locy = i.get_loc()
                if self._static_map[locx][locy] == TEAM2_FLAG:
                    self.blue_win = True
                    self.red_flag_captured = True
                    
        if not has_alive_entity:
            self.red_win = True
            self.blue_eliminated = True

        # Calculate Reward
        reward = self._create_reward()

        isDone = self.red_win or self.blue_win

        # Pass internal info
        info = {
                'blue_trajectory': self._blue_trajectory,
                'red_trajectory': self._red_trajectory,
                'static_map': self._static_map
            }

        self.run_step += 1
        
        return self.get_obs_blue, reward, isDone, info

    def _stoch_transition(self, loc):
        if self.STOCH_TRANSITIONS_MOD == 'random':
            return self.np_random.randint(0,len(self.ACTION))
        elif self.STOCH_TRANSITIONS_MOD == 'fix':
            return 0
        elif self.STOCH_TRANSITIONS_MOD == 'drift1':
            if loc[0] > self.map_size[0]//2 or loc[1] > self.map_size[1]//2:
                return self.np_random.choice([1,2])
            else:
                return self.np_random.choice([3,4])
        else:
            raise AttributeError('Unknown transition mod is used: {}'.format(self.STOCH_TRANSITIONS_MOD))


    def _interaction(self, entity):
        """
        Interaction 

        Checks if a unit is dead
        If configuration parameter 'STOCH_ATTACK' is true, the interaction becomes stochastic

        Parameters
        ----------
        self    : object
            CapEnv object
        entity_num  : int
            Represents where in the unit list is the unit to move
        team    : int
            Represents which team the unit belongs to

        Return
        ______
        bool    :
            Return true if the entity survived after the interaction
        """
        if entity.is_air:
            return True

        in_range = lambda i, j, r: i*i + j*j <= r*r + 1e-8
        loc = np.array(entity.get_loc())
        if self.STOCH_ATTACK:
            enemy_list = self._team_red if entity.team == TEAM1_BACKGROUND else self._team_blue
            friend_list = self._team_blue if entity.team == TEAM1_BACKGROUND else self._team_red

            n_enemies = 0
            for enemy in enemy_list:
                if enemy.is_air: continue
                if not enemy.isAlive: continue
                att_range = enemy.a_range
                enemy_loc = np.array(enemy.get_loc())

                if in_range(*(loc-enemy_loc), att_range):
                    n_enemies += enemy.get_advantage

            n_friends = 0
            for friend in friend_list:
                if friend.is_air: continue
                if not friend.isAlive: continue
                att_range = friend.a_range
                friend_loc = np.array(friend.get_loc())

                if np.all(friend_loc != loc) and in_range(*(loc-friend_loc), att_range):
                    n_friends += friend.get_advantage

            # Interaction 
            if n_enemies > 0:
                # Advantage bias for being in team territory
                if entity.team == self._static_map[entity.get_loc()]:
                    n_friends += self.STOCH_ATTACK_BIAS
                else:
                    n_enemies += self.STOCH_ATTACK_BIAS

                if self.np_random.rand() > n_friends/(n_friends + n_enemies):
                    #self._env[loc[0], loc[1], CHANNEL[DEAD]] = REPRESENT[DEAD]
                    return False 
        else:
            # Check if agent is in enemy's territory
            if self._static_map[tuple(loc)] == entity.team:
                return True

            enemy_list = self._team_red if entity.team == TEAM1_BACKGROUND else self._team_blue

            for enemy in enemy_list:
                if enemy.is_air: continue
                if not enemy.isAlive: continue
                att_range = enemy.a_range
                enemy_loc = np.array(enemy.get_loc())

                if in_range(*(loc-enemy_loc), att_range):
                    # If enemy is within attack range, declare dead
                    #self._env[loc[0], loc[1], CHANNEL[DEAD]] = REPRESENT[DEAD]
                    return False

        return True


    def _update_global_memory(self, env):
        """ 
        team memory map
        
        """
        
        l, b = self.blue_memory.shape
        for blue_agent in self._team_blue:
            b_obs = blue_agent.get_obs(env=env)
            leng, breth = b_obs.shape
            leng, breth = leng//2, breth//2
            b_coord_x, b_coord_y = blue_agent.get_loc()
            b_offset_x, b_offset_y = leng - b_coord_x, breth - b_coord_y
            b_obs = b_obs[b_offset_x: b_offset_x + l, b_offset_y: b_offset_y + b]   
            b_coord = b_obs!= const.UNKNOWN
            self.blue_memory[b_coord] = self._static_map[b_coord]
             
        l, b = self.red_memory.shape
        for red_agent in self._team_red:
            r_obs = red_agent.get_obs(env=env)
            leng, breth = r_obs.shape
            leng, breth = leng//2, breth//2
            r_coord_x, r_coord_y = red_agent.get_loc()
            r_offset_x, r_offset_y = leng - r_coord_x, breth - r_coord_y
            r_obs = r_obs[r_offset_x: r_offset_x + l, r_offset_y: r_offset_y + b]   
            r_coord = r_obs!= const.UNKNOWN
            self.red_memory[r_coord] = self._static_map[r_coord]
        
        return

    def _create_reward(self, mode='dense'):
        """
        Range (-100, 100)

        Parameters
        ----------
        self    : object
            CapEnv object
        """

        assert mode in ['dense', 'flag', 'combat', 'defense', 'capture']

        if mode == 'dense':
            # Dead enemy team gives .5/total units for each dead unit
            if self.red_win:
                return -100
            if self.blue_win:
                return 100
            reward = 0
            red_alive = sum([entity.isAlive for entity in self._team_red if not entity.is_air])
            blue_alive = sum([entity.isAlive for entity in self._team_blue if not entity.is_air])
            if self.mode != 'sandbox':
                reward += 50.0 * red_alive / (self.NUM_RED + self.NUM_RED_UGV2)
            reward -= 50.0 * blue_alive / (self.NUM_BLUE + self.NUM_BLUE_UGV2)
            return reward
        elif mode == 'flag':
            # Flag game reward
            if self.red_flag_captured:
                return 100
            if self.blue_flag_captured:
                return -100
        elif mode == 'combat':
            # Aggressive combat game. Elliminate enemy to win
            red_alive = sum([entity.isAlive for entity in self._team_red if not entity.is_air])
            return 100 * red_alive / self.NUM_RED
        elif mode == 'defense':
            # Lose reward if flag is lost.
            if self.blue_flag_captured:
                return -100
        elif mode == 'capture':
            # Reward only by capturing (sparse)
            if self.red_flag_captured:
                return 100


    def render(self, mode='human'):
        """
        Renders the screen options="obs, env"

        Parameters
        ----------
        self    : object
            CapEnv object
        mode    : string
            Defines what will be rendered
        """

        if (self.RENDER_INDIV_MEMORY == True and self.INDIV_MEMORY == "fog") or (self.RENDER_TEAM_MEMORY == True and self.TEAM_MEMORY == "fog"):
            SCREEN_W = 1200
            SCREEN_H = 600

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
                self.viewer.set_bounds(0, SCREEN_W, 0, SCREEN_H)
    
            self.viewer.draw_polygon([(0, 0), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), (0, SCREEN_H)], color=(0, 0, 0))

            self._env_render(self._static_map,
                            [7, 7], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_blue_render,
                            [7+1.49*SCREEN_H//3, 7], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_red_render,
                            [7+1.49*SCREEN_H//3, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_full_state,
                            [7, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])

            # ind blue agent memory rendering
            for num_blue, blue_agent in enumerate(self._team_blue):
                if num_blue < 2:
                    blue_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if blue_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(blue_agent.memory,
                                         [900+num_blue*SCREEN_H//4, 7], [SCREEN_H//4-10, SCREEN_H//4-10])
                else:
                    blue_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if blue_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(blue_agent.memory,
                                         [900+(num_blue-2)*SCREEN_H//4, 7+SCREEN_H//4], [SCREEN_H//4-10, SCREEN_H//4-10])

            # ind red agent memory rendering
            for num_red, red_agent in enumerate(self._team_red):
                if num_red < 2:
                    red_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if red_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(red_agent.memory,
                                         [900+num_red*SCREEN_H//4, 7+1.49*SCREEN_H//2], [SCREEN_H//4-10, SCREEN_H//4-10])
    
                else:
                    red_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if red_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(red_agent.memory,
                                         [900+(num_red-2)*SCREEN_H//4, 7+SCREEN_H//2], [SCREEN_H//4-10, SCREEN_H//4-10])

            if self.TEAM_MEMORY == "fog" and self.RENDER_TEAM_MEMORY == True:
                # blue team memory rendering
                self._env_render(self.blue_memory,
                                 [7+2.98*SCREEN_H//3, 7], [SCREEN_H//2-10, SCREEN_H//2-10])
                # red team memory rendering    
                self._env_render(self.red_memory,
                                 [7+2.98*SCREEN_H//3, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])
        else:
            SCREEN_W = 600
            SCREEN_H = 600
                
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
                self.viewer.set_bounds(0, SCREEN_W, 0, SCREEN_H)

            self.viewer.draw_polygon([(0, 0), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), (0, SCREEN_H)], color=(0, 0, 0))
            
            self._env_render(self._static_map,
                            [5, 10], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_blue_render,
                            [5+SCREEN_W//2, 10], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._agent_render(self.get_full_state,
                            [5+SCREEN_W//2, 10], [SCREEN_W//2-10, SCREEN_H//2-10], self._team_blue)
            self._env_render(self.get_obs_red_render,
                            [5+SCREEN_W//2, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._env_render(self.get_full_state,
                            [5, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._agent_render(self.get_full_state,
                            [5, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _env_render(self, env, rend_loc, rend_size):
        map_h = len(env[0])
        map_w = len(env)

        tile_w = rend_size[0] / len(env)
        tile_h = rend_size[1] / len(env[0])

        for y in range(map_h):
            for x in range(map_w):
                locx, locy = rend_loc
                locx += x * tile_w
                locy += y * tile_h
                cur_color = np.divide(COLOR_DICT[env[x][y]], 255.0)
                self.viewer.draw_polygon([
                    (locx, locy),
                    (locx + tile_w, locy),
                    (locx + tile_w, locy + tile_h),
                    (locx, locy + tile_h)], color=cur_color)

                if env[x][y] == TEAM1_UAV or env[x][y] == TEAM2_UAV:
                    self.viewer.draw_polyline([
                        (locx, locy),
                        (locx + tile_w, locy + tile_h)],
                        color=(0,0,0), linewidth=2)
                    self.viewer.draw_polyline([
                        (locx + tile_w, locy),
                        (locx, locy + tile_h)],
                        color=(0,0,0), linewidth=2)#col * tile_w, row * tile_h

    def _agent_render(self, env, rend_loc, rend_size, agents=None):
        if agents is None:
            agents = self._team_blue + self._team_red
        tile_w = rend_size[0] / len(env)
        tile_h = rend_size[1] / len(env[0])

        for entity in agents:
            if not entity.isAlive: continue
            x,y = entity.get_loc()
            locx, locy = rend_loc
            locx += x * tile_w
            locy += y * tile_h
            cur_color = COLOR_DICT[entity.unit_type]
            cur_color = np.divide(cur_color, 255.0)
            self.viewer.draw_polygon([
                (locx, locy),
                (locx + tile_w, locy),
                (locx + tile_w, locy + tile_h),
                (locx, locy + tile_h)], color=cur_color)

            if entity.is_air:
                self.viewer.draw_polyline([
                    (locx, locy),
                    (locx + tile_w, locy + tile_h)],
                    color=(0,0,0), linewidth=2)
                self.viewer.draw_polyline([
                    (locx + tile_w, locy),
                    (locx, locy + tile_h)],
                    color=(0,0,0), linewidth=2)#col * tile_w, row * tile_h
            if type(entity) == GroundVehicle_Tank:
                self.viewer.draw_polyline([
                    (locx, locy),
                    (locx + tile_w//2, locy + tile_h)],
                    color=(0,0,0), linewidth=3)
                self.viewer.draw_polyline([
                    (locx + tile_w//2, locy + tile_h),
                    (locx + tile_w, locy)],
                    color=(0,0,0), linewidth=3)

            if entity.marker is not None:
                ratio = 0.6
                color = np.divide(entity.marker, 255.0)
                self.viewer.draw_polygon([
                    (locx + tile_w * ratio, locy + tile_h * ratio),
                    (locx + tile_w, locy + tile_h * ratio),
                    (locx + tile_w, locy + tile_h),
                    (locx + tile_w * ratio, locy + tile_h)], color=color)

    def close(self):
        if self.viewer: self.viewer.close()

    def _env_flat(self, mask=None):
        # Return 2D representation of the state
        board = np.copy(self._static_map)
        if mask is not None:
            board[mask] = UNKNOWN
        for entity in self._team_blue+self._team_red:
            if not entity.isAlive: continue
            loc = entity.get_loc()
            if mask is not None and mask[loc]: continue
            board[loc] = entity.unit_type
        return board

    @property
    def get_full_state(self, mask=None):
        return self._env_flat()

    @property
    def get_full_state_channel(self):
        return np.copy(self._env)

    @property
    def get_full_state_rgb(self):
        w, h, ch = self._env.shape
        image = np.full(shape=[w, h, 3], fill_value=0, dtype=int)
        for element in CHANNEL.keys():
            channel = CHANNEL[element]
            rep = REPRESENT[element]
            image[self._env[:,:,channel]==rep] = np.array(COLOR_DICT[element])
        return image

    @property
    def get_team_blue(self):
        return np.copy(self._team_blue)

    @property
    def get_team_red(self):
        return np.copy(self._team_red)

    @property
    def get_team_grey(self):
        return np.copy(self.team_grey)

    @property
    def get_map(self):
        return np.copy(self._static_map)

    @property
    def get_obs_blue(self):
        blue_view = np.copy(self._env)

        if self.BLUE_PARTIAL:
            mask_channel = CHANNEL[UNKNOWN]
            mask_represent = REPRESENT[UNKNOWN]

            blue_view[self._blue_mask, :] = 0
            blue_view[self._blue_mask, mask_channel] = mask_represent

        return blue_view

    @property
    def get_obs_red(self):
        red_view = np.copy(self._env)

        if self.RED_PARTIAL:
            mask_represent = REPRESENT[UNKNOWN]
            mask_channel = CHANNEL[UNKNOWN]

            red_view[self._red_mask, :] = 0
            red_view[self._red_mask, mask_channel] = mask_represent

        # Change red's perspective same as blue
        swap = [CHANNEL[TEAM1_BACKGROUND], CHANNEL[TEAM1_UGV], CHANNEL[TEAM1_UAV], CHANNEL[TEAM1_FLAG], CHANNEL[TEAM1_UGV2]]

        for ch in swap:
            red_view[:,:,ch] *= -1

        return red_view

    @property
    def get_obs_blue_render(self):
        return self._env_flat(self._blue_mask)

    @property
    def get_obs_red_render(self):
        return self._env_flat(self._red_mask)

    @property
    def get_obs_grey(self):
        return np.copy(self.observation_space_grey)


    # def quit_game(self):
    #     if self.viewer is not None:
    #         self.viewer.close()
    #         self.viewer = None


# Different environment sizes and modes
# Random modes
class CapEnvGenerate(CapEnv):
    def __init__(self):
        super(CapEnvGenerate, self).__init__(map_size=20)


# State space for capture the flag
class Board(spaces.Space):
    """A Board in R^3 used for CtF """
    def __init__(self, shape=None, dtype=np.uint8):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        if shape is None:
            self.shape = (20, 20, NUM_CHANNEL)
        else:
            assert shape[2] == NUM_CHANNEL
            self.shape = tuple(shape)
        super(Board, self).__init__(self.shape, self.dtype)

    def __repr__(self):
        return "Board" + str(self.shape)

    def sample(self):
        map_obj = [NUM_BLUE, NUM_UAV, NUM_RED, NUM_UAV, NUM_GRAY]
        state, _, _ = gen_random_map('map',
                self.shape[0], rand_zones=False, map_obj=map_obj)
        return state

