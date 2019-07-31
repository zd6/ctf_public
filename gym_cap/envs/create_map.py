import numpy as np
import random
from .const import *

"""
This module generates a map given desire conditions:

gen_random_map:
    It creates random map given dimension size, number of obstacles,
    and number of agents for each team.
    The background can be divided in half, or random box.
custom_map:
    It generates map given a numpy array.
"""

def gen_random_map(name, dim=20, in_seed=None, rand_zones=False, np_random=None,
            map_obj=[NUM_BLUE, NUM_UAV, NUM_RED, NUM_UAV, NUM_GRAY]):
    """
    Method

    Generate map with given setting

    Parameters
    ----------
    name        : TODO
        Not used
    dim         : int
        Size of the map
    in_seed     : int
        Random seed between 0 and 2**32
    rand_zones  : bool
        True if zones are defined random
    map_obj     : list
        The necessary elements to build the map
        0   : blue UGV
        1   : blue UAV
        2   : red UGV
        3   : red UAV
        4   : gray units
    """

    # ASSERTION
    assert map_obj is not None

    # INITIALIZE THE SEED 
    if np_random is None:
        np_random = np.random
    if in_seed is not None:
        np.random.seed(in_seed)

    # PARAMETERS
    total_blue = map_obj[0] + map_obj[1] + 1  # No.elements to place in blue zone
    total_red = map_obj[2] + map_obj[3] + 1   # No.elements to place in red zone

    # CH 0 : UNKNOWN
    mask = np.zeros([dim, dim], dtype=int)

    # CH 1 : ZONE and OBSTACLE
    zone = np.ones([dim, dim], dtype=int)  # 1 for blue, -1 for red, 0 for obstacle
    static_map = np.zeros([dim, dim], dtype=int)
    if rand_zones:
        sx, sy = np_random.randint(dim//2, 4*dim//5, [2])
        lx, ly = np_random.randint(0, dim - max(sx,sy)-1, [2])
        zone[lx:lx+sx, ly:ly+sy] = -1
        static_map[lx:lx+sx, ly:ly+sy] = TEAM2_BACKGROUND
    else:
        zone[:,0:dim//2] = -1
        static_map[:,0:dim//2] = TEAM2_BACKGROUND
        zone = np.rot90(zone)
    if 0.5 < np_random.rand():
        zone = -zone  # Reverse
        static_map = -static_map+1  # TODO: not a safe method to reverse static_map

    num_obst = int(np.sqrt(dim))
    for i in range(num_obst):
        lx, ly = np_random.randint(0, dim, [2])
        sx, sy = np_random.randint(0, dim//5, [2]) + 1
        zone[lx-sx:lx+sx, ly-sy:ly+sy] = 0
        static_map[lx-sx:lx+sx, ly-sy:ly+sy] = OBSTACLE

    if dim < 20:
        element_count = dict(zip(*np.unique(zone, return_counts=True)))
        blue_capacity = element_count[1]
        red_capacity = element_count[-1]
        if blue_capacity < total_blue:
            raise Exception('Cannot fit all blue object in an given map.')
        if red_capacity < total_red:
            raise Exception('Cannot fit all red object in an given map.')

    # CH 2 : FLAG
    blue_pool = np.argwhere(zone== 1)
    blue_indices = np_random.choice(len(blue_pool), total_blue, replace=False)
    blue_coord = np.take(blue_pool, blue_indices, axis=0)

    red_pool = np.argwhere(zone==-1)
    red_indices = np_random.choice(len(red_pool), total_red, replace=False)
    red_coord = np.take(red_pool, red_indices, axis=0)

    num_flag = 1  # Parameter for later change
    flag = np.zeros([dim, dim], dtype=int)

    blue_flag_coord, blue_coord = blue_coord[:num_flag], blue_coord[num_flag:]
    flag[blue_flag_coord[:,0], blue_flag_coord[:,1]] = 1
    static_map[blue_flag_coord[:,0], blue_flag_coord[:,1]] = TEAM1_FLAG
    red_flag_coord, red_coord = red_coord[:num_flag], red_coord[num_flag:]
    flag[red_flag_coord[:,0], red_flag_coord[:,1]] = -1
    static_map[red_flag_coord[:,0], red_flag_coord[:,1]] = TEAM2_FLAG
    
    # CH 3 : UGV
    agent_locs = {}
    ugv = np.zeros([dim, dim], dtype=int)

    blue_ugv_coord, blue_coord = blue_coord[:map_obj[0]], blue_coord[map_obj[0]:]
    ugv[blue_ugv_coord[:,0], blue_ugv_coord[:,1]] = 1
    red_ugv_coord, red_coord = red_coord[:map_obj[2]], red_coord[map_obj[2]:]
    ugv[red_ugv_coord[:,0], red_ugv_coord[:,1]] = -1

    agent_locs[TEAM1_UGV] = blue_ugv_coord.tolist()
    agent_locs[TEAM2_UGV] = red_ugv_coord.tolist()
    
    # CH 4 : UAV
    uav = np.zeros([dim, dim], dtype=int)

    blue_uav_coord, blue_coord = blue_coord[:map_obj[1]], blue_coord[map_obj[1]:]
    uav[blue_uav_coord[:,0], blue_uav_coord[:,1]] = 1
    red_uav_coord, red_coord = red_coord[:map_obj[3]], red_coord[map_obj[3]:]
    uav[red_uav_coord[:,0], red_uav_coord[:,1]] = -1

    agent_locs[TEAM1_UAV] = blue_uav_coord.tolist()
    agent_locs[TEAM2_UAV] = red_uav_coord.tolist()

    # CH 5 : GRAY UGV (Neutral)
    gray = np.zeros([dim, dim], dtype=int)
    # TODO: Figure out gray agents

    new_map = np.stack([mask, zone, flag, ugv, uav, gray], axis=-1)

    return new_map, static_map, agent_locs

def custom_map(new_map):
    """
    Method
        Outputs static_map when new_map is given as input.
        Addtionally the number of agents will also be
        counted
    
    Parameters
    ----------
    new_map        : numpy array
        new_map
    The necessary elements:
        ugv_1   : blue UGV
        ugv_2   : red UGV
        uav_2   : red UAV
        gray    : gray units
        
    """
    
    # build object count array
    element_count = dict(zip(*np.unique(new_map, return_counts=True)))
    ugv_1 = element_count.get(TEAM1_UGV, 0)
    ugv_2 = element_count.get(TEAM2_UGV, 0)
    uav_1 = element_count.get(TEAM1_UAV, 0)
    uav_2 = element_count.get(TEAM2_UAV, 0)
    gray = element_count.get(TEAM3_UGV, 0)
    obj_arr = [ugv_1, uav_1, ugv_2, uav_2, gray]

    # Find locations
    team1_ugv_loc = new_map==TEAM1_UGV
    team1_uav_loc = new_map==TEAM1_UAV
    team2_ugv_loc = new_map==TEAM2_UGV
    team2_uav_loc = new_map==TEAM2_UAV
    team3_ugv_loc = new_map==TEAM3_UGV

    # build static map
    static_map = np.copy(new_map)
    static_map[team1_ugv_loc] = TEAM1_BACKGROUND
    static_map[team1_uav_loc] = TEAM1_BACKGROUND
    static_map[team2_ugv_loc] = TEAM2_BACKGROUND
    static_map[team2_uav_loc] = TEAM2_BACKGROUND
    static_map[team3_ugv_loc] = TEAM1_BACKGROUND # subject to change
    
    # build 3D new_map
    l, b = new_map.shape
    nd_map = np.zeros([l, b,NUM_CHANNEL], dtype = int)
    for elem in CHANNEL.keys():
        ch = CHANNEL[elem]
        const = REPRESENT[elem]
        if elem in new_map:
            nd_map[new_map==elem,ch] = const
    
    # location of agents
    agent_locs = {}
    agent_locs[TEAM1_UGV] = np.argwhere(team1_ugv_loc)
    agent_locs[TEAM1_UAV] = np.argwhere(team1_uav_loc)
    agent_locs[TEAM2_UGV] = np.argwhere(team2_ugv_loc)
    agent_locs[TEAM2_UAV] = np.argwhere(team2_uav_loc)
    
    nd_map[team1_ugv_loc, CHANNEL[TEAM1_BACKGROUND]] = REPRESENT[TEAM1_BACKGROUND]
    nd_map[team1_uav_loc, CHANNEL[TEAM1_BACKGROUND]] = REPRESENT[TEAM1_BACKGROUND]
    nd_map[team2_ugv_loc, CHANNEL[TEAM2_BACKGROUND]] = REPRESENT[TEAM2_BACKGROUND]
    nd_map[team2_uav_loc, CHANNEL[TEAM2_BACKGROUND]] = REPRESENT[TEAM2_BACKGROUND]
    
    return nd_map, static_map, obj_arr, agent_locs
