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
            map_obj=None):
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
    num_flag = 1
    total_blue, total_red = num_flag, num_flag
    for k, v in map_obj.items():
        total_blue += v[0]
        total_red += v[1]

    # CH 0 : UNKNOWN
    mask = np.zeros([dim, dim], dtype=int)

    can_fit = True
    while can_fit:
        # CH 1 : ZONE (included in static)
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
            #zone = np.rot90(zone)
        if 0.5 < np_random.rand():
            zone = -zone  # Reverse
            static_map = -static_map+1  # TODO: not a safe method to reverse static_map

        # CH 3 : OBSTACLE
        obst = np.zeros([dim, dim], dtype=int)
        num_obst = int(np.sqrt(dim))
        for i in range(num_obst):
            lx, ly = np_random.randint(0, dim, [2])
            sx, sy = np_random.randint(0, dim//5, [2]) + 1
            zone[lx-sx:lx+sx, ly-sy:ly+sy] = 0
            obst[lx-sx:lx+sx, ly-sy:ly+sy] = REPRESENT[OBSTACLE]
            static_map[lx-sx:lx+sx, ly-sy:ly+sy] = OBSTACLE

        ## Random Coord Create
        try: # Take possible coordinates for all elements
            blue_pool = np.argwhere(zone== 1)
            blue_indices = np_random.choice(len(blue_pool), total_blue, replace=False)
            blue_coord = np.take(blue_pool, blue_indices, axis=0)

            red_pool = np.argwhere(zone==-1)
            red_indices = np_random.choice(len(red_pool), total_red, replace=False)
            red_coord = np.take(red_pool, red_indices, axis=0)

            can_fit = False # Exit loop
        except ValueError as e:
            msg = "This warning occurs when the map is too small to allocate all elements."
            #raise ValueError(msg) from e

    # CH 2 : FLAG (included in static)
    flag = np.zeros([dim, dim], dtype=int)

    blue_flag_coord, blue_coord = blue_coord[:num_flag], blue_coord[num_flag:]
    flag[blue_flag_coord[:,0], blue_flag_coord[:,1]] = 1
    static_map[blue_flag_coord[:,0], blue_flag_coord[:,1]] = TEAM1_FLAG

    red_flag_coord, red_coord = red_coord[:num_flag], red_coord[num_flag:]
    flag[red_flag_coord[:,0], red_flag_coord[:,1]] = -1
    static_map[red_flag_coord[:,0], red_flag_coord[:,1]] = TEAM2_FLAG

    # Build New Map
    temp = np.zeros_like(mask)
    new_map = np.zeros([dim, dim, NUM_CHANNEL], dtype=int)
    new_map[:,:,0] = mask
    new_map[:,:,1] = zone
    new_map[:,:,2] = flag
    new_map[:,:,3] = obst
    
    ## Agents
    agent_locs = {}

    keys = [(TEAM1_UAV, TEAM2_UAV),
            (TEAM1_UGV, TEAM2_UGV),
            (TEAM1_UGV2, TEAM2_UGV2),
            (TEAM1_UGV3, TEAM2_UGV3),
            (TEAM1_UGV4, TEAM2_UGV4)]
    for k in keys:
        nb, nr = map_obj[k]
        
        channel = CHANNEL[k[0]]
        coord, blue_coord = blue_coord[:nb], blue_coord[nb:]
        new_map[coord[:,0], coord[:,1], channel] = REPRESENT[k[0]]
        agent_locs[k[0]] = coord.tolist()

        channel = CHANNEL[k[1]]
        coord, red_coord = red_coord[:nr], red_coord[nr:]
        new_map[coord[:,0], coord[:,1], channel] = REPRESENT[k[1]]
        agent_locs[k[1]] = coord.tolist()

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

    keys = {TEAM1_BACKGROUND: [TEAM1_UGV, TEAM1_UAV, TEAM1_UGV2, TEAM1_UGV3, TEAM1_UGV4],
            TEAM2_BACKGROUND: [TEAM2_UGV, TEAM2_UAV, TEAM2_UGV2, TEAM2_UGV3, TEAM2_UGV4] }
    obj_dict = {TEAM1_BACKGROUND: [],
                TEAM2_BACKGROUND: []}
    static_map = np.copy(new_map)
    agent_locs = {}
    l, b = new_map.shape

    # Build 3d map
    nd_map = np.zeros([l, b, NUM_CHANNEL], dtype = int)
    for elem in CHANNEL.keys():
        ch = CHANNEL[elem]
        const = REPRESENT[elem]
        if elem in new_map:
            nd_map[new_map==elem,ch] = const

    for team, elems in keys.items():
        for e in elems:
            count = element_count.get(e, 0)
            obj_dict[team].append(count)

            loc = new_map==e
            static_map[loc] = team

            agent_locs[e] = np.argwhere(loc)

            nd_map[loc, CHANNEL[team]] = REPRESENT[team]

    return nd_map, static_map, obj_dict, agent_locs

