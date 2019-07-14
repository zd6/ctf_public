import numpy as np
import random
from .const import *

class CreateMap:
    """This class generates and back-propogates a random map
    given dimension size, number of obstacles,
    and number of agents for each team"""

    @staticmethod
    def gen_map(name, dim=20, in_seed=None, rand_zones=False, np_random=None,
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
        channel = CHANNEL
        repr_const = REPRESENT

        # ASSERTION
        assert map_obj is not None
        assert channel[TEAM1_BACKGROUND] == channel[TEAM2_BACKGROUND]
        assert channel[TEAM1_UGV] == channel[TEAM2_UGV]
        assert channel[TEAM1_UAV] == channel[TEAM2_UAV]
        assert channel[TEAM1_FLAG] == channel[TEAM2_FLAG]

        # init the seed and set new_map to zeros
        if np_random == None:
            np_random = np.random
        if not in_seed == None:
            np.random.seed(in_seed)

        # zones init
        new_map = np.zeros([dim, dim, NUM_CHANNEL], dtype=int)
        new_map[:,:,channel[TEAM2_BACKGROUND]] = repr_const[TEAM2_BACKGROUND]
        if rand_zones:
            sx, sy = np_random.randint(dim//2, 4*dim//5, [2])
            lx, ly = np_random.randint(0, dim - max(sx,sy)-1, [2])
            new_map[lx:lx+sx, ly:ly+sy, channel[TEAM1_BACKGROUND]] = repr_const[TEAM1_BACKGROUND]
        else:
            new_map[:,0:dim//2, channel[TEAM1_BACKGROUND]] = repr_const[TEAM1_BACKGROUND]

        # obstacles init
        num_obst = int(np.sqrt(dim))
        for i in range(num_obst):
            lx, ly = np_random.randint(0, dim, [2])
            sx, sy = np_random.randint(0, dim//5, [2]) + 1
            new_map[lx-sx:lx+sx, ly-sy:ly+sy, channel[OBSTACLE]] = repr_const[OBSTACLE]
            new_map[lx-sx:lx+sx, ly-sy:ly+sy, channel[TEAM1_BACKGROUND]] = 0

        element_count = dict(zip(*np.unique(new_map[:,:,channel[TEAM1_BACKGROUND]], return_counts=True)))
        num_team1_space = element_count[repr_const[TEAM1_BACKGROUND]]
        num_team2_space = element_count[repr_const[TEAM2_BACKGROUND]]
        if num_team1_space < 1 + map_obj[0] + map_obj[1]:
            raise Exception('Cannot fit all blue object in an given map.')
        if num_team2_space  < 1 + map_obj[2] + map_obj[3]:
            raise Exception('Cannot fit all red object in an given map.')

        # define location of flags
        team_map = new_map[:,:,channel[TEAM1_BACKGROUND]]
        team1_pool = np.argwhere(team_map==repr_const[TEAM1_BACKGROUND]).tolist()
        team2_pool = np.argwhere(team_map==repr_const[TEAM2_BACKGROUND]).tolist()
        random.shuffle(team1_pool)
        random.shuffle(team2_pool)

        CreateMap.populate_map(new_map, team1_pool,
                repr_const[TEAM1_FLAG], channel[TEAM1_FLAG], 1)
        CreateMap.populate_map(new_map, team2_pool,
                repr_const[TEAM2_FLAG], channel[TEAM2_FLAG], 1)

        # define location of agents
        agent_locs = {}
        agent_locs[TEAM1_UGV] = CreateMap.populate_map(new_map, team1_pool,
                repr_const[TEAM1_UGV], channel[TEAM1_UGV], map_obj[0])
        agent_locs[TEAM1_UAV] = CreateMap.populate_map(new_map, team1_pool,
                repr_const[TEAM1_UAV], channel[TEAM1_UAV], map_obj[1])
        agent_locs[TEAM2_UGV] = CreateMap.populate_map(new_map, team2_pool,
                repr_const[TEAM2_UGV], channel[TEAM2_UGV], map_obj[2])
        agent_locs[TEAM2_UAV] = CreateMap.populate_map(new_map, team2_pool,
                repr_const[TEAM2_UAV], channel[TEAM2_UAV], map_obj[3])

        # TODO: change zone for grey team to complete map
        #new_map = CreateMap.populate_map(new_map,
        #                     TEAM2_BACKGROUND, TEAM3_UGV, map_obj[4])

        #np.save('map.npy', new_map)

        # build static map
        static_elem = [TEAM1_BACKGROUND, TEAM2_BACKGROUND, OBSTACLE, TEAM1_FLAG, TEAM2_FLAG]
        static_map = np.zeros([dim,dim], dtype=int)
        for elem in static_elem:
            ch = channel[elem]
            rep = repr_const[elem]
            static_map[new_map[:,:,ch]==rep] = elem

        return new_map, static_map, agent_locs
    
    @staticmethod
    def set_custom_map(new_map):
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
        
        channel = CHANNEL
        repr_const = REPRESENT
        
        # build object count array
        element_count = dict(zip(*np.unique(new_map, return_counts=True)))
        ugv_1 = element_count.get(TEAM1_UGV, 0)
        ugv_2 = element_count.get(TEAM2_UGV, 0)
        uav_1 = element_count.get(TEAM1_UAV, 0)
        uav_2 = element_count.get(TEAM2_UAV, 0)
        gray = element_count.get(TEAM3_UGV, 0)
        obj_arr = [ugv_1, uav_1, ugv_2, uav_2, gray]
                    
        # build static map
        static_map = np.copy(new_map)
        static_map[new_map==TEAM1_UGV] = TEAM1_BACKGROUND
        static_map[new_map==TEAM1_UAV] = TEAM1_BACKGROUND
        static_map[new_map==TEAM2_UGV] = TEAM2_BACKGROUND
        static_map[new_map==TEAM2_UAV] = TEAM2_BACKGROUND
        static_map[new_map==TEAM3_UGV] = TEAM1_BACKGROUND # subject to change
        
        # build 3D new_map
        l, b = new_map.shape
        nd_map = np.zeros([l, b,NUM_CHANNEL], dtype = int)
        for elem in channel.keys():
            ch = channel[elem]
            const = repr_const[elem]
            if elem in new_map:
                nd_map[new_map==elem,ch] = const
        
        # location of agents
        agent_locs = {}
        agent_locs[TEAM1_UGV] = np.argwhere(new_map==TEAM1_UGV)
        agent_locs[TEAM1_UAV] = np.argwhere(new_map==TEAM1_UAV)
        agent_locs[TEAM2_UGV] = np.argwhere(new_map==TEAM2_UGV)
        agent_locs[TEAM2_UAV] = np.argwhere(new_map==TEAM2_UAV)
        
        nd_map[agent_locs[TEAM1_UGV], CHANNEL[TEAM1_BACKGROUND]] = REPRESENT[TEAM1_BACKGROUND]
        nd_map[agent_locs[TEAM2_UGV], CHANNEL[TEAM2_BACKGROUND]] = REPRESENT[TEAM2_BACKGROUND]
        nd_map[agent_locs[TEAM1_UAV], CHANNEL[TEAM1_BACKGROUND]] = REPRESENT[TEAM1_BACKGROUND]
        nd_map[agent_locs[TEAM2_UAV], CHANNEL[TEAM2_BACKGROUND]] = REPRESENT[TEAM2_BACKGROUND]
        
        return nd_map, static_map, obj_arr, agent_locs

    @staticmethod
    def populate_map(new_map, code_where, code_what, channel, number=1):
        """
        Function
            Adds "code_what" to a random location of "code_where" at "new_map"

        Parameters
        ----------
        new_map     : 2d numpy array
            Map of the environment
        code_where  : list
            List of coordinate to put element 
        code_what   : int
            Value assigned to the random location of the map
        number      : int
            Number of element to place
        """
        if number == 0:
            return

        args = np.array(code_where[:number])
        del code_where[:number]

        new_map[args[:,0], args[:,1], channel] = code_what

        return args.tolist()
