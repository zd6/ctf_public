import unittest
import gym
import gym_cap

import numpy as np
import random
import time

import policy

ENV_NAME = 'cap-v0'

def repeat(times):
    " Wrapper for multiple test"
    def repeatwrapper(f):
        def callwrapper(*args):
            for i in range(times):
                f(*args)
        return callwrapper
    return repeatwrapper

class TestBuild(unittest.TestCase):
    """
    Test creating the environment under gym registry.
    Test building map and reseting the game with different configurations/settings.
    """

    def testBuild(self):
        " Test if environment build for any random seeds"
        test_epoch = 50 
        for epoch in range(test_epoch):
            env = gym.make(ENV_NAME)

    @repeat(10)
    def testMapSize(self):
        " Test if the environment can handle map size. (10~20)"
        test_epoch = 32 
        test_maxstep = 100
        for size in range(10, 20):
            for epoch in range(test_epoch):
                env = gym.make(ENV_NAME, map_size=size)

    def testCustomBoardRun(self):
        test_epoch = 4 
        env = gym.make(ENV_NAME, custom_board='test_maps/board1.txt')
        for epoch in range(test_epoch):
            env.reset(custom_board='test_maps/board1.txt')

    def testCustomBoardImport(self):
        test_maxstep = 100
        env = gym.make(ENV_NAME, policy_red=policy.random.Random(), custom_board='test_maps/board1.txt')
        render_state = env.get_full_state
        test_render_state = np.array([
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7]
            ])
        np.testing.assert_array_equal(render_state, test_render_state)

class TestRun(unittest.TestCase):

    def testStepWithPolicyProvided(self):
        test_maxstep = 150
        env = gym.make(
                ENV_NAME,
                policy_red=policy.random.Random(),
                policy_blue=policy.random.Random()
            )
        for step in range(test_maxstep):
            s,r,d,i = env.step()
            if d: break

    def testStepWithBlueActionSpecified(self):
        test_maxstep = 150
        env = gym.make(
                ENV_NAME,
                policy_red=policy.random.Random(),
            )
        for step in range(test_maxstep):
            action = env.action_space.sample()
            s,r,d,i = env.step(action)
            if d: break

class TestInteraction(unittest.TestCase):
    
    def testDeterministicInteractionRun(self):
        self.STOCH_ATTACK = False
        test_maxstep = 150 
        env = gym.make(
                ENV_NAME,
                policy_red=policy.roomba.Roomba(),
                policy_blue=policy.roomba.Roomba(),
            )
        for step in range(test_maxstep):
            action = env.action_space.sample()
            s,r,d,i = env.step(action)
            if d: break

    def testStochasticInteractionRun(self):
        self.STOCH_ATTACK = True
        test_maxstep = 150 
        env = gym.make(
                ENV_NAME,
                policy_red=policy.roomba.Roomba(),
                policy_blue=policy.roomba.Roomba(),
            )
        for step in range(test_maxstep):
            action = env.action_space.sample()
            s,r,d,i = env.step(action)
            if d: break

class TestAgentTeamMemory(unittest.TestCase):
    
    def testTeamMemory(self):
        "Testing Team memory"
        test_epoch = 1
        env = gym.make(ENV_NAME)
        for epoch in range(test_epoch):
            env.reset(map_size=20,
                      policy_blue=policy.Roomba(),
                      policy_red=policy.Random(),
                      custom_board='test_maps/board1.txt')
            env.TEAM_MEMORY = 'fog'
            env.step()
            b_map, r_map = env.blue_memory, env.red_memory
            b_sol, r_sol = np.loadtxt('test_maps/memory_sol/sol_2.txt'), np.loadtxt('test_maps/memory_sol/sol_3.txt')
            for idx, idy in np.ndindex(b_sol.shape):
                assert(b_sol[idx, idy] == b_map[idx, idy])
                assert(r_sol[idx, idy] == r_map[idx, idy])

class TestAgentIndivMemory(unittest.TestCase):
  
    def testIndivMemory(self):
        "Testing Individual memory"
        test_epoch = 1
        env = gym.make(ENV_NAME)
        for epoch in range(test_epoch):        
            env.reset(map_size=20,
                      policy_blue=policy.Roomba(),
                      policy_red=policy.Random(),
                      custom_board='test_maps/board1.txt')
            for agent in env._team_blue + env._team_red:
                agent.memory_mode = 'fog'
            env.step()
            for agent in env._team_blue + env._team_red:
                t_map = agent.memory
                sol = np.loadtxt('test_maps/memory_sol/sol_1.txt')
                for idx, idy in np.ndindex(t_map.shape):
                    assert(sol[idx, idy] == t_map[idx, idy])

class TestAgentGetObs(unittest.TestCase):

    @repeat(10)
    def testFrequency(self):
        " Communication frequency test"
        env = gym.make(ENV_NAME)
        env.COM_FREQUENCY = random.random()# Try random frequency
        env.reset()
        for entity in env.team_blue+env.team_red:
            entity.get_obs(env)
                
    @repeat(10)
    def testComAir(self):
        " Communication between ground and air test"
        env = gym.make(ENV_NAME)
        env.NUM_UAV = 2
        env.COM_AIR = True
        env.reset()
        for entity in env.team_blue+env.team_red:
            entity.get_obs(env)

    @repeat(10)
    def testComGround(self):
        " Communication between ground and air test"
        env = gym.make(ENV_NAME)
        env.NUM_UAV = 2
        env.COM_GROUND = True
        env.reset()
        for entity in env.team_blue+env.team_red:
            entity.get_obs(env)

if __name__ == '__main__':
    unittest.main()
