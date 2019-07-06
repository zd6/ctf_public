import unittest
import gym
import gym_cap
import numpy as np
import random
import time

import policy
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
            s, r, d, i = env.step()
            if d: break

    def testStepWithBlueActionSpecified(self):
        test_maxstep = 150
        env = gym.make(
            ENV_NAME,
            policy_red=policy.random.Random(),
        )
        for step in range(test_maxstep):
            action = env.action_space.sample()
            s, r, d, i = env.step(action)
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
            s, r, d, i = env.step(action)
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
            s, r, d, i = env.step(action)
            if d: break


class TestAgentTeamMemory(unittest.TestCase):
    pass


class TestAgentIndivMemory(unittest.TestCase):
    pass


class TestAgentGetObs(unittest.TestCase):

    @repeat(10)
    def testFrequency(self):
        " Communication frequency test"
        env = gym.make(ENV_NAME)
        env.COM_FREQUENCY = random.random()  # Try random frequency
        env.reset()
        for entity in env._team_blue + env._team_red:
            entity.get_obs(env)

    @repeat(10)
    def testComAir(self):
        " Communication between ground and air test"
        env = gym.make(ENV_NAME)
        env.NUM_UAV = 2
        env.COM_AIR = True
        env.reset(custom_board = 'test_maps/comms_board1.txt')
        for entity in env._team_blue+env._team_red:
            loc = 19, 19
            if entity.get_loc() == loc:
                obs = entity.get_obs(env)
                a = np.loadtxt('test_maps/solution_maps/solution2.txt')
                for ix, iy in np.ndindex(a.shape):
                    assert(a[ix, iy] == obs[ix, iy])

    @repeat(10)
    def testComGround(self):
        " Communication between ground and ground test"
        env = gym.make(ENV_NAME)
        env.NUM_UAV = 2
        env.COM_GROUND = True
        env.reset(custom_board='test_maps/board1.txt')
        for entity in env._team_blue+env._team_red:
            loc = 18, 18
            if entity.get_loc() == loc:
                obs = entity.get_obs(env)
                a = np.loadtxt('test_maps/solution_maps/solution1.txt')
                for ix, iy in np.ndindex(a.shape):
                    assert(a[ix, iy] == obs[ix, iy])

if __name__ == '__main__':
    unittest.main()
