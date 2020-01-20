import time
import os
from os.path import isfile, join
import sys
import gym
import gym_cap
import numpy as np
import random
from tqdm import tqdm


# the modules that you can use to generate the policy.
import policy
import policy_wip

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

rscore = []

# reset the environment and select the policies for each of the team
#env.RED_STEP = 1
#env.RED_DELAY = 1

map_path = 'test_maps/fair_uav'
map_paths = [join(map_path,f) for f in os.listdir(map_path) if isfile(join(map_path, f))]

eprun = 1000

try:
    #for _ in tqdm(range(eprun)):
    blue_policy = policy_wip.UAV()
    for _ in range(eprun):
    #while True:
        observation = env.reset(
                map_size=20,
                config_path='uav_settings.ini',
                custom_board=random.choice(map_paths),
                #policy_blue=blue_policy,
                policy_blue=policy.Roomba(),
                policy_red=policy.Roomba(),
            )
        t = 0
        done = False
        epreward = 0
        while not done:
            #you are free to select a random action
            # or generate an action using the policy
            # or select an action manually
            # and the apply the selected action to blue team
            # or use the policy selected and provided in env.reset
            #action = env.action_space.sample()  # choose random action
            #action = policy_blue.gen_action(env.team1,observation,map_only=env.team_home)
            #action = [0, 0, 0, 0]
            #observation, reward, done, info = env.step(action)
    
            observation, reward, done, info = env.step()  # feedback from environment
            epreward += reward
    
            # render and sleep are not needed for score analysis
            #env.render()
            #for i in range(1):
            #    print(env.get_obs_red[::-1,:,i])
            #input('')
            #time.sleep(.05)

            #print(reward, epreward)
            #for i in range(10):
            #    print(env.get_obs_blue[:,:,i])
            #for i in range(10):
            #    print(env.get_obs_red[:,:,i])
            #input('')
    
            t += 1
            if t == 150:
                break
    
        rscore.append(epreward)
        print("Time: %.2f s, score: %.2f" %
            ((time.time() - start_time),epreward))
        print(env.blue_win)
    print("Time: %.2f s, score: %.2f" %
        ((time.time() - start_time),np.asarray(rscore).mean()))

except KeyboardInterrupt:
    env.close()
    del gym.envs.registry.env_specs['cap-v0']

    print("CtF environment Closed")

