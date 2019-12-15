import time
import gym
import gym_cap
import numpy as np


# the modules that you can use to generate the policy.
import policy

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

done = False
rscore = []

# reset the environment and select the policies for each of the team
observation = env.reset(map_size=20,
                        policy_blue=policy.Defense(),
                        policy_red=policy.Random())

try:
    while True:
        t = 0
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
            locs = {
                "T1/R1":[1,1],
                "T1/R2":[1,2],
                "T1/R3":[1,3],
                "T1/R4":[1,4],
                "T2/R1":[10,1],
                "T2/R2":[10,2],
                "T2/R3":[10,3],
                "T2/R4":[10,4],}
            observation, reward, done, info = env.update_step(locs)  # feedback from environment
            env.render()
            input()
            locs = {
                "T1/R1":[1,1],
                "T1/R2":[1,3],
                "T1/R3":[1,5],
                "T1/R4":[1,7],
                "T2/R1":[10,1],
                "T2/R2":[10,3],
                "T2/R3":[10,5],
                "T2/R4":[10,7],}
            observation, reward, done, info = env.update_step(locs)  # feedback from environment

            # render and sleep are not needed for score analysis
            env.render()
            input()

            t += 1
            if t == 100:
                break

        env.reset()
        done = False
        rscore.append(reward)
        print("Time: %.2f s, score: %.2f" %
            ((time.time() - start_time),np.asarray(rscore).mean()))

except KeyboardInterrupt:
    env.close()
    del gym.envs.registry.env_specs['cap-v0']

    print("CtF environment Closed")
