import time
import gym
import gym_cap
import gym_cap.heuristic as policy

import numpy as np

class CustomPolicy(policy.Policy):
    def __init__(self): 
        super().__init__()

    def gen_action(self, agent_list, observation):
        # This is the method that is called at each step.
        # The method could be called manually as well.
        action_out = []
        flag_loc = self.get_flag_loc(0) # pass 0 to find red's flag
        if flag_loc is None:
            return np.random.randint(5, size=len(agent_list))
        else:
            for agent in agent_list:
                position = agent.get_loc()
                action = self.move_toward(position, flag_loc)
                if self.can_move(position, action):
                    action_out.append(action)
                else:
                    action_out.append(np.random.randint(0,5))
            return action_out
blue_policy = CustomPolicy()

# Initialize the environment
env = gym.make("cap-v0")

# Reset the environment and select the policies for each of the team
observation = env.reset(
        map_size=20,
        config_path='base_settings.ini',
        policy_blue=blue_policy,
        policy_red=policy.Random()
    )

num_match = 1
render = True

rscore = []
start_time = time.time()
for n in range(num_match):
    done = False
    rewards = []
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

        # Take a step and receive feedback
        # Action does not have to be explicitly given if policy is passed during reset.
        # Any provided actions override policy actions.
        actions = blue_policy.gen_action(env.get_team_blue, observation)
        observation, reward, done, info = env.step(actions)
        rewards.append(reward)

        # Render and sleep (not needed for score analysis)
        if render:
            env.render()
            time.sleep(.05)

    # Reset the game state
    env.reset()

    # Statistics
    rscore.append(sum(rewards))
    duration = time.time() - start_time 
    print("Time: %.2f s, Score: %.2f" % (duration, rscore[-1]))

print("Average Time: %.2f s, Average Score: %.2f"
        % (duration/num_match, sum(rscore)/num_match))
env.close()

