### Get_Observation  
>def get_obs(self, env, com_ground=False, com_air=False, distance=None, freq=1.0, *args):

The method returns a 39X39 matrix (map_size*2-1) centered around an individual agent. 
If communication is allowed between agents, 
observation for that agent is edited to include vision from other agents. Default settings are shown above and explained below.

#### Parameters:
**com_ground** and **com_air** (boolean): toggle communication between ground/air units, agent's vision matrix
will include information from other agents.  

**distance** (int): the maximum distance between units for which communication is allowed  

**freq** (0<int<1.0): the probability that communication is received

**obstacle_blocking** (boolean): the agent will not see past 
obstaces in horizontal and vertical directions 

---
###Policy
Each policy consists of intiate() and gen_action(). When a policy is used, initiate() is called. 
Following, Gen_action() will returns a list of actions, which the environment uses to carry out the game.   
####Making Custom Policies
Custom policies are children of the Policy class described in policy.py 
Within policy.py, common methods are defined which can be used in custom policies. 

#####Spiral Policy
Spiral is a search related policy that attempts to cover all unvisited areas on the map. The agent will move to the nearest (euclidean 
distance) unsearched area, resulting in a spiral shaped pattern. The locations the agent 
moves to is calculated in intitiate() and translated into an array of actions in 
gen_action()