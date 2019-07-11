### Get_Observation  
>def get_obs(self, env, com_ground=False, com_air=False, distance=None, freq=1.0, *args):

The method returns a 39X39 matrix (map_size*2-1) centered around individual agents. 
If communication is allowed between agents, 
observation for that agent is edited to include vision from other agents.

#### Parameters:
**com_ground** and **com_air** (boolean): toggle communication between ground/air units.  

**distance** (int): the maximum distance between units for which communication is allowed  

**freq** (0<int<1.0): the probability that communication is received

**obstacle_blocking** (boolean): the agent will not see past 
obstaces in horizontal and vertical directions 

---