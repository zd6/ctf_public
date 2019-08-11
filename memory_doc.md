### Individual Memory 
>def update_memory(self, env)

This method returns a 20x20 matrix (map_size x map_size). It returns the vision of an individual agent. It aids in exploring the env.
It uses get_obs method and masks it to get the vision/map for an agent. 
The default parameters are explained below.

#### Parameters:
**env** (env): environment

**get_obs** (method): The method returns a 39X39 matrix (map_size*2-1) centered around an individual agent.
If communication is allowed between agents, 
observation for that agent is edited to include vision from other agents 

![Individual Memory](https://github.com/raide-project/ctf_public/blob/gh-pages_memory/ind_memory.png)

---

### Global Memory 
>def _update_global_memory(self, env)

This method returns a 20x20 matrix (map_size x map_size). It returns the vision of the team. 
It aids agents communicate within the team and explore the env.
It implements update_memory method for all individual agents and creates the combined vision/map for a team. 
The default parameters are explained below.

#### Parameters:
**env** (env): environment

**get_obs** (method): The method returns a 39X39 matrix (map_size*2-1) centered around an individual agent.
If communication is allowed between agents, 
observation for that agent is edited to include vision from other agents

![Global Memory](https://github.com/raide-project/ctf_public/blob/gh-pages_memory/global_memory.png)

---
