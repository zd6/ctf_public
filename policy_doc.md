### Policy
Each policy consists of intiate() and gen_action(). When a policy is used, initiate() is called. 
Following, Gen_action() will returns a list of actions, which the environment uses to carry out the game.   

#### Making Custom Policies
Custom policies are children of the Policy class described in policy.py 
Within policy.py, common methods are defined which can be used in custom policies.

### AStar
AStar is a search based heuristic which seeks to find path to the enemy flag. 
This policy uses A* algorithm along with Manhattan distance to trace the path of the enemy flag.  
The route the agent takes is calculated in intitiate() and is translated into an array of actions in gen_action().
This method works under full observation setting

---

### Fighter
Fighter policy seeks to guard the home flag as well as capture the enemy agent in their territory.
The agent type is set in intitiate(). 
An 'aggr' agent seeks out to capture the enemy agent,
while a 'def' agent seeks to guard the home flag and capture the enemy wandering close to the home flag.
The 'aggr' agents use A* algorithm along with Manhattan distance to trace the path to the enemy agents.
The 'def' agents guard the home flag within a guard radius and hunt down the enemy when it comes into the down radius.
These are then translated into an array of actions in gen_action().
This method works under full observation setting

---
