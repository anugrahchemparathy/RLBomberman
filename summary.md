Here is a summary of the contents of every file

* env_tools - all functional code used to create the gym env and agents etc.
    - arenas.py - Two simple arenas (stored as np arrays) that can be modified and opened in another file
    - game_gui.py - Simple code to visualize rollouts
    - expert_agent.py - Deterministic exploration using BFS for use in behavior cloning
    - game_objects.py - Agent and Obstacle classes for use in game env
    - settings.py - Very quick solution to creating fixed game settings.
    - new_env.py - A custom game env for bomberman - loosely based off online implementations
    - run_agent.py - Simple script to run training
    - models.py - Simple architectures for the policy networks and for DQN
    - Other - Tools for Behavior cloning and partial DQN port