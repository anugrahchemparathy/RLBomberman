import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import registry, register

import sys
import numpy as np
from contextlib import closing
from io import StringIO
"""
game_settings = named tuple which contains variety of settings like game size and rewards
event_ids = named tuple which contains all possible events

You can access with
print(game_settings.rows)
print(event_ids.MOVED_UP)
"""
from settings import game_settings, event_ids, rewards
from game_objects import *
from arenas import *

NO_REPEAT_BASELINE = 1.5

# ACTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BOMB = 4
WAIT = 5

enum_2_action = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'BOMB',
    5: 'WAIT',
}

# STATES
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
RENDER_CORNERS = False
RENDER_HISTORY = True

ITEM_2_EMOJI = {
    FREE: "👣",
    WALL: "🟦",
    CRATE: "📦",
    BOMB: "🧨",
    EXPLOSION: "💥",
    PLAYER: "🤖",
    COIN: "🔸",
}


"""
Functions needed to train agents:
env.reset() - resets env, returns initial state
env.step(action) - returns next_state, reward, done, log
env._get_obs() - get current state
"""

use_curiosity_subreward = True
use_no_repeat_subreward = True
use_bomb_avoidance_subreward = True


class BombermanEnv(gym.Env):
    """
    fields:
    self.screen_height - height of the game screen
    self.screen_width - width of the game screen
    self.action_space - action space of the game (gym spaces)
    self.observation_space - observation space of the game (gym spaces)
    self.seed() - seed the game
    self.logger - logger for the game from


    methods:
    self.reset() - resets the game, returns initial state
    self.step(action) - returns next_state, reward, done, log
    self._get_obs() - get current state
    self.render() - render the game screen using helper functions

    
    """
    def __init__(self, bombermanrlSettings=game_settings):
        self.screen_height = bombermanrlSettings.rows
        self.screen_width = bombermanrlSettings.cols

        # see action space above
        self.action_space = spaces.Discrete(6)

        # NEED TO CHANGE THIS
        self.observation_space = spaces.Box(
            low=-3, high=3, shape=(2, 17, 17), dtype=np.int8
            )
        self.seed()
        self.logger = Log()

        # Start the first game
        self.reset()
        self.env = self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # gets arena randomly selected from arenas.py
    def generate_arena(self):
        self.arena = get_arena()
        self.coins = []
        agent_start_position = random.choice(np.argwhere(self.arena == 0))

        agent_x, agent_y = agent_start_position
        
        
        for x in range(game_settings.cols):
            for y in range(game_settings.rows):
                if self.arena[x,y] == 0 and (x != agent_x or y != agent_y):
                    self.coins.append(Coin((x,y))) # Adding coins everywhere

        return agent_start_position
        
        
    """
    given the current player and action, update the player's location
    and returns reward based on coins collected
    """
    def update_player_loc(self, action):
        reward = 0
        # print('action: ', action, self.tile_is_free(self.player.x, self.player.y + 1))
        scaled_bomb_distances = self.player.get_bomb_avoidance_reward(self.bombs)

        if action == LEFT and self.tile_is_free(self.player.x, self.player.y - 1):
            self.player.y -= 1
            self.player.events.append(event_ids.MOVED_LEFT)
            reward += rewards.valid_move

            self.player.update_past_positions()
            if use_bomb_avoidance_subreward:
                bomb_avoidance = self.player.get_bomb_avoidance_reward(self.bombs) - scaled_bomb_distances
                reward +=  bomb_avoidance

            if use_curiosity_subreward:
                curiosity = self.player.get_curiosity_reward()
                reward += curiosity

        elif action == RIGHT and self.tile_is_free(self.player.x, self.player.y + 1):
            self.player.y += 1
            self.player.events.append(event_ids.MOVED_RIGHT)
            reward += rewards.valid_move

            self.player.update_past_positions()
            if use_bomb_avoidance_subreward:
                bomb_avoidance = self.player.get_bomb_avoidance_reward(self.bombs) - scaled_bomb_distances
                reward +=  bomb_avoidance

            if use_curiosity_subreward:
                curiosity = self.player.get_curiosity_reward()
                reward += curiosity

        elif action == UP and self.tile_is_free(self.player.x - 1, self.player.y):
            self.player.x -= 1
            self.player.events.append(event_ids.MOVED_UP)
            reward += rewards.valid_move

            self.player.update_past_positions()
            if use_bomb_avoidance_subreward:
                bomb_avoidance = self.player.get_bomb_avoidance_reward(self.bombs) - scaled_bomb_distances
                reward +=  bomb_avoidance

            if use_curiosity_subreward:
                curiosity = self.player.get_curiosity_reward()
                reward += curiosity

        elif action == DOWN and self.tile_is_free(self.player.x + 1, self.player.y):
            self.player.x += 1
            self.player.events.append(event_ids.MOVED_DOWN)
            reward += rewards.valid_move

            self.player.update_past_positions()
            if use_bomb_avoidance_subreward:
                bomb_avoidance = self.player.get_bomb_avoidance_reward(self.bombs) - scaled_bomb_distances
                reward +=  bomb_avoidance

            if use_curiosity_subreward:
                curiosity = self.player.get_curiosity_reward()
                reward += curiosity
        
        elif action == BOMB and self.player.bombs_left > 0:
            self.logger.info(f'player <{self.player.id}> drops bomb at {(self.player.x, self.player.y)}')
            self.bombs.append(self.player.make_bomb())
            self.player.events.append(event_ids.BOMB_DROPPED)
            reward += rewards.place_bomb
        
        elif action == WAIT:
            self.player.events.append(event_ids.WAITED)
            reward += rewards.wait
        
        else:
            reward += rewards.invalid_action
        

        if use_no_repeat_subreward:
            no_repeat_reward = self.player.get_no_repeat_reward()
            reward += no_repeat_reward - NO_REPEAT_BASELINE

        # collect coins
        num_coins = 0
        for coin in self.coins:
            if coin.collectable:
                a = self.player
                if a.x == coin.x and a.y == coin.y:
                    coin.collectable = False
                    coin.collected = True
                    self.logger.info(f'Agent <{a.id}> picked up coin at {(a.x, a.y)} and receives 1 point')
                    a.update_score(game_settings.reward_coin)
                    a.events.append(event_ids.COIN_COLLECTED)
                    reward += rewards.collect_coin
                    num_coins = 1

        return reward, num_coins

    """
    explodes bomb and modifies state of all crates in blasted area. creates a new explosion.
    """
    def explode_bomb(self, bomb):
        self.logger.info(f'Agent <{bomb.owner.id}>`s bomb at {(bomb.x, bomb.y)} explodes')
        blast_coords = bomb.get_blast_coords(self.arena)
        # Clear crates
        for (x, y) in blast_coords:
            if self.arena[x, y] == 1:
                self.arena[x, y] = 0

        # Create explosion, no need to reward - agent was rewarded for placing the bomb
        self.explosions.append(Explosion(blast_coords, bomb.owner))
        bomb.active = False
        bomb.owner.bombs_left += 1

    """
    updates explosions (ie timer of explosions) and kills agents that lie in explosion range
    TODO: set reward for agent
    """
    def place_explosions(self):
        detonation = False
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                detonation = True
                a = self.player
                if a.alive:
                    if (a.x, a.y) in explosion.blast_coords:
                        a.alive = False
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(
                                f'Agent <{a.id}> blown up by own bomb')
                            a.events.append(event_ids.KILLED_SELF)
                        else:
                            self.logger.info(f'Agent <{a.id}> blown up by agent <{explosion.owner.id}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(game_settings.reward_kill)
                            explosion.owner.events.append(event_ids.KILLED_OPPONENT)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            explosion.timer -= 1
          
        self.explosions = [e for e in self.explosions if e.active]

    def all_players_dead(self):
        return not self.player.alive

    
    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + [self.player]:  # TODO Players...
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free
    
    def check_if_all_coins_collected(self):
        return len([c for c in self.coins if c.collected]) == len(self.coins)

    def all_players_dead(self):
        return not self.player.alive
    

    """
    ==========================================================
    CORE ENV FUNCTIONS
    ==========================================================
    """
    
    # resets env, returns initial state
    def reset(self, artificial_start_position=None):
        self.round = 0
        agent_start_position = self.generate_arena()
        # print('agent_start_position: ', agent_start_position)
        if artificial_start_position is not None:
            self.player = Agent(1, artificial_start_position)
        else:
            self.player = Agent(1, agent_start_position)

        self.bombs = []
        self.explosions = []

        self.update_player_loc(BOMB)

        return self._get_obs()
    
    # returns next_state, reward, done, log
    def step(self, action):
        reward = 0
        
        # player locations
        r, c = self.update_player_loc(action)
        reward += r
        
        # update bombs
        for bomb in self.bombs:
            bomb.timer -= 1
            if bomb.timer == 0:
                self.explode_bomb(bomb)
        self.bombs = [b for b in self.bombs if b.active]

        self.place_explosions()

        self.round = self.round+1
        success = self.check_if_all_coins_collected()
        done = self.check_if_all_coins_collected() or self.all_players_dead() or self.round > 500

        if self.round > 500:
            reward += rewards.game_timeout
        if not self.player.alive:
            reward += rewards.agent_died
        
        #print('did action', enum_2_action[action], 'got reward', reward, 'done', done, 'round', self.round)
        if self.all_players_dead():
            print('AGENT DIED, got coins', str(len([c for c in self.coins if c.collected])))
        if done:
            self.render()

        return (self._get_obs(), reward, done, {'coin': c, 'success': success})

    # get current state
    def _get_obs(self):
        agent_view = np.copy(self.arena)
        bomb_plane = np.copy(self.arena)
        
        # add coins
        for coin in self.coins:
            if coin.collectable:
                agent_view[coin.x, coin.y] = COIN
        
        # add bombs
        if game_settings.know_bomb_timer:
            for bomb in self.bombs:
                bomb_plane[bomb.x, bomb.y] = bomb.timer
        else:
            for bomb in self.bombs:
                bomb_plane[bomb.x, bomb.y] = BOMB
        for explosion in self.explosions:
            for e in explosion.blast_coords:
                bomb_plane[e[0], e[1]] = EXPLOSION
        
        # TODO add players
        agent_view[self.player.x, self.player.y] = PLAYER

        # stack rendered map and bomb plane along dimension 0
        rendered_map = np.stack((agent_view, bomb_plane), axis=0)

        return rendered_map


    """
    ==========================================================
    RENDER FUNCTIONS
    ==========================================================
    """

    def render(self, mode=None):
        rendered_map = np.copy(self.arena)
        
        # add coins
        for coin in self.coins:
            if coin.collectable:
                rendered_map[coin.x, coin.y] = COIN
        
        # add bombs
        for bomb in self.bombs:
            rendered_map[bomb.x, bomb.y] = BOMB
        for explosion in self.explosions:
            for e in explosion.blast_coords:
                rendered_map[e[0], e[1]] = EXPLOSION
        
        # add players
        rendered_map[self.player.x, self.player.y] = PLAYER

        out_string = ''
        for row in rendered_map:
            out_string += ''.join([ITEM_2_EMOJI[r] for r in row]) + '\n'
        
        print(out_string)
        return out_string


module_name = __name__
env_name = 'Bomberman-v1'
if env_name in registry:
    del registry[env_name]
register(
    id=env_name,
    entry_point=f'{module_name}:BombermanEnv',
)

if __name__ == "__main__":

    benv = BombermanEnv(game_settings)
    benv.render()
    benv.step(DOWN)
    benv.render()
    benv.step(DOWN)
    benv.render()
    benv.step(RIGHT)
    benv.render()
    benv.step(RIGHT)
    benv.render()
    benv.step(BOMB)
    benv.render()
    benv.step(LEFT)
    benv.render()
    benv.step(WAIT)
    benv.render()