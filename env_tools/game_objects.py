from settings import game_settings, rewards
import numpy as np

# manhattan distance between two positions
def dist(pos1, pos2):
    return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])

AVOID_BOMB_WEIGHT = 5
CURIOSITY_REWARD = 0.75
# maybe we should add true curiosity reward, like if the agent has never been in a certain state before



class Agent(object):
    def __init__(self, id, initial_pos):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.id = id
        self.bombs_left = 3
        self.alive = True
        self.events = []
        self.score = 0

        self.past_positions = [initial_pos]
        self.past_positions_set = set()


    def get_pos(self):
        return (self.x, self.y)

    def update_past_positions(self):
        pos = (self.x, self.y)
        self.past_positions.append(pos)
        self.past_positions_set.add(pos)

    def get_no_repeat_reward(self, gamma = CURIOSITY_REWARD):
        pos = (self.x, self.y)
        distances = [dist(pos, self.past_positions[~i]) * (gamma ** i) for i in range(len(self.past_positions))]
        return sum(distances)
    
    def get_curiosity_reward(self):
        pos = (self.x, self.y)
        if pos in self.past_positions_set:
            return rewards.new_position
        else:
            return rewards.repeat_position

    def get_bomb_avoidance_reward(self, bombs): # this can be tuned to prioritize soon to explode bombs more
        pos = (self.x, self.y)

        if not bombs:
            return 0

        # weight nearby bombs more, add e = 0.1 to denominator to avoid division by 0
        # bomb_distances = [1/(dist(pos, bomb.get_pos()) + 0.1) for bomb in bombs]
        bomb_distances = np.array([dist(pos, bomb.get_pos()) for bomb in bombs])
        bomb_distances = np.where(bomb_distances > 1, np.sqrt(bomb_distances-1), bomb_distances-1)
        return np.sum(bomb_distances) * AVOID_BOMB_WEIGHT

    def update_score(self, points):
        self.score = self.score+points

    def make_bomb(self):
        self.bombs_left -= 1
        return Bomb((self.x, self.y), self, game_settings.bomb_timer+1, game_settings.bomb_power)


class Item(object):
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]


class Bomb(Item):
    def __init__(self, pos, owner, timer, power):
        super(Bomb, self).__init__(pos)
        self.owner = owner
        self.timer = timer
        self.power = power
        self.active = True

    def get_pos(self):
        return (self.x, self.y)
    

    def get_state(self):
        # return ((self.x, self.y), self.timer, self.power, self.active, self.owner.name)
        return (self.x, self.y, self.timer)
    # arena np array, if is -1 hard

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x, y)]

        for i in range(1, self.power+1):
            if arena[x+i, y] == -1: break
            blast_coords.append((x+i, y))
        for i in range(1, self.power+1):
            if arena[x-i, y] == -1: break
            blast_coords.append((x-i, y))
        for i in range(1, self.power+1):
            if arena[x, y+i] == -1: break
            blast_coords.append((x, y+i))
        for i in range(1, self.power+1):
            if arena[x, y-i] == -1: break
            blast_coords.append((x, y-i))

        return blast_coords


class Coin(Item):
    def __init__(self, pos):
        super(Coin, self).__init__(pos)
        self.collectable = True
        self.collected = False

    def get_state(self):
        return (self.x, self.y)


class Explosion(object):
    def __init__(self, blast_coords, owner, explosion_timer=game_settings.explosion_timer):
        self.blast_coords = blast_coords
        self.owner = owner
        self.timer = explosion_timer
        self.active = True



class Log(object):
    def info(self, message):
        pass
        # print("INFO: "+str(message))

    def debug(self, message):
        pass
        # print("DEBUG: "+str(message))