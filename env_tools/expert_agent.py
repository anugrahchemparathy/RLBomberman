from new_env import BombermanEnv
import sys
env = BombermanEnv()

# STATES
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3

# ACTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BOMB = 4
WAIT = 5

dirs = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0)
}

def find_player(ob):
    playerX, playerY = None, None
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == PLAYER:
                playerX, playerY = i, j
    return playerX, playerY

def bfs_to_coin(ob):
    playerX, playerY = find_player(ob)
    #print("player", playerX, playerY)
    q = [(playerX, playerY, [])]
    ind = 0
    visited = set()

    while ind < len(q):
        curX, curY, path = q[ind]

        if ob[0][curX][curY] == COIN:
            return path
        
        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN:
                    q.append((curX+dirX, curY+dirY, path+[key]))
    
        ind+=1

    return []

power = 1
WAIT_TIME = 6
def get_blast_coords(x, y, arena):
        blast_coords = [(x, y)]

        for i in range(1, power+1):
            if arena[x+i, y] == -1: break
            blast_coords.append((x+i, y))
        for i in range(1, power+1):
            if arena[x-i, y] == -1: break
            blast_coords.append((x-i, y))
        for i in range(1, power+1):
            if arena[x, y+i] == -1: break
            blast_coords.append((x, y+i))
        for i in range(1, power+1):
            if arena[x, y-i] == -1: break
            blast_coords.append((x, y-i))

        return blast_coords

def place_bomb_and_dodge(ob, playerX, playerY, start=False):
    actions = [BOMB]
    if start:
        actions = []
    blast = set(get_blast_coords(playerX, playerY, ob[0]))
    #print("Bomb", playerX, playerY)
    #print(blast)

    q = [(playerX, playerY, [])]
    dodge_path = None
    visited = set()
    ind = 0
    while ind < len(q):
        curX, curY, path = q[ind]

        if not ((curX, curY) in blast):
            #print("Dodged")
            dodge_path = path
            break
        
        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN or val == PLAYER:
                    q.append((curX+dirX, curY+dirY, path+[key]))
        ind+=1
    
    # print("DODGE PATH", [enum_2_action[x] for x in dodge_path])
    if dodge_path is None:
        return actions
    #     raise "Couldnt dodge"

    og_len = len(dodge_path)
    while len(dodge_path) <= WAIT_TIME:
        dodge_path.append(WAIT)
    if not start:
        for i in reversed(range(og_len)):
            if dodge_path[i] == LEFT:
                dodge_path.append(RIGHT)
            elif dodge_path[i] == RIGHT:
                dodge_path.append(LEFT)
            elif dodge_path[i] == UP:
                dodge_path.append(DOWN)
            elif dodge_path[i] == DOWN:
                dodge_path.append(UP)
        
    actions.extend(dodge_path)

    return actions


def bfs_through_crate(ob):
    playerX, playerY = None, None
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == PLAYER:
                playerX, playerY = i, j
    ogX, ogY = playerX, playerY
    # print("player", playerX, playerY)
    q = [(playerX, playerY, [])]
    ind = 0
    visited = set()

    path_to_coin = None
    while ind < len(q):
        curX, curY, path = q[ind]

        if ob[0][curX][curY] == COIN:
            path_to_coin = path
            break

        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN or val == CRATE:
                    q.append((curX+dirX, curY+dirY, path+[key]))
        ind+=1
        
    actions = []
    if path_to_coin is None:
        return []
    
    for act in path_to_coin:
        playerX += dirs[act][0]
        playerY += dirs[act][1]
        if ob[0][playerX][playerY] == CRATE:
            expect_ob = ob[:]
            expect_ob[0][ogX, ogY] = FREE
            expect_ob[0][playerX - dirs[act][0], playerY - dirs[act][1]] = PLAYER
            actions.extend(place_bomb_and_dodge(expect_ob, playerX - dirs[act][0], playerY - dirs[act][1]))
            break
        else:
            actions.append(act)

    return actions

enum_2_action = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'BOMB',
    5: 'WAIT',
}

import numpy as np
import random
class ExpertAgent():
    def __init__(self) -> None:
        self.q = []

    def get_action(self, ob, sample=True):
        if len(ob.shape) == 4:
            ob = np.squeeze(ob)

        if len(self.q) > 0:
            x = self.q[0]
            self.q = self.q[1:]
            return x, dict()

        x, y = find_player(ob)
        
        if ob[1][x, y] > 1:
            # print("Starting mode")
            acts = place_bomb_and_dodge(ob, x, y, start=True)
        else:
            acts = bfs_to_coin(ob)
            if len(acts)==0:
                acts = bfs_through_crate(ob)
        if len(acts) == 0:
            return random.choice([LEFT, RIGHT, UP, DOWN]), dict()
        self.q = acts[1:]
        return acts[0], dict()
  


def count_coins(ob):
    count = 0
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == COIN:
                count += 1
    
    return count


if __name__ == "__main__":
    ob = env.reset()
    replay = ""
    replay += env.render() + "\n"
    agent = ExpertAgent()
    while True:
        action, _ = agent.get_action(ob)
        next_ob, reward, done, info = env.step(action)
        ob = next_ob
        replay += env.render() + "\n"
        print(enum_2_action[action])
        print(env.render())
        if done:
            with open('replays/replay2.txt', 'w') as f:
                f.write(replay)
            break
