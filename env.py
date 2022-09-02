import math
import os
from pdb import set_trace as TT
import sys
import random

from einops import rearrange, repeat
import gym
import networkx as nx
import numpy as np
import pygame
import os


class TileTypes:
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    ENEMY = 3
    GOLD = 4
    SPAWNER = 5
    MAX = 6  # The number of distinct tile types (increment if adding a new tile)


def gen_random_map(h, w):
    """Generate a random map of the given size."""
    # Generate terrain with some distribution of empty/wall tiles
    map_arr = np.random.choice(2, size=(h-2, w-2), p=[0.8, 0.2])
    # Create a border of wall around the map
    map_arr = np.pad(map_arr, 1, mode="constant", constant_values=TileTypes.WALL)
    # Spawn the player at a random position
    player_pos = np.array([random.randint(1, h-2), random.randint(1, w-2)], dtype=np.uint8)
    map_arr[player_pos[0], player_pos[1]] = TileTypes.PLAYER
    # empty_coords = np.argwhere(map_arr == TileTypes.EMPTY)
    # gold_xy = empty_coords[random.randint(0, len(empty_coords) - 1)]
    # map_arr[tuple(gold_xy)] = TileTypes.GOLD

    return map_arr, player_pos


def discrete_to_onehot(map_disc, n_chan):
    """Convert a discrete map to a onehot-encoded map."""
    return np.eye(n_chan)[map_disc].transpose(2, 0, 1)


class Env(gym.Env):
    h = w = 18
    # Loose upper bound on distance between enemy and player, for using dijkstra map between these two.
    max_enemy_dist = h * w + 1
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    traversable_tile_idxs = [TileTypes.EMPTY]
    tile_size = 40
    max_steps = 1000
    _n_tiles = TileTypes.MAX
    tile_colors = np.array([
        [255, 255, 255],  # Empty
        [0, 0, 0],  # Wall
        [255, 255, 255],  # Player
        [255, 255, 255],  # Enemy
        [255, 255, 255], # Gold
        [255, 255, 255],  # Spawner
    ])
    dummy_map = np.pad(np.zeros((h-2, w-2)), 1, mode='constant', constant_values=1)
    dummy_map[0,0] = dummy_map[0, w-1] = dummy_map[h-1, 0] = dummy_map[h-1, w-1] = 0
    border_coords = np.argwhere(dummy_map == 1)

    def __init__(self, env_cfg):
        self._n_gold = 4
        self._n_gold_to_spawn = self._n_gold
        self._xys_to_enemies = {}
        self._xys_to_spawners = {}
        # self.enemies = []
        self.spawn_lvl = 1
        self.n_enemies_to_spawn = self.spawn_lvl
        # Generate a discrete encoding of the map.
        self.map_disc, self.player_position = gen_random_map(h=self.h, w=self.w)
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)
        self.n_step = 0
        self.player_action_space = gym.spaces.Discrete(4)
        self.action_space = self.player_action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self._n_tiles, self.h, self.w), dtype=np.uint8)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.mouse_down = False
        self.resource_counter = 5
        self.gold_counter = 0
        self.gold_pos = set()

        pass

    def _spawn_gold(self):
        """Spawn gold on a random EMPTY tile."""
        empty_coords = np.argwhere(self.map_disc == TileTypes.EMPTY)
        if len(empty_coords) > 2:
            gold_xy = empty_coords[random.randint(0, len(empty_coords) - 1)]
            self.map_disc[tuple(gold_xy)] = TileTypes.GOLD
            self._update_map_onehot()
            self.gold_pos.add(tuple(gold_xy))

    def step(self, action):
        if np.sum(self.map_onehot[TileTypes.ENEMY]) != len(self._xys_to_enemies):
            TT()
        if len(self._xys_to_enemies) == 0 or self.map_onehot[TileTypes.ENEMY].sum() == 0:
            self._n_tick_no_enemy += 1
        else:
            self._n_tick_no_enemy = 0
        if self._n_tick_no_enemy > 1:
            TT()

        if self._n_gold_to_spawn > 0:
            [self._spawn_gold() for _ in range(self._n_gold_to_spawn)]
            self._n_gold_to_spawn = 0
        done = False
        # give positive reward as long as the agent is alive
        reward = 0.0  # TODO: TBD
        next_pos = self.player_position + self.directions[action]

        # Update the player's position if they are moving onto a valid tile.
        if self.map_disc[tuple(next_pos)] not in [TileTypes.WALL, TileTypes.ENEMY]:
            self.map_disc[tuple(self.player_position)] = TileTypes.EMPTY
            self.player_position = next_pos
            if self.map_disc[tuple(self.player_position)] == TileTypes.GOLD:
                reward += 5
                self._n_gold_to_spawn += 1
                self.gold_pos.remove(tuple(self.player_position))
                self.gold_counter += 1
                # qua
            self.map_disc[tuple(self.player_position)] = TileTypes.PLAYER
            self._update_map_onehot()

        # TODO: Check if player is neighboring an enemy(s), and respond accordingly.
        player_y_pos = self.player_position[0]
        player_x_pos = self.player_position[1]
        surrounded_check = self.map_disc[player_y_pos - 1:player_y_pos + 2, player_x_pos - 1:player_x_pos + 2]
        # player position is [1,1]
        enemies_idx_y, enemies_idx_x = np.where(surrounded_check == 3)
        surrounding_enemies_sum = np.sum(surrounded_check[enemies_idx_y, enemies_idx_x])

        # if this is >5 then the player is surrounded by enemies, thus it dies
        if surrounding_enemies_sum >= TileTypes.ENEMY * 2:
            reward -= 1
            done = True
        if surrounding_enemies_sum == TileTypes.ENEMY:
            # kill the enemy
            # surrounded_check should contain a portion of the original matrix
            # thus we are modifying the original martrix!!
            surrounded_check[enemies_idx_y, enemies_idx_x] = 0

            enemy_pos = (enemies_idx_y.item() + player_y_pos - 1, enemies_idx_x.item() + player_x_pos - 1)
            if enemy_pos not in self._xys_to_enemies:
                raise Exception
            self._xys_to_enemies.pop(enemy_pos)
            # reward += 1

            # for i in range(len(self.enemies)):
            #     enemy = self.enemies[i]
            #     if np.array_equal(enemy.pos,
            #                        np.array([enemies_idx_y[0] + player_x_pos - 1, enemies_idx_x[0] + player_y_pos - 1])):
            #         self.enemies.pop(i)
            #         reward += 1
            #         break

        if self.n_enemies_to_spawn > 0:
            self.n_enemies_to_spawn -= 1
            enemy = self.spawn_enemy_at_border()
        # If the player has defeated the current wave, set up the next wave.
        elif self.n_enemies_to_spawn == 0 and len(self._xys_to_enemies) == 0:
            self.spawn_lvl += 1
            self.n_enemies_to_spawn = self.spawn_lvl

        for spawner in list(self._xys_to_spawners.values()):
            # TODO: factor this out into generic movement code -SE
            spawner.update(map_onehot=self.map_onehot)
            self._update_map_onehot()
        if self.map_disc[tuple(self.player_position)] == TileTypes.SPAWNER:
            reward += 3
            self._xys_to_spawners.pop(tuple(self.player_position))


        enemy_dijkstra_map = flood_fill(self.map_onehot, Env.traversable_tile_idxs, trg_pos=self.player_position)

        for enemy in list(self._xys_to_enemies.values()):
            # TODO: factor this out into generic movement code -SE
            old_enemy_pos = enemy.pos
            self._xys_to_enemies.pop(tuple(old_enemy_pos))
            next_enemy_pos = enemy.update(map_onehot=self.map_onehot, dijkstra_map=enemy_dijkstra_map)
            assert tuple(next_enemy_pos) not in self._xys_to_enemies
            self.map_disc[tuple(old_enemy_pos)] = TileTypes.EMPTY
            self.map_disc[tuple(next_enemy_pos)] = TileTypes.ENEMY
            self._xys_to_enemies[tuple(next_enemy_pos)] = enemy
            self._update_map_onehot()

        # Add back the border in case the enemy has left a hole in the border after moving away from it.
        self.map_disc[0, :] = self.map_disc[-1, :] = TileTypes.WALL
        self.map_disc[:, 0] = self.map_disc[:, -1] = TileTypes.WALL
        # Redundant, just in case an enemy is trapped on the border (lol)
        self._place_enemies()

        done = done or self.n_step >= self.max_steps
        self.n_step += 1
        # if done:
        #     print(f"episode len {self.n_step}")

        return self._get_obs(), reward, done, {}

    def _place_enemies(self):
        for (x, y) in self._xys_to_enemies:
            self.map_disc[x, y] = TileTypes.ENEMY
        self._update_map_onehot()

    def spawn_enemy(self, enemy_pos):
        self.map_disc[tuple(enemy_pos)] = TileTypes.ENEMY
        self._update_map_onehot()
        enemy = Enemy(enemy_pos)
        self._xys_to_enemies[tuple(enemy.pos)] = enemy
        return enemy

    def spawn_enemy_at_border(self):
        """Spawn an enemy at a random position on the border of the map."""
        n_border_tiles = len(self.border_coords)
        enemy_pos = self.border_coords[random.randint(0, n_border_tiles - 1)]

        # self.map_disc[tuple(enemy_pos)] = TileTypes.ENEMY
        # self._update_map_onehot()
        # enemy = Enemy(enemy_pos)
        # self._xys_to_enemies[tuple(enemy.pos)] = enemy
        return self.spawn_enemy(enemy_pos)

    def create_enemy_spawner(self, spawner_pos=None):
        if spawner_pos is None:
            empty_coords = np.argwhere(self.map_disc == TileTypes.EMPTY)
            spawner_xy = empty_coords[random.randint(0, len(empty_coords) - 1)]
            spawner_pos = [spawner_xy[0], spawner_xy[1]]
        if len(empty_coords) > 2:
            self.map_disc[tuple(spawner_pos)] = TileTypes.SPAWNER
            self._update_map_onehot()
            spawner = Spawner(self, spawner_pos)
            self._xys_to_spawners[tuple(spawner.pos)] = spawner
            # print(self.map_disc)
            return spawner

    def _update_map_onehot(self):
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)

    def reset(self):
        self.gold_pos = set()
        self._n_gold_to_spawn = self._n_gold
        self._n_tick_no_enemy = 0
        self.spawn_lvl = 1
        self.n_enemies_to_spawn = 1
        self.n_step = 0
        self._xys_to_enemies = {}
        # self.enemies = []
        self.map_disc, self.player_position = gen_random_map(h=self.h, w=self.w)
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)
        # spawner = self.create_enemy_spawner()

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        # return self.map_onehot.astype(np.uint8)
        # TT()
        py = self.player_position[0] + self.h
        px = self.player_position[1] + self.w
        obs = np.pad(self.map_onehot, ((0, 0), (self.h, self.h), (self.w, self.w)), 'constant', constant_values=0)[:,
            py - math.floor(self.h/2): py + math.ceil(self.h/2),
            px - math.floor(self.w/2): px + math.ceil(self.w/2)].astype(np.uint8)
        # assert obs.shape == self.map_onehot.shape
        return obs

    def render(self, mode='human', is_interactive=True):
        tile_size = self.tile_size
        # self.rend_im = np.zeros_like(self.int_map)
        # Create an int map where the last tiles in `self.tiles` take priority.
        map_disc = self.map_disc
        self.rend_im = self.tile_colors[map_disc]
        self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        self.rend_im = np.pad(self.rend_im, ((0,0), (0,tile_size), (0,0)), constant_values=1)
        if mode == "human":
            if self.screen is None:
                pygame.init()
                # Set up the drawing window
                self.screen = pygame.display.set_mode([self.h*self.tile_size, (self.w+1)*self.tile_size])
            pygame_render_im(self.screen, self.rend_im, self)
            if is_interactive:
                for i in range(10):
                    # Handle Input Events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        # if event.type == pygame.QUIT:
                        #     going = False
                        # if we are pressing the mouse button, build the selected wall/trap
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.mouse_down = True
                            self.button_type = event.button
                        elif event.type == pygame.MOUSEBUTTONUP:
                            self.mouse_down = False
                        if self.mouse_down:
                            # TODO: walls are a little bit offset
                            y, x = pygame.mouse.get_pos()
                            y = round(y/self.tile_size -.5)
                            x = round(x/self.tile_size -.5)
                            # we cannot draw on top of other entities
                            if self.map_disc[y, x] not in [TileTypes.ENEMY, TileTypes.PLAYER, TileTypes.GOLD] and (x!=0 
                            and y!=0 and x !=self.w and y !=self.h): 
                            # TODO: fix the boundary checks in the bottom and right part
                                tile_type = TileTypes.EMPTY
                                if self.button_type == 1:  # left click
                                    if self.resource_counter > 0:
                                        tile_type = TileTypes.WALL
                                        # decrement the resource counter
                                        self.resource_counter -= 1
                                elif self.button_type == 3:  # right click
                                    self.resource_counter += 1
                                self.map_disc[y][x] = tile_type
                                self._update_map_onehot()
                    self.clock.tick(60)
                # self.clock.tick(20)

            return
        else:
            raise NotImplementedError

def add_images(surf, env, position, img):
    agent = pygame.transform.scale(img, (env.tile_size, env.tile_size))
    surf.blit(agent, (position[0]*env.tile_size, position[1]*env.tile_size))
    return surf


def pygame_render_im(screen, img, env):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    agent_img = pygame.image.load(os.path.join('Images', 'hercules.png'))
    enemy_img = pygame.image.load(os.path.join('Images', 'greek_zombie.png'))
    apple_img = pygame.image.load(os.path.join('Images', 'golden_apple.png'))
    temple_img = pygame.image.load(os.path.join('Images', 'ruined_temple.png'))
    surf = add_images(surf, env, env.player_position, agent_img)
    for enemy in list(env._xys_to_enemies.values()):
        surf = add_images(surf, env, enemy.pos, enemy_img)
    for s_gold_pos in env.gold_pos:
        surf = add_images(surf, env, s_gold_pos, apple_img)
    for spawner in list(env._xys_to_spawners.values()):
        surf = add_images(surf, env, spawner.pos, temple_img)


    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(f'Walls: {env.resource_counter}             Apples: {env.gold_counter}', True, [255,255,255])
    textRect = text.get_rect()  
    textRect.center = (env.w*env.tile_size // 2, (env.h+.5)*env.tile_size)
    surf.blit(text, textRect)
    
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()


def test():
    env = Env({})
    for _ in range(100):
        env.reset()
        # env.render()
        done = False
        while not done:
            env.render(is_interactive=True)
            action = env.player_action_space.sample()
            obs, rew, done, info = env.step(action)
            env.render()
        # TT()



class Enemy():
    def __init__(self, pos):
        self.pos = pos
        self.path = None
        self._sleep = 0

    def update(self, map_onehot, dijkstra_map):
        """Update the enemy's path to the player."""
        if self._sleep < 1:
            self._sleep += 1
            return self.pos
        # Pad so that we don't have to worry about going out of bounds.
        dijkstra_map = np.pad(dijkstra_map, pad_width=1, mode="constant", constant_values=Env.max_enemy_dist+1)
        # Adjust neighbors to account for padding.
        adj_xys = self.pos + Env.directions
        np.random.shuffle(adj_xys)
        adj_xys = adj_xys.T + 1
        new_pos_i = np.argmin(dijkstra_map[adj_xys[0], adj_xys[1]])
        new_pos_xy = adj_xys.T[new_pos_i] - 1
        self._sleep = 0
        

        # Update if moving onto a traversable tiles (note that an enemy may be blocking us)
        # TODO: for now we allow enemy to move through walls (random movements actually, thanks to argmin above)... ultimately we want to have
        #   them intentionally bonk the walls until they break. To make this work with a dijkstra map we need to look at
        #   each wall tile, and give it an activation of max(neihbor activation - epsilon) on the dijkstra map, so that 
        #   enemies attempt to move onto walls only if necessary. Then, we will have an array storing wall health, etc...
        if np.any(map_onehot[[Env.traversable_tile_idxs + [TileTypes.WALL]], new_pos_xy[0], new_pos_xy[1]] == 1):
        # if np.any(map_onehot[Env.traversable_tile_idxs, new_pos_xy[0], new_pos_xy[1]] == 1):
            self.pos = new_pos_xy
            
        # assert map_onehot[TileTypes.WALL, new_pos_xy[0], new_pos_xy[1]]
        # self.path_to_player = shortest_path(map_onehot, traversable_tile_idxs=[TileTypes.EMPTY], src_pos=self.pos, 
            # trg_pos=player_pos)
        # if len(self.path_to_player) > 1:
            # self.pos = self.path_to_player[1]
        return self.pos


class Spawner():
    def __init__(self, env, pos, time_between_spawns=10):
        self.env = env
        self.pos = pos
        self.path = None
        self.time_between_spawns = time_between_spawns
        self.time_since_last_spawn = 0
        self.do_spawn = False

    def update(self, map_onehot):
        self.time_since_last_spawn += 1
        self.do_spawn = False
        if self.time_since_last_spawn == self.time_between_spawns:
            self.do_spawn = True
            self.spawn_enemy_from_spawner(self.env) # TODO: the env can take care of this, would be easier, just access to self.time_to_spawn
            self.time_since_last_spawn = 0
            # spawn an enemy in a position adjacent to the spawner

    def spawn_enemy_from_spawner(self, env):
        """Spawn an enemy at a random position on the border of the map."""
        # n_border_tiles = len(self.border_coords)
        y_pos, x_pos = self.pos[0], self.pos[1]

        surrounding_check = env.map_disc[y_pos - 1:y_pos + 2, x_pos - 1:x_pos + 2]
        # player position is [1,1]
        empty_tile_y, empty_tile_x = np.where(surrounding_check == TileTypes.EMPTY)
        spawn_pos = np.random.randint(len(empty_tile_y))
        env.spawn_enemy(np.array([empty_tile_y[spawn_pos]+y_pos-1, empty_tile_x[spawn_pos]+x_pos-1]))
        # enemy_pos = self.border_coords[random.randint(0, n_border_tiles - 1)]
        # self.map_disc[tuple(enemy_pos)] = TileTypes.ENEMY
        # self._update_map_onehot()
        # enemy = Enemy(enemy_pos)
        # self._xys_to_enemies[tuple(enemy.pos)] = enemy
        # return enemy

def id_to_xy(idx, width):
    return idx // width, idx % width

def xy_to_id(x, y, width):
    return x * width + y


def flood_fill(map_onehot, traversable_tile_idxs, trg_pos):
    dijkstra_map = np.zeros(map_onehot.shape[1:]) + Env.max_enemy_dist
    dijkstra_map[tuple(trg_pos)] = 0
    frontier = trg_pos + Env.directions
    frontier = [(tuple(f), 1) for f in frontier]
    while len(frontier) > 0:
        f, dist = frontier.pop(0)
        
        # FIXME: wtf why is this happening?
        if 0 > f[0] or map_onehot.shape[1] <= f[0] or 0 > f[1] or map_onehot.shape[2] <= f[1]:
            continue 

        if dijkstra_map[f] != Env.max_enemy_dist or np.all(map_onehot[traversable_tile_idxs, f[0], f[1]] != 1):
            continue
        else:
            dijkstra_map[f] = dist
            for d in Env.directions:
                new_f = tuple(np.array(f) + d)
                frontier.append((new_f, dist + 1))
    return dijkstra_map


def shortest_path(map_onehot, traversable_tile_idxs, src_pos, trg_pos):
    src, trg = None, None
    graph = nx.Graph()
    _, width, height = map_onehot.shape
    size = width * height
    nontraversable_edge_weight = size
    graph.add_nodes_from(range(size))
    edges = []
    src, trg = xy_to_id(*src_pos, width), xy_to_id(*trg_pos, width)
    for u in range(size):
        ux, uy = id_to_xy(u, width)
        edge_weight = 1
        if np.all(map_onehot[traversable_tile_idxs, ux, uy] != 1):
            edge_weight = nontraversable_edge_weight
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        # adj_feats = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbs = [xy_to_id(x, y, width) for x, y in neighbs_xy]
        for v, (vx, vy) in zip(neighbs, neighbs_xy):
            if not 0 <= v < size or vx < 0 or vx >= width or vy < 0 or vy >= height:
                continue
            if np.all(map_onehot[traversable_tile_idxs, vx, vy] != 1):
                edge_weight = nontraversable_edge_weight
            graph.add_edge(u, v, weight=edge_weight)
            edges.append((u, v))
        edges.append((u, u))

    path = nx.shortest_path(graph, src, trg)
    path = np.array([id_to_xy(idx, width) for idx in path])

    return path


if __name__ == "__main__":
    test()