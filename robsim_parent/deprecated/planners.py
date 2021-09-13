from workspace import Workspace
from utility import dist_l1, dist_l2, dist_max, dist_names
from primitives.point import Point

from math import ceil
from random import random, choice, choices

from tqdm import tqdm

import multiprocessing as mp
import os

def propagate(i, chunk, steps, workspace, robot):
    result = []

    for j, ((a, alpha), state) in enumerate(chunk):
        path = robot.propagate(state, a, alpha, steps)

        if not all([workspace.is_free(Point(state[0], state[1])) for state in path]):
            continue
        
        result.append((j, path))

    return (i, result)

def check_paths(i, chunk, goal):
    paths = []

    for j, path in chunk:
        for k, state in enumerate(path):
            in_goal = max(abs(state[0] - goal[0]), abs(state[1] - goal[1])) < 0.05
            stopped = state[3] < 1e-1
            
            if in_goal and stopped:
                paths.append((i, j, path[:k]))
                break

    return paths

class Planner:
    pass

class PRM(Planner):
    def __init__(self, workspace: Workspace, delta: float = 0.01, k: int = 1, r: float = 0.0, distance_type: str = 'l2'):
        self.workspace = workspace
        self.delta = delta
        self.k = k
        self.r = r

        self.edges = []
        self.nodes = []

        self.dist = dist_names[distance_type]

    def make_path(self, a: Point, b: Point, d=None):
        if d == None:
            d = self.dist(a, b)

        N = ceil(d/self.delta)
        scales = [(n + 1) / N for n in range(N - 1)]

        return [Point(s * a.x + (1 - s) * b.x, s * a.y + (1 - s) * b.y) for s in scales]

    def construct(self, N: int):
        for _ in range(N):
            i = len(self.nodes)

            pt = Point(random() * self.workspace.dimensions[0], random() * self.workspace.dimensions[1])

            if not self.workspace.is_free(pt):
                continue

            i_pts = [{'i': j, 'pt': self.nodes[j], 'dist': self.dist(pt, self.nodes[j])} for j in range(i)]

            # Removing farther than r nodes.
            if self.r != 0:
                i_pts = [i_pt for i_pt in i_pts if i_pt['dist'] < self.r]

            # Reducing to k nodes at most.
            if self.k != 0:
                i_pts = sorted(i_pts, key=lambda i_pt: i_pt['dist'])
                i_pts = i_pts[:self.k]

            for i_pt in i_pts:
                path = self.make_path(pt, i_pt['pt'], i_pt['dist'])
                if all([self.workspace.is_free(p) for p in path]):
                    self.edges.append((i_pt['i'], i))
                    
            self.nodes.append(pt)

class RKMP(Planner):
    def __init__(self, workspace: Workspace, x0: tuple, goal: tuple, robot, steps: int, grid_size: float, check_state):
        self.workspace = workspace
        self.robot = robot
        self.steps = steps
        self.grid_size = grid_size
        self.check_state = check_state # Must take a state and return if the state is a valid goal state.
        self.goal = goal

        self.grid_width = ceil(float(workspace.dimensions[0]) / grid_size)
        self.grid_height = ceil(float(workspace.dimensions[1]) / grid_size)
        self.grid = [[[] for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        self.add_state(x0)

        self.paths = {}

    def get_grid_indeces(self, x: float, y: float) -> tuple:
        i = max(0, min(int(x / self.grid_size), self.grid_width - 1))
        j = max(0, min(int(y / self.grid_size), self.grid_height - 1))

        return i, j

    def add_state(self, state: tuple):
        i, j = self.get_grid_indeces(state[0], state[1])
        self.grid[j][i].append(state)

    def sample_state(self, k: int = 1) -> list:
        grid_cells = [self.grid[j][i] for i in range(self.grid_width) for j in range(self.grid_height) if len(self.grid[j][i]) > 0]
        cells = choices(grid_cells, k=k)

        return [(self.robot.sample_control(), choice(cell)) for cell in cells]

    def get_route(self, state):
        route = []

        while state in self.paths:
            state, path = self.paths[state]
            route += list(reversed(path))
        
        return list(reversed(route))

    def solve(self, N: int = 1e7, chunk_size: int = 1) -> list:
        pbar = tqdm(total=N)

        for _ in range(int(N)):
            chunk = self.sample_state(k=chunk_size)

            for (a, alpha), state in chunk:
                path = self.robot.propagate(state, a, alpha, self.steps)

                if not all([self.workspace.is_free(Point(state[0], state[1])) for state in path]):
                    continue

                route = []
                
                for i, s in enumerate(path):
                    complete = self.check_state(s)

                    if complete:
                        route = self.get_route(state)

                        return route + path[:i]

                end_state = path[-1]

                self.paths[end_state] = (state, path[:-1])
                self.add_state(end_state)

            pbar.update(len(chunk))

        pbar.close()

        return []

    def solve_mp(self, num_processes: int = 8, initial_chunk_size: int = 2, max_chunk_size: int = 5000, chunk_size_growth: float = 1.2, N: int = 1e7) -> list:
        pbar = tqdm(total=N)

        pool = mp.Pool(num_processes)

        n = 0
        chunk_size = initial_chunk_size
        while n < N:
            chunk_size = min(sum(len(self.grid[j][i]) for i in range(self.grid_width) for j in range(self.grid_height)) / (num_processes * 10), max_chunk_size)
            chunk_size = ceil(chunk_size)

            state_chunk = [(i, self.sample_state(k=int(chunk_size)), self.steps, self.workspace, self.robot) for i in range(num_processes)]

            paths_chunk = pool.starmap(propagate, state_chunk)
            
            paths = [(i, paths, self.goal) for i, paths in paths_chunk]

            results = pool.starmap_async(check_paths, paths)

            for states, (i, paths) in zip(state_chunk, paths_chunk):
                for (control, state), (j, path) in zip(states[1], paths):
                    end_state = path[-1]

                    self.paths[end_state] = (state, path[:-1])
                    self.add_state(end_state)

            results = results.get()

            successes = [paths for paths in results if len(paths) > 0]

            if len(successes) > 0:
                paths = []
                for p in successes:
                    paths += p

                routes = []
                for i, j, path in paths:
                    state = state_chunk[i][1][j][1]
                    route = self.get_route(state) + path
                    routes.append(route)
                
                route_lengths = [len(route) for route in routes]
                i_min = route_lengths.index(min(route_lengths))
                route = routes[i_min]

                pool.close()
                pool.join()

                pbar.close()

                return route

            pbar.update(chunk_size * num_processes)
            n += chunk_size * num_processes
            
        pool.close()
        pool.join()

        pbar.close()

        return []