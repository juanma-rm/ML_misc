from typing import Tuple, Dict, Optional, Iterable, Callable

import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt

from IPython.display import HTML

import gymnasium as gym
from gymnasium import spaces

import pygame
from pygame import gfxdraw

from typing import Dict, Tuple, Iterable, List
from random import shuffle, choice, uniform, random

class Maze(gym.Env):

    def __init__(self, exploring_starts: bool = False,
                 shaped_rewards: bool = False, size: int = 5,
                 wall_density: float = 0.01, branch_density: float = 0.999) -> None:
        super().__init__()
        self.exploring_starts = exploring_starts
        self.shaped_rewards = shaped_rewards
        self.size = size
        self.state = (size - 1, size - 1)
        self.goal = (size - 1, size - 1)
        self.maze = self._create_maze(size=size, wall_density=wall_density, branch_density=branch_density)
        self.distances = self._compute_distances(self.goal, self.maze)
        self.action_space = spaces.Discrete(n=4)
        self.action_space.action_meanings = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: "LEFT"}
        self.observation_space = spaces.MultiDiscrete([size, size])

        self.screen = None
        self.agent_transform = None

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict]:
        reward = self.compute_reward(self.state, action)
        self.state = self._get_next_state(self.state, action)
        terminated = self.state == self.goal
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[Tuple[int, int], Dict]:
        super().reset(seed=seed)
        if self.exploring_starts:
            while self.state == self.goal:
                self.state = tuple(self.observation_space.sample())
        else:
            self.state = (0, 0)
        return self.state, {}

    def render(self) -> Optional[np.ndarray]:
        screen_size = 600
        scale = screen_size / self.size

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))

        for row in range(self.size):
            for col in range(self.size):
                state = (row, col)
                for next_state in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                    if next_state not in self.maze[state]:
                        # Add the geometry of the edges and walls (i.e. the boundaries between
                        # adjacent squares that are not connected).
                        row_diff, col_diff = np.subtract(next_state, state)
                        left = (col + (col_diff > 0)) * scale - 2 * (col_diff != 0)
                        right = ((col + 1) - (col_diff < 0)) * scale + 2 * (col_diff != 0)
                        top = (self.size - (row + (row_diff > 0))) * scale - 2 * (row_diff != 0)
                        bottom = (self.size - ((row + 1) - (row_diff < 0))) * scale + 2 * (row_diff != 0)

                        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (255, 255, 255))

        # Add the geometry of the goal square to the viewer.
        left, right = scale * (self.size - 1) + 10, scale * self.size - 10
        top, bottom = scale - 10, 10
        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (40, 199, 172))

        # Add the geometry of the agent to the viewer.
        agent_row = int(screen_size - scale * (self.state[0] + .5))
        agent_col = int(scale * (self.state[1] + .5))
        gfxdraw.filled_circle(surf, agent_col, agent_row, int(scale * .6 / 2), (228, 63, 90))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            return - (self.distances[next_state] / self.distances.max())
        return - float(state != self.goal)

    def simulate_step(self, state: Tuple[int, int], action: int):
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        terminated = next_state == self.goal
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        if next_state in self.maze[state]:
            return next_state
        return state
    

    @staticmethod
    def _create_maze(size: int, wall_density: float = 0.01, branch_density: float = 0.999) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        """
        Creates a maze represented as a dictionary where keys are cells 
        (tuples of (row, col)) and values are lists of neighboring cells.

        Ensures there is always a valid path from start (0, 0) to end (size-1, size-1) 
        with a certain level of 'invalid' branches.

        Args:
            size: The size of the maze (number of rows and columns).
            wall_density: The probability of a wall between two adjacent cells.
            branch_density: The probability of creating an 'invalid' branch 
                           (a path that doesn't lead to the end).

        Returns:
            A dictionary representing the maze.
        """

        maze: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # Initialize maze with empty cells
        for row in range(size):
            for col in range(size):
                maze[(row, col)] = []

        # Create a list of all cells
        all_cells = [(row, col) for row in range(size) for col in range(size)]

        # Initialize visited set 
        visited = set()

        # Depth-First Search to create the maze
        def dfs(current):
            visited.add(current)

            neighbors = []
            if current[0] > 0:
                neighbors.append((current[0] - 1, current[1]))
            if current[0] < size - 1:
                neighbors.append((current[0] + 1, current[1]))
            if current[1] > 0:
                neighbors.append((current[0], current[1] - 1))
            if current[1] < size - 1:
                neighbors.append((current[0], current[1] + 1))

            shuffle(neighbors)  # Randomize neighbor order

            for neighbor in neighbors:
                if neighbor not in visited:
                    maze[current].append(neighbor)
                    maze[neighbor].append(current)
                    dfs(neighbor)

        # Start DFS from the top-left corner
        dfs(all_cells[0])

        # Add random walls (while maintaining connectivity)
        def add_walls():
            for row in range(size):
                for col in range(size):
                    if (row, col) != (0, 0) and (row, col) != (size - 1, size - 1):
                        for neighbor in maze[(row, col)][:]:
                            if random() < wall_density:
                                maze[(row, col)].remove(neighbor)
                                maze[neighbor].remove((row, col))
                                if not Maze._is_reachable(maze, (0, 0), (size - 1, size - 1)): 
                                    maze[(row, col)].append(neighbor)
                                    maze[neighbor].append((row, col))

        add_walls() 

        # Add 'invalid' branches (while ensuring path validity)
        def add_branches():
            for row in range(size):
                for col in range(size):
                    if (row, col) != (0, 0) and (row, col) != (size - 1, size - 1):
                        for neighbor in maze[(row, col)][:]:
                            if random() > branch_density:
                                # Temporarily remove the connection
                                maze[(row, col)].remove(neighbor)
                                maze[neighbor].remove((row, col))

                                # Check if path is still valid
                                if Maze._is_reachable(maze, (0, 0), (size - 1, size - 1)):
                                    # Keep the change if path remains valid
                                    pass 
                                else:
                                    # Revert the change if path is invalid
                                    maze[(row, col)].append(neighbor)
                                    maze[neighbor].append((row, col))

        add_branches()

        return maze

    @staticmethod
    def _is_reachable(maze, start, end):
        """Simple depth-first search to check if 'end' is reachable from 'start'."""
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            visited.add(current)
            for neighbor in maze[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        return False










    def _compute_distances(self, goal: Tuple[int, int],
                           maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]]) -> np.ndarray:
        distances = np.full((self.size, self.size), np.inf)
        visited = set()
        distances[goal] = 0.

        while visited != set(maze):
            sorted_dst = [(v // self.size, v % self.size) for v in distances.argsort(axis=None)]
            closest = next(x for x in sorted_dst if x not in visited)
            visited.add(closest)

            for neighbour in maze[closest]:
                distances[neighbour] = min(distances[neighbour], distances[closest] + 1)
        return distances


def plot_policy(probs_or_qvals, frame, action_meanings=None):
    if action_meanings is None:
        action_meanings = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    max_prob_actions = probs_or_qvals.argmax(axis=-1)
    probs_copy = max_prob_actions.copy().astype(object)
    for key in action_meanings:
        probs_copy[probs_copy == key] = action_meanings[key]
    sns.heatmap(max_prob_actions, annot=probs_copy, fmt='', cbar=False, cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.suptitle("Policy", size=18)
    plt.tight_layout()


def plot_values(state_values, environment):
    frame = environment.render()
    f, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={'size': 8}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

def display_video_html(frames):
    # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    matplotlib.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=50, blit=True, repeat=False)
    return HTML(anim.to_html5_video())

def display_frame_onscreen(environment):
    frame = environment.render()
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.clf()
    plt.imshow(frame)
    plt.pause(1)
    plt.close()

def display_video_onscreen(frames):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    for frame in frames:
        plt.clf()
        plt.imshow(frame)
        plt.pause(0.05)
    plt.close()

def test_agent(environment, policy, episodes=10, show_html=False):
    frames = []
    for episode in range(episodes):
        state, _ = environment.reset()
        terminated = truncated = False
        frames.append(environment.render())

        while not (terminated or truncated):
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)
            else:
                action = p
            next_state, reward, terminated, truncated, info = environment.step(action)
            img = environment.render()
            frames.append(img)
            state = next_state

    if (show_html):
        return display_video_html(frames)
    else:
        return display_video_onscreen(frames)