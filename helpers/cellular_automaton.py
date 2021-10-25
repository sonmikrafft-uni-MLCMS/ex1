from enum import Enum
from typing import Optional, Dict
import numpy as np
import pandas as pd
from ast import literal_eval as make_tuple


class CellState(Enum):
    EMPTY = 'E'
    PEDESTRIAN = 'P'
    OBSTACLE = 'O'
    TARGET = 'T'


def cell_state_to_visualized(x: Enum) -> str:
    """
    Helper function to convert Enum to its string value.

    :param x: Enum object to be converted
    :return: Corresponding string value of the object
    """
    if x == CellState.EMPTY:
        return ' '
    if x == CellState.PEDESTRIAN:
        return 'P'
    if x == CellState.OBSTACLE:
        return 'O'
    return 'T'


def _compute_distance(u: tuple[int, int], v: tuple[int, int]) -> float:
    """
    Compute the euclidean distance between 2 cells
    :param u: first cell
    :param v: second cell
    :return: euclidean distance between 2 cells
    """
    return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


class CellularAutomaton():
    def __init__(self, grid_size: tuple[int, int]) -> None:
        """
        Creates cellular automaton by setting up an empty grid of states and utilities.

        :param grid_size: Tuple of 2D grid dimensions
        """
        self.grid = np.full(grid_size, CellState.EMPTY)
        self.curr_iter = 0  # current iteration of the simulation
        self.grid_history = {}  # type: Dict[int, np.ndarray]

    def add(self, what: CellState, pos_idx: tuple[int, int]) -> None:
        """
        Fill a yet empty cell with either a pedestrian, an obstace, or a target.

        :param what: One of CellState to indicate what to fill
        :param pos_idx: Tuple of 2D index on which cell to fill
        :raises ValueError: If cell is not empty
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        self._check_empty(pos_idx)
        self.grid[(pos_idx)] = what

    def visualize_grid(self, iteration: Optional[int] = None) -> None:
        """
        Visualizes the state grid by printing it on the console nicely.

        Optionally, one can pass an iteration number at most of the current iteration to visualize the grid at that
        point.

        :param iteration: Optionally specify to visualize the grid at that specific tieration in the past
        :raises ValueError: If specified iteration number is not within [0, current iteration]
        """
        grid_to_visualize = self.grid.copy()

        if iteration is not None:
            self._check_iteration_number(iteration)
            grid_to_visualize = self.grid_history[iteration].copy()

        vfunc = np.vectorize(cell_state_to_visualized)
        visualized_grid = vfunc(grid_to_visualize)
        output_str = '[' + ']\n['.join(['  '.join([str(cell) for cell in row]) for row in visualized_grid]) + ']'
        print(output_str)

    def simulate_next_n(self, n: int, target_absorbs: bool = True) -> None:
        """
        Simulate next n steps by calling `simulate_next` n-times.

        :param n: Number of iterations to simulate
        :param target_absorbs: Flag that tells if a pedestrian is absorbed when going onto the target or not.
        """
        for _ in range(n):
            self.simulate_next(target_absorbs=target_absorbs)

    def simulate_next(self, target_absorbs: bool = True) -> None:
        """
        Propogate states of pedestrian one forward and add new grid state into the history.

        Pedestrians move to the cell with the best utility value in their direct surrounding.
        Only cells that are no obstacles are considered.

        :param target_absorbs: Flag that tells if a pedestrian is absorbed when going onto the target or not.
        """
        self._save_to_grid_history()
        utility_grid = self._get_distance_based_utility_grid()
        next_grid = self.grid.copy()

        # Iterate over current grid
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                curr_idx = (row, col)
                # check if pedestrian
                if self.grid[curr_idx] == CellState.PEDESTRIAN:
                    surrounding_idx = self._get_surrounding_idx(curr_idx)

                    # look around and keep track of cell with best utility
                    best_utility = utility_grid[curr_idx]
                    best_idx = curr_idx
                    for potential_next_idx in surrounding_idx:
                        if next_grid[potential_next_idx] in [CellState.OBSTACLE, CellState.PEDESTRIAN]:
                            continue
                        if not target_absorbs and next_grid[potential_next_idx] == CellState.TARGET:
                            continue
                        if utility_grid[potential_next_idx] < best_utility:
                            best_utility = utility_grid[potential_next_idx]
                            best_idx = potential_next_idx

                    # nothing to do if already on best cell
                    if best_idx == curr_idx:
                        continue

                    # otherwise propagate forward
                    next_grid[row, col] = CellState.EMPTY
                    if not self.grid[best_idx] == CellState.TARGET:
                        next_grid[best_idx] = CellState.PEDESTRIAN

        self.grid = next_grid
        self.curr_iter += 1
        self._save_to_grid_history()

    def reset_to_iteration(self, i_reset: int) -> None:
        """
        Reset the state of the cellular automaton to the specified iteration number.

        By resetting the grid property to the grid entry in the grid_history dictionary, deleting entries after that
        iteration and setting the current iteration property back to that value.

        :param i_reset: Iteration to be resetted to
        :raises ValueError: If specified iteration number is not within [0, current iteration]
        """
        self._check_iteration_number(i_reset)
        self.grid = self.grid_history[i_reset].copy()

        for i in range(i_reset + 1, self.curr_iter + 1):
            del self.grid_history[i]

        self.curr_iter = i_reset

    def _get_distance_based_utility_grid(self) -> np.ndarray:
        """
        Get utility grid filled by euclidean distance to closest target.

        Distance is calulated in an naive way ignoreing obstacles.

        - target cells are filled with 0
        - obstacle cells are filled with infinity
        - other cells are filled with euclidean distance to closest target

        :return: Numpy array of same shape as state grid filled with utility values
        """
        utility_grid = np.zeros(self.grid.shape)

        # get idx of all targets
        idx_targets = [tuple(a) for a in np.argwhere(self.grid == CellState.TARGET)]
        if len(idx_targets) == 0:
            raise ValueError('No target in grid.')

        # get idx of all non target and non obstacle cells
        idx_movables = [tuple(a) for a in np.argwhere(
            (self.grid != CellState.TARGET) & (self.grid != CellState.OBSTACLE))]
        # iterate over movable cells
        for idx_movable in idx_movables:
            smallest_distance = None

            # iterate over target cells
            for idx_target in idx_targets:
                eucl_distance = np.linalg.norm(np.array(idx_movable) - np.array(idx_target))
                if smallest_distance is None or eucl_distance < smallest_distance:
                    smallest_distance = eucl_distance

            utility_grid[idx_movable] = smallest_distance

        # treat obstacles
        utility_grid[self.grid == CellState.OBSTACLE] = np.infty

        return utility_grid

    def _get_surrounding_idx(self, pos_idx: tuple[int, int]) -> set[tuple[int, int]]:
        """
        Given the current position in the 2D grid as a tuple of indices, return a set of valid surrounding
        positions.

        Example:
        >>> myCellularAutomaton = CellularAutomaton((2,2))
        >>> myCellularAutomaton._get_surrounding_idx((0, 0))
        {(0, 1), (1, 1), (1, 0)}

        :param pos_idx: 2D Tuple of current position as indices
        :return: Set of valid surrounding positions
        """
        self._check_valid_idx(pos_idx)
        moves = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]
        moves_applied = [tuple(map(sum, zip(pos_idx, move))) for move in moves]
        surrounding_idx = {move for move in moves_applied if self._check_valid_idx(
            move, surpess_error=True)}  # type: ignore
        return surrounding_idx  # type: ignore

    def _save_to_grid_history(self) -> None:
        """
        Add copy of current grid property as an entry to the grid history dict, where the key is the curr_iter.
        """
        self.grid_history[self.curr_iter] = self.grid.copy()

    def _check_iteration_number(self, i_to_check: int) -> None:
        """
        Check if iteration number is within [0, current iteration], i.e. valid.

        :raises ValueError: If specified iteration number is not within [0, current iteration]
        """
        if i_to_check not in range(0, self.curr_iter + 1):
            raise ValueError(f'Please specify an iteration number within [0,{self.curr_iter}]')

    def _check_valid_idx(self, pos_idx: tuple[int, int], surpess_error: bool = False) -> Optional[bool]:
        """
        Checks if we can access the grid with the given indices.

        :param pos_idx: Index that we want to access, specified as 2D tuple
        :param surpess_error: Flag if to surpress raising an IndexError and return False instead
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        :return: True if valid, False invalid and surpress_error == True
        """
        if not all((idx < grid) and (idx >= 0) for (idx, grid) in zip(pos_idx, self.grid.shape)):
            if not surpess_error:
                raise IndexError(
                    f'Trying to access grid position {pos_idx} but grid is only of size {self.grid.shape}.')
            else:
                return False
        return True

    def _check_empty(self, pos_idx: tuple[int, int]) -> None:
        """
        Checks if cell is empty.

        :param pos_idx: Index that we want to check
        :raises ValueError: If cell is not empty
        """
        self._check_valid_idx(pos_idx)
        if self.grid[pos_idx] != CellState.EMPTY:
            raise ValueError(f'Trying to write value into grid position {pos_idx}, but not empty.')

    def set_utilities(self, obstacle_avoidance: bool) -> None:
        """
        Set the utilities of each cell according to their respective distance to the closest target
        """
        # dev: self.utilities can be initialized with the following line in the init method
        self.utilities = np.full_like(self.utilities, np.inf)

        targets = [tuple(a) for a in np.argwhere(self.grid == CellState.TARGET)]

        for target in targets:
            self.set_dijkstra_for_one_target(target, obstacle_avoidance)

        # very basic obstacle avoidance
        self.utilities[self.grid == CellState.OBSTACLE] = np.inf

    def print_utilities(self) -> None:
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(self.utilities)

    def set_dijkstra_for_one_target(self, target: tuple[int, int], obstacle_avoidance: bool) -> None:
        """
        Set utilities of each cell with Dijkstra's algorithm to find minimal distance to a single target

        Adapted from: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/ (24/10/2021)
        """
        # initialize a distance array (computed distances to the target) and an array of visited cells
        dist = np.full_like(self.utilities, np.inf)
        dist[target] = 0
        self.utilities[target] = 0
        visited = np.full_like(self.utilities, False)

        for _ in self.utilities.flat:
            u = self.find_minimal_distance_cell(dist, visited)
            visited[u] = True

            neighbors = self._get_surrounding_idx(u)

            for v in neighbors:
                if obstacle_avoidance and self.grid[v] == CellState.OBSTACLE:
                    continue
                if not visited[v] and dist[v] > dist[u] + _compute_distance(u, v):
                    dist[v] = dist[u] + _compute_distance(u, v)

                    # update a cell's utility if the cell is closer to this target than a previous target
                    if self.utilities[v] > dist[v]:
                        self.utilities[v] = dist[v]

    def find_minimal_distance_cell(self, dist, visited) -> tuple[int, int]:
        """
        Determine cell with minimal distance to the target that will be visited next
        :param dist: array of distances between cells and the target
        :param visited: array that indicates whether a cell has been visited or not
        :return: cell with minimal distance to the target
        """
        min_dist = np.inf
        min_dist_cell = (0, 0)

        for fst, _ in enumerate(self.utilities):
            for snd, _ in enumerate(self.utilities[fst]):
                u = (fst, snd)
                if dist[u] < min_dist and not visited[u]:
                    min_dist = dist[u]
                    min_dist_cell = u
        return min_dist_cell


def fill_from_scenario_file(scenario_file: str) -> CellularAutomaton:
    """
    Read specified scenario file and creates a matching CellularAutomaton object.

    Scenario file needs to have the following columns:
    - 'grid_size', one row only, specifies grid size
    - 'initial_position_obstacles', tuples of form (x, y), one tuple per row, specifies obstacles positions
    - 'position_target_zone', tuples of form (x, y), one tuple per row, specifies target positions
    - 'initial_position_pedestrian', tuples of form (x, y), one tuple per row, specifies pedestrian positions

    :param scenario_file: Path to .csv file
    :return: CellularAutomaton object matching the scenario file configuration
    """
    df = pd.read_csv(scenario_file, delimiter=';')

    grid_size = make_tuple(df['grid_size'][0])
    obstacle_positions = df['initial_position_obstacles']
    target_positions = df['position_target_zone']
    pedestrian_positions = df['initial_position_pedestrian']

    my_cellular_automaton = CellularAutomaton(grid_size)

    for obstacle_position in obstacle_positions:
        my_cellular_automaton.add(CellState.OBSTACLE, make_tuple(obstacle_position))

    for target_position in target_positions:
        my_cellular_automaton.add(CellState.TARGET, make_tuple(target_position))

    for pedestrian_position in pedestrian_positions:
        my_cellular_automaton.add(CellState.PEDESTRIAN, make_tuple(pedestrian_position))

    return my_cellular_automaton
