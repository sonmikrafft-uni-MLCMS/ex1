from enum import Enum
from typing import Optional, Dict
import numpy as np
import pandas as pd
from ast import literal_eval as make_tuple
from copy import deepcopy
from collections import namedtuple


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
        self.state_grid = np.full(grid_size, CellState.EMPTY)
        self.curr_iter = 0  # current iteration of the simulation
        self.state_grid_history = {}  # type: Dict[int, np.ndarray]
        self.pedestrians = []  # type: list[dict]
        self.pedestrians_history = {}  # type: dict[int, list[dict]]

    def add_obstacle(self, pos_idx: tuple[int, int]) -> None:
        """
        Fill a yet empty cell with an obstacle.

        :param pos_idx: Tuple of 2D index on which cell to fill
        :raises ValueError: If cell is not empty
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        self._check_empty(pos_idx)
        self.state_grid[(pos_idx)] = CellState.OBSTACLE

    def add_target(self, pos_idx: tuple[int, int]) -> None:
        """
        Fill a yet empty cell with a target.

        :param pos_idx: Tuple of 2D index on which cell to fill
        :raises ValueError: If cell is not empty
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        self._check_empty(pos_idx)
        self.state_grid[(pos_idx)] = CellState.TARGET

    def add_pedestrian(self, pos_idx: tuple[int, int], speed: float) -> None:
        """
        Fill a yet empty cell with either a pedestrian, an obstace, or a target.

        :param pos_idx: Tuple of 2D index on which cell to fill
        :param speed: Speed of the pedestrian in cell units per iteration
        :raises ValueError: If cell is not empty
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        self._check_empty(pos_idx)
        self.state_grid[(pos_idx)] = CellState.PEDESTRIAN

        pedestrian = {
            'speed': speed,  # average speed when moving
            'pos': pos_idx,  # current position as tuple of row, col id
            'travelled': 0,  # keeps track of total distance travelled
            'skips': 0,  # keeps track of iterations where not able to move
        }

        self.pedestrians.append(pedestrian)

    def visualize_state_grid(self, iteration: Optional[int] = None) -> None:
        """
        Visualizes the state grid by printing it on the console nicely.

        Optionally, one can pass an iteration number at most of the current iteration to visualize the grid at that
        point.

        :param iteration: Optionally specify to visualize the grid at that specific tieration in the past
        :raises ValueError: If specified iteration number is not within [0, current iteration]
        """
        grid_to_visualize = self.state_grid.copy()

        if iteration is not None:
            self._check_iteration_number(iteration)
            grid_to_visualize = self.state_grid_history[iteration].copy()

        vfunc = np.vectorize(cell_state_to_visualized)
        visualized_grid = vfunc(grid_to_visualize)
        output_str = '[' + ']\n['.join(['  '.join([str(cell) for cell in row]) for row in visualized_grid]) + ']'
        print(output_str)

    def simulate_until_no_change(self, smart_obstacle_avoidance: bool = True, target_absorbs: bool = True) -> None:
        """
        Simulate until there is no change.

        :param smart_obstacle_avoidance: Optional flag wether intelligent obstacle avoidance is active
            Defaults to true
        :param target_absorbs: Optional flag that tells if a pedestrian is absorbed when going onto the target or not
            Defaults to true
        """
        while(True):
            change = self.simulate_next(smart_obstacle_avoidance=smart_obstacle_avoidance,
                                        target_absorbs=target_absorbs)
            if (change is not None) and (not change):
                return

    def simulate_next_n(self, n: int, stop_when_no_change: bool = True, smart_obstacle_avoidance: bool = True,
                        target_absorbs: bool = True) -> Optional[bool]:
        """
        Simulate next n steps by calling `simulate_next` n-times.

        :param n: Number of iterations to simulate
        :param stop_when_no_change: Optional flag wether to stop when there is no change happening and to then return
            False
        :param smart_obstacle_avoidance: Optional flag wether intelligent obstacle avoidance is active
            Defaults to true
        :param target_absorbs: Optional flag that tells if a pedestrian is absorbed when going onto the target or not
            Defaults to true
        :return: Optionally return True / False when stop_when_no_change is set.
            True = During n steps, there always was a change
            False = Interrupted since there was no change at some point
        """
        for _ in range(n):
            change = self.simulate_next(smart_obstacle_avoidance=smart_obstacle_avoidance,
                                        target_absorbs=target_absorbs)
            if not change and stop_when_no_change:
                return False

        if stop_when_no_change:
            return True

        return None

    def simulate_next(self, stop_when_no_change: bool = True, smart_obstacle_avoidance: bool = True,
                      target_absorbs: bool = True) -> Optional[bool]:
        """
        Propogate states of pedestrian one forward and add new grid state into the history.

        Pedestrians move to the cell with the best utility value in their direct surrounding.
        Only cells that are no obstacles are considered.

        @TODO make pedestrian check implicit by adding a pedestrian based utiliy on top of positional utility grid
        :param stop_when_no_change: Optional flag wether to stop when there is no change happening and to then return
            False
        :param smart_obstacle_avoidance: Optional flag wether intelligent obstacle avoidance is active
            Defaults to true
        :param target_absorbs: Optional flag that tells if a pedestrian is absorbed when going onto the target or not
            Defaults to true
        :return: Optionally return True / False when stop_when_no_change is set.
            True = There was a change
            False = There was no change
        """
        self._save_to_history()
        state_grid = self.state_grid
        utility_grid = self._get_dijkstra_utility_grid(state_grid, smart_obstacle_avoidance)
        next_grid = self.state_grid.copy()
        LastStep = namedtuple('LastStep', 'error state_grid pedestrians')

        # 1 >>>> Iterate over pedestrians >>>>

        deleted = 0  # keeping track of deleted pedestrians that reached target
        skip_pedestrian = [False] * len(self.pedestrians)  # keep track of skips
        if len(self.pedestrians) == 0 and stop_when_no_change:
            return False
        for i in range(len(self.pedestrians)):
            pedestrian = self.pedestrians[i - deleted]  # -deleted since we are manipulating the list while iterating
            curr_idx = pedestrian['pos']
            assert self.state_grid[curr_idx] == CellState.PEDESTRIAN, 'Pedestrian list does not match cell states grid.'

            # 2 >>>> For one pedestrian, check how many steps it should go >>>>
            error_not_moving = abs(pedestrian['travelled'] / (self.curr_iter + 1) - pedestrian['speed'])
            last_step = LastStep(error_not_moving, next_grid.copy(), deepcopy(self.pedestrians))

            potential_next_grid = next_grid.copy()
            potential_next_pedestrians = self.pedestrians.copy()
            num_steps = 1
            while(True):

                # 3 >>>> For one step, iterate over surrounding cells, find best utility >>>>
                surrounding_idx = self._get_surrounding_idx(curr_idx)
                best_utility = utility_grid[curr_idx]
                best_idx = curr_idx
                for potential_next_idx in surrounding_idx:
                    if potential_next_grid[potential_next_idx] in [CellState.PEDESTRIAN]:
                        continue
                    if not target_absorbs and potential_next_grid[potential_next_idx] == CellState.TARGET:
                        continue
                    if utility_grid[potential_next_idx] < best_utility:
                        best_utility = utility_grid[potential_next_idx]
                        best_idx = potential_next_idx
                # 3 <<<< For one step, iterate over surrounding cells, find best utility <<<<

                # nothing to do if already on best cell
                # we use last step as optimum = nothing to do
                if best_idx == curr_idx:
                    skip_pedestrian[i] = True
                    next_grid = last_step.state_grid.copy()
                    self.pedestrians = deepcopy(last_step.pedestrians)
                    if num_steps == 1:
                        self.pedestrians[i]['skips'] += 1  # only when not moving
                    break

                # otherwise propagate forward
                potential_next_grid[curr_idx] = CellState.EMPTY
                pedestrian['pos'] = best_idx
                pedestrian['travelled'] += np.linalg.norm(np.array(curr_idx) - np.array(best_idx))
                error = abs(pedestrian['travelled'] / (self.curr_iter + 1 - pedestrian['skips']) - pedestrian['speed'])

                # if pedestrian moves to regular cell
                if not self.state_grid[best_idx] == CellState.TARGET:
                    potential_next_grid[best_idx] = CellState.PEDESTRIAN
                # if pedestrian steps on target and is absorbed, we cannot check any further steps
                # therefore we compare this step with previous one and choose better
                else:
                    if error <= last_step.error:
                        del potential_next_pedestrians[i - deleted]
                        deleted += 1  # manual correction for the deletion, needed since we are iterating over it
                        next_grid = potential_next_grid.copy()
                        self.pedestrians = deepcopy(potential_next_pedestrians)
                        break

                # if error is again increasing, use last step as optimum
                if error > last_step.error:
                    next_grid = last_step.state_grid.copy()
                    self.pedestrians = deepcopy(last_step.pedestrians)
                    break
                # otherwise continue with next step
                last_step = LastStep(error, potential_next_grid.copy(), deepcopy(potential_next_pedestrians))
                # set for next iteration
                curr_idx = best_idx
                num_steps += 1

            # 2 <<<< For one pedestrian, check how many steps it should go <<<<

        # 1 <<<< Iterate over pedestrians <<<<

        # check if no pedestrian moved => stop simulation if flag is set
        if np.all(skip_pedestrian) and stop_when_no_change and (np.array_equal(next_grid, self.state_grid)):
            return False

        # end this one simulation step by assigning the current grid state back
        self.state_grid = next_grid
        self.curr_iter += 1
        self._save_to_history()

        if stop_when_no_change:
            return True

        return None

    def reset_to_iteration(self, i_reset: int) -> None:
        """
        Reset the state of the cellular automaton to the specified iteration number.

        By resetting the grid property to the grid entry in the grid_history dictionary, deleting entries after that
        iteration and setting the current iteration property back to that value.

        :param i_reset: Iteration to be resetted to
        :raises ValueError: If specified iteration number is not within [0, current iteration]
        """
        self._check_iteration_number(i_reset)

        # check nothing has been saved so far, then do nothing
        if len(self.state_grid_history) == 0:
            return

        self.state_grid = self.state_grid_history[i_reset].copy()
        self.pedestrians = deepcopy(self.pedestrians_history[i_reset])

        for i in range(i_reset + 1, self.curr_iter + 1):
            del self.state_grid_history[i]
            del self.pedestrians_history[i]

        self.curr_iter = i_reset

    def print_utilities(self, smart_obstacle_avoidance: bool = True, iteration: Optional[int] = None) -> None:
        """
        Calculates and prints the utility grid based on the state grid.

        :param smart_obstacle_avoidance: Optional flag wether intelligent obstacle avoidance is active
            Defaults to true
        :param iteration: Optionally specify the iteration number where you want to get the utility grid for
        """
        state_grid = self.state_grid.copy()
        if iteration is not None:
            self._check_iteration_number(iteration)
            state_grid = self.state_grid_history[iteration].copy()

        utility_grid = self._get_dijkstra_utility_grid(state_grid, smart_obstacle_avoidance)

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(utility_grid)

    def _get_distance_based_utility_grid(self) -> np.ndarray:
        """
        Get utility grid filled by euclidean distance to closest target.

        REMARK: This is the old, naive calculation. See _get_dijkstra_utility_grid for a proper implementation
        Distance is calulated in an naive way ignoring obstacles..

        - target cells are filled with 0
        - obstacle cells are filled with infinity
        - other cells are filled with euclidean distance to closest target

        :return: Numpy array of same shape as state grid filled with utility values
        """
        utility_grid = np.zeros(self.state_grid.shape)

        # get idx of all targets
        idx_targets = [tuple(a) for a in np.argwhere(self.state_grid == CellState.TARGET)]
        if len(idx_targets) == 0:
            raise ValueError('No target in grid.')

        # get idx of all non target and non obstacle cells
        idx_movables = [tuple(a) for a in np.argwhere(
            (self.state_grid != CellState.TARGET) & (self.state_grid != CellState.OBSTACLE))]
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
        utility_grid[self.state_grid == CellState.OBSTACLE] = np.infty

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

    def _save_to_history(self) -> None:
        """
        Add copy of current grid property as an entry to the grid history dict, where the key is the curr_iter.
        """
        self.state_grid_history[self.curr_iter] = self.state_grid.copy()
        self.pedestrians_history[self.curr_iter] = deepcopy(self.pedestrians)

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
        if not all((idx < grid) and (idx >= 0) for (idx, grid) in zip(pos_idx, self.state_grid.shape)):
            if not surpess_error:
                raise IndexError(
                    f'Trying to access grid position {pos_idx} but grid is only of size {self.state_grid.shape}.')
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
        if self.state_grid[pos_idx] != CellState.EMPTY:
            raise ValueError(f'Trying to write value into grid position {pos_idx}, but not empty.')

    def _get_dijkstra_utility_grid(self, state_grid: np.ndarray, smart_obstacle_avoidance: bool) -> np.ndarray:
        """
        Set the utilities of each cell according to their respective distance to the closest target, based on Dijkstra's
        algorithm.

        :param state_grid: 2D array of state grid with CellStates
        :smart_obstacle_avoidance: Flag wether intelligent obstacle avoidance is active
        """
        utitliy_grid = np.full(state_grid.shape, np.inf)

        targets = [tuple(a) for a in np.argwhere(state_grid == CellState.TARGET)]

        for target in targets:
            utitliy_grid = self._set_dijkstra_for_one_target(target, state_grid, utitliy_grid, smart_obstacle_avoidance)

        # very basic obstacle avoidance
        utitliy_grid[state_grid == CellState.OBSTACLE] = np.inf
        return utitliy_grid

    def _set_dijkstra_for_one_target(self, target: tuple[int, int], state_grid: np.ndarray, utitliy_grid: np.ndarray,
                                     smart_obstacle_avoidance: bool) -> np.ndarray:
        """
        Set utilities of each cell with Dijkstra's algorithm to find minimal distance to a single target

        Adapted from: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/ (24/10/2021)

        :param taget: Tuple of ids of target cell
        :param state_grid: numpy array of state grid filled with CellStates
        :param utility_grid: current utility grade (based on other targets)
        :param smart_obstacle_avoidance: Flag wether intelligent obstacle avoidance is active
        """
        # initialize a distance array (computed distances to the target) and an array of visited cells
        dist = np.full_like(utitliy_grid, np.inf)
        dist[target] = 0
        utitliy_grid[target] = 0
        visited = np.full_like(utitliy_grid, False)

        for _ in utitliy_grid.flat:
            u = self._find_minimal_distance_cell(dist, visited, utitliy_grid)
            visited[u] = True

            neighbors = self._get_surrounding_idx(u)

            for v in neighbors:
                if smart_obstacle_avoidance and state_grid[v] == CellState.OBSTACLE:
                    continue
                if not visited[v] and dist[v] > dist[u] + _compute_distance(u, v):
                    dist[v] = dist[u] + _compute_distance(u, v)

                    # update a cell's utility if the cell is closer to this target than a previous target
                    if utitliy_grid[v] > dist[v]:
                        utitliy_grid[v] = dist[v]

        return utitliy_grid

    def _find_minimal_distance_cell(self, dist, visited, utitliy_grid) -> tuple[int, int]:
        """
        Determine cell with minimal distance to the target that will be visited next.

        :param dist: array of distances between cells and the target
        :param visited: array that indicates whether a cell has been visited or not
        :return: cell with minimal distance to the target
        """
        min_dist = np.inf
        min_dist_cell = (0, 0)

        for fst, _ in enumerate(utitliy_grid):
            for snd, _ in enumerate(utitliy_grid[fst]):
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
    - 'avg_velocity_pedestrian', float value for avg. pedestrian speed in cell units per iteration

    Number of pedestrian entries needs to match.

    :param scenario_file: Path to .csv file
    :raises ValueError: If number of values for pedestrians do not match
    :return: CellularAutomaton object matching the scenario file configuration
    """
    df = pd.read_csv(scenario_file, delimiter=';')

    grid_size = make_tuple(df['grid_size'][0])
    obstacle_positions = df['initial_position_obstacles'].dropna()
    target_positions = df['position_target_zone'].dropna()
    pedestrian_positions = df['initial_position_pedestrian'].dropna()
    pedestrian_speeds = df['avg_velocity_pedestrian'].dropna()

    if len(pedestrian_positions) != len(pedestrian_speeds):
        raise ValueError('Need same amount of entries for all pedestrian values')

    my_cellular_automaton = CellularAutomaton(grid_size)

    for obstacle_position in obstacle_positions:
        my_cellular_automaton.add_obstacle(make_tuple(obstacle_position))

    for target_position in target_positions:
        my_cellular_automaton.add_target(make_tuple(target_position))

    for pedestrian_position, speed in zip(pedestrian_positions, pedestrian_speeds):
        my_cellular_automaton.add_pedestrian(make_tuple(pedestrian_position), float(speed))

    return my_cellular_automaton
