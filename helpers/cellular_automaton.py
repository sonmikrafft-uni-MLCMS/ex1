from enum import Enum
import numpy as np


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


class CellularAutomaton():
    def __init__(self, grid_size: tuple[int, int]) -> None:
        """
        Creates cellular automaton by setting up an empty grid of states and utilities.

        :param grid_size: Tuple of 2D grid dimensions
        """
        self.grid = np.full(grid_size, CellState.EMPTY)
        self.utilities = np.zeros(grid_size)

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

    def _check_valid_idx(self, pos_idx: tuple[int, int]) -> None:
        """
        Checks if we can access the grid with the given indices.

        :param pos_idx: Index that we want to access, specified as 2D tuple
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        if not all((idx < grid) and (idx >= 0) for (idx, grid) in zip(pos_idx, self.grid.shape)):
            raise IndexError(f'Trying to access grid position {pos_idx} but grid is only of size {self.grid.shape}.')

    def _check_empty(self, pos_idx: tuple[int, int]) -> None:
        """
        Checks if cell is empty.

        :param pos_idx: Index that we want to check
        :raises ValueError: If cell is not empty
        """
        self._check_valid_idx(pos_idx)
        if self.grid[pos_idx] != CellState.EMPTY:
            raise ValueError(f'Trying to write value into grid position {pos_idx}, but not empty.')

    def visualize_grid(self) -> None:
        """
        Visualizes the current state grid by printing it on the console nicely.
        """
        vfunc = np.vectorize(cell_state_to_visualized)
        visualized_grid = vfunc(self.grid)
        output_str = '[' + ']\n['.join(['  '.join([str(cell) for cell in row]) for row in visualized_grid]) + ']'
        print(output_str)

    def set_utilities(self, grid_size: tuple[int, int]) -> None:
        """
        Set the utilities of each cell according to their respective distance to the closest target

        :param grid_size: Defines the size of the utility array
        """
        # dev: self.utilities can be initialized with the following line in the init method
        self.utilities = np.full(grid_size, np.inf)

        # set target cells to zero
        targets = zip(*np.where(self.grid == CellState.TARGET))
        self.utilities[targets] = 0

        for target in range(targets):
            self.set_dijkstra_for_one_target(target)

        print(self.utilities)

    def set_dijkstra_for_one_target(self, target: tuple[int, int]) -> None:
        """
        Set utilities of each cell with Dijkstra's algorithm to find minimal distance to a single target

        Adapted from: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/ (24/10/2021)
        """
        # initialize a distance array (computed distances to the target) and an array of visited cells
        dist = np.full_like(self.utilities, np.inf)
        dist[target] = 0
        visited = np.full_like(self.utilities, False)

        for _ in range(self.utilities):
            u = self.find_minimal_distance_cell(dist, visited)
            visited[u] = True

        # TODO find neighbors
        neighbors = []

        for v in range(neighbors):
            if not visited[v] and dist[v] > dist[u] + self.compute_distance(u, v):
                dist[v] = dist[u] + self.compute_distance(u, v)

                # update a cell's utility if the cell is closer to this target than a previous target
                if self.utilities[v] > dist[v]:
                    self.utilities[v] = dist[v]

    def _find_minimal_distance_cell(self, dist, visited) -> tuple[int, int]:
        """
        Determine cell with minimal distance to the target that will be visited next
        :param dist: array of distances between cells and the target
        :param visited: array that indicates whether a cell has been visited or not
        :return: cell with minimal distance to the target
        """
        min_dist = np.inf

        for u in range(self.utilities):
            if dist[u] < min_dist and not visited[u]:
                min_dist = dist[u]
                min_dist_cell = u
        return min_dist_cell

    def _compute_distance(self, u: tuple[int, int], v: tuple[int, int]) -> float:
        """
        Compute the euclidean distance between 2 cells
        :param u: first cell
        :param v: second cell
        :return: euclidean distance between 2 cells
        """
        return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)

