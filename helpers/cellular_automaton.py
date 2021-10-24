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

    def _get_distance_based_utility_grid(self) -> np.ndarray:
        """
        Get utility grid filled by euclidean distance to closest target.

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

    def visualize_grid(self) -> None:
        """
        Visualizes the current state grid by printing it on the console nicely.
        """
        vfunc = np.vectorize(cell_state_to_visualized)
        visualized_grid = vfunc(self.grid)
        output_str = '[' + ']\n['.join(['  '.join([str(cell) for cell in row]) for row in visualized_grid]) + ']'
        print(output_str)
