from enum import Enum
import numpy as np


class CellState(Enum):
    EMPTY = 'E'
    PEDESTRIAN = 'P'
    OBSTACLE = 'O'
    TARGET = 'T'


class CellularAutomaton():
    def __init__(self, grid_size: tuple[int, int]):
        """
        Creates cellular automaton by setting up an empty grid of states and utilities.

        :param grid_size: Tuple of 2D grid dimensions
        """
        self.grid = np.full(grid_size, CellState.EMPTY)
        self.utilities = np.zeros(grid_size)

    def add(self, what: CellState, pos_idx: tuple[int, int]):
        """
        Fill a yet empty cell with either a pedestrian, an obstace, or a target.

        :param what: One of CellState to indicate what to fill
        :param pos_idx: Tuple of 2D index on which cell to fill
        :raises ValueError: If cell is not empty
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        self._check_empty(pos_idx)
        self.grid[(pos_idx)] = what

    def _check_valid_idx(self, pos_idx: tuple[int, int]):
        """
        Checks if we can access the grid with the given indices.

        :param pos_idx: Index that we want to access, specified as 2D tuple
        :raises IndexError: If given index is invalid, either because index is negative or out of bounds
        """
        if not all((idx < grid) and (idx >= 0) for (idx, grid) in zip(pos_idx, self.grid.shape)):
            raise IndexError(f'Trying to access grid position {pos_idx} but grid is only of size {self.grid.shape}.')

    def _check_empty(self, pos_idx: tuple[int, int]):
        """
        Checks if cell is empty.

        :param pos_idx: Index that we want to check
        :raises ValueError: If cell is not empty
        """
        self._check_valid_idx(pos_idx)
        if self.grid[pos_idx] != CellState.EMPTY:
            raise ValueError(f'Trying to write value into grid position {pos_idx}, but not empty.')
