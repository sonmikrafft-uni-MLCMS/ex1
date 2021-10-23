from enum import Enum
from typing import Optional
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
        self.grid_history = [self.grid.copy()]  # keeps history of grid state along simulated iterations
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

    def _check_valid_idx(self, pos_idx: tuple[int, int], surpess_error: bool = False) -> Optional[bool]:
        """
        Checks if we can access the grid with the given indices.

        :param pos_idx: Index that we want to access, specified as 2D tuple
        :param surpress_error: Flag if to surpress raising an IndexError and return False instead
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

    def visualize_grid(self) -> None:
        """
        Visualizes the current state grid by printing it on the console nicely.
        """
        vfunc = np.vectorize(cell_state_to_visualized)
        visualized_grid = vfunc(self.grid)
        output_str = '[' + ']\n['.join(['  '.join([str(cell) for cell in row]) for row in visualized_grid]) + ']'
        print(output_str)

    def simulate_next_n(self, n: int) -> None:
        """
        Simulate next n steps by calling `simulate_next` n-times.

        :param n: Number of iterations to simulate
        """
        for _ in range(n):
            self.simulate_next()

    def simulate_next(self) -> None:
        """
        Propogate states of pedestrian one forward and add new grid state into the history.

        Pedestrians move to the cell with the best utility value in their direct surrounding.
        Only cells that are no obstacles are considered.
        """
        next_grid = self.grid.copy()

        # Iterate over current grid
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                curr_idx = (row, col)
                # check if pedestrian
                if self.grid[curr_idx] == CellState.PEDESTRIAN:
                    surrounding_idx = self._get_surrounding_idx(curr_idx)

                    # look around and keep track of cell with best utility
                    best_utility = self.utilities[curr_idx]
                    best_idx = curr_idx
                    for potential_next_idx in surrounding_idx:
                        if self.grid[potential_next_idx] in [CellState.OBSTACLE, CellState.PEDESTRIAN]:
                            continue
                        if self.utilities[potential_next_idx] < best_utility:
                            best_utility = self.utilities[potential_next_idx]
                            best_idx = potential_next_idx

                    # nothing to do if already on best cell
                    if best_idx == curr_idx:
                        continue

                    # otherwise propagate forward
                    next_grid[row, col] = CellState.EMPTY
                    if not self.grid[best_idx] == CellState.TARGET:
                        next_grid[best_idx] = CellState.PEDESTRIAN

        self.grid_history.append(next_grid.copy())
        self.grid = next_grid

    def _get_surrounding_idx(self, pos_idx: tuple[int, int]) -> set[tuple[int, int]]:
        """ Given the current position in the 2D grid as a tuple of indices, return a set of valid surrounding
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
