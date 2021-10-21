from typing import Optional


class CellularAutomaton():
    def __init__(self, config_file: Optional[str] = None, grid_size: Optional[tuple[int, int]] = None):
        if config_file is None and grid_size is None:
            raise ValueError('Either need "config_file" or "grid_size" to be specified.')

        if config_file is not None:
            # parse config file
            pass
        else:
            # create empty grid according to grid_size
            pass
