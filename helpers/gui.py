from helpers.cellular_automaton import CellState, fill_from_scenario_file
import tkinter as tk


class GUI:
    def __init__(self, root, scenario_file: str):
        self.my_cellular_automaton = fill_from_scenario_file(scenario_file)

        n_rows, n_cols = self.my_cellular_automaton.grid.shape
        n_pedestrians = (self.my_cellular_automaton.grid == CellState.PEDESTRIAN).sum()
        n_obstacles = (self.my_cellular_automaton.grid == CellState.OBSTACLE).sum()
        n_targets = (self.my_cellular_automaton.grid == CellState.TARGET).sum()

        self.setup_container(root)
        self.setup_canvas()
        self.setup_grid(n_rows, n_cols)
        self.add_static_text_descriptors(scenario_file, n_rows, n_cols, n_pedestrians, n_obstacles, n_targets)

        # visualize start state by iterating over our grid
        self.visualize_state()

    def visualize_state(self):
        rows, cols = self.my_cellular_automaton.grid.shape
        for ix in range(0, rows):
            for iy in range(0, cols):
                cell = self.my_cellular_automaton.grid[ix, iy]
                self.add_item_to_grid(ix, iy, cell)

    def add_item_to_grid(self, row, col, cell_state):
        if cell_state == CellState.EMPTY:
            return

        if cell_state == CellState.PEDESTRIAN:
            cell_color = "red"
        elif cell_state == CellState.OBSTACLE:
            cell_color = "violet"
        elif cell_state == CellState.TARGET:
            cell_color = "yellow"
        else:
            cell_color = "white"

        item_id = self.gui_rect[row][col]
        # TODO add one of {'P', 'O', 'T'} as text of rectangle
        self.myCanvas.itemconfig(item_id, fill=cell_color)

    def add_static_text_descriptors(self, scenario_file: str, n_rows, n_cols, n_pedestrians, n_obstacles, n_targets):
        self.myCanvas.create_text(700, 10, text='Scenario Details:', font="Times 15 bold", anchor=tk.N)
        self.myCanvas.create_text(700, 35, text=f'Displayed File: {scenario_file}', font="Times 10 bold", anchor=tk.N)
        self.myCanvas.create_text(700, 70, text=f'Grid Size: {n_rows} x  {n_cols}', font="Times 12 bold", anchor=tk.N)
        self.myCanvas.create_text(
            700, 90, text=f'Number of Pedestrians: {n_pedestrians}', font="Times 12 bold", anchor=tk.N)
        self.myCanvas.create_text(
            700, 110, text=f'Number of Obstacle Fields: {n_obstacles}', font="Times 12 bold", anchor=tk.N)
        self.myCanvas.create_text(
            700, 130, text=f'Number of Target Fields: {str(n_targets)}', font="Times 12 bold", anchor=tk.N)

    def setup_container(self, root):
        self.container = tk.Frame(root)
        self.container.pack()

    def setup_canvas(self):
        window_width = 1000
        window_height = 1000

        self.myCanvas = tk.Canvas(self.container, width=window_width, height=window_height, highlightthickness=0)
        self.myCanvas.pack(side="top", fill="both", expand="true")

    def setup_grid(self, n_rows: int, n_cols: int):
        cell_width = 10
        cell_height = 10

        self.gui_rect = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for column in range(n_cols):
            for row in range(n_rows):
                x1 = column * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                self.gui_rect[row][column] = self.myCanvas.create_rectangle(x1, y1, x2, y2, fill="white", tags="rect")
