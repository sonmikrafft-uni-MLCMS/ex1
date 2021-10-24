from tkinter import font
from tkinter.constants import ANCHOR, N
from helpers.cellular_automaton import CellularAutomaton, CellState
import tkinter as tk
import pandas as pd
from ast import literal_eval as make_tuple


def fill_from_scenario_file(scenario_file: str) -> CellularAutomaton:
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
        self.add_dynamic_elements()

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

    def add_dynamic_elements(self):
        btn_start_simulation = tk.Button(self.myCanvas, text = "Start Simulation", font="Times 12 bold", command=self.start_simulation)
        btn_start_simulation.place(x=600, y=180, anchor=N)

        btn_reset_simulation = tk.Button(self.myCanvas, text = "Reset Simulation", font="Times 12 bold", command=self.reset_simulation)
        btn_reset_simulation.place(x=800, y=180, anchor=N) 

        btn_simulation_settings = tk.Button(self.myCanvas, text = "Simulation Settings", font="Times 12 bold", command=self.simulation_settings)
        btn_simulation_settings.place(x=610, y=240, anchor=N)  

    def redraw_grid(self, n_current_time_step):
        #TODO: redraw the GUI for current timestamp
        pass
        #display current time stamp in gui
        self.myCanvas.create_text(700, 200, text=f'Current Time Step: {n_current_time_step}', font="Times 15 bold", anchor=tk.N)   

    def simulation_settings(self):
        #TODO: gets user input for start and end -> Missing: 1. Run internally whole simulation 2. access position history array 3. display 
        master = tk.Tk()
        master.title("Specifiy your Simulation Time Frame ")
        master.geometry("375x80")

        tk.Label(master, text="Start Time Step").grid(row=0)
        tk.Label(master, text="End Time Step").grid(row=1)
        
        entry1 = tk.Entry(master)
        entry2 = tk.Entry(master)

        entry1.grid(row=0, column=1)
        entry2.grid(row=1, column=1)

        tk.Button(master, text='Start Simulation', command=lambda:[self.start_specified_simulation(entry1.get(),entry2.get()), master.destroy()]).grid(row=3, column=0, sticky=tk.W, pady=8)
        tk.Button(master, text='Cancel', command= master.destroy).grid(row=3, column=1, sticky=tk.W, pady=8)        

        master.mainloop()

    def start_specified_simulation(self, n_start_time, n_end_time):
        #TODO: write fct to run simulation from user input time stamp till user given time step
        print(n_start_time)
        print(n_end_time)

    def start_simulation(self):
        #TODO: write fct to run and visualize whole simulation from start to end 
        pass
    
    def reset_simulation(self):
        #TODO: write fct to rerun the simulation from beginning to end
        pass

    def setup_container(self, root):
        self.container = tk.Frame(root)
        self.container.pack()

    def setup_canvas(self):
        window_width = 1000
        window_height = 1000

        self.myCanvas = tk.Canvas(self.container, width=window_width, height=window_height, highlightthickness=0, background="grey")
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
