from posixpath import relpath
from tkinter import font
from tkinter.filedialog import askopenfilename
from tkinter.constants import ANCHOR, N
from helpers.cellular_automaton import CellState, CellularAutomaton,  fill_from_scenario_file
import pandas as pd
import tkinter as tk
import os

class GUI:
    def __init__(self, root):

        n_rows, n_cols = 0, 0
        n_pedestrians = 0
        n_obstacles = 0
        n_targets = 0

        self.setup_container(root)
        self.setup_canvas()
        self.add_static_text_descriptors()
        self.add_dynamic_text_descriptors("No Scenario File Selected", n_rows, n_cols, n_pedestrians, n_obstacles, n_targets)
        self.add_dynamic_elements()
        self.time_step_label = None

    def update_scenario(self, scenario_file: str):
        
        #Read in new scenariofile 
       
        #rel_scenario_file_path = "\mlcs-ex1-modeling-crowds-2\scenario_0.csv"

        self.my_cellular_automaton = fill_from_scenario_file(scenario_file=scenario_file)
        
        #Prepare dynamic text labels for new update values
        self.delete_dynamic_text_descriptors()

        #Assign Values from scenario
        self.n_rows, self.n_cols = self.my_cellular_automaton.state_grid.shape
        n_pedestrians = (self.my_cellular_automaton.state_grid == CellState.PEDESTRIAN).sum()
        n_obstacles = (self.my_cellular_automaton.state_grid == CellState.OBSTACLE).sum()
        n_targets = (self.my_cellular_automaton.state_grid == CellState.TARGET).sum()
        print(self.n_cols)
        #add new values to canvas for visualisation
        self.add_dynamic_text_descriptors(scenario_file, self.n_rows, self.n_cols, n_pedestrians, n_obstacles, n_targets)
        
        # visualize start state by iterating over our state_grid
        self.setup_grid(self.n_rows, self.n_cols)
        self.visualize_state()    

    def delete_dynamic_text_descriptors(self):
        
        #Delete old text labels before update 
        self.myCanvas.delete(self.file_label)
        self.myCanvas.delete(self.grid_size_label)
        self.myCanvas.delete(self.n_pedestrians_label)
        self.myCanvas.delete(self.n_obstacles_label)
        self.myCanvas.delete(self.n_targets_label)

    def visualize_state(self):
        rows, cols = self.my_cellular_automaton.state_grid.shape
        for ix in range(0, rows):
            for iy in range(0, cols):
                cell = self.my_cellular_automaton.state_grid[ix, iy]
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

    def add_static_text_descriptors(self):

        #Adds text to GUI in order to describe the choosen scenario
        self.title_label = self.myCanvas.create_text(700, 10, text='Scenario Details:', font="Times 15 bold", anchor=tk.N)
        self.settings_label = self.myCanvas.create_text(620, 290, text='Simulation Settings:', font="Times 15 bold", anchor=tk.N)

        #Visualize Caption to explain colored rectangles as used in grid visualization  
        self.myCanvas.create_rectangle(540,160,555,175, fill='red')
        self.myCanvas.create_text(
            597, 157, text=f'Pedestrian', font="Times 12 bold", anchor=tk.N)

        self.myCanvas.create_rectangle(645,160,660,175, fill='violet')
        self.myCanvas.create_text(
            695, 157, text=f'Obstacle', font="Times 12 bold", anchor=tk.N)

        self.myCanvas.create_rectangle(740,160,755,175, fill='yellow')
        self.myCanvas.create_text(
            785, 157, text=f'Target', font="Times 12 bold", anchor=tk.N)


    def add_dynamic_text_descriptors(self, scenario_file: str, n_rows, n_cols, n_pedestrians, n_obstacles, n_targets):
        rel_scenario_file_path = os.path.basename(scenario_file)
        self.file_label = self.myCanvas.create_text(700, 35, text=f'Displayed File: {rel_scenario_file_path}', font="Times 10 bold", anchor=tk.N)
        self.grid_size_label = self.grid_size = self.myCanvas.create_text(700, 70, text=f'Grid Size: {n_rows} x  {n_cols}', font="Times 12 bold", anchor=tk.N)
        self.n_pedestrians_label = self.myCanvas.create_text(
            700, 90, text=f'Number of Pedestrians: {n_pedestrians}', font="Times 12 bold", anchor=tk.N)
        self.n_obstacles_label=self.myCanvas.create_text(
            700, 110, text=f'Number of Obstacle Fields: {n_obstacles}', font="Times 12 bold", anchor=tk.N)
        self.n_targets_label=self.myCanvas.create_text(
            700, 130, text=f'Number of Target Fields: {str(n_targets)}', font="Times 12 bold", anchor=tk.N)

    def add_dynamic_elements(self):

        #Adds Buttons to GUI and connects specific commands
        btn_start_simulation = tk.Button(self.myCanvas, text="Start Simulation",
                                         font="Times 12 bold", command=self.start_simulation)
        btn_start_simulation.place(x=600, y=200, anchor=N)

        btn_reset_simulation = tk.Button(self.myCanvas, text="Reset Simulation",
                                         font="Times 9 bold", command=self.reset_simulation)
        btn_reset_simulation.place(x=740, y=240, anchor=N)

        btn_Choose_filename = tk.Button(self.myCanvas, text="Choose Filename",
                                            font="Times 10 bold", command=self.get_scenario_path)
        btn_Choose_filename.place(x=900, y=25, anchor=N)

        btn_previous_simulation_step = tk.Button(self.myCanvas, text="Previous Step",
                                            font="Times 9 bold", command=self.visualize_previous_step)
        btn_previous_simulation_step.place(x=580, y=240, anchor=N)

        btn_next_simulation_step = tk.Button(self.myCanvas, text="Next Step",
                                            font="Times 9 bold", command=self.visualize_next_step)
        btn_next_simulation_step.place(x=655, y=240, anchor=N)

        #Initialise and Place Checkbox to GUI for Simulations parameters Obstacle Avoidance and Target Absorbation
        
        self.cbx_val_obstacle_avoidance = tk.BooleanVar()
        self.cbx_val_obstacle_avoidance.set(True)

        cbx_obstacle_avoidance = tk.Checkbutton(self.myCanvas, text="Obstacle Avoidance", variable=self.cbx_val_obstacle_avoidance, command=self.cbx_update_obstacle_avoidance)                       
        cbx_obstacle_avoidance.place(x=607, y=330, anchor=N)                                          

        self.cbx_val_target_absorbation = tk.BooleanVar()
        self.cbx_val_target_absorbation.set(True)

        cbx_target_absorbation = tk.Checkbutton(self.myCanvas, text="Target Absorbation",variable=self.cbx_val_target_absorbation, command=self.cbx_update_target_absorbation)                                      
        cbx_target_absorbation.place(x=605, y=360, anchor=N)                                   
        

    def get_scenario_path(self):
        #Chosoe a scenario file
        scenario_file_path= askopenfilename()
        print(scenario_file_path)
        self.update_scenario(scenario_file = scenario_file_path)


    def cbx_update_obstacle_avoidance(self):
        self.bool_obstacle_avoidance = self.cbx_val_obstacle_avoidance.get()
        
        

    def cbx_update_target_absorbation(self):
        self.bool_target_absorbation = self.cbx_val_target_absorbation.get()


    def visualize_previous_step(self):
        
        if self.current_grid_id != 1:
            previous_id = self.current_grid_id - 1
            self.setup_grid(self.n_rows, self.n_cols)
    
            for grid in range(previous_id):
                self.visualize_grid_state(self.my_cellular_automaton.state_grid_history[grid])
            
            self.update_time_step_label(previous_id)


    def visualize_next_step(self):

        if self.current_grid_id != self.last_grid_id:
            next_id = self.current_grid_id + 1
            self.setup_grid(self.n_rows, self.n_cols)
    
            for grid in range(next_id):
                self.visualize_grid_state(self.my_cellular_automaton.state_grid_history[grid])
            
            self.update_time_step_label(next_id)


    def update_time_step_label(self, current_grid):

        self.current_grid_id = current_grid

        if self.time_step_label is not None:
            self.myCanvas.delete(self.time_step_label)
        self.time_step_label = self.myCanvas.create_text(
            715, 205, text=f'Time Step: {current_grid} of {self.last_grid_id}', font="Times 10 bold", anchor=tk.N)

    def start_simulation(self):

        self.cbx_update_obstacle_avoidance()
        self.cbx_update_target_absorbation()

        self.my_cellular_automaton.simulate_until_no_change(smart_obstacle_avoidance=self.bool_obstacle_avoidance, target_absorbs=self.bool_target_absorbation)
        simulated_grid_states = self.my_cellular_automaton.state_grid_history
        self.last_grid_id = max(simulated_grid_states, key=int)

        self.update_time_step_label(self.last_grid_id)
        
        self.setup_grid(self.n_rows, self.n_cols)
        for current_grid in simulated_grid_states:
            self.visualize_grid_state(simulated_grid_states[current_grid])
            
    def visualize_grid_state(self, grid):
        rows, cols = grid.shape
        for ix in range(0, rows):
            for iy in range(0, cols):
                cell = grid[ix, iy]
                self.add_item_to_grid(ix, iy, cell)          

    def reset_simulation(self):
        self.setup_grid(self.n_rows, self.n_cols)
        self.visualize_grid_state(self.my_cellular_automaton.state_grid_history[0])
        self.update_time_step_label(1)

    def setup_container(self, root):
        self.container = tk.Frame(root)
        self.container.pack()

    def setup_canvas(self):
        window_width = 1000
        window_height = 1000


        self.myCanvas = tk.Canvas(self.container, width=window_width, height=window_height,
                                  highlightthickness=0, background="grey")
        self.myCanvas.pack(side="top", fill="both", expand="true")

    def setup_grid(self, n_rows: int, n_cols: int):
        cell_width = 20
        cell_height = 20

        self.gui_rect = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for column in range(n_cols):
            for row in range(n_rows):
                x1 = column * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                self.gui_rect[row][column] = self.myCanvas.create_rectangle(x1, y1, x2, y2, fill="white", tags="rect")
