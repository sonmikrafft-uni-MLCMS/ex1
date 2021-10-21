from helpers.cellular_automaton import CellularAutomaton, CellState
import tkinter as tk


# def fill_from_scenario_file(scenario_file: str):
#     # create pandas dataframe
#     data = pd.read_csv(scenario_file, delimiter=';')
#     data = data.loc[0]

#     # extract values from dataframe
#     grid_size = data.grid_size

#     # grid_x, grid_y = get_values_from_csv_array(self, grid_size)

#     self.grid_rows, self.grid_columns = 50, 50
#     pedestrian_id = data.pedestrian_id
#     pedestrain_position = data.initial_position_pedestrian
#     print(self.grid_rows)


# def get_values_from_csv_array(self, array_name):
#     list_of_array_values = list(array_name[1:-1].split(", "))
#     return list_of_array_values[0], list_of_array_values[1]

#     # draw the grid according to scenario details


# def add_text_descriptors(self):
#     self.myCanvas.create_text(700, 10, text='Scenario Details:', font="Times 15 bold", anchor=tk.N)
#     self.myCanvas.create_text(700, 35, text='Displayed File: ' +
#                               str(self.scenario_file), font="Times 10 bold", anchor=tk.N)
#     self.myCanvas.create_text(700, 70, text='Grid Size: ' + str(self.grid_rows) +
#                               " x " + str(self.grid_columns), font="Times 12 bold", anchor=tk.N)
#     self.myCanvas.create_text(700, 90, text='Number of Pedestrians: ' +
#                               str(self.pedestrian_counter), font="Times 12 bold", anchor=tk.N)
#     self.myCanvas.create_text(700, 110, text='Number of Obstacle Fields: ' +
#                               str(self.obstacle_counter), font="Times 12 bold", anchor=tk.N)
#     self.myCanvas.create_text(700, 130, text='Number of Target Fields: ' +
#                               str(self.pedestrian_counter), font="Times 12 bold", anchor=tk.N)


class GUI:
    def __init__(self, root, scenario_file: str):
        # self.scenario_file = scenario_file
        # self.my_cellular_automaton = fill_from_scenario_file(scenario_file)

        # create cellular automaton
        self.my_cellular_automaton = CellularAutomaton((50, 50))
        self.my_cellular_automaton.add(CellState.OBSTACLE, (10, 10))
        self.my_cellular_automaton.add(CellState.PEDESTRIAN, (20, 10))
        self.my_cellular_automaton.add(CellState.TARGET, (30, 10))

        self.setup_container(root)
        self.setup_canvas()
        self.setup_grid()

        # self.pedestrian_counter = 0
        # self.obstacle_counter = 0
        # self.target_counter = 0

        # visualize start state by iterating over our grid
        self.visualize_state()

    def visualize_state(self):
        rows, cols = self.my_cellular_automaton.grid.shape
        for ix in range(0, rows):
            for iy in range(0, cols):
                cell = self.my_cellular_automaton.grid[ix, iy]
                self.add_item_to_grid(ix, iy, cell)

    def add_item_to_grid(self, row, col, cell_state):
        # simple case switch to determine color for visualisation purposes in the grid
        if cell_state == CellState.EMPTY:
            return

        if cell_state == CellState.PEDESTRIAN:
            cell_color = "red"
            # self.pedestrian_counter += 1
        elif cell_state == CellState.OBSTACLE:
            cell_color = "violet"
            # self.obstacle_counter += 1
        elif cell_state == CellState.TARGET:
            cell_color = "yellow"
            # self.target_counter += 1

        # find cell and fill with state dependent color
        item_id = self.gui_rect[row][col]
        self.myCanvas.itemconfig(item_id, fill=cell_color)

    # self.add_text_descriptors()

    # pick scenario csv file
    # self.scenario_file = "scenario_0.csv"

    # define grid structures
    # self.rect = {}

    # gets csv scenario file and extract following scenario parameters:
    # general parameters:  grid_size, position_target_zone, initial_position_obstacles
    # person specific parameters: pedestrian_id, initial_position_pedestrian, avg_velocity_pedestrian

    # get the two inner values of a string defined in the csv as following tuple "(first_value, second_value)"

    def setup_container(self, root):
        self.container = tk.Frame(root)
        self.container.pack()

    def setup_canvas(self):
        window_width = 1000
        window_height = 1000

        self.myCanvas = tk.Canvas(self.container, width=window_width, height=window_height, highlightthickness=0)
        self.myCanvas.pack(side="top", fill="both", expand="true")

    def setup_grid(self):
        cell_width = 10
        cell_height = 10

        rows, cols = self.my_cellular_automaton.grid.shape
        self.gui_rect = [[None for _ in range(cols)] for _ in range(rows)]
        for column in range(cols):
            for row in range(rows):
                x1 = column * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                self.gui_rect[row][column] = self.myCanvas.create_rectangle(x1, y1, x2, y2, fill="white", tags="rect")
