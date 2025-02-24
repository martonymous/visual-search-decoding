from psychopy import visual, event, core, monitors
from psychopy.iohub import launchHubServer
from psychopy.core import getTime, wait
from titta import Titta

import os
import sys
import random
import numpy as np
import pandas as pd
import csv
import datetime
import uuid
import time

class Experiment:
    def __init__(self, num_trials=5, eye_tracking=False):
        # =================== Setup ======================
        # Monitor configuration
        self.monitor = monitors.Monitor("testMonitor", width=53.0, distance=60.0)  # Adjust for your setup
        self.win = visual.Window(size=(1920, 1080), monitor=self.monitor, units="deg", fullscr=False, color="gray")
        self.mouse = event.Mouse(win=self.win)
        self.eye_tracking = eye_tracking

        """ Eye-tracking setup """
        # simulation mode
        if not eye_tracking:
            self.iohub_config = {
                "eyetracker.hw.mouse.EyeTracker": {},
            }
            self.io = launchHubServer(**self.iohub_config)
            self.eyetracker = self.io.devices.eyetracker
        # tobii eyetracker
        else:
            self.et_settings = Titta.get_defaults('Tobii Pro X3-120')
            self.eyetracker = Titta.Connect(self.et_settings)
            self.eyetracker.init()
            self.eyetracker.calibrate(self.win)

        # Object parameters
        self.grid_x = 7
        self.grid_y = 4
        self.num_objects = self.grid_x * self.grid_y
        self.grid_scaler_x = 13              # Grid spacing
        self.grid_scaler_y = 12              # Grid spacing
        self.jitter = 1.8                    # Position jitter
        self.object_scaler = 1.0             # Object size
        self.center_scaler = 1.4             # Center offset
        self.center_x = 0.5  * self.object_scaler * self.center_scaler
        self.center_y = 0.75 * self.object_scaler * self.center_scaler
        
        # Experiment parameters
        self.task_duration = 60.0             # Duration of search task in each trial
        self.target_present = True           # Whether the target is present
        self.color_difference = 100          # Degree of color differences
        self.num_trials = 100                  # Number of trials in the experiment

        # intiialize Target attributes
        self.target_attributes = {
            "color": [255, 0, 0],
            "size": 2.5,
            "masked": False,
            "target": True,
            "orientation": 0,
        }

        # Generate random distractor attributes
        self.base_color = [127, 127, 127]    # RGB for gray
        self.saturated_colors = [
            (255, 0, 0),                # Red
            (0, 255, 0),                # Green
            (0, 0, 255),                # Blue
            (255, 255, 0),              # Yellow
            (255, 0, 255),              # Magenta
            (0, 255, 255),              # Cyan
            # (255, 127, 0),              # Orange
            # (127, 0, 255),              # Purple
            # (0, 255, 127),              # Spring Green
            # (255, 0, 127),              # Deep Pink
            # (127, 255, 0),              # Chartreuse
            # (0, 127, 255),              # Sky Blue
        ]

        # discrete values for size and orientation
        self.sizes = [2.0, 2.5, 3.0, 3.5, 4.0]
        self.orientations = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        # range values for cointinuous attributes
        self.size_range = [3.0, 4.25]
        self.orientation_range = [-90, 90]
        self.convex_range = [-1.5, 2.0]

    @staticmethod
    def get_convex_curve(start, end, convexity=0.9, direction=1, num_points=10):
        """
        Generate a convex curve between two points using a quadratic Bézier curve.

        Parameters:
            start (tuple): (x, y) coordinates of the start vertex.
            end (tuple): (x, y) coordinates of the end vertex.
            convexity (float): How convex the curve should be (0 = straight line).
            direction (int): 1 for outward convexity, -1 for inward convexity.
            num_points (int): Number of points along the curve.

        Returns:
            List of (x, y) points forming the convex curve.
        """
        start = np.array(start)
        end = np.array(end)

        # Midpoint of the line segment
        mid = (start + end) / 2

        # Compute perpendicular vector
        perp_vector = np.array([-(end[1] - start[1]), end[0] - start[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)  # Normalize

        # Move midpoint in the perpendicular direction by 'convexity' factor
        control = mid + perp_vector * convexity * direction

        # Quadratic Bézier curve formula
        t_vals = np.linspace(0, 1, num_points)
        curve_points = [(1 - t)**2 * start + 2 * (1 - t) * t * control + t**2 * end for t in t_vals]

        return curve_points

    def create_L_stimulus(self, attributes, pos):
        """
        Create a single object based on target and distractor attributes. The object consists
        of a set of 10 vertices that form an irregular shape. The parameters of the shape are 
        determined by the attributes provided, and allow parametric creation of the object.
        """

        object_vertices = [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 3.0),
            (1.0, 3.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (1.0, 0.0)
        ]
        narrow_vertices = [
            (0.0, 0.0),
            (0.25, 1.0),
            (0.25, 2.0),
            (0.25, 3.0),
            (.75, 3.0),
            (.75, 2.0),
            (1.0, .75),
            (2.0, .75),
            (2.0, 0.25),
            (1.0, 0.25)
        ]


        # Randomly use some of the narrow vertices
        if attributes["target"] == False:
            x = random.randint(0, 5)
            random_vertex_indices = random.sample(range(0, len(object_vertices)), x)
            object_vertices = [narrow_vertices[i] if i in random_vertex_indices else object_vertices[i] for i in range(len(object_vertices))]

        # center the object (for rotation accuracy)
        object_vertices = [(x - 1.0, y - 1.5) for x, y in object_vertices]

        # Scale the object based on the size attribute
        object_vertices = [(x * attributes["size"], y * attributes["size"]) for x, y in object_vertices]

        # Create a ShapeStim object with the specified attributes
        stim = visual.ShapeStim(
            win=self.win,
            vertices=object_vertices,
            fillColor=attributes["color"],
            lineColor=None,
            pos=pos,
            ori=attributes["orientation"],
            opacity=attributes["opacity"],
        )

        return [stim]

    def create_diamond_stimulus(self, attributes, pos):
        """
        Create a diamond-shaped stimulus with elongated proportions. 
        Lines extend from each corner of the diamond to a central point.
        The exact parameters of the shape are determined by the attributes provided.
        """

        convexity = attributes.get("convexity", .2)  # Adjust curvature
        direction = attributes.get("direction", 1)   # 1 for outward, -1 for inward
        elongation_x = attributes.get("elongation_x", 1.2)
        elongation_y = attributes.get("elongation_y", .9)
        center_point = attributes.get("center_point", (0.0, 0.0))
        missing_vertex = attributes.get("missing_vertex", None)

        diamond_vertices = [
            (0.0, 1.5),  # Top
            (.75, 0.0),  # Right
            (0.0, -1.0), # Bottom
            (-.75, 0.0), # Left
        ]
        diamond_vertices = [(x * elongation_x, y * elongation_y) for x, y in diamond_vertices]
        diamond_vertices = [(x * attributes["size"] * self.object_scaler, y * attributes["size"] * self.object_scaler) for x, y in diamond_vertices]

        # Center the object for rotations
        # all_vertices = [(x - 0.0, y - 0.0) for x, y in all_vertices]

        shape_points = []
        for i in range(len(diamond_vertices)):
            if missing_vertex is not None:
                if i == missing_vertex:
                    shape_points.append(center_point)
                    continue
                if random.random() >= 0.0 and (i + 1) % len(diamond_vertices) == missing_vertex:
                    shape_points.append(center_point)
                    continue
            
            start = diamond_vertices[i]
            end = diamond_vertices[(i + 1) % len(diamond_vertices)]
            curve_points = self.get_convex_curve(start, end, convexity, direction)
            _ = [shape_points.append(x) for x in curve_points]

        color = [x/255 for x in attributes["color"]]
        shape = visual.ShapeStim(
            win=self.win,
            vertices=shape_points,
            lineColor="black",
            lineWidth=2,
            fillColor=color,
            pos=pos,
            ori=attributes["orientation"],
        )

        # Create lines connecting each vertex to the center
        line_stims = []
        for vertex in diamond_vertices:
            if missing_vertex is not None and diamond_vertices.index(vertex) == missing_vertex:
                continue
            line_stim = visual.ShapeStim(
                win=self.win,
                pos=pos,
                vertices=[center_point, vertex],
                lineColor="black",
                lineWidth=2,
                ori=attributes["orientation"],
            )
            line_stims.append(line_stim)

        return [shape] + line_stims

    def create_masks(self, attributes, pos):
        stimuli = []

        # Create a rectangle mask using a polygon
        mask_vertices = [
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5)
        ]
        mask_orientation = random.choice([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345])
        mask = visual.ShapeStim(
            win=self.win,
            vertices=mask_vertices,
            fillColor=self.win.color,  # Match the window's background color
            lineColor=None,       # No border for the mask
            pos=pos,
            ori=mask_orientation,  # Set random orientation for the mask
            opacity=1.0,
        )
        stimuli.append(mask)  # Add the mask to the list of stimuli

        return stimuli

    def create_circular_noise_mask(self, radius, square_size, pos=(0, 0)):
        """
        Creates and draws a checkered noise mask that fills a circular area.

        Parameters:
        - win: The PsychoPy window where the mask will be drawn.
        - radius: The radius of the circular area to fill.
        - square_size: The size of the individual squares in the checkered pattern.
        - pos: The position of the circular mask's center.
        """
        # Calculate the number of squares in each direction based on the circle's radius
        num_squares = int((2 * radius) / square_size) + 1

        # Generate the grid positions
        squares = []
        for x in range(-num_squares, num_squares + 1):
            for y in range(-num_squares, num_squares + 1):
                # Calculate square center
                square_x = x * square_size + pos[0]
                square_y = y * square_size + pos[1]
                
                # Check if the square's center is within the circular radius
                if (square_x - pos[0])**2 + (square_y - pos[1])**2 <= radius**2:
                    # Randomly generate a color for the square
                    ranvar = random.random()
                    if ranvar < 0.4:
                        if random.random() < 0.5:
                            color = "black"
                        else:
                            color = "white"
                    elif ranvar < 0.7:
                        color = random.choice(self.saturated_colors)
                    else:
                        var = random.random()
                        if var < 0.33:
                            color = [random.uniform(0.25, 1), random.uniform(0, .5), random.uniform(0, .5)]
                        elif var < 0.66:
                            color = [random.uniform(0, .5), random.uniform(0.25, 1), random.uniform(0, .5)]
                        else:
                            color = [random.uniform(0, .5), random.uniform(0, .5), random.uniform(0.25, 1)]
                    
                    # Create a square stimulus
                    square = visual.Rect(
                        win=self.win,
                        width=square_size,
                        height=square_size,
                        fillColor=color,
                        lineColor=None,
                        pos=(square_x, square_y),
                        opacity=2.0
                    )
                    squares.append(square)

        for square in squares:
            square.draw()

    def create_checkered_noise_mask(self, square_size):
        """
        Creates and draws a checkered noise mask that fills a circular area.

        Parameters:
        - win: The PsychoPy window where the mask will be drawn.
        - square_size: The size of the individual squares in the checkered pattern.
        """
        # Calculate the number of squares in each direction based on the display size
        num_squares_x = int(self.win.size[0] / square_size) + 1
        num_squares_y = int(self.win.size[1] / square_size) + 1

        # Generate the grid positions
        squares = []

        for x in range(-num_squares_x, num_squares_x + 1):
            for y in range(-num_squares_y, num_squares_y + 1):
                # Calculate square center
                square_x = x * square_size
                square_y = y * square_size

                # Randomly generate a color for the square
                ranvar = random.random()
                if ranvar < 0.4:
                    if random.random() < 0.5:
                        color = "black"
                    else:
                        color = "white"
                elif ranvar < 0.7:
                    color = random.choice(self.saturated_colors)
                else:
                    var = random.random()
                    if var < 0.33:
                        color = [random.uniform(0.25, 1), random.uniform(0, .5), random.uniform(0, .5)]
                    elif var < 0.66:
                        color = [random.uniform(0, .5), random.uniform(0.25, 1), random.uniform(0, .5)]
                    else:
                        color = [random.uniform(0, .5), random.uniform(0, .5), random.uniform(0.25, 1)]
                
                # Create a square stimulus
                square = visual.Rect(
                    win=self.win,
                    width=square_size,
                    height=square_size,
                    fillColor=color,
                    lineColor=None,
                    pos=(square_x, square_y),
                    opacity=2.0
                )
                squares.append(square)

        for square in squares:
            square.draw()

    def generate_stimuli(self, stimulus_function, target_present = True):
        """Generate target and distractor stimuli arranged in a grid."""
        target = []
        stimuli = []
        stimulus_data = []

        # Generate grid positions
        grid_positions = [
            [(x + 0.5 - (self.grid_x / 2)) * self.grid_scaler_x, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler_y]
            for x in range(self.grid_x)
            for y in range(self.grid_y)
        ]
        random.shuffle(grid_positions)

        for i, coordinate_pos in enumerate(grid_positions):
            pos = [None, None]
            for j, val in enumerate(coordinate_pos):
                pos[j] = val + random.uniform(-self.jitter, self.jitter)

            if self.target_present and i == 0:
                color = random.choice(self.saturated_colors)
                self.target_color = [max(min((int(color[x] + random.randint(-self.color_difference, self.color_difference))), 255), 0) for x in range(3)]
                missing_vertex = random.randint(0, 3) if random.random() < 0.5 else None
                target_attributes = {
                    "color": self.target_color,
                    "size": random.uniform(self.size_range[0], self.size_range[1]),
                    "orientation": random.uniform(self.orientation_range[0], self.orientation_range[1]),
                    "convexity": random.uniform(self.convex_range[0], self.convex_range[1]),
                    "center_point": (random.uniform(-self.center_x, self.center_x), random.uniform(-self.center_y, self.center_y)),
                    "missing_vertex": missing_vertex,
                    "masked": False,
                    "target": True}
                stimuli.extend(stimulus_function(target_attributes, pos))
                target.extend(stimulus_function(target_attributes, (0,0)))
                stimulus_data.append({
                    "target_present": self.target_present,
                    "is_target": True,
                    "coordinates": coordinate_pos,
                    "true_position": pos,
                    "color": target_attributes["color"],
                    "size": target_attributes["size"],
                    "orientation": target_attributes["orientation"],
                    "convexity": target_attributes["convexity"],
                    "center_point": target_attributes["center_point"],
                    "missing_vertex": missing_vertex,
                    "masked": False,
                })
            else:
                # do only once if target is not present
                if i == 0:
                    self.target_color = random.choice(self.saturated_colors)
                    fake_target_color = [max(min((int(self.target_color[x] + random.randint(-self.color_difference, self.color_difference))), 255), .0) for x in range(3)]
                    missing_vertex = random.randint(0, 3) if random.random() < 0.5 else None
                    fake_target_attributes = {
                        "color": fake_target_color,
                        "size": random.uniform(self.size_range[0], self.size_range[1]),
                        "orientation": random.uniform(self.orientation_range[0], self.orientation_range[1]),
                        "convexity": random.uniform(self.convex_range[0], self.convex_range[1]),
                        "center_point": (random.uniform(-self.center_x, self.center_x), random.uniform(-self.center_y, self.center_y)),
                        "missing_vertex": missing_vertex,
                        "masked": False,
                        "target": False,
                    }
                    target.extend(stimulus_function(fake_target_attributes, (0,0)))
                    stimulus_data.append({
                        "target_present": self.target_present,
                        "is_target": True,
                        "coordinates": coordinate_pos,
                        "true_position": pos,
                        "color": fake_target_attributes["color"],
                        "size": fake_target_attributes["size"],
                        "orientation": fake_target_attributes["orientation"],
                        "convexity": fake_target_attributes["convexity"],
                        "center_point": fake_target_attributes["center_point"],
                        "missing_vertex": missing_vertex,
                        "masked": False,
                    })
                else:
                    distractor_color = [max(min((int(self.target_color[x] + random.randint(-self.color_difference, self.color_difference))), 255), 0) for x in range(3)]
                    missing_vertex = random.randint(0, 3) if random.random() < 0.5 else None
                    distractor_attributes = {
                        "color": distractor_color,
                        "size": random.uniform(self.size_range[0], self.size_range[1]),
                        "orientation": random.uniform(self.orientation_range[0], self.orientation_range[1]),
                        "convexity": random.uniform(self.convex_range[0], self.convex_range[1]),
                        "center_point": (random.uniform(-self.center_x, self.center_x), random.uniform(-self.center_y, self.center_y)),
                        "missing_vertex": missing_vertex,
                        "masked": False,
                        "target": False,
                    }
                    stimuli.extend(stimulus_function(distractor_attributes, pos))
                    stimulus_data.append({
                        "target_present": self.target_present,
                        "is_target": False,
                        "coordinates": coordinate_pos,
                        "true_position": pos,
                        "color": distractor_attributes["color"],
                        "size": distractor_attributes["size"],
                        "orientation": distractor_attributes["orientation"],
                        "convexity": distractor_attributes["convexity"],
                        "center_point": distractor_attributes["center_point"],
                        "missing_vertex": missing_vertex,
                        "masked": False,
                    })
        return stimuli, target, stimulus_data

    def measure_performance(self):
        """
        Implements the learning phase of the task:
        1. Present a stimulus for a short duration.
        2. Present a noise mask with colors.
        3. Wait with a fixation cross.
        4. Present a stimulus, either identical or dissimilar.
        5. Wait for participant input indicating if the objects are the same.
        6. Wait briefly before the next trial.
        """

        # Parameters for the learning phase
        num_trials = 4
        stimulus_duration = 0.5  # Duration to show the stimulus in seconds
        mask_duration = 0.5  # Duration of the noise mask
        fixation_duration = 0.5  # Duration of the fixation cross
        response_duration = 1.5  # Time allowed for participant response
        similarity_threshold = 0.2  # Degree of dissimilarity for "different" stimuli

        for trial_num in range(num_trials):
            # Step 1: Present the initial stimulus
            attributes = {
                "color": random.choice(self.saturated_colors),
                "size": random.choice(self.sizes),
                "opacity": 1.0,
                "orientation": random.choice(self.orientations),
                "elongation_x": random.uniform(0.75, 2.0),
                "elongation_y": random.uniform(0.75, 2.0),
            }
            pos = (0.0, 0.0)  # Central position for simplicity
            initial_stimulus = self.create_diamond_stimulus(attributes, pos)
            for stim in initial_stimulus:
                stim.draw()
            self.win.flip()
            core.wait(stimulus_duration)

            # Step 2: Present the noise mask
            # Parameters for the noise mask
            mask_radius = 25.0  # Radius of the circular area
            square_size = 1.25  # Size of individual squares
            mask_position = (0.0, 0.0)  # Center position of the mask

            # Draw the checkered noise mask
            self.create_checkered_noise_mask(self.win, radius=mask_radius, square_size=square_size, pos=mask_position)
            self.win.flip()
            core.wait(mask_duration)

            # Step 3: Present the fixation cross
            fixation = visual.TextStim(self.win, text="+", color="white")
            fixation.draw()
            self.win.flip()
            core.wait(fixation_duration)

            # Step 4: Present the second stimulus
            if trial_num % 2 == 1:
                is_same = random.choice([True, False])  # Decide if the stimulus is the same or dissimilar
                if is_same:
                    second_attributes = attributes
                else:
                    second_attributes = {
                        "color": [min(1.0, max(0.0, c + random.uniform(-similarity_threshold, similarity_threshold))) for c in attributes["color"]],
                        "size": attributes["size"] * random.uniform(.75, 1.25),
                        "opacity": attributes["opacity"],
                        "orientation": (attributes["orientation"] + random.choice(self.orientations)) % 360,
                        "elongation_x": attributes["elongation_x"] * random.uniform(0.75, 1.5),
                        "elongation_y": attributes["elongation_y"] * random.uniform(0.75, 1.5),
                    }
                second_stimulus = self.create_diamond_stimulus(second_attributes, pos)
                for stim in second_stimulus:
                    stim.draw()
                self.win.flip()
            else:
                correct_side = random.choice(["left", "right"])  # Randomize correct answer

                # Create the more similar stimulus
                similar_attributes = attributes.copy()
                similar_attributes["size"] *= random.uniform(0.9, 1.1)
                similar_attributes["orientation"] = (attributes["orientation"] + random.uniform(-10, 10)) % 360
                similar_attributes["elongation_x"] *= random.uniform(0.9, 1.1)
                similar_attributes["elongation_y"] *= random.uniform(0.9, 1.1)

                # Create the more different stimulus
                different_attributes = {
                    "color": [min(1.0, max(0.0, c + random.uniform(-similarity_threshold * 2, similarity_threshold * 2))) for c in attributes["color"]],
                    "size": attributes["size"] * random.uniform(0.6, 1.4),
                    "opacity": attributes["opacity"],
                    "orientation": (attributes["orientation"] + random.choice(self.orientations)) % 360,
                    "elongation_x": attributes["elongation_x"] * random.uniform(0.6, 1.4),
                    "elongation_y": attributes["elongation_y"] * random.uniform(0.6, 1.4),
                }

                # Assign positions
                left_stimulus = self.create_diamond_stimulus(similar_attributes, (-0.5, 0.0)) if correct_side == "left" else self.create_diamond_stimulus(different_attributes, (-4.5, 0.0))
                right_stimulus = self.create_diamond_stimulus(similar_attributes, (0.5, 0.0)) if correct_side == "right" else self.create_diamond_stimulus(different_attributes, (4.5, 0.0))

                # Draw both stimuli
                for stim in left_stimulus + right_stimulus:
                    stim.draw()
                self.win.flip()

            # Step 5: Wait for participant response
            response = None
            start_time = core.getTime()
            while core.getTime() - start_time < response_duration:
                keys = event.getKeys(keyList=["s", "d"])
                if keys:
                    response = keys[0]  # Record the response
                    break

            # Evaluate response
            correct_response = "s" if is_same else "d"
            feedback_text = "Correct!" if response == correct_response else "Incorrect!"
            feedback_color = "green" if response == correct_response else "red"
            if response is None:
                feedback_text = "Response not recorded in time. Try again."
            feedback = visual.TextStim(self.win, text=feedback_text, color=feedback_color)
            feedback.draw()
            self.win.flip()
            core.wait(1.0)

            # Step 6: Wait briefly before the next trial
            self.win.flip()
            core.wait(0.5)

    def run_trial(self, save_destination: str, trial_num):
        """Run a dual visual search trial, usign the same display with two difficulty levels."""

        trial_id = f"{trial_num}_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(f"{save_destination}/{trial_id}")
        
        # Generate target and stimuli and save the data
        stimuli, target, stimulus_data = self.generate_stimuli(self.create_diamond_stimulus, target_present=True)

        # create pandas row from each stimulus entry
        df = pd.DataFrame(stimulus_data)
        df.to_csv(f"{save_destination}/{trial_id}/stimulus_data.csv", index=False)

        # Display instructions
        instructions = ("In this task, you will first see a target object. Try to remember its shape, color, and orientation.\n\n"
                        "Next, a brief visual mask will appear. After which a fixation cross will be shown. Focus your gaze on it. \n\n"
                        "After a short pause, a grid of objects will appear.\n\n")
        bold_instructions = ("Search (with your eyes) for the target object within the grid. Once you find it,"
                             "fixate your gaze on it and press the Q button.\n")
        instructions_2 = ("If you don't find it, keep searching until the trial ends.\n\n"
                         "You may press SPACE to begin.")

        visual.TextStim(self.win, text=instructions, color="white", height=1.5, wrapWidth=70, pos=(0,6)).draw()
        visual.TextStim(self.win, text=bold_instructions, bold = True, color="white", height=1.5, wrapWidth=70).draw()
        visual.TextStim(self.win, text=instructions_2, color="white", height=1.5, wrapWidth=70, pos=(0,-6)).draw()

        self.win.flip()
        event.waitKeys(keyList=["space"])

        # Show target object
        for stim in target:
            stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames(f"{save_destination}/{trial_id}/target.png")
        core.wait(1.0)
    
        # Parameters for the noise mask
        mask_radius = 20.0  # Radius of the circular area
        square_size = 1.2  # Size of individual squares
        mask_position = (0.0, 0.0)  # Center position of the mask

        # Draw the checkered noise mask
        self.create_checkered_noise_mask(self.win, radius=mask_radius, square_size=square_size, pos=mask_position)
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames(f"{save_destination}/{trial_id}/mask.png")
        core.wait(.25)
        
        # Show fixation cross
        fixation = visual.TextStim(self.win, text="+", color="white", height=6.0)
        fixation.draw()
        self.win.flip()
        wait(2.0)

        # Start eye-tracking recording
        if self.eye_tracking:
            self.eyetracker.start_recording(gaze=True)

        # Display stimuli
        for stim in stimuli:
            stim.draw()  # Draw each component of the stimulus group
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames(f"{save_destination}/{trial_id}/stimulus.png")

        # TODO: Live data recording
        stime = getTime()
        while getTime() - stime < self.task_duration:
            samples = self.eyetracker.buffer.peek_N('gaze', 1)
            left_gaze_x = (samples['left_gaze_point_on_display_area_x'][0] - 0.5) * 100
            left_gaze_y = -(samples['left_gaze_point_on_display_area_y'][0] - 0.6) * 40
            right_gaze_x = (samples['right_gaze_point_on_display_area_x'][0] - 0.5) * 100
            right_gaze_y = -(samples['right_gaze_point_on_display_area_y'][0] - 0.6) * 40
            # draw two circles at the gaze points
            for stim in stimuli:
                stim.draw()
            visual.Circle(self.win, pos=(left_gaze_x, left_gaze_y), radius=1, lineColor='red', lineWidth=4).draw()
            visual.Circle(self.win, pos=(right_gaze_x, right_gaze_y), radius=1, lineColor='blue', lineWidth=4).draw()
            self.win.flip()


            # Wait for participant to press button
            keys = event.getKeys(keyList=["q"])

            # defaults
            response = None
            duration = self.task_duration
            if keys:
                duration = getTime() - stime
                samples = self.eyetracker.buffer.peek_N('gaze', 1)
                left_gaze_x = (samples['left_gaze_point_on_display_area_x'][0] - 0.5) * 100
                left_gaze_y = -(samples['left_gaze_point_on_display_area_y'][0] - 0.5) * 40
                # left_gaze_user_coordinates_x = samples['left_gaze_point_in_user_coordinates_x']
                # left_gaze_user_coordinates_y = samples['left_gaze_point_in_user_coordinates_y']   
                # left_gaze_user_coordinates_z = samples['left_gaze_point_in_user_coordinates_z']
                right_gaze_x = (samples['right_gaze_point_on_display_area_x'][0] - 0.5) * 100
                right_gaze_y = -(samples['right_gaze_point_on_display_area_y'][0] - 0.5) * 40
                # right_gaze_user_coordinates_x = samples['right_gaze_point_in_user_coordinates_x']
                # right_gaze_user_coordinates_y = samples['right_gaze_point_in_user_coordinates_y']
                # right_gaze_user_coordinates_z = samples['right_gaze_point_in_user_coordinates_z']

                # print gaze data
                print(f"Left gaze: ({left_gaze_x}, {left_gaze_y})\nRight gaze: ({right_gaze_x}, {right_gaze_y})")
                
                # # draw two circles at the gaze points
                # for stim in stimuli:
                #     stim.draw()
                # visual.Circle(self.win, pos=(left_gaze_x, left_gaze_y), radius=1.5, fillColor='red').draw()
                # visual.Circle(self.win, pos=(right_gaze_x, right_gaze_y), radius=1.5, fillColor='blue').draw()
                # self.win.flip()

                # detect if gaze is on target
                max_target_distance = ((self.grid_scaler_x + self.grid_scaler_y) / 2)
                target_position = stimulus_data[0]["true_position"]
                left_distance = np.sqrt((left_gaze_x - target_position[0])**2 + (left_gaze_y - target_position[1])**2)
                right_distance = np.sqrt((right_gaze_x - target_position[0])**2 + (right_gaze_y - target_position[1])**2)
                if left_distance < max_target_distance or right_distance < max_target_distance:
                    print("Gaze on target")
                    # create green square at target position
                    visual.Rect(self.win, width=8, height=8, lineColor=(0,255,0), lineWidth=5, pos=target_position).draw()
                    response = True
                else:
                    print("Gaze not on target")
                    # create square at object nearest to the gaze position
                    object_coords = [stimulus_data[i]["true_position"] for i in range(1, len(stimulus_data))]
                    nearest_object_position = min(object_coords, key=lambda x: np.sqrt((left_gaze_x - x[0])**2 + (left_gaze_y - x[1])**2))
                    visual.Rect(self.win, width=8, height=8, lineColor=(255,0,0), lineWidth=5, pos=nearest_object_position).draw()
                    response = False
                break
            wait(0.01)

        # Stop recording
        if self.eye_tracking:
            self.eyetracker.save_data(f"{save_destination}/{trial_id}/trial_{trial_num}")
            self.eyetracker.stop_recording()

        # Wait for participant to press space to proceed
        visual.TextStim(self.win, text=f"Trial {trial_num} complete. Press SPACE to continue.", color="white").draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])

        return response, duration

    def stimulus_variability(self):
        """
        Generate a series of stimuli with varying attributes for visual inspection.
        """
        self.grid_x = 9
        self.grid_y = 5

        red_color_range = [
            [255, 128, 0],
            [255, 96, 0],
            [255, 64, 0],
            [255, 32, 0],
            [255, 0, 0],
            [255, 0, 32],
            [255, 0, 64],
            [255, 0, 96],
            [255, 0, 128]
        ]
        green_color_range = [
            [224, 255, 0],
            [160, 255, 0],
            [96, 255, 0],
            [48, 255, 0],
            [0, 255, 0],
            [0, 255, 48],
            [0, 255, 96],
            [0, 255, 160],
            [0, 255, 224]
        ]
        blue_color_range = [
            [0, 128, 255],
            [0, 96, 255],
            [0, 64, 255],
            [0, 32, 255],
            [0, 0, 255],
            [32, 0, 255],
            [64, 0, 255],
            [96, 0, 255],
            [128, 0, 255]
        ]
        yellow_color_range = [
            [255, 128, 0],
            [255, 160, 0],
            [255, 192, 0],
            [255, 224, 0],
            [255, 255, 0],
            [224, 255, 0],
            [192, 255, 0],
            [160, 255, 0],
            [128, 255, 0]
        ]
        orange_color_range = [
            [255, 255, 0],
            [255, 224, 0],
            [255, 192, 0],
            [255, 160, 0],
            [255, 128, 0], 
            [255, 96, 0],
            [255, 64, 0],
            [255, 32, 0],
            [255, 0, 0]
        ]
        purple_color_range = [
            [255, 0, 128],
            [255, 0, 160],
            [255, 0, 192],
            [255, 0, 224],
            [255, 0, 255],
            [224, 0, 255],
            [192, 0, 255],
            [160, 0, 255],
            [128, 0, 255]
        ]

        # convexity display
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in red_color_range[x]],
                    "size": self.sizes[y],
                    "convexity": self.convex_range[0] + (self.convex_range[1] - self.convex_range[0]) * (x / (self.grid_x - 1)),
                    "opacity": 1.0,
                    "orientation": self.orientations[0],
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()

        #save image
        self.win.getMovieFrame()
        self.win.saveMovieFrames("convexity.png")

        event.waitKeys(keyList=["space"])

        # center point display 
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in orange_color_range[x]],
                    "size": self.sizes[y],
                    "center": (self.center_x * (y - 2) * 0.35*self.sizes[y], (self.center_y * (x - 4) * 0.15*self.sizes[y])),
                    "opacity": 1.0,
                    "orientation": self.orientations[0],
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames("center.png")

        event.waitKeys(keyList=["space"])

        # elongation display
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in yellow_color_range[x]],
                    "size": self.sizes[y],
                    "elongation_x": .9 + (x / (self.grid_x - 1))/2,
                    "elongation_y": .6 + (y / (self.grid_y - 1))/2,
                    "opacity": 1.0,
                    "orientation": self.orientations[0],
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames("elongation.png")

        event.waitKeys(keyList=["space"])

        # orientation + convexity display
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in green_color_range[x]],
                    "size": self.sizes[y],
                    "convexity": self.convex_range[0] + (self.convex_range[1] - self.convex_range[0]) * (x / (self.grid_x - 1)),
                    "orientation": self.orientations[x],
                    "opacity": 1.0,
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames("orientation_convexity.png")

        event.waitKeys(keyList=["space"])

        # orientation + center display
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in blue_color_range[x]],
                    "size": self.sizes[y],
                    "center": (self.center_x * (y - 2) * 0.35*self.sizes[y], (self.center_y * (x - 4) * 0.15*self.sizes[y])),
                    "orientation": self.orientations[x],
                    "opacity": 1.0,
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames("orientation_center.png")

        event.waitKeys(keyList=["space"])
        
        # center + conexity display 
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                attributes = {
                    "color": [(z-y*32)/255 for z in purple_color_range[x]],
                    "size": self.sizes[y],
                    "center": (self.center_x * (y - 2) * 0.35*self.sizes[y], (self.center_y * (x - 4) * 0.15*self.sizes[y])),
                    "convexity": self.convex_range[0] + (self.convex_range[1] - self.convex_range[0]) * (x / (self.grid_x - 1)),
                    "opacity": 1.0,
                    "orientation": self.orientations[0],
                }

                pos = ((x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler)
                stimuli = self.create_diamond_stimulus(attributes, pos)
                for stim in stimuli:
                    stim.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames("center_convexity.png")

        event.waitKeys(keyList=["space"])

if __name__ == "__main__":
    exp = Experiment(num_trials=4, eye_tracking=True)

    # Display stimulus variability
    # exp.stimulus_variability()

    # Learning phase
    # exp.measure_performance()

    # Run the experiment
    # participant_id = input("Enter participant ID: ")
    participant_id = "test"
    session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_destination = f"data/tobii_recordings/{participant_id}/{session_id}"
    os.makedirs(save_destination, exist_ok=True)

    responses, durations = [], []
    for trial in range(1, exp.num_trials + 1):
        response, duration = exp.run_trial(save_destination, trial)
        responses.append(response)
        durations.append(duration)

    # Save responses as a csv file
    df = pd.DataFrame({"trial": list(range(1, exp.num_trials + 1)), "response": responses, "duration": durations})
    df.to_csv(f"{save_destination}/responses.csv", index=False)

    exp.win.close()
    core.quit()
