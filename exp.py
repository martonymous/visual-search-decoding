from psychopy import visual, event, core, monitors
from psychopy.iohub import launchHubServer
from psychopy.core import getTime, wait

import random
import numpy as np


class Experiment:
    def __init__(self):
        # =================== Setup ======================
        # Monitor configuration
        self.monitor = monitors.Monitor("testMonitor", width=53.0, distance=60.0)  # Adjust for your setup
        self.win = visual.Window(size=(3440, 1440), monitor=self.monitor, units="deg", fullscr=False, color="gray")
        self.mouse = event.Mouse(win=self.win)

        # # Eye-tracking setup (simulation mode)
        self.iohub_config = {
            "eyetracker.hw.mouse.EyeTracker": {},
        }

        # setup tobii eyetracker
        # iohub_config = {
        #     "eyetracker.hw.tobii.EyeTracker": {
        #         "name": "tracker",
        #         "runtime_settings": {
        #             "sampling_rate": 120,
        #             "track_eyes": "both",
        #             "events_enabled": True,
        #         },
        #     },
        # }
        self.io = launchHubServer(**self.iohub_config)
        self.eyetracker = self.io.devices.eyetracker

        # Experiment parameters
        self.grid_x = 8
        self.grid_y = 4
        self.num_objects = self.grid_x * self.grid_y
        self.grid_scaler = 12.0              # Grid spacing
        self.object_scaler = 1.25            # Object size
        self.center_scaler = 1.2             # Center offset
        self.center_x = 0.5  * self.object_scaler * self.center_scaler
        self.center_y = 0.75 * self.object_scaler * self.center_scaler

        self.target_present = True           # Whether the target is present
        self.color_difference = 255          # Degree of color differences
        self.num_trials = 5                  # Number of trials in the experiment

        # Target attributes
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
            (255, 127, 0),              # Orange
            (127, 0, 255),              # Purple
            (0, 255, 127),              # Spring Green
            (255, 0, 127),              # Deep Pink
            (127, 255, 0),              # Chartreuse
            (0, 127, 255),              # Sky Blue
            (255, 51, 51),              # Bright Red
            (51, 255, 51),              # Bright Green
            (51, 51, 255),              # Bright Blue
        ]

        self.sizes = [2.0, 2.5, 3.0, 3.5, 4.0]
        self.orientations = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        self.size_range = [3.0, 6.0]
        self.orientation_range = [0, 360]
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
        center_point = attributes.get("center", (0.0, 0.0))

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
            start = diamond_vertices[i]
            end = diamond_vertices[(i + 1) % len(diamond_vertices)]
            curve_points = self.get_convex_curve(start, end, convexity, direction)
            _ = [shape_points.append(x) for x in curve_points]
        
        shape = visual.ShapeStim(
            win=self.win,
            vertices=shape_points,
            lineColor="black",
            lineWidth=2,
            fillColor=attributes["color"],
            pos=pos,
            ori=attributes["orientation"],
        )

        # Create lines connecting each vertex to the center
        line_stims = []
        for vertex in diamond_vertices:
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

    def create_checkered_noise_mask(self, win, radius, square_size, pos=(0, 0)):
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
                    if random.random() < 0.4:
                        if random.random() < 0.5:
                            color = "black"
                        else:
                            color = "white"
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

    def generate_stimuli(self, stimulus_function, jitter: float = 3.0):
        """Generate target and distractor stimuli arranged in a grid."""
        stimuli = []

        # Generate grid positions
        grid_positions = [
            [(x + 0.5 - (self.grid_x / 2)) * self.grid_scaler, (y + 0.5 - (self.grid_y / 2)) * self.grid_scaler]
            for x in range(self.grid_x)
            for y in range(self.grid_y)
        ]
        random.shuffle(grid_positions)

        target_pos = grid_positions[0] if self.target_present else None

        for i, pos in enumerate(grid_positions):
            for j, val in enumerate(pos):
                pos[j] = val + random.uniform(-jitter, jitter)

            if self.target_present and i == 0:
                color = random.choice(self.saturated_colors)
                self.target_color = [max(min((int(color[x] + random.uniform(-self.color_difference, self.color_difference)))/255, 1.0), 0.0) for x in range(3)]
                target_attributes = {
                    "color": self.target_color,
                    "size": random.randint(self.size_range[0], self.size_range[1]),
                    "orientation": random.uniform(self.orientation_range[0], self.orientation_range[1]),
                    "convexity": random.uniform(self.convex_range[0], self.convex_range[1]),
                    "center_point": (random.uniform(-self.center_x, self.center_x), random.uniform(-self.center_y, self.center_y)),
                    "masked": False,
                    "target": True}
                stimuli.extend(stimulus_function(target_attributes, target_pos))
            else:
                if not self.target_present:
                    self.target_color = random.choice(self.saturated_colors)

                distractor_color = [self.target_color[x] + random.uniform(-self.color_difference, self.color_difference)/255 for x in range(3)]
                print(distractor_color)
                distractor_attributes = {
                    "color": distractor_color,
                    "size": random.randint(self.size_range[0], self.size_range[1]),
                    "orientation": random.uniform(self.orientation_range[0], self.orientation_range[1]),
                    "convexity": random.uniform(self.convex_range[0], self.convex_range[1]),
                    "center_point": (random.uniform(-self.center_x, self.center_x), random.uniform(-self.center_y, self.center_y)),
                    "masked": False,
                    "target": False,
                }
                stimuli.extend(stimulus_function(distractor_attributes, pos))
                data = {
                    "target_present": self.target_present,
                    "target_pos": target_pos,
                    "distractor_attributes": distractor_attributes,
                    "distractor_pos": pos,
                }
        return stimuli, data

    def learning_phase(self):
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
        num_learning_trials = 4
        stimulus_duration = 0.5  # Duration to show the stimulus in seconds
        mask_duration = 0.5  # Duration of the noise mask
        fixation_duration = 0.5  # Duration of the fixation cross
        response_duration = 1.5  # Time allowed for participant response
        similarity_threshold = 0.2  # Degree of dissimilarity for "different" stimuli

        for trial_num in range(num_learning_trials):
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

    def run_trial(self, trial_num):
        """Run a single visual search trial."""

        # Generate stimuli
        stimuli, data = self.generate_stimuli(self.create_diamond_stimulus)

        # Show fixation cross
        fixation = visual.TextStim(self.win, text="+", color="white", height=2.0)
        fixation.draw()
        self.win.flip()
        core.wait(0.1)

        # Start eye-tracking recording
        self.eyetracker.setRecordingState(True)
        stime = getTime()
        while getTime() - stime < 1.0:
            for e in self.eyetracker.getEvents():
                print(e)

        # Display stimuli
        for stimgroup in stimuli:
            for stim in stimuli:
                stim.draw()  # Draw each component of the stimulus group
        self.win.flip()

        # Wait for participant to click
        mouse = event.Mouse(win=self.win)
        while True:
            if mouse.getPressed()[0]:  # Left mouse button clicked
                click_pos = mouse.getPos()
                break
            wait(0.01)

        # Stop recording
        self.eyetracker.setRecordingState(False)

        # Wait for participant to press space to proceed
        visual.TextStim(self.win, text=f"Trial {trial_num} complete. Press SPACE to continue.", color="white").draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])

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
    exp = Experiment()

    # Display stimulus variability
    exp.stimulus_variability()

    # Learning phase
    exp.learning_phase()

    # Run the experiment
    for trial in range(1, exp.num_trials + 1):
        exp.run_trial(trial)

    exp.win.close()
    core.quit()
