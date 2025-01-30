from psychopy import visual, event, core, monitors
from psychopy.iohub import launchHubServer
from psychopy.core import getTime, wait

import random
import numpy as np

# =================== Setup ======================
# Monitor configuration
monitor = monitors.Monitor("testMonitor", width=53.0, distance=60.0)  # Adjust for your setup
win = visual.Window(size=(3440, 1440), monitor=monitor, units="deg", fullscr=False, color="gray")
mouse = event.Mouse(win=win)

# # Eye-tracking setup (simulation mode)
iohub_config = {
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
io = launchHubServer(**iohub_config)
eyetracker = io.devices.eyetracker

# Experiment parameters
grid_x = 8
grid_y = 4
num_objects = grid_x * grid_y
grid_scaler = 15.0              # Grid spacing
object_scaler = .5              # Object size

target_present = True           # Whether the target is present
color_difference = 192          # Degree of color differences
num_trials = 5                  # Number of trials in the experiment

# Target attributes
target_attributes = {
    "color": [255, 0, 0],
    "size": 2.5,
    "masked": False,
    "target": True,
    "orientation": 0,
}

# Generate random distractor attributes
base_color = [127, 127, 127]    # RGB for gray
saturated_colors = [
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

sizes = [2.0, 2.5, 3.0, 3.5, 4.0]
size_range = [2.0, 4.0]
orientations = [0, 45, 90, 135, 180, 225, 270, 315]
orientation_range = [0, 360]

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

def create_L_stimulus(attributes, pos):
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
        win,
        vertices=object_vertices,
        fillColor=attributes["color"],
        lineColor=None,
        pos=pos,
        ori=attributes["orientation"],
        opacity=attributes["opacity"],
    )

    return [stim]

def create_diamond_stimulus(attributes, pos):
    """
    Create a diamond-shaped stimulus with elongated proportions. 
    Lines extend from each corner of the diamond to a central point.
    The exact parameters of the shape are determined by the attributes provided.
    """

    convexity = attributes.get("convexity", 1.5)  # Adjust curvature
    direction = attributes.get("direction", 1)   # 1 for outward, -1 for inward
    elongation_factor = attributes.get("elongation", 1.5)

    diamond_vertices = [
        (0.0, 1.5),  # Top
        (.75, 0.0),  # Right
        (0.0, -1.0), # Bottom
        (-.75, 0.0), # Left
    ]
    diamond_vertices = [(x * elongation_factor, y) for x, y in diamond_vertices]
    diamond_vertices = [(x * attributes["size"], y * attributes["size"]) for x, y in diamond_vertices]

    # Center the object for rotations
    # all_vertices = [(x - 0.0, y - 0.0) for x, y in all_vertices]

    shape_points = []
    for i in range(len(diamond_vertices)):
        start = diamond_vertices[i]
        end = diamond_vertices[(i + 1) % len(diamond_vertices)]
        curve_points = get_convex_curve(start, end, convexity, direction)
        _ = [shape_points.append(x) for x in curve_points]
    
    shape = visual.ShapeStim(
        win,
        vertices=shape_points,
        lineColor="black",
        lineWidth=2,
        fillColor=attributes["color"],
        pos=pos,
        ori=attributes["orientation"],
    )

    # Create lines connecting each vertex to the center
    center_point = (random.uniform(-2, 2), random.uniform(-2, 2))
    line_stims = []
    for vertex in diamond_vertices:
        line_stim = visual.ShapeStim(
            win,
            pos=pos,
            vertices=[center_point, vertex],
            lineColor="black",
            lineWidth=2,
            ori=attributes["orientation"],
        )
        line_stims.append(line_stim)

    return [shape] + line_stims

def create_masks(attributes, pos):
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
        win,
        vertices=mask_vertices,
        fillColor=win.color,  # Match the window's background color
        lineColor=None,       # No border for the mask
        pos=pos,
        ori=mask_orientation,  # Set random orientation for the mask
        opacity=1.0,
    )
    stimuli.append(mask)  # Add the mask to the list of stimuli

    return stimuli

def create_checkered_noise_mask(win, radius, square_size, pos=(0, 0)):
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
                    win,
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

def generate_stimuli(stimulus_function, jitter: float = 3.0):
    """Generate target and distractor stimuli arranged in a grid."""
    stimuli = []

    # Generate grid positions
    grid_positions = [
        [(x + 0.5 - (grid_x / 2)) * grid_scaler, (y + 0.5 - (grid_y / 2)) * grid_scaler]
        for x in range(grid_x)
        for y in range(grid_y)
    ]
    random.shuffle(grid_positions)

    target_pos = grid_positions[0] if target_present else None

    for i, pos in enumerate(grid_positions):
        for j, val in enumerate(pos):
            pos[j] = val + random.uniform(-jitter, jitter)

        if target_present and i == 0:
            target_attributes = {
                "color": random.choice(saturated_colors),
                "size": random.randint(size_range[0], size_range[1]),
                "masked": False,
                "orientation": random.choice(orientations),
                "target": True}
            stimuli.extend(stimulus_function(target_attributes, target_pos))
        else:
            if target_present:
                target_color = target_attributes["color"]
            else:
                target_color = random.choice(saturated_colors)

            distractor_color = [
                max(min((int(target_color[0] + random.uniform(-color_difference, color_difference)))/255, 1.0), 0.0),
                max(min((int(target_color[1] + random.uniform(-color_difference, color_difference)))/255, 1.0), 0.0),
                max(min((int(target_color[2] + random.uniform(-color_difference, color_difference)))/255, 1.0), 0.0),
            ]
            print(distractor_color)
            distractor_attributes = {
                "color": distractor_color,
                "size": random.choice(sizes),
                "orientation": random.choice(orientations),
                "masked": False,
                "target": False,
            }
            stimuli.extend(stimulus_function(distractor_attributes, pos))
            data = {
                "target_present": target_present,
                "target_pos": target_pos,
                "distractor_attributes": distractor_attributes,
                "distractor_pos": pos,
            }
    return stimuli, data

def learning_phase():
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
    num_learning_trials = 1
    stimulus_duration = 0.5  # Duration to show the stimulus in seconds
    mask_duration = 0.5  # Duration of the noise mask
    fixation_duration = 0.5  # Duration of the fixation cross
    response_duration = 1.5  # Time allowed for participant response
    similarity_threshold = 0.2  # Degree of dissimilarity for "different" stimuli

    for _ in range(num_learning_trials):
        # Step 1: Present the initial stimulus
        attributes = {
            "color": random.choice(saturated_colors),
            "size": random.choice(sizes),
            "opacity": 1.0,
            "orientation": random.choice(orientations),
            "elongation": random.uniform(1.0, 2.0),
        }
        pos = (0.0, 0.0)  # Central position for simplicity
        initial_stimulus = create_diamond_stimulus(attributes, pos)
        for stim in initial_stimulus:
            stim.draw()
        win.flip()
        core.wait(stimulus_duration)

        # Step 2: Present the noise mask
        # Parameters for the noise mask
        mask_radius = 25.0  # Radius of the circular area
        square_size = 1.25  # Size of individual squares
        mask_position = (0.0, 0.0)  # Center position of the mask

        # Draw the checkered noise mask
        create_checkered_noise_mask(win, radius=mask_radius, square_size=square_size, pos=mask_position)
        win.flip()
        core.wait(mask_duration)

        # Step 3: Present the fixation cross
        fixation = visual.TextStim(win, text="+", color="white")
        fixation.draw()
        win.flip()
        core.wait(fixation_duration)

        # Step 4: Present the second stimulus
        is_same = random.choice([True, False])  # Decide if the stimulus is the same or dissimilar
        if is_same:
            second_attributes = attributes
        else:
            second_attributes = {
                "color": [min(1.0, max(0.0, c + random.uniform(-similarity_threshold, similarity_threshold))) for c in attributes["color"]],
                "size": attributes["size"] * random.uniform(1.0, 4.0),
                "opacity": attributes["opacity"],
                "orientation": (attributes["orientation"] + random.choice(orientations)) % 360,
                "elongation": attributes["elongation"] * random.uniform(0.9, 1.1),
            }
        second_stimulus = create_diamond_stimulus(second_attributes, pos)
        for stim in second_stimulus:
            stim.draw()
        win.flip()

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
        if response is None:
            feedback_text = "Too slow!"
        feedback = visual.TextStim(win, text=feedback_text, color="white")
        feedback.draw()
        win.flip()
        core.wait(1.0)

        # Step 6: Wait briefly before the next trial
        win.flip()
        core.wait(0.5)

def run_trial(trial_num):
    """Run a single visual search trial."""

    # Generate stimuli
    stimuli, data = generate_stimuli(create_diamond_stimulus)

    # Show fixation cross
    fixation = visual.TextStim(win, text="+", color="white", height=2.0)
    fixation.draw()
    win.flip()
    core.wait(0.1)

    # Start eye-tracking recording
    eyetracker.setRecordingState(True)
    stime = getTime()
    while getTime() - stime < 1.0:
        for e in eyetracker.getEvents():
            print(e)

    # Display stimuli
    for stimgroup in stimuli:
        for stim in stimuli:
            stim.draw()  # Draw each component of the stimulus group
    win.flip()

    # Wait for participant to click
    mouse = event.Mouse(win=win)
    while True:
        if mouse.getPressed()[0]:  # Left mouse button clicked
            click_pos = mouse.getPos()
            break
        wait(0.01)

    # Stop recording
    eyetracker.setRecordingState(False)

    # Wait for participant to press space to proceed
    visual.TextStim(win, text=f"Trial {trial_num} complete. Press SPACE to continue.", color="white").draw()
    win.flip()
    event.waitKeys(keyList=["space"])

if __name__ == "__main__":
    # Learning phase
    # learning_phase()

    # Run the experiment
    for trial in range(1, num_trials + 1):
        run_trial(trial)

    win.close()
    core.quit()
