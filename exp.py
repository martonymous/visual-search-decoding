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
num_objects = 16  # Total objects match grid size
grid_size = 4  # Grid dimensions (4x4)
grid_scaler = 16.0  # Grid spacing
object_scaler = .6  # Object size
target_present = False  # Whether the target is present
color_difference = 0.99  # Degree of color differences (0-1)
num_trials = 2  # Number of trials in the experiment

# Target attributes
target_attributes = {
    "color": [1.0, 0.0, 0.0],  # RGB for red
    "size": 2.5,
    "opacity": 1.0, 
    "masked": False,  # Target is not masked
    "target": True,
}

# Generate random distractor attributes
base_color =    [0.5, 0.5, 0.5]  # Base gray color
red =           [1.0, 0.0, 0.0]  # Red color
blue =          [0.0, 0.0, 1.0]  # Blue color
green =         [0.0, 1.0, 0.0]  # Green color

sizes = [3.0, 4.0, 5.0, 6.0]
opacities = [0.8, 0.9, 1.0]
orientations = [0, 45, 90, 135, 180, 225, 270, 315]

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

    # Define the vertices for a diamond shape
    diamond_vertices = [
        (0.0, 1.5),  # Top
        (.75, 0.0),  # Right
        (0.0, -1.0), # Bottom
        (-.75, 0.0), # Left
    ]

    # Elongate the diamond shape if specified
    elongation_factor = attributes.get("elongation", 1.5)  # Default elongation factor
    all_vertices = [(x * elongation_factor, y) for x, y in diamond_vertices]

    # Scale the shape based on the size attribute
    all_vertices = [(x * attributes["size"], y * attributes["size"]) for x, y in all_vertices]

    # Center the object for rotation accuracy
    # all_vertices = [(x - 0.0, y - 0.0) for x, y in all_vertices]

    # Create a ShapeStim object with the specified attributes
    stim = visual.ShapeStim(
        win,
        vertices=all_vertices,
        fillColor=attributes["color"],
        lineColor=None,
        pos=pos,
        ori=attributes["orientation"],
        opacity=attributes["opacity"],
    )

    # Create lines connecting each vertex to the center
    center_point = (random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5))
    line_stims = []
    for vertex in all_vertices:
        line_stim = visual.ShapeStim(
            win,
            vertices=[center_point, vertex],
            lineColor="black",
            lineWidth=2,  # Adjust thickness of the lines
            pos=pos,
            ori=attributes["orientation"],
            opacity=attributes["opacity"],
        )
        line_stims.append(line_stim)

    return [stim] + line_stims

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
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        color = [0, 0, 0]
                    else:
                        color = [10, 10, 10]
                else:
                    color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                
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

    # Draw the squares
    for square in squares:
        square.draw()

    # # Optionally add a circular boundary (not required)
    # boundary = visual.Circle(win, radius=radius, edges=128, lineColor="black", pos=pos)
    # boundary.draw()

def generate_stimuli(stimulus_function, jitter: float = 3.0):
    """Generate target and distractor stimuli arranged in a grid."""
    stimuli = []

    # Generate grid positions
    grid_positions = [
        [(x + 0.5 - (grid_size / 2)) * grid_scaler, (y + 0.5 - (grid_size / 2)) * grid_scaler]
        for x in range(grid_size)
        for y in range(grid_size)
    ]
    random.shuffle(grid_positions)

    target_pos = grid_positions.pop() if target_present else None

    for i, pos in enumerate(grid_positions):
        for j, val in enumerate(pos):
            pos[j] = val + random.uniform(-jitter, jitter)

        if target_present and i == 0:
            stimuli.extend(stimulus_function(target_attributes, target_pos))
        else:
            distractor_color = [
                min(1.0, max(0.0, base_color[0] + random.uniform(-color_difference, color_difference))),
                min(1.0, max(0.0, base_color[1] + random.uniform(-color_difference, color_difference))),
                min(1.0, max(0.0, base_color[2] + random.uniform(-color_difference, color_difference))),
            ]
            distractor_attributes = {
                "color": distractor_color,
                "size": random.choice(sizes),
                "opacity": random.choice(opacities),
                "masked": random.choice([True, False]),  # Randomly decide if the stimulus is masked
                "orientation": random.choice(orientations),
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
    num_learning_trials = 10
    stimulus_duration = 0.5  # Duration to show the stimulus in seconds
    mask_duration = 0.5  # Duration of the noise mask
    fixation_duration = 0.5  # Duration of the fixation cross
    response_duration = 1.5  # Time allowed for participant response
    similarity_threshold = 0.2  # Degree of dissimilarity for "different" stimuli

    for _ in range(num_learning_trials):
        # Step 1: Present the initial stimulus
        attributes = {
            "color": random.choice([red, blue, green]),
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
        square_size = 1.5  # Size of individual squares
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
                "size": attributes["size"] * random.uniform(0.9, 1.1),
                "opacity": attributes["opacity"],
                "orientation": (attributes["orientation"] + random.choice([45, 90, 135])) % 360,
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
    if trial_num % 2 == 0:
        stimuli, data = generate_stimuli(create_diamond_stimulus)
    else:
        stimuli, data = generate_stimuli(create_L_stimulus)

    # Show fixation cross
    fixation = visual.TextStim(win, text="+", color="white")
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
    learning_phase()

    # Run the experiment
    for trial in range(1, num_trials + 1):
        run_trial(trial)

    win.close()
    core.quit()
