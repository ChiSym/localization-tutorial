# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ProbComp Localization Tutorial
#
# This notebook aims to give an introduction to probabilistic computation (ProbComp).  This term refers to a way of expressing probabilistic constructs in a computational paradigm, made precise by a probablistic programming language (PPL).  The programmer can thus encode their probabilistic intuition for solving a problem into an algorithm.  Back-end language work automates the routine but error-prone derivations.

# Dependencies are specified in pyproject.toml.
# %%
# Global setup code

from __future__ import annotations
import json
import genstudio.plot as Plot
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import genjax

import os
from math import sin, cos, pi
from typing import Optional

import numpy as np


# Ensure a location for image generation.
os.makedirs("imgs", exist_ok=True)

# %% [markdown]
# ## The "real world"
#
# We assume given
# * a map of a space, together with
# * some clutters that sometimes unexpectedly exist in that space.
#
# We also assume given a description of a robot's behavior via
# * an estimated initial pose (= position + heading), and
# * a program of controls (= advance distance, followed by rotate heading).
#
# *In addition to the uncertainty in the initial pose, we are uncertain about the true execution of the motion of the robot.*
#
# Below, we will also introduce sensors.

# %% [markdown]
# ### Load map and robot data
#
# Generally speaking, we keep general code and specific examples in separate cells, as signposted here.

# %%
# General code here

class Pose:
    def __init__(self, p: np.ndarray, hd: Optional[float] = None, dp: Optional[np.ndarray] = None):
        """
        Initializes a Pose object either from a heading (hd) or a direction vector (dp).

        Args:
            p (np.ndarray): The position as a numpy array [x, y].
            hd (float, optional): The heading in radians. Optional if dp is provided.
            dp (np.ndarray, optional): The direction vector as a numpy array [dx, dy]. Optional if hd is provided.

        Raises:
            ValueError: If both 'hd' and 'dp' are None.
        """
        self.p = p
        if hd is not None:
            self.hd = hd % (2 * pi)  # Ensuring the heading is within 0 to 2π
            self.dp = np.array([np.cos(self.hd), np.sin(self.hd)])
        elif dp is not None:
            self.hd = np.arctan2(dp[1], dp[0])
            self.dp = dp
        else:
            raise ValueError("Either 'hd' (heading) or 'dp' (direction vector) must be provided, not both None.")

    def __repr__(self):
        return f"Pose(p={self.p}, hd={self.hd})"

    def step_along(self, s: float) -> Pose:
        """
        Moves along the direction of the pose by a scalar and returns a new Pose.

        Args:
            s (float): The scalar distance to move along the pose's direction.

        Returns:
            Pose: A new Pose object representing the moved position.
        """
        new_p = self.p + s * self.dp
        return Pose(new_p, hd=self.hd)

    def rotate(self, a: float) -> Pose:
        """
        Rotates the pose by angle 'a' (in radians) and returns a new Pose.

        Args:
            a (float): The angle in radians to rotate the pose.

        Returns:
            Pose: A new Pose object representing the rotated pose.
        """
        new_hd = self.hd + a
        return Pose(self.p, hd=new_hd)

# Example usage:
pose = Pose(np.array([1.0, 2.0]), hd=1.57)
print(pose)

# Move the pose along its direction
print(pose.step_along(5))

# Rotate the pose
print(pose.rotate(pi / 2))

class Segment:
    def __init__(self, p1, p2):
        # If p1 and p2 are Pose objects, extract their positions.
        if isinstance(p1, Pose) and isinstance(p2, Pose):
            self.p1 = p1.p
            self.p2 = p2.p
        else:
            self.p1 = np.array(p1)
            self.p2 = np.array(p2)
        self.dp = self.p2 - self.p1

    def __repr__(self):
        return f"Segment({self.p1}, {self.p2})"
# %%

class Control:
    def __init__(self, ds: float, dhd: float):
        self.ds = ds
        self.dhd = dhd

def create_segments(verts, loop_around=False):
    verts_np = np.array(verts)
    segs = [Segment(p1, p2) for p1, p2 in zip(verts_np[:-1], verts_np[1:])]
    if loop_around:
        segs.append(Segment(verts_np[-1], verts_np[0]))
    return segs

def make_world(walls_vec, clutters_vec, start, controls, loop_around=False):
    """
    Constructs the world by creating segments for walls and clutters, calculates the bounding box, and prepares the simulation parameters.

    Args:
    - walls_vec (list of list of float): A list of 2D points representing the vertices of walls.
    - clutters_vec (list of list of list of float): A list where each element is a list of 2D points representing the vertices of a clutter.
    - start (Pose): The starting pose of the robot.
    - controls (list of Control): A list of control actions for the robot.
    - loop_around (bool, optional): Whether to connect the last and first vertices of walls and clutters. Defaults to False.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    # Create segments for walls and clutters
    walls = create_segments(walls_vec, loop_around=loop_around)
    clutters = [create_segments(clutter, loop_around=loop_around) for clutter in clutters_vec]
    walls_clutters = walls + [item for sublist in clutters for item in sublist]

    # Combine all points for bounding box calculation
    all_points_np = np.vstack((np.array(walls_vec), np.concatenate(clutters_vec), np.array([start.p])))
    x_min, y_min = np.min(all_points_np, axis=0)
    x_max, y_max = np.max(all_points_np, axis=0)

    # Calculate bounding box, box size, and center point
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])

    # Determine the total number of control steps
    T = len(controls)

    return ({'walls': walls, 'clutters': clutters, 'walls_clutters': walls_clutters,
             'bounding_box': bounding_box, 'box_size': box_size, 'center_point': center_point},
            {'start': start, 'controls': controls},
            T)

def load_world(file_name, loop_around=False):
    """
    Loads the world configuration from a specified file and constructs the world.

    Args:
    - file_name (str): The name of the file containing the world configuration.
    - loop_around (bool, optional): Whether to connect the last and first vertices of walls and clutters. Defaults to False.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)

    walls_vec = [np.array(vert, dtype=float) for vert in data["wall_verts"]]
    clutters_vec = [np.array(clutter, dtype=float) for clutter in data["clutter_vert_groups"]]
    start = Pose(np.array(data["start_pose"]["p"], dtype=float), float(data["start_pose"]["hd"]))
    controls = [Control(float(control["ds"]), float(control["dhd"])) for control in data["program_controls"]]

    return make_world(walls_vec, clutters_vec, start, controls, loop_around=loop_around)
# %%
# Specific example code here

world, robot_inputs, T = load_world("../example_20_program.json");

# %% [markdown]
# ### Integrate a path from a starting pose and controls
#
# If the motion of the robot is determined in an ideal manner by the controls, then we may simply integrate to determine the resulting path.  Naïvely, this results in the following.

# %%
def integrate_controls_unphysical(robot_inputs):
    """
    Integrates the controls to generate a path from the starting pose.

    This function takes the initial pose and a series of control steps (ds for distance, dhd for heading change)
    and computes the resulting path by applying each control step sequentially.

    Args:
    - robot_inputs (dict): A dictionary containing the starting pose and control steps.

    Returns:
    - list: A list of Pose instances representing the path taken by applying the controls.
    """
    # Initialize the path with the starting pose
    path = [robot_inputs['start']]

    # Iterate over each control step to compute the new pose and add it to the path
    for control in robot_inputs['controls']:
        # Compute the new position (p) by applying the distance change (ds) in the direction of dp
        # Note: dp is derived from the current heading (hd) to ensure movement in the correct direction
        p = path[-1].p + control.ds * path[-1].dp
        # Compute the new heading (hd) by adding the heading change (dhd)
        hd = path[-1].hd + control.dhd
        # Create a new Pose with the updated position and heading, and add it to the path
        path.append(Pose(p, hd))

    return path

# %% [markdown]
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance from the point of contact in the normal direction to the wall.

# %%

def solve_lines(p, u, q, v, PARALLEL_TOL=1.0e-10):
    """
    Solves for the intersection of two lines defined by points and direction vectors.

    Args:
    - p, u: Point and direction vector defining the first line.
    - q, v: Point and direction vector defining the second line.
    - PARALLEL_TOL: Tolerance for determining if lines are parallel.

    Returns:
    - s, t: Parameters for the line equations at the intersection point. None if lines are parallel.
    """
    det = u[0] * v[1] - u[1] * v[0]
    if abs(det) < PARALLEL_TOL:
        return None, None
    else:
        s = (v[0] * (p[1]-q[1]) - v[1] * (p[0]-q[0])) / det
        t = (u[1] * (q[0]-p[0]) - u[0] * (q[1]-p[1])) / det
        return s, t

def distance(p, seg):
    """
    Computes the distance from a pose to a segment, considering the pose's direction.

    Args:
    - p: The Pose object.
    - seg: The Segment object.

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    s, t = solve_lines(p.p, p.dp, seg.p1, seg.dp)
    if s is None or s < 0 or not (0 <= t <= 1):
        return np.inf
    else:
        return s

def physical_step(p1, p2, hd, world_inputs):
    """
    Computes a physical step considering wall collisions and bounces.

    Args:
    - p1, p2: Start and end points of the step.
    - hd: Heading direction.
    - world_inputs: dict containing world configuration, including walls and bounce distance.

    Returns:
    - Pose: The new pose after taking the step, considering potential wall collisions.
    """
    step_direction = np.subtract(p2, p1)
    step_pose = Pose(p1, dp=step_direction)
    distances = [distance(step_pose, wall) for wall in world_inputs['walls']]
    closest_wall_distance, closest_wall_index = min((dist, idx) for (idx, dist) in enumerate(distances))
    step_length = np.linalg.norm(step_direction)

    if closest_wall_distance >= step_length:
        return Pose(p2, hd)
    else:
        collision_point = np.add(p1, np.multiply(closest_wall_distance, step_pose.dp))
        wall_normal_direction = world_inputs['walls'][closest_wall_index].dp
        normalized_wall_direction = np.divide(wall_normal_direction, np.linalg.norm(wall_normal_direction))
        wall_normal = np.array([-normalized_wall_direction[1], normalized_wall_direction[0]])

        if np.cross(step_pose.dp, wall_normal_direction) < 0:
            wall_normal = -wall_normal

        bounce_off_point = np.add(collision_point, np.multiply(world_inputs['bounce'], wall_normal))
        return Pose(bounce_off_point, hd)

#%%
def integrate_controls(robot_inputs, world_inputs):
    """
    Integrates controls to generate a path, taking into account physical interactions with walls.

    Args:
    - robot_inputs: Dictionary containing the starting pose and control steps.
    - world_inputs: Dictionary containing the world configuration.

    Returns:
    - list: A list of Pose instances representing the path taken by applying the controls.
    """
    path = [robot_inputs['start']]
    for control in robot_inputs['controls']:
        next_position = path[-1].p + control.ds * path[-1].dp
        next_heading = path[-1].hd + control.dhd
        path.append(physical_step(path[-1].p, next_position, next_heading, world_inputs))
    return path

# %%

# How bouncy the walls are in this world.
world_inputs = {'walls': world['walls'], 'bounce': 0.1}

path_integrated = integrate_controls(robot_inputs, world_inputs)

# %% [markdown]
# ### Plot such data

# %%

def clutter_points(clutter):
    points = []
    for segment in clutter:
        points.append(segment.p1)
    points.append(clutter[-1].p2)
    return points

def arrowhead_line(point, heading, wingLength=0.4, wingAngle=pi/4, **kwargs):
    leftWingAngle = heading + wingAngle
    rightWingAngle = heading - wingAngle

    leftWingEnd = [point[0] - wingLength * cos(leftWingAngle),
                   point[1] - wingLength * sin(leftWingAngle)]
    rightWingEnd = [point[0] - wingLength * cos(rightWingAngle),
                    point[1] - wingLength * sin(rightWingAngle)]

    return Plot.line([leftWingEnd, point, rightWingEnd], **kwargs)

def pose_arrow(p, r=0.5, **kwargs):
    start = p.p
    end = p.step_along(r).p
    opts = {'strokeWidth': 2, **kwargs}
    return Plot.line([start, end], **opts) + arrowhead_line(end, p.hd, **opts)

# Plot the world with walls only
world_plot = Plot.new(
    [Plot.line([wall.p1, wall.p2], strokeWidth=1, stroke=Plot.constantly('walls'),) for wall in world['walls']],
        {'title': "Given data",
         'width': 500,
         'height': 500,
         'margin': 0,
         'inset': 50},
         Plot.color_legend,
         Plot.color_map({'walls': '#000000',
                         'clutters': 'magenta',
                         'path from integrating controls': 'lightgreen',
                         'given start pose': 'darkgreen'}),
         Plot.frame(strokeWidth=4, stroke="#ddd"))

# Plot of the starting pose of the robot
starting_pose_plot = pose_arrow(robot_inputs['start'], stroke=Plot.constantly('given start pose'))

# Plot of the path from integrating controls
controls_path_plot = Plot.dot([pose.p for pose in path_integrated], fill=Plot.constantly('path from integrating controls'))

# Plot of the clutters
clutters_plot = [Plot.line(clutter_points(clutter), fill=Plot.constantly('clutters'))
              for clutter in world['clutters']]

# Save the figure to a file
# plt.savefig("imgs/given_data")

# Following this initial display of the given data, we suppress the clutters until much later in the notebook.

world_plot + controls_path_plot + starting_pose_plot + clutters_plot

# %%


# %% [markdown]
# We can also visualize the behavior of the model of physical motion:
#
# ![](../imgs_stable/physical_motion.gif)

# %% [markdown]
# ## Gen basics
#
# As said initially, we are uncertain about the true initial position and subsequent motion of the robot.  In order to reason about these, we now specify a model using `Gen`.
#
# Each piece of the model is declared as a *generative function* (GF).  The `Gen` library provides two DSLs for constructing GFs: the dynamic DSL using the decorator `@gen` on a function declaration, and the static DSL similarly decorated with `@gen (static)`.  The dynamic DSL allows a rather wide class of program structures, whereas the static DSL only allows those for which a certain static analysis may be performed.
#
# The library offers two basic constructs for use within these DSLs: primitive *distributions* such as "Bernoulli" and "normal", and the sampling operator `~`.  Recursively, GFs may sample from other GFs using `~`.

# %% [markdown]
# ### Components of the motion model
#
# We start with the two building blocks: the starting pose and individual steps of motion.

# %%


# %%
@static_gen_fn
def start_pose_prior(start, motion_settings):
    p = genjax.mv_normal(start.p, motion_settings['p_noise']**2 * np.eye(2)) @ "p"
    hd = genjax.normal(start.hd, motion_settings['hd_noise']) @ "hd"
    return Pose(p, hd)


@static_gen_fn
def step_model(start, c, world_inputs, motion_settings):
    p = genjax.mvnormal(start.p + c.ds * start.dp, motion_settings['p_noise']**2 * np.eye(2)) @ "p"
    hd = genjax.normal(start.hd + c.dhd, motion_settings['hd_noise']) @ "hd"
    return physical_step(start.p, p, hd, world_inputs)

# %% [markdown]
# Returning to the code, we can call a GF like a normal function and it will just run stochastically:


# %%
# Generate points on the unit circle
theta = np.linspace(0, 2*np.pi, 500)
unit_circle_xs = np.cos(theta)
unit_circle_ys = np.sin(theta)

# Function to create a circle with center p and radius r
def make_circle(p, r):
    return (p[0] + r * unit_circle_xs, p[1] + r * unit_circle_ys)

# %%
# Set the motion settings
motion_settings = {'p_noise': 0.5, 'hd_noise': 2*np.pi / 360}

# Generate N_samples of starting poses from the prior
N_samples = 50
key = jax.random.PRNGKey(314159)
key, *sub_keys = jax.random.split(key, N_samples + 1)
pose_samples = [start_pose_prior.simulate(k, (robot_inputs['start'], motion_settings)) for k in sub_keys]




sampler(start_pose_prior, robot_inputs['start'], motion_settings)
start_pose_prior.simulate(key, (robot_inputs['start'], motion_settings)).get_retval()

# editorTextFocus &&  !notebookEditorFocused && isWorkspaceTrusted && jupyter.ownsSelection && !findInputFocussed && !replaceInputFocussed && editorLangId == 'python' &&

# Calculate the radius of the 95% confidence region
std_devs_radius = 2.5 * motion_settings['p_noise']

# Plot the world, starting pose samples, and 95% confidence region
the_plot = plot_world(world, "Start pose prior (samples)")
circle_xs, circle_ys = make_circle(robot_inputs['start'].p, std_devs_radius)
plt.fill(circle_xs, circle_ys, color='red', alpha=0.25, label="95% region")
plt.plot([pose.choices['p'].value[0] for pose in pose_samples], [pose.choices['p'].value[1] for pose in pose_samples], 'r.', label="start pose samples")
plt.legend()
plt.savefig("imgs/start_prior.png")
plt.show()

# %%
from pathlib import Path

def get_function_def(func_name):
    source = Path(__file__).read_text()
    lines = source.split('\n')
    # Python functions start with 'def' followed by the function name and a colon
    start_index = next((i for i, line in enumerate(lines) if line.strip().startswith(f"def {func_name}(")), None)
    if start_index is None:
        return None  # Function not found
    # Find the end of the function by looking for a line that is not indented
    end_index = next((i for i, line in enumerate(lines[start_index+1:], start_index+1) if not line.startswith((' ', '\t'))), None)
    # If the end is not found, assume the function goes until the end of the file
    end_index = end_index or len(lines)
    return '\n'.join(lines[start_index:end_index])

print(get_function_def("start_pose_prior"))

## source code highlighter as one reusable visualizer.
## initial state - grab source (eg. from current file given function def)
## then set highlights at different times t.
## the time-series db will show all the visualizers according to the current time (step).
# %%
