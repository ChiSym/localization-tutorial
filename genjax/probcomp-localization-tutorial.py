# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# %%
# Global setup code

# The dependencies consist of the following Python packages.
import datetime
import json  
import matplotlib.pyplot as plt  
import genjax 
import os
from math import cos, sin, atan2, pi, sqrt
from typing import Optional
from __future__ import annotations
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

def norm(v):
    return sqrt(sum(v**2))

class Pose:
    def __init__(self, p: list[float], hd: Optional[float] = None, dp: Optional[list[float]] = None):
        """
        Initializes a Pose object either from a heading (hd) or a direction vector (dp).
        
        :param p: The position as a list of floats [x, y].
        :param hd: The heading in radians. Optional if dp is provided.
        :param dp: The direction vector as a list of floats [dx, dy]. Optional if hd is provided.
        """
        self.p = p
        if hd is not None:
            self.hd = hd % (2 * pi)  # Ensuring the heading is within 0 to 2π
            self.dp = [cos(self.hd), sin(self.hd)]
        elif dp is not None:
            self.hd = atan2(dp[1], dp[0])
            self.dp = dp
        else:
            raise ValueError("Either 'hd' (heading) or 'dp' (direction vector) must be provided, not both None.")

    def __repr__(self):
        return f"Pose(p={self.p}, hd={self.hd})"

    def step_along(self, s: float) -> Pose:
        """Moves along the direction of the pose by a scalar and returns a new Pose."""
        new_p = [self.p[0] + s * self.dp[0], self.p[1] + s * self.dp[1]]
        return Pose(new_p, hd=self.hd)

    def rotate(self, a: float) -> Pose:
        """Rotates the pose by angle 'a' (in radians) and returns a new Pose."""
        new_hd = self.hd + a
        return Pose(self.p, hd=new_hd)

# Example usage:
pose = Pose([1.0, 2.0], hd=1.57)
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
            self.p1 = p1
            self.p2 = p2
        self.dp = [p2_i - p1_i for p1_i, p2_i in zip(self.p1, self.p2)]

    def __repr__(self):
        return f"Segment({self.p1}, {self.p2})"

# %%

# %% 

class Control:
    def __init__(self, ds: float, dhd: float):
        self.ds = ds
        self.dhd = dhd

def create_segments(verts, loop_around=False):
    segs = [Segment(p1, p2) for p1, p2 in zip(verts[:-1], verts[1:])]
    if loop_around:
        segs.append(Segment(verts[-1], verts[0]))
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
    all_points = walls_vec + [item for sublist in clutters_vec for item in sublist] + [start.p]
    x_min = min(p[0] for p in all_points)
    x_max = max(p[0] for p in all_points)
    y_min = min(p[1] for p in all_points)
    y_max = max(p[1] for p in all_points)
    
    # Calculate bounding box, box size, and center point
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
    
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
    
    walls_vec = [list(map(float, vert)) for vert in data["wall_verts"]]
    clutters_vec = [[list(map(float, vert)) for vert in clutter] for clutter in data["clutter_vert_groups"]]
    start = Pose(list(map(float, data["start_pose"]["p"])), float(data["start_pose"]["hd"]))
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
        p = [path[-1].p[0] + control.ds * path[-1].dp[0], path[-1].p[1] + control.ds * path[-1].dp[1]]
        # Compute the new heading (hd) by adding the heading change (dhd)
        hd = path[-1].hd + control.dhd
        # Create a new Pose with the updated position and heading, and add it to the path
        path.append(Pose(p, hd))
    
    return path

# %%
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
        return float('inf')
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
        wall_normal = [-normalized_wall_direction[1], normalized_wall_direction[0]]
        
        if np.cross(step_pose.dp, wall_normal_direction) < 0:
            wall_normal = -np.array(wall_normal)
        
        bounce_off_point = np.add(collision_point, np.multiply(world_inputs['bounce'], wall_normal))
        return Pose(bounce_off_point.tolist(), hd)

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
        next_position = np.add(path[-1].p, np.multiply(control.ds, path[-1].dp)).tolist()
        next_heading = path[-1].hd + control.dhd
        path.append(physical_step(path[-1].p, next_position, next_heading, world_inputs))
    return path

# %%

# Define the world's inputs including walls and how bouncy they are.
world_inputs = {'walls': world['walls'], 'bounce': 0.1}

# Integrate the path based on robot inputs and world configuration.
path_integrated = integrate_controls(robot_inputs, world_inputs)

def flatten(segments_list):
        flat_list = []
        for item in segments_list:
            if isinstance(item, list):
                flat_list.extend(flatten(item))
            else:
                flat_list.append(item)
        return flat_list

# Function to plot a segment or a list of segments.
def plot_segments(segments, color='black', label=None, linewidth=0.5):
    """
    Plots segments on a matplotlib plot with a specified line width. Can handle a single segment or a list of segments.
    Each segment is expected to be an instance of the Segment class or a similar structure with p1 and p2 attributes.
    
    Args:
    - segments: A single Segment instance or a list of Segment instances to be plotted.
    - color: The color of the segments.
    - label: The label for the segments.
    - linewidth: The width of the lines used to plot the segments.
    """
    segments = [segments] if isinstance(segments, Segment) else flatten(segments)

    for i, seg in enumerate(segments):
        current_label = label if i == 0 else None
        plt.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], color=color, label=current_label, linewidth=linewidth)

        
# Define arrow style options for plotting a pose.
arrow_options = {
    'head_width': 0.5, 
    'head_length': 0.5, 
    'fill': False, 
    'overhang': 1, 
    'linewidth': 0.5
}

# Function to plot a pose with an optional radius and additional arguments.
def plot_pose(p, r=0.5, color='blue', label=None):
    """
    Plots a pose as an arrow on a matplotlib plot.
    The pose is represented by a point and a direction, optionally with a specified radius to extend the arrow.
    Adjusted the head width and length to make the arrow more visible.
    """
    end_point = p.step_along(r).p
    plt.arrow(p.p[0], p.p[1], end_point[0] - p.p[0], end_point[1] - p.p[1], color=color, label=label, **arrow_options)

# Function to plot the world with optional labels for walls and clutters.
def plot_world(world, title, label_world=False, show_clutters=False):
    """
    Plots the world configuration including walls and optionally clutters.
    The world is represented within a specified bounding box, with an optional title and labels for walls and clutters.
    """
    border = world['box_size'] * (3./19.)
    plt.figure(figsize=(5, 5))
    plt.axis('equal')
    plt.grid(False)
    plt.xlim(world['bounding_box'][0]-border, world['bounding_box'][1]+border)
    plt.ylim(world['bounding_box'][2]-border, world['bounding_box'][3]+border)
    plt.title(title)
    plot_segments(world['walls'], color='black', label="walls" if label_world else None)
    if show_clutters:
        plot_segments(world['clutters'], color='magenta', label="clutters")
    

# %%

# Following this initial display of the given data, we suppress the clutters until much later in the notebook.

# Plot the world without clutters
the_plot = plot_world(world, "Given data", label_world=True, show_clutters=True)

# Plot the starting pose of the robot
plot_pose(robot_inputs['start'], color='green', label="given start pose")

# Plot the path from integrating controls
x_coords = [pose.p[0] for pose in path_integrated]
y_coords = [pose.p[1] for pose in path_integrated]
plt.scatter(x_coords, y_coords, color='lightgreen', label="path from integrating controls", s=3)

plt.legend(loc='lower left', fontsize='small')

# Save the figure to a file
plt.savefig("imgs/given_data")

# Show the plot
plt.show()

# %%
