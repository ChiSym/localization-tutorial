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
#
# Dependencies are specified in pyproject.toml.
# %%
# Global setup code

from __future__ import annotations
import json
import genstudio.plot as Plot

import functools
import jax
import jax.numpy as jnp
import genjax
from genjax import SelectionBuilder as S, ChoiceMapBuilder as C
from penzai import pz

import os
from math import sin, cos, pi


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


@pz.pytree_dataclass
class Pose(genjax.Pytree):
    p: genjax.typing.FloatArray
    hd: genjax.typing.FloatArray

    def __repr__(self):
        return f"Pose(p={self.p}, hd={self.hd})"

    def dp(self):
        return jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])

    def step_along(self, s: float) -> Pose:
        """
        Moves along the direction of the pose by a scalar and returns a new Pose.

        Args:
            s (float): The scalar distance to move along the pose's direction.

        Returns:
            Pose: A new Pose object representing the moved position.
        """
        dp = self.dp()
        new_p = self.p + s * dp
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
pose = Pose(jnp.array([1.0, 2.0]), hd=1.57)
print(pose)

# Move the pose along its direction
print(pose.step_along(5))

# Rotate the pose
print(pose.rotate(pi / 2))


# %%
# %%
@pz.pytree_dataclass
class Control(genjax.Pytree):
    ds: genjax.typing.FloatArray
    dhd: genjax.typing.FloatArray


def create_segments(points, loop_around=False):
    """Given an array of points of shape (N, 2), return an array of
    pairs of points. [p_1, p_2, p_3, ...] -> [[p_1, p_2], [p_2, p_3], ...]
    where each p_i is is [x_i, y_i]"""
    a = jnp.stack([points, jnp.roll(points, shift=-1, axis=0)], axis=1)
    return a if loop_around else a[:-1]


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
    clutters = jax.vmap(create_segments)(clutters_vec)
    # clutters = [create_segments(clutter, loop_around=loop_around) for clutter in clutters_vec]

    # Combine all points for bounding box calculation
    all_points_np = jnp.vstack(
        (jnp.array(walls_vec), jnp.concatenate(clutters_vec), jnp.array([start.p]))
    )
    x_min, y_min = jnp.min(all_points_np, axis=0)
    x_max, y_max = jnp.max(all_points_np, axis=0)

    # Calculate bounding box, box size, and center point
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = jnp.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])

    # Determine the total number of control steps
    T = len(controls.ds)

    return (
        {
            "walls": walls,
            "clutters": clutters,
            "bounding_box": bounding_box,
            "box_size": box_size,
            "center_point": center_point,
        },
        {"start": start, "controls": controls},
        T,
    )


def load_world(file_name, loop_around=False):
    """
    Loads the world configuration from a specified file and constructs the world.

    Args:
    - file_name (str): The name of the file containing the world configuration.
    - loop_around (bool, optional): Whether to connect the last and first vertices of walls and clutters. Defaults to False.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    with open(file_name, "r") as file:
        data = json.load(file)

    walls_vec = jnp.array(data["wall_verts"])
    clutters_vec = jnp.array(data["clutter_vert_groups"])
    start = Pose(
        jnp.array(data["start_pose"]["p"], dtype=float), float(data["start_pose"]["hd"])
    )
    controls = Control(
        jnp.array([control["ds"] for control in data["program_controls"]]),
        jnp.array([control["dhd"] for control in data["program_controls"]]),
    )

    return make_world(walls_vec, clutters_vec, start, controls, loop_around=loop_around)


# %%
# Specific example code here

world, robot_inputs, T = load_world("../example_20_program.json")

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
    path = [robot_inputs["start"]]

    # Iterate over each control step to compute the new pose and add it to the path

    controls = robot_inputs["controls"]
    for i in range(len(controls.ds)):
        p = path[-1].p + controls.ds[i]
        hd = path[-1].hd + controls.dhd[i]
        path.append(Pose(p, hd))

    for control in robot_inputs["controls"]:
        # Compute the new position (p) by applying the distance change (ds) in the direction of dp
        # Note: dp is derived from the current heading (hd) to ensure movement in the correct direction
        p = path[-1].p + control.ds * path[-1].dp()
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
    TODO: update commentary
    """
    det = u[0] * v[1] - u[1] * v[0]
    return jnp.where(
        det < PARALLEL_TOL,
        jnp.array([-jnp.inf, -jnp.inf]),
        jnp.array(
            [
                (v[0] * (p[1] - q[1]) - v[1] * (p[0] - q[0])) / det,
                (u[1] * (q[0] - p[0]) - u[0] * (q[1] - p[1])) / det,
            ]
        ),
    )

def distance(p, seg):
    """
    Computes the distance from a pose to a segment, considering the pose's direction.

    Args:
    - p: The Pose object.
    - seg: The Segment object.

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    a = solve_lines(p.p, p.dp(), seg[0], seg[1] - seg[0])
    return jnp.where(
        (a[0] >= 0.0) & (a[1] >= 0.0) & (a[1] <= 1.0),
        a[0],
        jnp.inf,
    )
    # if s is None or s < 0 or not (0 <= t <= 1):
    #     return jnp.inf
    # else:
    #     return s


def compute_wall_normal(wall_normal_direction):
    normalized_wall_direction = jnp.divide(
        wall_normal_direction, jnp.linalg.norm(wall_normal_direction)
    )
    return jnp.array([-normalized_wall_direction[1], normalized_wall_direction[0]])


@jax.jit
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
    step_direction = p2 - p1
    step_pose = Pose(p1, jnp.atan2(step_direction[1], step_direction[0]))

    # this should be a vmap of distance over world_inputs['vwalls'] with step_pose held constant
    # using in_axes
    distances = jax.vmap(distance, in_axes=(None, 0))(step_pose, world_inputs["walls"])

    closest_wall_index = jnp.argmin(distances)
    closest_wall_distance = distances[closest_wall_index]
    closest_wall = jax.tree.map(lambda v: v[closest_wall_index], world_inputs["walls"])
    wall_normal_direction = closest_wall[1] - closest_wall[0]
    wall_normal = compute_wall_normal(wall_normal_direction)
    step_length = jnp.linalg.norm(step_direction)
    collision_point = jnp.add(p1, jnp.multiply(closest_wall_distance, step_pose.dp()))
    wall_normal = jnp.where(
        jnp.cross(step_pose.dp(), wall_normal_direction) < 0, -wall_normal, wall_normal
    )
    bounce_off_point = jnp.add(
        collision_point, jnp.multiply(world_inputs["bounce"], wall_normal)
    )

    return Pose(
        jnp.where(closest_wall_distance >= step_length, p2, bounce_off_point), hd
    )


# %%
def integrate_controls(robot_inputs, world_inputs):
    """
    Integrates controls to generate a path, taking into account physical interactions with walls.

    Args:
    - robot_inputs: Dictionary containing the starting pose and control steps.
    - world_inputs: Dictionary containing the world configuration.

    Returns:
    - list: A list of Pose instances representing the path taken by applying the controls.
    """
    path = [robot_inputs["start"]]

    controls = robot_inputs["controls"]
    for i in range(len(controls.ds)):
        next_position = path[-1].p + controls.ds[i] * path[-1].dp()
        next_heading = path[-1].hd + controls.dhd[i]
        path.append(
            physical_step(path[-1].p, next_position, next_heading, world_inputs)
        )

    # for control in robot_inputs['controls']:
    #     next_position = path[-1].p + control.ds * path[-1].dp()
    #     next_heading = path[-1].hd + control.dhd
    #     path.append(physical_step(path[-1].p, next_position, next_heading, world_inputs))
    return path


# %%

# How bouncy the walls are in this world.
world_inputs = {
    "walls": world["walls"],
    "bounce": 0.1,
}

path_integrated = integrate_controls(robot_inputs, world_inputs)

# %% [markdown]
# ### Plot such data

# %%


def clutter_points(clutter):
    points = []
    for segment in clutter:
        points.append(segment[0])
    points.append(clutter[-1][1])
    return points


def arrowhead_line(point, heading, wingLength=0.4, wingAngle=pi / 4, **kwargs):
    leftWingAngle = heading + wingAngle
    rightWingAngle = heading - wingAngle

    leftWingEnd = [
        point[0] - wingLength * cos(leftWingAngle),
        point[1] - wingLength * sin(leftWingAngle),
    ]
    rightWingEnd = [
        point[0] - wingLength * cos(rightWingAngle),
        point[1] - wingLength * sin(rightWingAngle),
    ]

    return Plot.line([leftWingEnd, point, rightWingEnd], **kwargs)


def pose_arrow(p, r=0.5, **kwargs):
    start = p.p
    end = p.step_along(r).p
    opts = {"strokeWidth": 2, **kwargs}
    return Plot.line([start, end], **opts) + arrowhead_line(end, p.hd, **opts)


# Plot the world with walls only
world_plot = Plot.new(
    [
        Plot.line(wall, strokeWidth=1, tip=False, stroke=Plot.constantly("walls"))
        for wall in world["walls"]
    ],
    {"title": "Given data", "width": 500, "height": 500, "margin": 0, "inset": 50},
    Plot.color_legend,
    Plot.color_map(
        {
            "walls": "#000000",
            "clutters": "magenta",
            "path from integrating controls": "lightgreen",
            "given start pose": "darkgreen",
        }
    ),
    Plot.frame(strokeWidth=4, stroke="#ddd"),
)
world_plot + controls_path_plot + starting_pose_plot + clutters_plot
# %%

# Plot of the starting pose of the robot
starting_pose_plot = pose_arrow(
    robot_inputs["start"], stroke=Plot.constantly("given start pose")
)

# Plot of the path from integrating controls
controls_path_plot = Plot.dot(
    [pose.p for pose in path_integrated],
    fill=Plot.constantly("path from integrating controls"),
)

# Plot of the clutters
clutters_plot = [
    Plot.line(clutter_points(clutter), fill=Plot.constantly("clutters"))
    for clutter in world["clutters"]
]

# Save the figure to a file
# plt.savefig("imgs/given_data")

# Following this initial display of the given data, we suppress the clutters until much later in the notebook.

# world_plot + controls_path_plot + starting_pose_plot + clutters_plot
world_plot
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
@genjax.gen
def start_pose_prior(start, motion_settings):
    p = genjax.mv_normal(start.p, motion_settings["p_noise"] ** 2 * jnp.eye(2)) @ "p"
    hd = genjax.normal(start.hd, motion_settings["hd_noise"]) @ "hd"
    return Pose(p, hd)


@genjax.gen
def step_model(start, c, world_inputs, motion_settings):
    p = (
        genjax.mv_normal(
            start.p + c.ds * start.dp(), motion_settings["p_noise"] ** 2 * jnp.eye(2)
        )
        @ "p"
    )
    hd = genjax.normal(start.hd + c.dhd, motion_settings["hd_noise"]) @ "hd"
    return physical_step(start.p, p, hd, world_inputs)


# %% [markdown]
# Returning to the code, we can call a GF like a normal function and it will just run stochastically:


# %%
# Generate points on the unit circle
theta = jnp.linspace(0, 2 * jnp.pi, 500)
unit_circle_xs = jnp.cos(theta)
unit_circle_ys = jnp.sin(theta)


# Function to create a circle with center p and radius r
def make_circle(p, r):
    return (p[0] + r * unit_circle_xs, p[1] + r * unit_circle_ys)


# %%
# Set the motion settings
motion_settings = {"p_noise": 0.5, "hd_noise": 2 * jnp.pi / 36.0}

# Generate N_samples of starting poses from the prior
N_samples = 50
key = jax.random.PRNGKey(314159)

sub_keys = jax.random.split(key, N_samples + 1)
key = sub_keys[0]
# pose_samples = [start_pose_prior.simulate(k, (robot_inputs['start'], motion_settings)) for k in sub_keys]
pose_samples = jax.vmap(start_pose_prior.simulate, in_axes=(0, None))(
    sub_keys[1:], (robot_inputs["start"], motion_settings)
)


poses = pose_samples.get_retval()
poses


def poses_to_plots(poses: Pose):
    return list(map(lambda p, hd: pose_arrow(Pose(p, hd)), poses.p, poses.hd))


poses_plot = functools.reduce(lambda p, q: p + q, poses_to_plots(poses))

# Plot the world, starting pose samples, and 95% confidence region

# Calculate the radius of the 95% confidence region
confidence_circle = Plot.scaled_circle(
    robot_inputs["start"].p[0],
    robot_inputs["start"].p[1],
    r=2.5 * motion_settings["p_noise"],
    opacity=0.25,
    fill="red",
)

world_plot + confidence_circle + poses_plot


# %%
# %% [markdown]
# ### Traces: choice maps
#
# We can also perform *traced execution* of a generative function using the construct `Gen.simulate`.  This means that certain information is recorded during execution and packaged into a *trace*, and this trace is returned instead of the bare return value sample.
#
# The foremost information stored in the trace is the *choice map*, which is an associative array from labels to the labeled stochastic choices, i.e. occurrences of the `~` operator, that were encountered.  It is accessed by `Gen.get_choices`:

# %%
# `simulate` takes the GF plus a tuple of args to pass to it.
# trace = simulate(start_pose_prior, (robot_inputs.start, motion_settings))
# get_choices(trace)

# NOTE(colin): we don't need to run simulate again; we already have
# a trace from the plot above. We'll reuse that in what follows.

# %% [markdown]
# The choice map being the point of focus of the trace in most discussions, we often abusively just speak of a *trace* when we really mean its *choice map*.

# %% [markdown]
# ### GenJAX API for traces
#
# One can access the primitive choices in a trace using the method `get_choices`.
# One can access from a trace the GF that produced it using `Gen.get_gen_fn`, along with with arguments that were supplied using `Gen.get_args`, and the return value sample of the GF using the method `get_retval()`.  See below the fold for examples of all these.

# %%

pose_choices = pose_samples.get_choices()


# %%
pose_choices["hd"]

# %%
pose_choices["p"]
# %%
pose_samples.get_gen_fn()

# %%
pose_samples.get_args()

# %%
pose_samples.get_retval()

# TODO(colin,huebert): We have used a vector trace here, instead of a single
# trace. That shows the JAX-induced structure. It might be simpler to simulate
# to get a single trace without vmap, or go ahead and show the JAX stuff here.

# %%

# %% [markdown]
# ### Traces: scores/weights/densities
#
# Traced execution of a generative function also produces a particular kind of score/weight/density.  It is very important to be clear about which score/weight/density value is to be expected, and why.  Consider the following generative function
# ```
# p = 0.25
# @genjax.gen
# def g(x,y):
#   flip = genjax.flip(p) @ 'flip'
#   return jax.lax.select(flip, x, y)
# end
# ```
# that, given two inputs `x` and `y`, flips a coin with weight `p`, and accordingly returns `x` or `y`.  When `x` and `y` are unequal, a sensible reporting of the score/weight/density in the sampling process would produce `p` or `1.0-p` accordingly.  If the user supplied equal values `x == y`, then which score/weight/density should be returned?
#
# One tempting view identifies a GF with a *distribution over its return values*.  In this view, the correct score/weight/density of `g` above would be $1$.
#
# The mathematical picture would be as follows.  Given a stochastic function $g$ from $X$ to $X'$, the results of calling $g$ on the input $x$ are described by a probability distribution $k_{g;x}$ on $X'$.  A family of probability distributions of this form is called a *probability kernel* and is indicated by the dashed arrow $k_g \colon X \dashrightarrow X'$.  And for some $x,x'$ we would be seeking the density $k_{g;x}(x')$ with which the sample $x' \sim k_{g;x}$ occurs.  Pursuing this approach requires knowlege of all execution histories that $g$ that might have followed from $x$ to $x'$, and then performing a sum or integral over them.  For some small finite situations this may be fine, but this general problem of computing marginalizations is computationally impossible.
#
# The marginalization question is especially forced upon us when trying to compose stochastic functions.  Given a second stochastic function $g'$ from $X'$ to $X''$, corresponding to a probability kernel $k_{g'} \colon X' \dashrightarrow X''$, the composite $g' \circ g$ from $X$ to $X''$ should correspond to the following probability kernel $k_{g' \circ g} \colon X \dashrightarrow X''$.  To sample $x'' \sim k_{g' \circ g;x}$ means "sample $x' \sim k_{g;x}$, then sample $x'' \sim k_{g';x'}$, then return $x''$".  However, computing the density $k_{g' \circ g;x}(x'')$, even if one can compute $k_{g;x}(x')$ and $k_{g';x'}(x'')$ for any given $x,x',x''$, would require summing or integrating over all possible intermediate values $x'$ (which manifests an "execution history" of $g' \circ g$) that could have intervened in producing $x''$ given $x$.
#
# Therefore, distribution-over-return-values is ***not the viewpoint of Gen***, and the score/weight/density being introduced here is a ***different number***.
#
# The only thing a program can reasonably be expected to know is the score/weight/density of its arriving at its return value *via the particular stochastic computation path* that got it there, and the approach of Gen is to report this number.  The corresponding mathematical picture imagines GFs as factored into *distributions over choice maps*, whose score/weight/density is computable, together with *deterministic functions on these data* that produce the return value from them.  In mathematical language, a GF $g$ from $X$ to $X'$ corresponds to the data of an auxiliary space $U_g$ containing all of the choice map information, a probability kernel $k_g \colon X \dashrightarrow U_g$ (with computable density) embodying the stochastic execution history, and a deterministic function that we will (somewhat abusively) denote $g \colon X \times U_g \to X'$ embodying extraction of the return value from the particular stochastic execution choices.
#
# In the toy example `g` above, choice map consists of `flip` so the space $U_g$ is binary; the deterministic computation $g$ amounts to the `return` statement; and the score/weight/density is `p` or `1.0-p`, regardless of whether the inputs are equal.
#
# Tractable compositionality holds in this formulation; let's spell it out.  If another GF $g'$ from $X'$ to $X''$ has data $U_{g'}$, $k_{g'}$, and $g' \colon X' \times U_{g'} \to X''$, then the composite GF $g' \circ g$ from $X$ to $X''$ has the following data.
# * The auxiliary space is $U_{g' \circ g} := U_g \times U_{g'}$.
# * The kernel $k_{g' \circ g}$ is defined by "sample $u \sim k_{g;x}$, then compute $x' = \text{ret}_g(x,u)$, then sample $u' \sim k_{g';x'}$, then return $(u,u')$", and
# * its density is computed via $k_{g' \circ g; x}(u,u') := k_{g;x}(u) \cdot k_{g';g(x,u)}(u')$.
# * The return value function is $(g' \circ g)(x,(u,u')) := g'(g(x,u),u')$.
#
# As one composes more GFs, the auxiliary space accumulates more factors $U$, reflecting how the "execution history" consists of longer and longer records.
#
# In this picture, one may still be concerned with the distribution on return values as in the straw man viewpoint.  This information is still embodied in the aggregate of the stochastic executions that lead to any return value, together with their weights.  (Consider that this is true even in the toy example!  More math?)  In a sense, when we kick the can of marginalization down the road, we can proceed without difficulty.
#
# A final caveat: The common practice of confusing traces with their choice maps continues here, and we speak of a GF inducing a "distribution over traces".

# %% [markdown]
# Let's have a look at the score/weight/densities in our running example.
#
# A pose consists of a pair $z = (z_\text p, z_\text{hd})$ where $z_\text p$ is a position vector and $z_\text{hd}$ is an angle.  A control consists of a pair $(s, \eta)$ where $s$ is a distance of displacement and $\eta$ is a change in angle.  Write $u(\theta) = (\cos\theta, \sin\theta)$ for the unit vector in the direction $\theta$.  We are given a "world" $w$ and "motion settings" parameters $\nu = (\nu_\text p, \nu_\text{hd})$.
#
# The models `start_pose_prior` and `step_model` correspond to distributions over their traces, respectively written $\text{start}$ and $\text{step}$.  In both cases these traces consist of the choices at addresses `:p` and `:hd`, so they may be identified with poses $z$ as above.  The distributions are defined as follows, when $y$ is a pose:
# * $z \sim \text{start}(y, \nu)$ means that $z_\text p \sim \text{mvnormal}(y_\text p, \nu_\text p^2 I)$ and $z_\text{hd} \sim \text{normal}(y_\text{hd}, \nu_\text{hd})$ independently.
# * $z \sim \text{step}(y, (s, \eta), w, \nu)$ means that $z_\text p \sim \text{mvnormal}(y_\text p + s\,u(y_\text{hd}), \nu_\text p^2 I)$ and $z_\text{hd} \sim \text{normal}(y_\text{hd} + \eta, \nu_\text {hd})$ independently.
#
# The return values $\text{retval}(z)$ of these models are obtained from traces $z$ by reducing $z_\text{hd}$ modulo $2\pi$, and in the second case applying collision physics (relative to $w$) to the path from $y_\text p$ to $z_\text p$.  (We invite the reader to imagine if PropComp required us to compute the marginal density of the return value here!)  We have the following closed form for the density functions:
# $$\begin{align*}
# P_\text{start}(z; y, \nu)
# &= P_\text{mvnormal}(z_\text p; y_\text p, \nu_\text p^2 I)
# \cdot P_\text{normal}(z_\text{hd}; y_\text{hd}, \nu_\text{hd}), \\
# P_\text{step}(z; y, (s, \eta), w, \nu)
# &= P_\text{mvnormal}(z_\text p; y_\text p + s\,u(y_\text{hd}), \nu_\text p^2 I)
# \cdot P_\text{normal}(z_\text{hd}; y_\text{hd} + \eta, \nu_\text{hd}).
# \end{align*}$$
#
# In general, the density of any trace factors as the product of the densities of the individual primitive choices that appear in it.  Since the primitive distributions of the language are equipped with efficient probability density functions, this overall computation is tractable.  It is represented by `Gen.get_score`:

# %%
pose_samples.get_score()


# %% [markdown]
# #### Subscores/subweights/subdensities
#
# Instead of (the log of) the product of all the primitive choices made in a trace, one can take the product over just a subset using `Gen.project`.  See below the fold for examples.

# %%

# jax.random.split(jax.random.PRNGKey(3333), N_samples).shape

ps0 = jax.tree.map(lambda v: v[0], pose_samples)
(
    ps0.project(jax.random.PRNGKey(2), S[()]),
    ps0.project(jax.random.PRNGKey(2), S["p"]),
    ps0.project(jax.random.PRNGKey(2), S["p"] | S["hd"]),
)

# ps0.get_choices()

# jax.vmap(pose_samples.project, in_axes=(0, None))(
#     jax.random.split(jax.random.PRNGKey(2), N_samples),
#     all
# )

# jax.vmap(pose_samples.project, in_axes=(0, None))(
#     jax.random.split(jax.random.PRNGKey(3333), N_samples),
#     all
# )
# project(trace, select())

# %%
# project(trace, select(:p))

# %%
# project(trace, select(:hd))


# %%
# project(trace, select(:p, :hd)) == get_score(trace)


@genjax.gen
def path_model_start(robot_inputs, motion_settings):
    return start_pose_prior(robot_inputs["start"], motion_settings) @ (
        "initial",
        "pose",
    )


def make_path_model_step(world_inputs, motion_settings):
    @genjax.scan_combinator(max_length=T)
    @genjax.gen
    def path_model_step(previous_pose, control):
        return step_model(previous_pose, control, world_inputs, motion_settings) @ (
            "steps",
            "pose",
        ), None

    return path_model_step


# prefix_address(t, rest) = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest)
# get_path(trace) = [trace[prefix_address(t, :pose)] for t in 1:(get_args(trace)[1]+1)];

key, sub_key1, sub_key2 = jax.random.split(key, 3)
initial_pose = path_model_start.simulate(sub_key1, (robot_inputs, motion_settings))
path_model_step = make_path_model_step(world_inputs, motion_settings)

# %%

step_model.simulate(
    jax.random.PRNGKey(222),
    (
        initial_pose.get_retval(),
        jax.tree.map(lambda v: v[0], robot_inputs["controls"]),
        world_inputs,
        motion_settings,
    ),
)

# %%
jitted = jax.jit(path_model_step.simulate)

# %%

arg_tuple = initial_pose.get_retval(), robot_inputs["controls"]
key, sub_key = jax.random.split(key)

steps = jitted(sub_key, arg_tuple)

world_plot + poses_to_plots(steps.inner.get_retval()[0])
# %%

key, sub_key = jax.random.split(key)
#Plot.autoGrid(
Plot.new(
[
    world_plot + poses_to_plots(jitted(key, arg_tuple).inner.get_retval()[0])
    for key in jax.random.split(key, 10)
])

# %%
