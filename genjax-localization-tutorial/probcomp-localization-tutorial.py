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
# pyright: reportUnusedExpression=false

# %% [markdown]
# # ProbComp Localization Tutorial
#
# This notebook provides an introduction to probabilistic computation (ProbComp). This term refers to a way of expressing probabilistic constructs in a computational paradigm, made precise by a probabilistic programming language (PPL). The programmer can encode their probabilistic intuition for solving a problem into an algorithm. Back-end language work automates the routine but error-prone derivations.
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
from genjax import SelectionBuilder as S
from genjax import ChoiceMapBuilder as C
from genjax.typing import FloatArray, PRNGKey
from penzai import pz

import os
from math import sin, cos, pi, atan2

# Ensure a location for image generation.
os.makedirs("imgs", exist_ok=True)

# %% [markdown]
# ## The "real world"
#
# We assume given:
# * A map of a space, together with
# * Some clutters that sometimes unexpectedly exist in that space.
#
# We also assume given a description of a robot's behavior via:
# * An estimated initial pose (position + heading), and
# * A program of controls (advance distance, followed by rotate heading).
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
    p: FloatArray
    hd: FloatArray

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
pose = Pose(jnp.array([1.0, 2.0]), hd=jnp.array(1.57))
print(pose)

# Move the pose along its direction
print(pose.step_along(5))

# Rotate the pose
print(pose.rotate(pi / 2))


# %%
@pz.pytree_dataclass
class Control(genjax.Pytree):
    ds: FloatArray
    dhd: FloatArray


def create_segments(points):
    """
    Given an array of points of shape (N, 2), return an array of
    pairs of points. [p_1, p_2, p_3, ...] -> [[p_1, p_2], [p_2, p_3], ...]
    where each p_i is [x_i, y_i]
    """
    return jnp.stack([points, jnp.roll(points, shift=-1, axis=0)], axis=1)


def make_world(walls_vec, clutters_vec, start, controls):
    """
    Constructs the world by creating segments for walls and clutters, calculates the bounding box, and prepares the simulation parameters.

    Args:
    - walls_vec (list of list of float): A list of 2D points representing the vertices of walls.
    - clutters_vec (list of list of list of float): A list where each element is a list of 2D points representing the vertices of a clutter.
    - start (Pose): The starting pose of the robot.
    - controls (list of Control): Control actions for the robot.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    # Create segments for walls and clutters
    walls = create_segments(walls_vec)
    clutters = jax.vmap(create_segments)(clutters_vec)

    # Combine all points for bounding box calculation
    all_points = jnp.vstack(
        (jnp.array(walls_vec), jnp.concatenate(clutters_vec), jnp.array([start.p]))
    )
    x_min, y_min = jnp.min(all_points, axis=0)
    x_max, y_max = jnp.max(all_points, axis=0)

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


def load_world(file_name):
    """
    Loads the world configuration from a specified file and constructs the world.

    Args:
    - file_name (str): The name of the file containing the world configuration.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    with open(file_name, "r") as file:
        data = json.load(file)

    walls_vec = jnp.array(data["wall_verts"])
    clutters_vec = jnp.array(data["clutter_vert_groups"])
    start = Pose(
        jnp.array(data["start_pose"]["p"], dtype=float),
        jnp.array(data["start_pose"]["hd"], dtype=float),
    )

    cs = jnp.array([[c["ds"], c["dhd"]] for c in data["program_controls"]])
    controls = Control(cs[:, 0], cs[:, 1])

    return make_world(walls_vec, clutters_vec, start, controls)


# %%
# Specific example code here

world, robot_inputs, T = load_world("../example_20_program.json")

# %% [markdown]
# ### Integrate a path from a starting pose and controls
#
# If the motion of the robot is determined in an ideal manner by the controls, then we may simply integrate to determine the resulting path. Naïvely, this results in the following.


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
        p = path[-1].p + controls.ds[i] * path[-1].dp()
        hd = path[-1].hd + controls.dhd[i]
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
    - s, t: Parameters for the line equations at the intersection point.
            Returns [-inf, -inf] if lines are parallel.
    """
    det = u[0] * v[1] - u[1] * v[0]
    return jnp.where(
        jnp.abs(det) < PARALLEL_TOL,
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


def compute_wall_normal(wall_direction):
    normalized_wall_direction = wall_direction / jnp.linalg.norm(wall_direction)
    return jnp.array([-normalized_wall_direction[1], normalized_wall_direction[0]])


@jax.jit
def physical_step(p1: FloatArray, p2: FloatArray, hd, world_inputs):
    """
    Computes a physical step considering wall collisions and bounces.

    Args:
    - p1, p2: Start and end points of the step.
    - hd: Heading direction.
    - world_inputs: dict containing world configuration, including walls and bounce distance.

    Returns:
    - Pose: The new pose after taking the step, considering potential wall collisions.
    """
    # Calculate step direction and length
    step_direction = p2 - p1
    step_length = jnp.linalg.norm(step_direction)
    step_pose = Pose(p1, jnp.arctan2(step_direction[1], step_direction[0]))

    # Calculate distances to all walls
    distances = jax.vmap(distance, in_axes=(None, 0))(step_pose, world_inputs["walls"])

    # Find the closest wall
    closest_wall_index = jnp.argmin(distances)
    closest_wall_distance = distances[closest_wall_index]
    closest_wall = jax.tree.map(lambda v: v[closest_wall_index], world_inputs["walls"])

    # Calculate wall normal and collision point
    wall_direction = closest_wall[1] - closest_wall[0]
    wall_normal = compute_wall_normal(wall_direction)
    collision_point = p1 + closest_wall_distance * step_pose.dp()

    # Ensure wall_normal points away from the robot's direction
    wall_normal = jnp.where(
        jnp.dot(step_pose.dp(), wall_normal) > 0, -wall_normal, wall_normal
    )

    # Calculate bounce off point
    bounce_off_point = collision_point + world_inputs["bounce"] * wall_normal

    # Determine final position based on whether a collision occurred
    final_position = jnp.where(
        closest_wall_distance >= step_length, p2, bounce_off_point
    )

    return Pose(final_position, hd)


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


def arrow_plot(start, end, wing_angle, wing_length, constants={}, **mark_options):
    mark_options = {"strokeWidth": 1.25, **mark_options}

    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = atan2(dy, dx)

    left_wing_angle = angle + wing_angle
    right_wing_angle = angle - wing_angle

    left_wing_end = {
        "x": end[0] - wing_length * cos(left_wing_angle),
        "y": end[1] - wing_length * sin(left_wing_angle),
        **constants,
    }
    right_wing_end = {
        "x": end[0] - wing_length * cos(right_wing_angle),
        "y": end[1] - wing_length * sin(right_wing_angle),
        **constants,
    }

    return Plot.line(
        [
            {**constants, "x": start[0], "y": start[1]},
            {**constants, "x": end[0], "y": end[1]},
            left_wing_end,
            {**constants, "x": end[0], "y": end[1]},
            right_wing_end,
        ],
        {"x": "x", "y": "y", **mark_options},
    )


def pose_arrow(p, r=0.5, constants={}, **opts):
    end = p.p
    start = p.step_along(-r).p
    wing_angle = pi / 4
    wing_length = 0.4

    return arrow_plot(start, end, wing_angle, wing_length, constants, **opts)


walls_plot = Plot.new(
    [
        Plot.line(
            [wall[0], wall[1]],
            strokeWidth=1,
            stroke=Plot.constantly("walls"),
        )
        for wall in world["walls"]
    ],
    {"margin": 0, "inset": 50, "width": 500, "height": 500},
    Plot.color_map(
        {
            "walls": "#ccc",
            "clutters": "magenta",
            "path from integrating controls": "lightgreen",
            "given start pose": "darkgreen",
        }
    ),
)
# Plot the world with walls only
world_plot = Plot.new(
    walls_plot,
    Plot.color_legend(),
    Plot.frame(strokeWidth=4, stroke="#ddd"),
)

# %%

# Plot of the starting pose of the robot
starting_pose_plot = pose_arrow(
    robot_inputs["start"],
    stroke=Plot.constantly("given start pose"),
    constants={"frame": 0},
)

# Plot of the path from integrating controls
controls_path_plot = Plot.dot(
    [pose.p for pose in path_integrated],
    fill=Plot.constantly("path from integrating controls"),
)

# Plot of the clutters
clutters_plot = [
    Plot.line(c[:, 0], fill=Plot.constantly("clutters")) for c in world["clutters"]
]

world_plot + controls_path_plot + starting_pose_plot + clutters_plot

# Save the figure to a file
# plt.savefig("imgs/given_data")

# Following this initial display of the given data, we suppress the clutters until much later in the notebook.

# world_plot + controls_path_plot + starting_pose_plot + clutters_plot
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


def poses_to_plots(poses: Pose, constants={}, **plot_opts):
    return list(
        map(
            lambda i, p, hd: pose_arrow(
                Pose(p, hd), constants={"step": i, **constants}, **plot_opts
            ),
            range(len(poses.p)),
            poses.p,
            poses.hd,
        )
    )


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


# %%

key, sub_key1, sub_key2 = jax.random.split(key, 3)
initial_pose = path_model_start.simulate(sub_key1, (robot_inputs, motion_settings))
path_model_step = make_path_model_step(world_inputs, motion_settings)
step_model.simulate(
    sub_key2,
    (
        initial_pose.get_retval(),
        jax.tree.map(lambda v: v[0], robot_inputs["controls"]),
        world_inputs,
        motion_settings,
    ),
)

# %%

path_model_step_simulate = jax.jit(path_model_step.simulate)


def generate_path_trace(key: PRNGKey) -> genjax.Trace:
    key, start_key = jax.random.split(key)
    initial_pose = path_model_start.simulate(start_key, (robot_inputs, motion_settings))
    key, step_key = jax.random.split(key)
    return path_model_step_simulate(
        step_key, (initial_pose.get_retval(), robot_inputs["controls"])
    )

def path_from_trace(tr: genjax.Trace) -> Pose:
    # TODO(colin): can we use one of @sritchie's new combinators to avoid
    # using `inner` to get at the trace steps?
    return tr.inner.get_retval()[0]

def generate_path(key: PRNGKey) -> Pose:
    return path_from_trace(generate_path_trace(key))


def nth(t, n: int):
    return jax.tree.map(lambda v: v[n], t)


N_samples = 12
key, sub_key = jax.random.split(key)
sample_paths_v = jax.vmap(generate_path)(jax.random.split(sub_key, N_samples))

Plot.Grid(
    [walls_plot + poses_to_plots(nth(sample_paths_v, n)) for n in range(N_samples)]
)

# Leaving this in as an a reference for animation;
# Julia animated this, but a grid seems easier to eyeball here.
# N_steps = len(robot_inputs["controls"].ds) - 1
# (
#     world_plot
#     + [poses_to_plots(path,
#                       constants={'frame': i},
#                       filter=Plot.js("({frame, step}) => frame === $state.frame && step <= $state.step")) for i, path in enumerate(sample_paths)]
#     | Plot.Slider("frame", label="Frame", range=[0, N_samples - 1], fps=2)
#     | Plot.Slider("step", label="Step", range=[0, N_steps], init=N_steps)
# )

# %%
# Animation showing a single path with confidence circles


def animate_path_with_confidence(path, motion_settings):
    frames = [
        (
            walls_plot
            # Prior poses in black
            + [
                pose_arrow(Pose(p, hd))
                for p, hd in zip(path.p[: step + 1], path.hd[: step + 1])
            ]
            # 95% confidence circle for next pose
            + [
                Plot.scaled_circle(
                    path.p[step][0],
                    path.p[step][1],
                    r=2.5 * motion_settings["p_noise"],
                    opacity=0.25,
                    fill="red",
                )
            ]
            # Next pose in red
            + [pose_arrow(Pose(path.p[step + 1], path.hd[step + 1]), stroke="red")]
            + {"axis": None}
        )
        for step in range(len(path.p) - 1)
    ]

    return Plot.Frames(frames, fps=2)


# Generate a single path
key, sample_key = jax.random.split(key)
animate_path_with_confidence(generate_path(sample_key), motion_settings)

# %% [markdown]
# ### Modfying traces
#
# The metaprogramming approach of Gen affords the opportunity to explore alternate stochastic
# execution histories.  Namely, `Gen.update` takes as inputs a trace, together with modifications
# to its arguments and primitive choice values, and returns an accordingly modified trace.
# It also returns (the log of) the ratio of the updated trace's density to the original trace's
# density, together with a precise record of the resulting modifications that played out.

# %% [markdown]
# One could, for instance, consider just the placement of the first pose, and replace its stochastic
# choice of heading with a specific value.

# %%

key, sub_key = jax.random.split(key)
trace = start_pose_prior.simulate(sub_key, (robot_inputs["start"], motion_settings))
key, sub_key = jax.random.split(key)
rotated_trace, rotated_trace_weight_diff, _, _ = trace.update(
    sub_key, C["hd"].set(jnp.pi / 2.0)
)

Plot.new(
    world_plot
    + pose_arrow(trace.get_retval(), stroke="green")
    + pose_arrow(rotated_trace.get_retval(), stroke="red"),
    Plot.color_map(
        {"walls": "#ccc", "some pose": "green", "with heading modified": "red"}
    ),
)

# %% [markdown]
# The original trace was typical under the pose prior model, whereas the modified one is
# rather less likely.  This is the log of how much unlikelier:

# %%
rotated_trace_weight_diff

# %% [markdown]
# It is worth carefully thinking through a tricker instance of this.  Suppose instead,
# within the full path, we replaced the $t = 0$ step's stochastic choice of heading
# with some specific value.

# %%
key, sub_key = jax.random.split(key)
trace = generate_path_trace(sub_key)
key, sub_key = jax.random.split(key)

# This doesn't work (See GEN-339)
try:
    rotated_first_step, rotated_first_step_weight_diff, _, _ = trace.update(
        sub_key, C[0, "steps", "pose", "hd"].set(jnp.pi / 2.0)
    )
except AttributeError as ae:
    ae

# trace.get_choices()[0, 'steps', 'pose', 'hd']
# trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))
# rotated_first_step, rotated_first_step_weight_diff, _, _ =
#     update(trace,
#            (T, robot_inputs, world_inputs, motion_settings), (NoChange(), NoChange(), NoChange(), NoChange()),
#            choicemap((:steps => 1 => :pose => :hd, π/2.)))
# the_plot = plot_world(world, "Modifying another heading")
# plot!(get_path(trace); color=:green, label="some path")
# plot!(get_path(rotated_first_step); color=:red, label="with heading at first step modified")
# savefig("imgs/modify_trace_1")
# the_plot

# %% [markdown]
# Another capability of `Gen.update` is to modify the *arguments* to the generative function
# used to produce the trace.  In our example, we might have on hand a very long list of
# controls, and we wish to explore the space of paths incrementally in the timestep:

# %%

# TODO(colin): Seek approval for the following modification of the notebook:
# I propose we skip this demonstration for the following reasons.
# It is un-JAX-like to use a for loop like this. There are probably
# better examples of wanting to change an argument than this. Further,
# this seems to step around the `scan` structure of our path generation,
# and overall looks a bit like an anti-pattern from the JAX point of view.
# The tests that are done in the loop don't reveal much about the
# statistics of the situation, so I think getting this to work is more
# trouble than it's worth.
#
# Furthermore, I think it is much more valuable to understand using
# choicemaps to adjust random choices in Gen than it is to fiddle with
# parameters.

# change_only_T = (UnknownChange(), NoChange(), NoChange(), NoChange())

# trace = simulate(path_model_loop, (0, robot_inputs, world_inputs, motion_settings))
# for t in 1:T
#     trace, _, _, _ = update(trace, (t, robot_inputs, world_inputs, motion_settings), change_only_T, choicemap())
#     # ...
#     # Do something with the trace of the partial path up to time t.
#     # ...
#     @assert has_value(get_choices(trace), :steps => t => :pose => :p)
#     @assert !has_value(get_choices(trace), :steps => (t+1) => :pose => :p)
# end

# println("Success");

# TODO(colin): discuss the following with the team.
#
# The next part of the notebook covers using the Unfold combinator in Julia
# to achieve high performance. But we have already done that by using scan
# in GenJAX, so I am skipping this section of the tutorial as contributing
# nothing new. We may, however, go back to earlier parts of the tutorial to
# explain how our choices of data structure were chosen to allow JAX to work
# at highest performance.
#
# We return to the Julia notebook at the point where sensors are introduced.

# %% [markdown]
# ### Ideal sensors
#
# We now, additionally, assume the robot is equipped with sensors that cast
# rays upon the environment at certain angles relative to the given pose,
# and return the distance to a hit.
#
# We first describe the ideal case, where the sensors return the true
# distances to the walls.

# %%

sensor_settings = {
    "fov": 2 * jnp.pi * (2 / 3),
    "num_angles": 41,
    "box_size": world["box_size"],
}


def sensor_distance(pose, walls, box_size):
    distances = jax.vmap(distance, in_axes=(None, 0))(pose, walls)
    d = jnp.min(distances)
    # Capping to a finite value avoids issues below.
    return jnp.where(jnp.isinf(d), 2.0 * box_size, d)


# This represents a "fan" of sensor angles, with given field of vision, centered at angle 0.


def make_sensor_angles(sensor_settings):
    na = sensor_settings["num_angles"]
    return sensor_settings["fov"] * (jnp.arange(na) - jnp.floor(na / 2)) / (na - 1)


sensor_angles = make_sensor_angles(sensor_settings)


def ideal_sensor(pose: Pose):
    walls = world['walls']
    box_size = sensor_settings['box_size']
    def reading(angle):
        return sensor_distance(pose.rotate(angle), walls, box_size)

    return jax.vmap(reading)(sensor_angles)


# %%
# Plot sensor data.


def plot_sensors(pose: Pose, sensor):
    readings = sensor(pose)
    projections = [
        pose.rotate(sensor_angles[j]).step_along(s) for j, s in enumerate(readings)
    ]
    return (
        [Plot.line([pose.p, p.p], stroke=Plot.constantly("#ddd")) for p in projections]
        + [Plot.dot([pose.p for pose in projections], fill="#f80")]
        + [Plot.dot([pose.p], fill="#0f0")]
    )


walls_plot + plot_sensors(initial_pose.get_retval(), ideal_sensor)
# %%


def animate_path_with_sensor(trace: genjax.Trace, sensor):
    path = path_from_trace(trace)
    # how should sensor work? If we have POSE, we can do ideal_sensor on it; if
    # we have TRACE, we should harvest the readings. since we have path from trace,
    # we can make it so it works from a trace. Let's try that.

    frames = [
        (
            walls_plot
            # Prior poses in black
            + [
                pose_arrow(Pose(p, hd))
                for p, hd in zip(path.p[: step], path.hd[: step])
            ]
            + plot_sensors(Pose(path.p[step], path.hd[step]), sensor)
            # Next pose in red
            + [pose_arrow(Pose(path.p[step], path.hd[step]), stroke="red")]
            + {"axis": None}
        )
        for step in range(1, len(path.p))
    ]

    return Plot.Frames(frames, fps=2)


key, sample_key = jax.random.split(key)
animate_path_with_sensor(generate_path_trace(sample_key), ideal_sensor)

# %% [markdown]
# ### Noisy sensors
#
# We assume that the sensor readings are themselves uncertain, say, the distances only knowable
# up to some noise.  We model this as follows.

# %%
@genjax.gen
def sensor_model(pose, angle):
    sensor_pose = pose.rotate(angle)
    return genjax.normal(sensor_distance(sensor_pose, world['walls'], sensor_settings['box_size']), sensor_settings['s_noise']) @ 'distance'

sensor_model_v = sensor_model.vmap(in_axes=(None, 0))

def noisy_sensor(pose):
    trace = sensor_model_v.simulate(key, (pose, sensor_angles))
    return trace

# # %% [markdown]
# The trace contains many choices corresponding to directions of sensor reading from the input pose.
# TODO(colin): The original notebook contains the following comment:
# To reduce notebook clutter, here we just show a subset of 5 of them:
# But I don't think we need to worry about this with penzai. We aren't relying on
# print statments to condense this data.

# %%
sensor_settings['s_noise'] = 0.10

key, sub_key = jax.random.split(key)
trace = sensor_model_v.simulate(sub_key, (robot_inputs['start'], sensor_angles))
# get_selected(get_choices(trace), select((1:5)...))
trace

# %% [markdown]
# The mathematical picture is as follows.  Given the parameters of a pose $y$, walls $w$,
# and settings $\nu$, one gets a distribution $\text{sensor}(y, w, \nu)$ over the traces
# of `sensor_model`, and when $z$ is a motion model trace we set
# $\text{sensor}(z, w, \nu) := \text{sensor}(\text{retval}(z), w, \nu)$.
# Its samples are identified with vectors $o = (o^{(1)}, o^{(2)}, \ldots, o^{(J)})$, where
# $J := \nu_\text{num\_angles}$, each $o^{(j)}$ independently following a certain normal
# distribution (depending, notably, on the distance from the pose to the nearest wall).
# Thus the density of $o$ factors into a product of the form
# $$
# P_\text{sensor}(o) = \prod\nolimits_{j=1}^J P_\text{normal}(o^{(j)})
# $$
# where we begin a habit of omitting the parameters to distributions that are implied by the code.
#
# Visualizing the traces of the model is probably more useful for orientation, so we do this now.

# %%
readings = trace.get_choices()[...,'distance']
readings

# function frame_from_sensors_trace(world, title, poses, poses_color, poses_label, pose, trace; show_clutters=false)
#     readings = [trace[j => :distance] for j in 1:sensor_settings.num_angles]
#     return frame_from_sensors(world, title, poses, poses_color, poses_label, pose,
#                              readings, "trace sensors", get_args(trace)[3];
#                              show_clutters=show_clutters)
# end;

key, sub_key = jax.random.split(key)
trace = generate_path_trace(sub_key)
path = path_from_trace(trace)
animate_path_with_sensor(generate_path_trace(sub_key), ideal_sensor)

# *** WHERE WE LEFT OFF ***
# We want to send a function to animate_path_with_sensor that would get the readings
# from the trace. ideal_sensor currently takes a pose argument which makes that
# inconvenient. Maybe we should send traces not paths to animate path with sensor,
# have it extract the path and sensor information from path with a function we supply,
# basically a lambda like readings above, otherwise tr -> pose -> ideal_sensor.


# %%
ani = Animation()
for pose in path_integrated
    trace = simulate(sensor_model, (pose, world.walls, sensor_settings))
    frame_plot = frame_from_sensors_trace(
        world, "Sensor model (samples)",
        path_integrated, :green2, "some path",
        pose, trace)
    frame(ani, frame_plot)
end
gif(ani, "imgs/sensor_1.gif", fps=1)




# %%
sensor_angles
# %%
