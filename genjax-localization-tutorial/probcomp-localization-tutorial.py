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

import itertools
import jax
import jax.numpy as jnp
import genjax
from genjax import SelectionBuilder as S
from genjax import ChoiceMapBuilder as C
from genjax.typing import FloatArray, PRNGKey
from penzai import pz
from typing import Any, Iterable

import os
from math import sin, cos, pi, atan2

import penzai.treescope

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


def nth(x, idx):
    return jax.tree_util.tree_map(lambda v: v[idx], x)


class PythonicPytree:
    """
    A class that adds support for bracket indexing/slicing, sequence-like operations,
    and concatenation to make working with pytrees more Pythonic.
    """

    def __getitem__(self, idx):
        return nth(self, idx)

    def __len__(self):
        return len(jax.tree_util.tree_leaves(self)[0])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

        def concat_leaves(x, y):
            return jnp.concatenate([x, y])

        return jax.tree_util.tree_map(concat_leaves, self, other)

    def prepend(self, child):
        return jax.tree.map(lambda x: x[jnp.newaxis], child) + self


@pz.pytree_dataclass
class Pose(genjax.Pytree, PythonicPytree):
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

    def apply_control(self, control):
        return Pose(self.p + control.ds * self.dp(), self.hd + control.dhd)

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


@pz.pytree_dataclass
class Control(genjax.Pytree, PythonicPytree):
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
    center_point = Pose(
        jnp.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]), jnp.array(0.0)
    )

    # How bouncy the walls are in this world.
    bounce = 0.1

    # Determine the total number of control steps
    T = len(controls.ds)

    return (
        {
            "walls": walls,
            "clutters": clutters,
            "bounding_box": bounding_box,
            "box_size": box_size,
            "center_point": center_point,
            "bounce": bounce,
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

noop_control = Control(jnp.array([0.0]), jnp.array([0.0]))


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
    return jax.lax.scan(
        lambda pose, control: (
            pose.apply_control(control),
            pose.apply_control(control),
        ),
        robot_inputs["start"],
        # Prepend a no-op control to include the first pose in the result
        noop_control + robot_inputs["controls"],
    )[1]


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


def compute_wall_normal(wall_direction) -> FloatArray:
    normalized_wall_direction = wall_direction / jnp.linalg.norm(wall_direction)
    return jnp.array([-normalized_wall_direction[1], normalized_wall_direction[0]])


@jax.jit
def physical_step(p1: FloatArray, p2: FloatArray, hd):
    """
    Computes a physical step considering wall collisions and bounces.

    Args:
    - p1, p2: Start and end points of the step.
    - hd: Heading direction.

    Returns:
    - Pose: The new pose after taking the step, considering potential wall collisions.
    """
    # Calculate step direction and length
    step_direction = p2 - p1
    step_length = jnp.linalg.norm(step_direction)
    step_pose = Pose(p1, jnp.arctan2(step_direction[1], step_direction[0]))

    # Calculate distances to all walls
    distances = jax.vmap(distance, in_axes=(None, 0))(step_pose, world["walls"])

    # Find the closest wall
    closest_wall_index = jnp.argmin(distances)
    closest_wall_distance = distances[closest_wall_index]
    closest_wall = world["walls"][closest_wall_index]

    # Calculate wall normal and collision point
    wall_direction = closest_wall[1] - closest_wall[0]
    wall_normal = compute_wall_normal(wall_direction)
    collision_point = p1 + closest_wall_distance * step_pose.dp()

    # Ensure wall_normal points away from the robot's direction
    wall_normal = jnp.where(
        jnp.dot(step_pose.dp(), wall_normal) > 0, -wall_normal, wall_normal
    )

    # Calculate bounce off point
    bounce_off_point: FloatArray = collision_point + world["bounce"] * wall_normal

    # Determine final position based on whether a collision occurred
    final_position = jnp.where(
        closest_wall_distance >= step_length, p2, bounce_off_point
    )

    return Pose(final_position, hd)


# %%
def integrate_controls_physical(robot_inputs):
    """
    Integrates controls to generate a path, taking into account physical interactions with walls.

    Args:
    - robot_inputs: Dictionary containing the starting pose and control steps.

    Returns:
    - Pose: A Pose object representing the path taken by applying the controls.
    """
    return jax.lax.scan(
        lambda pose, control: (
            new_pose := physical_step(
                pose.p, pose.p + control.ds * pose.dp(), pose.hd + control.dhd
            ),
            new_pose,
        ),
        robot_inputs["start"],
        noop_control + robot_inputs["controls"],
    )[1]


# %%

path_integrated = integrate_controls_physical(robot_inputs)

# %% [markdown]
# ### Plot such data
# %%
def pose_plot(p, r=0.5, constants={}, fill: str | Any = "black", **opts):
    WING_ANGLE, WING_LENGTH = jnp.pi/12, 0.6
    center = p.p
    angle = jnp.arctan2(*(center - p.step_along(-r).p)[::-1])

    # Calculate wing endpoints
    wing_ends = [
        center - WING_LENGTH * jnp.array([jnp.cos(angle + a), jnp.sin(angle + a)])
        for a in [WING_ANGLE, -WING_ANGLE]
    ]

    # Draw wings
    wings = Plot.line(
        [wing_ends[0], center, wing_ends[1]],
        strokeWidth=2,
        stroke=fill,
        opacity=0.3
    )

    # Draw center dot
    dot = Plot.scaled_circle(*center, r=0.14, fill=fill, **opts)

    return wings + dot


walls_plot = Plot.new(
    [
        Plot.line(
            [wall[0], wall[1]],
            strokeWidth=2,
            stroke="#ccc",
        )
        for wall in world["walls"]
    ],
    {"margin": 0, "inset": 50, "width": 500, "axis": None, "aspectRatio": 1},
    Plot.domain([0, 20]),
)
# Plot the world with walls only
world_plot = Plot.new(
    walls_plot, Plot.frame(strokeWidth=4, stroke="#ddd"), Plot.color_legend()
)

# %%

# Plot of the starting pose of the robot
starting_pose_plot = pose_plot(
    robot_inputs["start"],
    fill=Plot.constantly("given start pose"),
    constants={"frame": 0},
) + Plot.color_map({"given start pose": "blue"})

# Plot of the path from integrating controls
controls_path_plot = Plot.dot(
    [pose.p for pose in path_integrated],
    fill=Plot.constantly("path from integrating controls"),
) + Plot.color_map({"path from integrating controls": "#0c0"})

# Plot of the clutters
clutters_plot = (
    [Plot.line(c[:, 0], fill=Plot.constantly("clutters")) for c in world["clutters"]],
    Plot.color_map({"clutters": "magenta"}),
)

(
    world_plot
    + controls_path_plot
    + starting_pose_plot
    + clutters_plot
    + {"title": "Given Data"}
)

# %% [markdown]

# TODO(jay): Include code visualization
# %% [markdown]
# ## Gen basics
#
# As said initially, we are uncertain about the true initial position and subsequent motion of the robot.  In order to reason about these, we now specify a model using `Gen`.
#
# Each piece of the model is declared as a *generative function* (GF).  The `GenJAX` library provides a DSL for constructing GFs signalled by the use of the `@genjax.gen` decorator on an ordinary Python function. As we shall see, in order for the functions we write to be compilable for a GPU, there are certain constraints we must follow in the use of control flow, which we will discuss soon.


# The library offers two basic constructs for use within the DSL: primitive *distributions* such as "bernoulli" and "normal", and the *sampling operator* `@`.  Recursively, GFs may sample from other GFs using `@`.
# %% [markdown]
# ### Components of the motion model
#
# We start with the two building blocks: the starting pose and individual steps of motion.
# %%
@genjax.gen
def start_pose_prior(start, motion_settings):
    p = genjax.mv_normal(start.p, motion_settings["p_noise"] ** 2 * jnp.eye(2)) @ "p"
    hd = genjax.normal(start.hd, motion_settings["hd_noise"]) @ "hd"
    return Pose(p, hd)


@genjax.gen
def step_model(start, c, motion_settings):
    p = (
        genjax.mv_normal(
            start.p + c.ds * start.dp(), motion_settings["p_noise"] ** 2 * jnp.eye(2)
        )
        @ "p"
    )
    hd = genjax.normal(start.hd + c.dhd, motion_settings["hd_noise"]) @ "hd"
    return physical_step(start.p, p, hd)


# Set the motion settings
default_motion_settings = {"p_noise": 0.5, "hd_noise": 2 * jnp.pi / 36.0}

# %% [markdown]
# Returning to the code: we find that our function cannot be called directly--it is now a stochastic function!--so we must supply a source of randomness, in the form of a *key*, followed by a tuple of the function's expected arguments, illustrated here:

# %%
key = jax.random.PRNGKey(0)
start_pose_prior.simulate(
    key, (robot_inputs["start"], default_motion_settings)
).get_retval()

# %% [markdown]

# We called `get_retval()` on the result, which is a *trace*, a data structure with which we become much more familiar before we are done.

# %%
# Generate points on the unit circle
theta = jnp.linspace(0, 2 * jnp.pi, 500)
unit_circle_xs = jnp.cos(theta)
unit_circle_ys = jnp.sin(theta)


# Function to create a circle with center p and radius r
def make_circle(p, r):
    return (p[0] + r * unit_circle_xs, p[1] + r * unit_circle_ys)


# %%

# Generate N_samples of starting poses from the prior
N_samples = 50
key, sub_key = jax.random.split(key)
# pose_samples = [start_pose_prior.simulate(k, (robot_inputs['start'], motion_settings)) for k in sub_keys]
pose_samples = jax.vmap(step_model.simulate, in_axes=(0, None))(
    jax.random.split(sub_key, N_samples),
    (robot_inputs["start"], robot_inputs["controls"][0], default_motion_settings),
)


def poses_to_plots(poses: Iterable[Pose], constants={}, **plot_opts):
    return [
        pose_plot(pose, constants={"step": i, **constants}, **plot_opts)
        for i, pose in enumerate(poses)
    ]


# Plot the world, starting pose samples, and 95% confidence region


# Calculate the radius of the 95% confidence region
def confidence_circle(pose: Pose, motion_settings: dict):
    # TODO
    # should this also take into account the hd_noise?
    return Plot.scaled_circle(
        *pose.p,
        fill=Plot.constantly("95% confidence region"),
        r=2.5 * motion_settings["p_noise"],
    ) + Plot.color_map({"95% confidence region": "rgba(255,0,0,0.25)"})


(
    world_plot
    + poses_to_plots([robot_inputs["start"]], fill=Plot.constantly("step from here"))
    + confidence_circle(
        robot_inputs["start"].apply_control(robot_inputs["controls"][0]),
        default_motion_settings,
    )
    + poses_to_plots(pose_samples.get_retval(), fill=Plot.constantly("step samples"))
    + Plot.color_map({"step from here": "#000", "step samples": "red"})
)

# %% [markdown]
# ### Traces: choice maps
#
# The actual return value of `step_model.simulate` is a *trace*, which records certain information obtained during execution of the function.
#
# The foremost information stored in the trace is the *choice map*, which is tree of labels mapping to the corresponding stochastic choices, i.e. occurrences of the `@` operator, that were encountered.  It is accessed by `get_choices`:

# %%
# `simulate` takes the GF plus a tuple of args to pass to it.
key, sub_key = jax.random.split(key)
trace = start_pose_prior.simulate(
    sub_key, (robot_inputs["start"], default_motion_settings)
)
trace.get_choices()


# %% [markdown]
# The choice map being the point of focus of the trace in most discussions, we often abusively just speak of a *trace* when we really mean its *choice map*.

# %% [markdown]
# ### GenJAX API for traces
#
# One can access the primitive choices in a trace using the method `get_choices`.
# One can access from a trace the GF that produced it using `trace.get_gen_fn()`, along with with arguments that were supplied using `trace.get_args()`, and the return value sample of the GF using the method `get_retval()`.  See below the fold for examples of all these.

# %%
pose_choices = trace.get_choices()
# %%
pose_choices["hd"]
# %%
pose_choices["p"]
# %%
trace.get_gen_fn()
# %%
trace.get_args()
# %%
trace.get_retval()
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
trace.get_score()


# %% [markdown]
# #### Subscores/subweights/subdensities
#
# Instead of (the log of) the product of all the primitive choices made in a trace, one can take the product over just a subset using `Gen.project`.  See below the fold for examples.

# %% [hide-input]

# jax.random.split(jax.random.PRNGKey(3333), N_samples).shape

ps0 = jax.tree.map(lambda v: v[0], pose_samples)
(
    ps0.project(jax.random.PRNGKey(2), S[()]),
    ps0.project(jax.random.PRNGKey(2), S["p"]),
    ps0.project(jax.random.PRNGKey(2), S["p"] | S["hd"]),
)

key, sub_key = jax.random.split(key)
trace.project(key, S[()])
# %%
key, sub_key = jax.random.split(key)
trace.project(key, S[("p")])
# %%
key, sub_key = jax.random.split(key)
trace.project(key, S["p"] | S["hd"])

# %% [markdown]
# If in fact all of those projections resulted in the same number, you have encountered the issue [GEN-316](https://linear.app/chi-fro/issue/GEN-316/project-is-broken-in-genjax).
#
# ### Modeling a full path
#
# The model contains all information in its trace, rendering its return value redundant.  The noisy path integration will just be a wrapper around its functionality, extracting what it needs from the trace.
#
# (It is worth acknowledging two strange things in the code below: the use of the suffix `.accumulate()` in path_model and the use of that auxiliary function itself.
# %%

@genjax.gen
def path_model_start(robot_inputs, motion_settings):
    return start_pose_prior(robot_inputs["start"], motion_settings) @ (
        "initial",
        "pose",
    )

@genjax.gen
def path_model_step(motion_settings, previous_pose, control):
    return step_model(previous_pose, control, motion_settings) @ (
        "steps",
        "pose",
    )


def gen_partial(gen_fn, closed_over):
    @genjax.gen
    def inner(*args):
        return gen_fn.inline(closed_over, *args)
    return inner

path_model = gen_partial(path_model_step, default_motion_settings).accumulate()

# TODO(colin,huebert): talk about accumulate, what it does, and _why_ from the point of view of acceleration. This is the flow control modification we were hinting at above, and it constrains the step function to have the two-argument signature that it does, which is why we reached for `partial` in the first place. Emphasize that this small bit of preparation allows massively parallel execution on a GPU and so it's worth the hassle.

key, sub_key1, sub_key2 = jax.random.split(key, 3)
initial_pose = path_model_start.simulate(
    sub_key1, (robot_inputs, default_motion_settings)
)
step_model.simulate(
    sub_key2,
    (
        initial_pose.get_retval(),
        robot_inputs["controls"][0],
        default_motion_settings,
    ),
)

# %%


def generate_path_trace(key: PRNGKey) -> genjax.Trace:
    key, start_key = jax.random.split(key)
    initial_pose = path_model_start.simulate(
        start_key, (robot_inputs, default_motion_settings)
    )
    key, step_key = jax.random.split(key)
    return path_model.simulate(
        step_key, (initial_pose.get_retval(), robot_inputs["controls"])
    )


def path_from_trace(tr: genjax.Trace) -> Pose:
    return tr.get_retval()


def generate_path(key: PRNGKey) -> Pose:
    return path_from_trace(generate_path_trace(key))


# %%
key, sub_key = jax.random.split(key)
generate_path_trace(sub_key)
# %%
N_samples = 12
key, sub_key = jax.random.split(key)
sample_paths_v = jax.vmap(generate_path)(jax.random.split(sub_key, N_samples))

Plot.Grid([walls_plot + poses_to_plots(path) for path in sample_paths_v])

# %%
# Animation showing a single path with confidence circles


def plot_path_with_confidence(path: Pose, step: int, motion_settings: dict):
    plot = (
        world_plot
        + [pose_plot(path[i]) for i in range(step + 1)]
        + Plot.color_map({"next pose": "red"})
    )
    if step < len(path) - 1:
        plot += [
            confidence_circle(
                path[step].apply_control(robot_inputs["controls"][step]),
                motion_settings,
            ),
            pose_plot(path[step + 1], fill=Plot.constantly("next pose")),
        ]
    return plot


def animate_path_with_confidence(path: Pose, motion_settings: dict):
    frames = [
        plot_path_with_confidence(path, step, motion_settings)
        for step in range(len(path.p))
    ]

    return Plot.Frames(frames, fps=2)


# Generate a single path
key, sample_key = jax.random.split(key)
path = generate_path(sample_key)
Plot.Frames(
    [
        plot_path_with_confidence(path, step, default_motion_settings)
        + Plot.title("Motion model (samples)")
        for step in range(len(path))
    ],
    fps=2,
)

# %% [markdown]
# ### Modifying traces
#
# The metaprogramming approach of Gen affords the opportunity to explore alternate stochastic execution histories.  Namely, `trace.update` takes as inputs a source of randomness, together with modifications to its arguments and primitive choice values, and returns an accordingly modified trace. It also returns (the log of) the ratio of the updated trace's density to the original trace's density, together with a precise record of the resulting modifications that played out.
#
# One could, for instance, consider just the placement of the first step, and replace its stochastic choice of heading with an updated value. The original trace was typical under the pose prior model, whereas the modified one may be rather less likely. This plot is annotated with log of how much unlikelier, the score ratio:
# %%

key, sub_key = jax.random.split(key)
trace = start_pose_prior.simulate(
    sub_key, (robot_inputs["start"], default_motion_settings)
)
key, sub_key = jax.random.split(key)
rotated_trace, rotated_trace_weight_diff, _, _ = trace.update(
    sub_key, C["hd"].set(jnp.pi / 2.0)
)

# TODO(huebert): try using a slider to choose the heading we set (initial value is 0.0)

(
    Plot.new(
        world_plot
        + pose_plot(trace.get_retval(), fill=Plot.constantly("some pose"))
        + pose_plot(
            rotated_trace.get_retval(), fill=Plot.constantly("with heading modified")
        )
        + Plot.color_map({"some pose": "green", "with heading modified": "red"})
        + Plot.title("Modifying a heading")
    )
    | f"score ratio: {rotated_trace_weight_diff}"
)

# %% [markdown]
# It is worth carefully thinking through a trickier instance of this.  Suppose instead, within the full path, we replaced the first step's stochastic choice of heading with some specific value.
# %%

key, sub_key = jax.random.split(key)
trace = generate_path_trace(sub_key)
key, sub_key = jax.random.split(key)

rotated_first_step, rotated_first_step_weight_diff, _, _ = trace.update(
    sub_key, C[0, "steps", "pose", "hd"].set(jnp.pi / 2.0)
)

# %%
(
    world_plot
    + [
        pose_plot(pose, fill=Plot.constantly("with heading modified"))
        for pose in path_from_trace(rotated_first_step)
    ]
    + [
        pose_plot(pose, fill=Plot.constantly("some path"))
        for pose in path_from_trace(trace)
    ]
    + Plot.color_map({"some path": "green", "with heading modified": "red"})
) | f"score ratio: {rotated_first_step_weight_diff}"

# %%

# Another capability of `trace.update` is to modify the *arguments* to the generative function
# used to produce the trace.  In our example, we might have on hand a very long list of
# controls, and we wish to explore the space of paths incrementally in the timestep:

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
    walls = world["walls"]
    box_size = sensor_settings["box_size"]

    def reading(angle):
        return sensor_distance(pose.rotate(angle), walls, box_size)

    return jax.vmap(reading)(sensor_angles)


# %%
# Plot sensor data.


def plot_sensors(pose: Pose, readings):
    projections = [
        pose.rotate(angle).step_along(s) for angle, s in zip(sensor_angles, readings)
    ]
    return (
        [
            Plot.line([pose.p, p.p], stroke=Plot.constantly("sensor rays"))
            for p in projections
        ],
        [
            Plot.dot(
                [pose.p for pose in projections],
                r=2.75,
                fill=Plot.constantly("sensor readings"),
            )
        ],
        Plot.color_map({"sensor rays": "rgba(0,0,0,0.1)", "sensor readings": "#f80"}),
    )


world_plot + plot_sensors(
    initial_pose.get_retval(), ideal_sensor(initial_pose.get_retval())
)

# %%


def animate_path_with_sensor(path, readings):
    frames = [
        (
            world_plot
            + [pose_plot(pose) for pose in path[:step]]
            + plot_sensors(pose, readings[step])
            + [pose_plot(pose, fill="red")]
        )
        for step, pose in enumerate(path)
    ]
    return Plot.Frames(frames, fps=2)


key, sample_key = jax.random.split(key)
path = generate_path(sample_key)
readings = jax.vmap(ideal_sensor)(path)
animate_path_with_sensor(generate_path(sample_key), readings)

# %% [markdown]
# ### Noisy sensors
#
# We assume that the sensor readings are themselves uncertain, say, the distances only knowable
# up to some noise.  We model this as follows.
# %%

@genjax.gen
def sensor_model_one(pose, angle):
    sensor_pose = pose.rotate(angle)
    return (
        genjax.normal(
            sensor_distance(sensor_pose, world["walls"], sensor_settings["box_size"]),
            sensor_settings["s_noise"],
        )
        @ "distance"
    )


sensor_model = sensor_model_one.vmap(in_axes=(None, 0))


def noisy_sensor(pose):
    trace = sensor_model.simulate(key, (pose, sensor_angles))
    return trace


# %% [markdown]
# The trace contains many choices corresponding to directions of sensor reading from the input pose. To explore the trace values, open the "folders" within the trace by clicking on the small triangles until you see the 41-element array of sensor values.
# %%

sensor_settings["s_noise"] = 0.10

key, sub_key = jax.random.split(key)
trace = sensor_model.simulate(sub_key, (robot_inputs["start"], sensor_angles))

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

key, sub_key = jax.random.split(key)
path = generate_path(sub_key)
readings = jax.vmap(noisy_sensor)(path).get_retval()
animate_path_with_sensor(path, readings)

# %%
# TODO: annotate plot with title "Sensor model (samples)"
# %% [markdown]
# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", whose traces describe simulations of the entire robot situation as we have described it.
# %%

@genjax.gen
def full_model(motion_settings):

    motion_settings = motion_settings.const

    @genjax.gen
    def full_model_initial():
        pose = start_pose_prior(robot_inputs["start"], motion_settings) @ "pose"
        sensor_model(pose, sensor_angles) @ "sensor"
        return pose

    @genjax.gen
    def full_model_kernel(state, control):
        pose = step_model(state, control, motion_settings) @ "pose"
        sensor_model(pose, sensor_angles) @ "sensor"
        return pose, pose

    initial = full_model_initial() @ "initial"
    return full_model_kernel.scan(n=T)(initial, robot_inputs["controls"]) @ "steps"

def get_path(trace):
    p = trace.get_subtrace(("initial",)).get_retval()
    ps = trace.get_retval()[1]
    return ps.prepend(p)


def get_sensors(trace):
    ch = trace.get_choices()
    return jnp.concatenate((
        ch["initial", "sensor", ..., "distance"][jnp.newaxis],
        ch["steps", ..., "sensor", ..., "distance"]
    ))

key, sub_key = jax.random.split(key)
tr = full_model.simulate(sub_key, (genjax.Const(default_motion_settings),))

pz.ts.display(tr)
# %% [markdown]
# Again, the trace of the full model contains many choices, so we have used the Penzai visualization library to render the result. Click on the various nesting arrows and see if you can find the path within. For our purposes, we will supply a function `get_path` which will extract the list of Poses that form the path.

# %% [markdown]
# In the math picture, `full_model` corresponds to a distribution $\text{full}$ over its traces.  Such a trace is identified with of a pair $(z_{0:T}, o_{0:T})$ where $z_{0:T} \sim \text{path}(\ldots)$ and $o_t \sim \text{sensor}(z_t, \ldots)$ for $t=0,\ldots,T$.  The density of this trace is then
# $$\begin{align*}
# P_\text{full}(z_{0:T}, o_{0:T})
# &= P_\text{path}(z_{0:T}) \cdot \prod\nolimits_{t=0}^T P_\text{sensor}(o_t) \\
# &= \big(P_\text{start}(z_0)\ P_\text{sensor}(o_0)\big)
#   \cdot \prod\nolimits_{t=1}^T \big(P_\text{step}(z_t)\ P_\text{sensor}(o_t)\big).
# \end{align*}$$
#
# By this point, visualization is essential.
# %%

key, sub_key = jax.random.split(key)
tr = full_model.simulate(sub_key, (genjax.Const(default_motion_settings),))


def animate_full_trace(trace, frame_key=None):
    # TODO: get the motion settings from trace.get_args()
    path = get_path(trace)
    readings = get_sensors(trace)
    motion_settings = trace.get_args()[0].const

    frames = [
        plot_path_with_confidence(path, step, motion_settings)
        + plot_sensors(pose, readings[step])
        for step, pose in enumerate(path)
    ]

    return Plot.Frames(frames, fps=2, key=frame_key)

animate_full_trace(tr)
# %% [markdown]
# ## The data
#
# Let us generate some fixed synthetic motion data that, for pedagogical purposes, we will work with as if it were the actual path of the robot.  We will generate two versions, one each with low or high motion deviation.
# %%

motion_settings_low_deviation = {
    "p_noise": 0.05,
    "hd_noise": (1 / 10.0) * 2 * jnp.pi / 360,
}
key, k_low, k_high = jax.random.split(key, 3)
trace_low_deviation = full_model.simulate(k_low, (genjax.Const(motion_settings_low_deviation),))

motion_settings_high_deviation = {"p_noise": 0.25, "hd_noise": 2 * jnp.pi / 360}
trace_high_deviation = full_model.simulate(k_high, (genjax.Const(motion_settings_high_deviation),))

animate_full_trace(trace_low_deviation)
# frames_low = frames_from_full_trace(world, "Low motion deviation", trace_low_deviation)
# frames_high = frames_from_full_trace(world, "High motion deviation", trace_high_deviation)
# ani = Animation()
# for (low, high) in zip(frames_low, frames_high)
#     frame_plot = plot(low, high; size=(1000,500), plot_title="Two synthetic data sets")
#     frame(ani, frame_plot)
# end
# gif(ani, "imgs/the_data.gif", fps=2)


# %%

# TODO: next task is to create a side-by-side animation of the low and high deviation paths.

animate_full_trace(trace_high_deviation)
# %% [markdown]
# Since we imagine these data as having been recorded from the real world, keep only their extracted data, *discarding* the traces that produced them.
# %%

# These are are what we hope to recover...
path_low_deviation = get_path(trace_low_deviation)
path_high_deviation = get_path(trace_high_deviation)

# ...using these data.
observations_low_deviation = get_sensors(trace_low_deviation)
observations_high_deviation = get_sensors(trace_high_deviation)

# Encode sensor readings into choice map.


def constraint_from_sensors(readings):
    angle_indices = jnp.arange(len(sensor_angles))
    return jax.vmap(
        lambda ix, v: C["steps", ix, "sensor", angle_indices, "distance"].set(v)
    )(
        jnp.arange(T), readings[1:]
    ) + C['initial', 'sensor', angle_indices, 'distance'].set(readings[0])


constraints_low_deviation = constraint_from_sensors(observations_low_deviation)
constraints_high_deviation = constraint_from_sensors(observations_high_deviation)

# %% [markdown]
# We summarize the information available to the robot to determine its location. On the one hand, one has to produce a guess of the start pose plus some controls, which one might integrate to produce an idealized guess of path. On the other hand, one has the sensor data.

# %%
def animate_bare_sensors(path, plot_base=[]):
    def frame(pose, readings1, readings2):
        def plt(readings):
            return Plot.new(
                plot_base,
                plot_sensors(pose, readings),
                Plot.domain([0, 20]),
                {"width": 400, "height": 400},
            )

        return plt(readings1) & plt(readings2)

    frames = [
        frame(*scene)
        for scene in zip(path, observations_low_deviation, observations_high_deviation)
    ]
    return Plot.Frames(frames, fps=2)


animate_bare_sensors(itertools.repeat(world["center_point"]))
# %% [markdown]
# ## Inference
# ### Why we need inference: in a picture
#
# The path obtained by integrating the controls serves as a proposal for the true path, but it is unsatisfactory, especially in the high motion deviation case. The picture gives an intuitive sense of the fit:
# %%

animate_bare_sensors(path_integrated, walls_plot)
# %% [markdown]
# It would seem that the fit is reasonable in low motion deviation, but really breaks down in high motion deviation.
#
# We are not limited to visual judgments here: the model can quantitatively assess how good a fit the integrated path is for the data.  In order to do this, we detour to explain how to produce samples from our model that agree with the fixed observation data.

# %% [markdown]
# ### Producing samples with constraints
#
# We have seen how `simulate` performs traced execution of a generative function: as the program runs, it draws stochastic choices from all required primitive distributions, and records them in a choice map.
#
# Given a choice map of *constraints* that declare fixed values of some of the primitive choices, the operation `importance` proposes traces of the generative function that are consistent with these constraints.

# %%

full_model_importance = jax.jit(full_model.importance)

key, sub_key = jax.random.split(key)
sample, log_weight = full_model_importance(sub_key, constraints_low_deviation, (genjax.Const(motion_settings_low_deviation),))

animate_full_trace(sample) | f"log_weight: {log_weight}"
# %% [markdown]
# A trace resulting from a call to `importance` is structurally indistinguishable from one drawn from `simulate`.  But there is a key situational difference: while `get_score` always returns the frequency with which `simulate` stochastically produces the trace, this value is **no longer equal to** the frequency with which the trace is stochastically produced by `importance`.  This is both true in an obvious and less relevant sense, as well as true in a more subtle and extremely germane sense.
#
# On the superficial level, since all traces produced by `importance` are consistent with the constraints, those traces that are inconsistent with the constraints do not occur at all, and in aggregate the traces that are consistent with the constraints are more common.
#
# More deeply and importantly, the stochastic choice of the *constraints* under a run of `simulate` might have any density, perhaps very low.  This constraints density contributes as always to the `get_score`, whereas it does not influence the frequency of producing this trace under `importance`.
#
# The ratio of the `get_score` of a trace to the probability density that `importance` would produce it with the given constraints, is called the *importance weight*.  For convenience, (the log of) this quantity is returned by `importance` along with the trace.
#
# We stress the basic invariant:
# $$
# \text{get\_score}(\text{trace})
# =
# (\text{weight from importance})
# \cdot
# (\text{frequency simulate creates this trace}).
# $$
# %% [markdown]
# The preceding comments apply to generative functions in wide generality.  We can say even more about our present examples, because further assumptions hold.
# 1. There is no untraced randomness.  Given a full choice map for constraints, everything else is deterministic.  In particular, the importance weight is the `get_score`.
# 2. The generative function was constructed using GenJAX's DSL and primitive distributions.  Ancestral sampling; `importance` with empty constraints reduces to `simulate` with importance weight $1$.
# 3. Combined, the importance weight is directly computed as the `project` of the trace upon the choice map addresses that were constrained in the call to `importance`.
#
#   In our running example, the projection in question is $\prod_{t=0}^T P_\text{sensor}(o_t)$.
# %%
# TODO: this calculation doesn't work in GenJAX currently
# log_weight - project(trace, select([prefix_address(i, :sensor) for i in 1:(T+1)]...))

# %% [markdown]
# ### Why we need inference: in numbers
#
# We return to how the model offers a numerical benchmark for how good a fit the integrated path is.
#
# In words, the data are incongruously unlikely for the integrated path.  The (log) density of the measurement data, given the integrated path...

# %%
path_integrated
sample.get_choices()
constraints_path_integrated = C[""]

def constraint_from_path(path):

    c_ps = jax.vmap(
        lambda ix, p: C["steps", ix, "pose", "p"].set(p)
    )(jnp.arange(T), path.p[1:])

    c_hds = jax.vmap(
        lambda ix, hd: C["steps", ix, "pose", "hd"].set(hd)
    )(jnp.arange(T), path.hd[1:])

    c_p = C["initial", "pose", "p"].set(path.p[0])
    c_hd = C["initial", "pose", "hd"].set(path.hd[0])

    return c_ps + c_hds + c_p + c_hd

constraints_path_integrated = constraint_from_path(path_integrated)
constraints_path_integrated_observations_low_deviation = constraints_path_integrated + constraints_low_deviation
constraints_path_integrated_observations_high_deviation = constraints_path_integrated + constraints_high_deviation

key, sub_key = jax.random.split(key)
trace_path_integrated_observations_low_deviation, w_low = full_model_importance(sub_key, constraints_path_integrated_observations_low_deviation, (genjax.Const(motion_settings_low_deviation),))
key, sub_key = jax.random.split(key)
trace_path_integrated_observations_high_deviation, w_high = full_model_importance(sub_key, constraints_path_integrated_observations_high_deviation, (genjax.Const(motion_settings_high_deviation),))

w_low, w_high
# TODO: Jay then does two projections to compare the log-weights of these two things,
# in order to show that we can be quantitative about the quality of the paths generated
# by the two models. Unfortunately we can't, and so we should raise the priority of the
# blocking bug
# %%

Plot.Row(
    *[(Plot.Hiccup("div.f3.b.tc", title)
       | animate_full_trace(trace, frame_key="frame")
       | f"score: {score:,.2f}")
     for (title, trace, motion_settings, score) in
     [["Low deviation",
       trace_path_integrated_observations_low_deviation,
       motion_settings_low_deviation,
       w_low],
       ["High deviation",
       trace_path_integrated_observations_high_deviation,
       motion_settings_high_deviation,
       w_high]]]) | Plot.Slider("frame", T, fps=2)


# %%
#pz.ts.display(sample.get_choices()["steps", ..., "pose", "p"])
key, sub_key = jax.random.split(key)
u_tr, u_w, _, _ = sample.update(sub_key, constraint_from_path(path_integrated))
pz.ts.display(u_tr.get_choices()["initial", "pose", "p"])


# %%
constraints_path_integrated =
    choicemap(((prefix_address(t, :pose => :p), path_integrated[t].p) for t in 1:(T+1))...,
              ((prefix_address(t, :pose => :hd), path_integrated[t].hd) for t in 1:(T+1))...)

constraints_path_integrated_observations_low_deviation =
    merge(constraints_path_integrated, merged_constraints_low_deviation)
constraints_path_integrated_observations_high_deviation =
    merge(constraints_path_integrated, merged_constraints_high_deviation)

trace_path_integrated_observations_low_deviation, _ =
    generate(full_model, (T, full_model_args...), constraints_path_integrated_observations_low_deviation)
trace_path_integrated_observations_high_deviation, _ =
    generate(full_model, (T, full_model_args...), constraints_path_integrated_observations_high_deviation);

selection = select((prefix_address(i, :sensor) for i in 1:(T+1))...)

println("Log density of low deviation observations assuming integrated path: $(project(trace_path_integrated_observations_low_deviation, selection))")
println("Log density of high deviation observations assuming integrated path: $(project(trace_path_integrated_observations_high_deviation, selection))");



# %%
animate_full_trace(trace_low_deviation, motion_settings_low_deviation)
# %%
pz.ts.display(sample.get_choices())

# %%
