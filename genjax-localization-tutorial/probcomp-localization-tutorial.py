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


def indexable(cls):
    """
    A decorator that adds support for bracket indexing/slicing to a class.
    This allows for numpy/jax style indexing on Pytree-like objects.
    """

    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda v: v[idx], self)

    cls.__getitem__ = __getitem__
    return cls


def iterable(cls):
    """
    A decorator that adds sequence-like operations to a class.
    This provides `len` and `__iter__` methods for Pytree-like objects.

    Note: The `__len__` method assumes that all leaves in the Pytree have the same length.
    If leaves have different lengths, it will return the length of the first leaf encountered.
    """

    def __len__(self):
        return len(jax.tree_util.tree_leaves(self)[0])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    cls.__len__ = __len__
    cls.__iter__ = __iter__
    return cls


def concatable(cls):
    def __add__(self, other):
        if not isinstance(other, cls):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

        def concat_leaves(x, y):
            return jnp.concatenate([x, y])

        return jax.tree_util.tree_map(concat_leaves, self, other)

    cls.__add__ = __add__
    return cls


def pythonic_pytree(cls):
    """
    A decorator that composes indexable, iterable, and concatable decorators
    to make working with pytrees more Pythonic.
    """
    return indexable(iterable(concatable(cls)))


@pythonic_pytree
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


# Example usage:
pose = Pose(jnp.array([1.0, 2.0]), hd=jnp.array(1.57))
print(pose)

# Move the pose along its direction
print(pose.step_along(5))

# Rotate the pose
print(pose.rotate(pi / 2))


# %%
@pythonic_pytree
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
        # Prepend a no-op control
        Control(jnp.array([0.0]), jnp.array([0.0])) + robot_inputs["controls"],
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
            physical_step(
                pose.p, pose.p + control.ds * pose.dp(), pose.hd + control.dhd
            ),
            physical_step(
                pose.p, pose.p + control.ds * pose.dp(), pose.hd + control.dhd
            ),
        ),
        robot_inputs["start"],
        # Prepend a no-op control
        Control(jnp.array([0.0]), jnp.array([0.0])) + robot_inputs["controls"],
    )[1]


# %%

path_integrated = integrate_controls_physical(robot_inputs)

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
    {"margin": 0, "inset": 50, "width": 500, "height": 500, "axis": None},
    Plot.domain([0, 20]),
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
def step_model(start, c, motion_settings):
    p = (
        genjax.mv_normal(
            start.p + c.ds * start.dp(), motion_settings["p_noise"] ** 2 * jnp.eye(2)
        )
        @ "p"
    )
    hd = genjax.normal(start.hd + c.dhd, motion_settings["hd_noise"]) @ "hd"
    return physical_step(start.p, p, hd)


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
    return [
        pose_arrow(pose, constants={"step": i, **constants}, **plot_opts)
        for i, pose in enumerate(poses)
    ]


poses_plot = poses_to_plots(poses)

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


def make_path_model_step(motion_settings):
    @genjax.gen
    def path_model_step(previous_pose, control):
        return step_model(previous_pose, control, motion_settings) @ (
            "steps",
            "pose",
        )

    return path_model_step.accumulate()


# prefix_address(t, rest) = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest)
# get_path(trace) = [trace[prefix_address(t, :pose)] for t in 1:(get_args(trace)[1]+1)];


# %%

key, sub_key1, sub_key2 = jax.random.split(key, 3)
initial_pose = path_model_start.simulate(sub_key1, (robot_inputs, motion_settings))
path_model_step = make_path_model_step(motion_settings)
step_model.simulate(
    sub_key2,
    (
        initial_pose.get_retval(),
        jax.tree.map(lambda v: v[0], robot_inputs["controls"]),
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
    return tr.get_retval()


def generate_path(key: PRNGKey) -> Pose:
    return path_from_trace(generate_path_trace(key))


def nth(t, n: int):
    return jax.tree.map(lambda v: v[n], t)


# %%
generate_path_trace(key)
# %%
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


# %%


def animate_path_with_confidence(path: Pose, motion_settings):
    frames = [
        # prior poses in black
        (walls_plot + [pose_arrow(path[i]) for i in range(step + 1)])
        # confidence circle
        + [
            Plot.scaled_circle(
                *path[step].apply_control(robot_inputs["controls"][step]).p,
                opacity=0.25,
                fill="red",
                r=2.5 * motion_settings["p_noise"],
            ),
            # next pose
            pose_arrow(path[step + 1], stroke="red"),
        ]
        if step < len(path.p) - 1
        else None
        for step in range(len(path.p))
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
# One could, for instance, consider just the placement of the first step, and replace its stochastic
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
# It is worth carefully thinking through a trickier instance of this.  Suppose instead,
# within the full path, we replaced the $t = 0$ step's stochastic choice of heading
# with some specific value.

# %%
key, sub_key = jax.random.split(key)
trace = generate_path_trace(sub_key)
key, sub_key = jax.random.split(key)

rotated_first_step, rotated_first_step_weight_diff, _, _ = trace.update(
    sub_key, C[0, "steps", "pose", "hd"].set(jnp.pi / 2.0)
)

original_path = path_from_trace(trace)
rotated_first_step_path = path_from_trace(rotated_first_step)
# %%
Plot.new(
    world_plot
    + [
        pose_arrow(Pose(p, hd), stroke="red")
        for p, hd in zip(rotated_first_step_path.p, rotated_first_step_path.hd)
    ]
    + [
        pose_arrow(Pose(p, hd), stroke="green")
        for p, hd in zip(original_path.p, original_path.hd)
    ],
    Plot.color_map(
        {
            "walls": "#ccc",
            "some pose": "green",
            "with heading at first step modified": "red",
        }
    ),
)

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
    walls = world["walls"]
    box_size = sensor_settings["box_size"]

    def reading(angle):
        return sensor_distance(pose.rotate(angle), walls, box_size)

    return jax.vmap(reading)(sensor_angles)


# %%
# Plot sensor data.


def plot_sensors(pose: Pose, readings):
    projections = [
        pose.rotate(sensor_angles[j]).step_along(s) for j, s in enumerate(readings)
    ]
    return (
        [Plot.line([pose.p, p.p], stroke=Plot.constantly("#ddd")) for p in projections]
        + [Plot.dot([pose.p for pose in projections], fill="#f80")]
        + [Plot.dot([pose.p], fill="#0f0")]
    )


walls_plot + plot_sensors(
    initial_pose.get_retval(), ideal_sensor(initial_pose.get_retval())
)
# %%


def animate_path_with_sensor(path, readings):
    frames = [
        (
            walls_plot
            # Prior poses in black
            + [pose_arrow(pose) for pose in path[:step]]
            + plot_sensors(Pose(path.p[step], path.hd[step]), readings[step])
            # Next pose in red
            + [pose_arrow(Pose(path.p[step], path.hd[step]), stroke="red")]
            + {"axis": None}
        )
        for step in range(1, len(path.p))
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


# # %% [markdown]
# The trace contains many choices corresponding to directions of sensor reading from the input pose.
# TODO(colin): The original notebook contains the following comment:
# To reduce notebook clutter, here we just show a subset of 5 of them:
# But I don't think we need to worry about this with penzai. We aren't relying on
# print statments to condense this data.

# %%
sensor_settings["s_noise"] = 0.10

key, sub_key = jax.random.split(key)
trace = sensor_model.simulate(sub_key, (robot_inputs["start"], sensor_angles))
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
# function frame_from_sensors_trace(world, title, poses, poses_color, poses_label, pose, trace; show_clutters=false)
#     readings = [trace[j => :distance] for j in 1:sensor_settings.num_angles]
#     return frame_from_sensors(world, title, poses, poses_color, poses_label, pose,
#                              readings, "trace sensors", get_args(trace)[3];
#                              show_clutters=show_clutters)
# end;

key, sub_key = jax.random.split(key)
path = generate_path(sub_key)
readings = jax.vmap(noisy_sensor)(path).get_retval()
animate_path_with_sensor(path, readings)
# TODO: annotate plot with title "Sensor model (samples)"
# %% [markdown]
# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", whose traces describe simulations of the entire robot situation as we have described it.

# %%


def make_full_model(motion_settings):
    @genjax.gen
    def full_model_initial(motion_settings):
        pose = start_pose_prior(robot_inputs["start"], motion_settings) @ "pose"
        sensor_model(pose, sensor_angles) @ "sensor"
        return pose

    # The reason why we have this function structure is to place
    # motion_settings in scope for the full_model_kernel which is
    # meant to be used by with the scan combinator. Scanned generative
    # functions are best implemented by functions of two arguments,
    # a state and an input; without the scope done here, we would have
    # to carry the motion_settings as a constant component of the state.

    @genjax.gen
    def full_model_kernel(state, control):
        pose = step_model(state, control, motion_settings) @ "pose"
        _ = sensor_model(pose, sensor_angles) @ "sensor"
        return pose, None

    @genjax.gen
    def full_model():
        initial = full_model_initial(motion_settings) @ "initial"
        full_model_kernel.scan(n=T)(initial, robot_inputs["controls"]) @ "steps"

    return full_model


key, sub_key = jax.random.split(key)
full_model = make_full_model(motion_settings)
tr = full_model.simulate(sub_key, ())
ch = tr.get_choices()

# %% [markdown]
# Again, the trace of the full model contains many choices, so we just show a subset of them: the initial pose plus 2 timesteps, and 5 sensor readings from each.

# TODO(colin): we haven't done this. Why not let penzai handle the presentation.

# %%


def get_path(trace):
    ch = trace.get_choices()
    return Pose(ch["steps", ..., "pose", "p"], ch["steps", ..., "pose", "hd"])


def get_sensors(trace):
    return trace.get_choices()["steps", ..., "sensor", ..., "distance"]


get_path(tr)

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


def animate_full_trace(trace):
    path = get_path(trace)
    readings = get_sensors(trace)
    motion_settings = trace.get_subtrace(("initial",)).get_args()[0]

    noiseless_steps = jax.vmap(lambda pose, c: pose.p + c.ds * pose.dp())(
        path, robot_inputs["controls"]
    )

    std_devs_radius = 2.5 * motion_settings["p_noise"]

    # TODO: legend

    frames = [
        (
            walls_plot
            # Prior poses in black
            + [pose_arrow(Pose(p, hd)) for p, hd in zip(path.p[:step], path.hd[:step])]
            + plot_sensors(Pose(path.p[step], path.hd[step]), readings[step])
            # Next pose in red
            + [pose_arrow(Pose(path.p[step], path.hd[step]), stroke="red")]
            + Plot.scaled_circle(
                noiseless_steps[step - 1][0],
                noiseless_steps[step - 1][1],
                r=std_devs_radius,
                opacity=0.25,
                fill="red",
            )
            + {"axis": None}
        )
        for step in range(1, len(path.p))
    ]

    return Plot.Frames(frames, fps=2)


animate_full_trace(tr)

# %% [markdown]
# ## The data
#
# Let us generate some fixed synthetic motion data that, for pedagogical purposes,
# we will work with as if it were the actual path of the robot.  We will generate
# two versions, one each with low or high motion deviation.

# %%
motion_settings_low_deviation = {
    "p_noise": 0.05,
    "hd_noise": (1 / 10.0) * 2 * jnp.pi / 360,
}
key, k_low, k_high = jax.random.split(key, 3)
low_deviation_model = make_full_model(motion_settings_low_deviation)
trace_low_deviation = low_deviation_model.simulate(k_low, ())

motion_settings_high_deviation = {"p_noise": 0.25, "hd_noise": 2 * jnp.pi / 360}
high_deviation_model = make_full_model(motion_settings_high_deviation)
trace_high_deviation = high_deviation_model.simulate(k_high, ())

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
        jnp.arange(T), readings
    )  # + C['initial', 'sensor', angle_indices, 'distance'].set(readings[0])


constraints_low_deviation = constraint_from_sensors(observations_low_deviation)
constraints_high_deviation = constraint_from_sensors(observations_high_deviation)

# %% [markdown]
# We summarize the information available to the robot to determine its location. On the one hand,
# one has to produce a guess of the start pose plus some controls, which one might integrate to
# produce an idealized guess of path. On the other hand, one has the sensor data.

# %%
world_plot + plot_sensors(world["center_point"], observations_low_deviation[0])


# %%
def animate_bare_sensors(path, plot_base=[]):
    def frame(pose, readings1, readings2):
        def plt(readings):
            return Plot.new(
                plot_base + plot_sensors(pose, readings),
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
# The path obtained by integrating the controls serves as a proposal for the true path,
# but it is unsatisfactory, especially in the high motion deviation case.
# The picture gives an intuitive sense of the fit:

# %%

animate_bare_sensors(path_integrated, walls_plot)

# %% [markdown]

# At this point in the Julia notebook, Jay appeaals to a BlackBox inference
# technique which uses a Gen.jl library for particle filtering. The scoping
# for MathCamp suggests that we try importance sampling and stop there. Might
# as well give it a try?

low_dev_model_importance = jax.jit(low_deviation_model.importance)
high_dev_model_importance = jax.jit(high_deviation_model.importance)

# %%
N_importance_samples = 1
key, sub_key = jax.random.split(key)
samples = jax.vmap(high_dev_model_importance, in_axes=(0, None, None))(
    jax.random.split(sub_key, N_importance_samples), constraints_high_deviation, ()
)
# %%
amax = jnp.argmax(samples[0].get_score())
animate_full_trace(nth(samples[0], amax))
# %%
animate_full_trace(trace_low_deviation)
# %%
