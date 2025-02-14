# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Mapping tutorial

# The goal of this tutorial is – in a sense – the opposite of the localization tutorial. We consider the following scenario:
# * we have a two-dimensional map consisting of walls and free space
# * a robot is located at the center of the map (so it knows its position)
# * the robot is equipped with a sensor that can (noisily) measure the distance to the nearest wall in several directions
# * we want to infer the locations of the walls in the map
#
# As a simplifying assumption, we assume that
# * the map is a $21 \times 21$ grid of unit blocks ("pixels")
# * each pixel is either a wall or a free space
#
# This tutorial will teach you how to:
# * model this scenario in GenJAX
# * perform Bayesian inference to infer the walls on the map from noisy sensor measurements:
#   * first, via importance sampling (which turns out to perform poorly)
#   * then, via Gibbs sampling (which turns out to perform well)
#
# Potential future extensions:
# * include observations from several points of view
# * include uncertainty in the robot's position
# * allow the robot to move around (including uncertainty in the motion)

# +
# Global setup code

import genstudio.plot as Plot
import itertools
import jax
import jax.numpy as jnp
import genjax
from urllib.request import urlopen
from genjax import SelectionBuilder as S
from genjax import ChoiceMapBuilder as C
from genjax import pretty, Target
from genjax.typing import IntArray, FloatArray, PRNGKey
from penzai import pz
from typing import Any, Iterable

import os

pretty()

html = Plot.Hiccup
Plot.configure({"display_as": "html", "dev": False})

# Ensure a location for image generation.
os.makedirs("imgs", exist_ok=True)


# -

# ## Basic data structures
#
# We represent the world (i.e. the grid of pixels) as an array of pixels.
# Each pixel has a position $[x, y]$ and a flag (`true` or `false`) indicating whether it is a wall or not.
# It will turn out to be useful to represent the presence of a wall as a real number between 0 and 1 to quantify uncertainty later on.
# In order to easily manipulate pixels in GenJAX, we define them as a `Pytree`.
# `Pytree`s are structures that may contain tensors and that JAX can traverse and manipulate easily.

@pz.pytree_dataclass
class Pixel(genjax.PythonicPytree):
    pos: IntArray
    is_wall: FloatArray



# ## Loading the map
#
# The following functions are useful to load the map from a string (which makes editing the walls by hand easier than the array representation).

# +

GRID = r"""
.....................
.....................
.XXX..........X......
.X.X..........X......
.X............XXX....
.X..............X....
.X..............X....
.XX............XXX...
..X..............X...
..X..............X...
..X..............X...
..X..............X...
..XXXXX......XXXXX...
......X......X.......
......X....XXX.......
......X....X.........
......X....X.........
......XXXXXX.........
.....................
.....................
.....................
"""

# Maximum absolute x- and y-coordinate of the pixels:
GRID_RADIUS = 10

def all_pixel_coordinates(radius):
    """Returns a `(2 * radius + 1) x 2` array of all pixel coordinates."""
    return jnp.stack(jnp.meshgrid(jnp.arange(-radius, radius + 1), jnp.arange(-radius, radius + 1)), axis=-1).reshape(-1, 2)

ALL_GRID_POINTS = all_pixel_coordinates(GRID_RADIUS)

def extract_points_from_grid_string(grid_string):
    """Extracts the points from the grid string."""
    lines = grid_string.strip().split("\n")
    height = len(lines)
    width = len(lines[0])
    is_wall = jnp.zeros((height, width))
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "X":
                is_wall = is_wall.at[i, j].set(1)
    return is_wall.flatten()

true_walls = extract_points_from_grid_string(GRID)

true_pixels = Pixel(ALL_GRID_POINTS, true_walls)
# -

# Now, `true_walls` is a 21 x 21 array of 0s and 1s, where 1s indicate the presence of a wall.
# Similarly, `true_pixels` contains the coordinates of all `21 * 21 = 441` pixels and the presence of a wall.

# ## Plotting
#
# In order to understand the problem, our code, and for debugging, it is useful to have visualizations.
# Hence we define the following functions to plot the map.

# +
OFFSETS = jnp.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
])

OFFSETS_CLOSED = jnp.concat([OFFSETS, OFFSETS[:1]], axis=0)

def plot_walls(pixels: Pixel):
    pixel_vertices = pixels.pos[:, None, :] + OFFSETS_CLOSED[None, :, :]
    return [
        Plot.line(vertices, fill="black", fillOpacity=is_wall)
        for vertices, is_wall in zip(pixel_vertices, pixels.is_wall)
        if is_wall > 0
    ] + Plot.ellipse([[0, 0]], r=0.2, fill="red")

def line_segments_of_pixel(pixel: IntArray):
    vertices = pixel[None, :] + OFFSETS
    return jnp.stack([vertices, jnp.roll(vertices, 1, axis=0)], axis=1)

line_segments = jax.vmap(line_segments_of_pixel, in_axes=0)

def make_plot(p):
    return Plot.new(
        p,
        {"width": 500, "aspectRatio": 1},
        Plot.domain([-10, 10], [-10, 10])
    )


# -

make_plot(plot_walls(true_pixels))


# ## Prior
#
# For a Bayesian model, we first need to specify our prior belief about the map.
# We will keep it very simple here and assume that each pixel is a wall with probability 0.5.

# +
@genjax.gen
def pixel_prior_single(position: IntArray):
    is_wall = genjax.bernoulli(0.0) @ ()
    return Pixel(position, is_wall)

pixels_prior = pixel_prior_single.vmap(in_axes=0)
# -

# We first define a generative function `pixel_prior_single` (using the `@genjax.gen` decorator) that generates a single pixel for a given position by sampling from a Bernoulli distribution.
# (Note that `bernoulli` takes a logit, not a probability, so we need to pass `0.0` to flip a fair coin.)
# The resulting sample is given the (empty) address `()`.
# (Addresses will be explained in more detail later.)
#
# TODO: Are empty addresses an anti-pattern? I tried to name it `"is_wall"` and to `.inline` the model below to avoid nested addresses, but this didn't work.
#
# Then we use the `vmap` combinator to map this generative function over all pixels, resulting in a generative function sampling each pixel indendently and identically distributed (i.i.d.).
#
# In order to get a better understanding, let us sample from the prior.
# To do this, JAX requires a random seed, called "key", which we pass to the `.simulate` method of the generative function.
# The result is a **trace**, which contains the return value of the function, the score, and the choice ("sampled values during execution").
# The score is the log probability of the choices.

key = jax.random.key(0)
sample_prior_jitted = jax.jit(pixels_prior.simulate)
tr = sample_prior_jitted(key, (ALL_GRID_POINTS,))
display(tr.get_score())
display(tr.get_choices())
display(tr.get_retval())

# Let's visualize the map of pixels that was sampled from the prior.

pixels = tr.get_retval()
make_plot(plot_walls(pixels))


# We can see that, in fact, about half the pixels are walls. So the prior seems to work as intended.
#
# POSSIBLE EXTENSION: However, these maps don't look like real-world maps. The prior could be refined in various ways:
# * Probably, less than 50% of the pixels are walls, so the probability could be lowered.
# * "Wall pixels" are likely connected, so we could generate line segments in the prior rather than individual pixels.

# ## Exact sensor model
#
# As mentioned at the start, the robot located at the origin has a sensor to measure the distance to the nearest wall in several directions.
# We can model this as follows.
# The math in the next cell is standard geometry, but not relevant for the overall understanding of modeling and inference in GenJAX, so feel free to skip the details.

# +
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


def distance(dir, seg):
    """
    Computes the distance from the origin to a segment, in a given direction.

    Args:
    - dir: The direction: `[dir_x, dir_y]`.
    - seg: The Segment object: `[[start_x, start_y], [end_x, end_y]]`.

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    pos = jnp.array([0, 0])
    a = solve_lines(pos, dir, seg[0], seg[1] - seg[0])
    return jnp.where(
        (a[0] >= 0.0) & (a[1] >= 0.0) & (a[1] <= 1.0),
        a[0],
        jnp.inf,
    )

def unit_dir(angle):
    """Unit vector in the direction of `angle`."""
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])



# -

# The robot's sensor can measure the distance in `NUM_DIRECTIONS` equidistant directions.
#
# The following code computes the sensor readings.
# It makes use of the JAX functions `jax.lax.cond` (instead of `if`) and `jax.vmap` (instead of a for-loop) because we want JAX to JIT-compile and vectorize these operations.

# +
NUM_DIRECTIONS = 50
ANGLES = jnp.arange(0, 1, 1 / NUM_DIRECTIONS) * 2 * jnp.pi
MAX_DISTANCE = GRID_RADIUS * 2

def distance_to_pixel(dir, pixel):
    """Distance from the origin to the `pixel` if the ray in direction `dir` hits the pixel and it is a wall."""
    segs = line_segments_of_pixel(pixel.pos)
    dists = jax.vmap(lambda seg: distance(dir, seg))(segs)
    return jax.lax.cond(
        pixel.is_wall > 0,
        lambda: jax.lax.cond(
            jnp.array_equal(pixel.pos, jnp.array([0, 0])),
            lambda: 0.0, # we are inside the wall
            lambda: jnp.min(dists, axis=0),
        ),
        lambda: jnp.inf
    )

def distance_to_pixels(dir, pixels):
    """Distance from the origin to the nearest wall among `pixels` in direction `dir`.
    
    The distance is capped at `MAX_DISTANCE` to avoid infinities in the calculations."""
    return jnp.minimum(
        jnp.min(jax.vmap(lambda pixel: distance_to_pixel(dir, pixel))(pixels), axis=0),
        MAX_DISTANCE
    )

def sensor_distances(pixels):
    """Sensor distances in all directions (as specified by `ANGLES`) for the map given by `pixels`."""
    return jax.vmap(lambda angle: distance_to_pixels(unit_dir(angle), pixels))(ANGLES)


# -

# As before, it is useful to have visualizations.

# +
def plot_sensors(readings):
    """Plot the sensor readings."""
    unit_vecs = jax.vmap(unit_dir, in_axes=0)(ANGLES)
    ray_endpoints = unit_vecs * readings[:, None]
    return [
        Plot.line([[0, 0], [x, y]], stroke=Plot.constantly("sensor rays"))
        for x, y in ray_endpoints
    ] + [
        Plot.ellipse([endpoint], r=0.1, fill=Plot.constantly("sensor readings"))
        for endpoint in ray_endpoints
    ]

def world_and_sensors_plot(pixels, readings):
    return plot_walls(pixels) + plot_sensors(readings)

true_readings = sensor_distances(true_pixels)

make_plot(world_and_sensors_plot(true_pixels, true_readings))
# -

# ## Noisy sensor model
#
# A real sensor is going to be noisy, which we model with a normal distribution.
# As before, we specify the model for a single direction/angle first, and then map it over all directions.

# +
NOISE = 0.5

@genjax.gen
def sensor_model_single(pixels, angle):
    exact_distance = distance_to_pixels(unit_dir(angle), pixels)
    return genjax.normal(exact_distance, NOISE) @ ()

sensor_model = sensor_model_single.vmap(in_axes=(None, 0))
# -

# Let's sample from the sensor model to see what data the robot receives.
# We can see that the sensor readings are no longer exact, but contain quite a bit of noise.
# The noise level can be controlled by changing the `NOISE` variable above.

key = jax.random.key(0)
trace = sensor_model.simulate(key, (true_pixels, ANGLES,))
observed_readings = trace.get_retval()
make_plot(world_and_sensors_plot(true_pixels, observed_readings))

# To get an idea of what the robot sees, let's remove the walls. How can we infer the walls?

make_plot(plot_sensors(observed_readings))


# ## Full model
#
# Now, we can put the pieces together by combining the prior and the sensor model.

@genjax.gen
def full_model():
    pixels = pixels_prior(ALL_GRID_POINTS) @ "is_wall"
    readings = sensor_model(pixels, ANGLES) @ "readings"
    return (pixels, readings)



# This model samples a map from the prior and then samples sensor readings from the sensor model.
# The samples are given the addresses `"is_wall"` and `"readings"`.
# When inspecting the trace, we can see that the sampled values are stored under these addresses.

key = jax.random.key(1)
trace = full_model.simulate(key, ())
pixels, readings = trace.get_retval()
make_plot(world_and_sensors_plot(pixels, readings))
trace.get_choices()


# ## Importance sampling (self-normalized importance sampling)
#
# One of the simplest inference methods is importance sampling.
# To do this, we constrain our model to the observed data.
# For samples corresponding to observations (i.e. the sensor readings), we don't actually sample, but instead use the observed value and record the likelihood of seeing that observation.
# GenJAX provides a method for this: `model.importance(key, constraints, args)` runs the model with the random seed `key` and arguments `args`, but constrains the sampled values to `constraints` (the observations).
# It returns a trace with the sampled values and a log weight $\log p(observations \mid pixels)$.
#
# If we run `.importance` several times, some traces (with a better prior sample) will have a higher weight and others (with a worse prior choice) will have lower weight.
# Suppose these weighted samples are $(pixels_i, \log(w_i))$ for $i = 1 \ldots N$.
# We can **normalize** them as follows: $w'_i := \frac{w_i}{\sum_{i} w_i}$.
# Then posterior expected values can be approximated using the weighted samples: $\mathbb{E}_{pixels \sim p(pixels \mid readings)} [f(pixels)] \approx \sum_{i=1}^N w'_i f(pixels_i)$.
#
# We can also obtain unweighted samples approximating the posterior by **resampling**: sampling from a categorical distribution where each category $pixels_i$ has $w'_i$.
#
# Let's try this.

# +
def importance_sampling(key, observed_readings, N = 100):
    """Very naive MAP estimation. Just try N random samples and take the one with the highest weight."""
    model_importance = jax.jit(full_model.importance)
    keys = jax.random.split(key, N)
    constraints = C["readings"].set(observed_readings)
    traces, log_weights = jax.vmap(lambda key: model_importance(key, constraints, ()))(keys)
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    best_index = jnp.argmax(log_weights)
    best_trace = jax.tree_util.tree_map(lambda trace: trace[best_index], traces)
    pixels, _readings = best_trace.get_retval()
    return traces, log_weights

key, subkey = jax.random.split(key)
traces, log_weights = importance_sampling(subkey, observed_readings, N=10000)
subkeys = jax.random.split(subkey, 100)
resampled_indices = [jax.random.categorical(subkeys[i], log_weights) for i in range(100)]
display(resampled_indices)
# -

# If we look at the resampled indices, we see that they are all the same.
# What this means is that one trace had such a high weight that none of the other traces had a chance to be resampled.
# This makes sense, because it is extremely unlikely to guess a "good" map just by sampling from the prior.
# So one (bad) sample will get lucky and have a high weight by chance and be selected every time.
# We can confirm this by looking at a histogram of the weights.

import matplotlib.pyplot as plt
plt.hist(log_weights, bins=100, range=(-500, 0))
plt.show()


# As we can see, the next best trace has a **log** weight of around -100, orders of magnitude worse than the best sample.
#
# This is clearly bad. We need better inference methods.

# ## Automated importance sampling with prior proposal
#
# GenJAX has automation for importance sampling.
#
# TODO: `ImportanceK` gives weird results. I'm not sure what's going wrong here.

from genjax.inference.smc import ImportanceK
num_samples = 100
constraints = C["readings"].set(observed_readings)
target = Target(full_model, (), constraints)
alg = ImportanceK(target, k_particles=num_samples)
key, *subkeys = jax.random.split(key, num_samples + 1)
subkeys = jnp.array(subkeys)
posterior_samples = jax.jit(jax.vmap(alg(target)))(subkeys)
is_wall = jnp.mean(posterior_samples["is_wall"], axis=0)
pixels = Pixel(ALL_GRID_POINTS, is_wall)
make_plot(world_and_sensors_plot(pixels, observed_readings))


# ## Simple Gibbs sampling
#
# As a better inference algorithm, we turn to Gibbs sampling.
# Gibbs sampling is a simple algorithm to sample from a joint distribution $p(x_1, \dots, x_n)$, assuming one can sample from the conditional distributions $p(x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n)$.
# A Gibbs starts with initial values $x_1^{(0)}, \dots, x_n^{(0)}$ and then iteratively updates each $x_i$ from the conditional distribution $p(x_i \mid x_1^{(t)}, \dots, x_{i-1}^{(t)}, x_{i+1}^{(t - 1)}, \dots, x_n^{(t - 1)})$ until convergence.
#
# In our case, the $x_i$'s are the pixels.
# We can sample from $p(x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = \frac{p(x_1, \dots, x_n)}{p(x_1, \dots, x_{i-1}, 0, x_{i+1}, \dots, x_n) + p(x_1, \dots, x_{i-1}, 1, x_{i+1}, \dots, x_n)}$ by enumeration: evaluating the joint density at $x_i = 0$ and $x_i = 1$ (free space or wall).
# To evaluate the density, GenJAX provides the method `model.assess(choice_map, args)`, which runs the model with all samples constrained to `choice_map` and returns the likelihood.
#
# TODO: it would probably be better to use `update` or `edit` here? Not sure how exactly yet.
#
# TODO: is there a more idiomatic way to write Gibbs? Should the proposal be a @genjax.gen model?

# +
full_model_assess = jax.jit(full_model.assess)

def gibbs_update_pixel(key, readings, is_wall, i):
    is_wall_false = is_wall.at[i].set(0)
    is_wall_true = is_wall_false.at[i].set(1)
    chm_false = C["readings"].set(readings) ^ C["is_wall"].set(is_wall_false)
    (false_weight, _) = full_model_assess(chm_false, ())
    chm_true = C["readings"].set(readings) ^ C["is_wall"].set(is_wall_true)
    (true_weight, _) = full_model_assess(chm_true, ())
    # categorical automatically normalizes the weights
    pixel_is_wall = genjax.categorical.sample(key, jnp.array([false_weight, true_weight])).astype(jnp.float32)
    return pixel_is_wall

def simple_gibbs_sweep(key, readings, is_wall):
    subkeys = jax.random.split(key, ALL_GRID_POINTS.shape[0])
    is_wall = jax.lax.fori_loop(
        0,
        ALL_GRID_POINTS.shape[0],
        lambda i, is_wall: is_wall.at[i].set(gibbs_update_pixel(subkeys[i], readings, is_wall, i)),
        is_wall
    )
    return is_wall


# -

# Starting from any map, applying Gibbs sweeps will (under certain conditions) yield approximate samples from the posterior distribution.
# Let's try this out.

def run_gibbs_chain(key, gibbs_update, num_samples=100):
    is_wall = jnp.zeros(ALL_GRID_POINTS.shape[0])
    key = jax.random.key(1)
    key, *subkeys = jax.random.split(key, num_samples + 1)
    subkeys = jnp.array(subkeys)
    gibbs_update_jitted = jax.jit(gibbs_update)
    _, gibbs_chain = jax.lax.scan(lambda is_wall, key: (gibbs_update_jitted(key, observed_readings, is_wall), is_wall), is_wall, subkeys)
    return gibbs_chain
gibbs_chain = run_gibbs_chain(jax.random.key(0), simple_gibbs_sweep)


# +
def plot_ground_truth_walls(pixels):
    pixel_vertices = pixels.pos[:, None, :] + OFFSETS_CLOSED[None, :, :]
    return [Plot.line(vertices, stroke="green", strokeWidth=3) for vertices, is_wall in zip(pixel_vertices, pixels.is_wall) if is_wall > 0]

ground_truth_plot = plot_ground_truth_walls(true_pixels)
sensors_plot = plot_sensors(observed_readings)
# -

# Use the slider at the bottom of the first plot to visualize the iterations of the Gibbs sampler.
# The second plot averages over the chain and gives an idea of how likely each pixel is to be a wall.

num_frames = 100
thinning = gibbs_chain.shape[0] // num_frames
animation = Plot.Frames([
    make_plot(plot_walls(Pixel(ALL_GRID_POINTS, sample)) + sensors_plot + ground_truth_plot)
    for sample in gibbs_chain[::thinning]
])
display(animation)
plot = make_plot(plot_walls(Pixel(ALL_GRID_POINTS, jnp.mean(gibbs_chain, axis=0))) + sensors_plot + ground_truth_plot)
display(plot)


# We can see that it takes about 10 iterations before all the walls at the bottom are removed.

# ## Smarter Gibbs sampling
#
# We can make Gibbs sampling converge faster by updating pixels in a different order.
# Note that if a pixel near the origin is a wall, it casts a "shadow" and all the pixels in its shadow are irrelevant (and thus sampled from the prior as fair coin flips).
# If the pixel near the origin flips and becomes free space, all the pixels in its shadow are suddenly "visible", but will be randomly initialized.
#
# For this reason it is better to update the pixels from the center outwards.
# This way, the pixels in the center are updated **before** the pixels in their shadow, so the pixels in the shadow can be updated in the same sweep.
# Specifically, we first update the pixels according to their distance to the origin.
# First, the origin itself, then the "diamond" around it, then the next layer etc.

# +
def smarter_gibbs_update_distance(key, readings, is_wall, distance):
    subkeys = jax.random.split(key, ALL_GRID_POINTS.shape[0])
    updated_is_wall = jax.lax.fori_loop(
        0,
        ALL_GRID_POINTS.shape[0],
        lambda i, is_wall: jax.lax.cond(
            jnp.abs(ALL_GRID_POINTS[i, 0]) + jnp.abs(ALL_GRID_POINTS[i, 1]) == distance,
            lambda: is_wall.at[i].set(gibbs_update_pixel(subkeys[i], readings, is_wall, i)),
            lambda: is_wall
        ),
        is_wall
    )
    return updated_is_wall

def smarter_gibbs_update(key, readings, is_wall):
    subkeys = jax.random.split(key, 2 * GRID_RADIUS + 1)
    is_wall = jax.lax.fori_loop(
        0,
        2 * GRID_RADIUS + 1,
        lambda distance, is_wall: smarter_gibbs_update_distance(subkeys[distance], readings, is_wall, distance),
        is_wall
    )
    return is_wall


# -

gibbs_chain = run_gibbs_chain(jax.random.key(0), smarter_gibbs_update)

ground_truth_plot = plot_ground_truth_walls(true_pixels)
sensors_plot = plot_sensors(observed_readings)
animation = Plot.Frames([
    make_plot(plot_walls(Pixel(ALL_GRID_POINTS, sample)) + ground_truth_plot + sensors_plot + Plot.ellipse([[0, 0]], r=0.2, fill="red"))
    for sample in gibbs_chain
])
display(animation)
plot = make_plot(plot_walls(Pixel(ALL_GRID_POINTS, jnp.mean(gibbs_chain, axis=0))) + ground_truth_plot + sensors_plot)
display(plot)


# Here it only takes 2 iterations before all the walls at the bottom are (correctly) removed.

# ## Block Gibbs sampling (NOT READY FOR REVIEW)
#
# TODO: discuss other optimization: block Gibbs for pixels with the same L_infinity distance to the origin.

# +
def block_gibbs_update_distance(key, readings, is_wall, distance):
    subkeys = jax.random.split(key, ALL_GRID_POINTS.shape[0])
    pixel_is_wall = jax.vmap(
        lambda key, i: jax.lax.cond(
            jnp.maximum(jnp.abs(ALL_GRID_POINTS[i, 0]), jnp.abs(ALL_GRID_POINTS[i, 1])) == distance,
            lambda: gibbs_update_pixel(key, readings, is_wall, i),
            lambda: is_wall[i] # TODO: this seems wasteful, but I don't know how to do this statically in JAX
        ),
        in_axes=(0, 0)
    )(subkeys, jnp.arange(ALL_GRID_POINTS.shape[0]))
    return pixel_is_wall

def block_gibbs_update(key, readings, is_wall):
    is_wall = jax.lax.fori_loop(0, GRID_RADIUS + 1, lambda distance, is_wall: block_gibbs_update_distance(key, readings, is_wall, distance), is_wall)
    return is_wall


# -

gibbs_chain = run_gibbs_chain(jax.random.key(0), block_gibbs_update)

ground_truth_plot = plot_ground_truth_walls(true_pixels)
sensors_plot = plot_sensors(observed_readings)
animation = Plot.Frames([
    make_plot(plot_walls(Pixel(ALL_GRID_POINTS, sample)) + ground_truth_plot + sensors_plot + Plot.ellipse([[0, 0]], r=0.2, fill="red"))
    for sample in gibbs_chain
])
display(animation)
plot = make_plot(plot_walls(Pixel(ALL_GRID_POINTS, jnp.mean(gibbs_chain, axis=0))) + ground_truth_plot + sensors_plot)
display(plot)

# Here it only takes 3 iterations before all the walls at the bottom are (correctly) removed.
#
# TODO: is it OK to parallelize Gibbs in this way?
