# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (Poetry)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mapping tutorial

# %% [markdown]
# The goal of this tutorial is – in a sense – the opposite of the localization tutorial. We consider the following scenario:
# * We have a two-dimensional map consisting of walls and free space.
# * A robot is located at the center of the map (so it knows its position).
# * The robot is equipped with a sensor that can (noisily) measure the distance to the nearest wall in several directions.
# * We want to infer the locations of the walls in the map.
#
# As a simplifying assumption, we assume that
# * the map is a $21 \times 21$ grid of unit blocks ("pixels") and
# * each pixel is either a wall or a free space.
#
# This tutorial will teach you how to:
# * model this scenario in GenJAX,
# * perform Bayesian inference to infer the walls on the map from noisy sensor measurements:
#   * first, via importance sampling (which turns out to perform poorly)
#   * then, via **Gibbs sampling** (which turns out to perform well),
# * using visualizations and interactivity in GenStudio.
#
# We will explore various variants of Gibbs sampling, so this can also be seen as a tutorial for Gibbs sampling.
#
# A **fully interactive version** of this tutorial can be found at the very end of this notebook.
#
# Potential future extensions:
# * include observations from several points of view
# * include uncertainty in the robot's position
# * allow the robot to move around (including uncertainty in the motion)

# %%
# Global setup code

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import jax.random as jrand
import genjax
from genjax import ChoiceMapBuilder as C
from genjax import pretty
from genjax.typing import IntArray
from penzai import pz

pretty()

Plot.configure({"display_as": "html", "dev": False})


# %% [markdown]
# ## Loading the map
#
# The following function is useful to load the ground truth map from a string (which makes editing the walls by hand easier than the array representation).

# %%

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

GRID_SIZE = 21
CENTER = jnp.array([GRID_SIZE // 2, GRID_SIZE // 2])

def extract_points_from_grid_string(grid_string):
    """Extracts the points from the grid string."""
    lines = grid_string.strip().split("\n")
    height = len(lines)
    width = len(lines[0])
    walls = jnp.zeros((height, width))
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "X":
                walls = walls.at[i, j].set(1)
    return walls

true_walls = extract_points_from_grid_string(GRID)
true_walls

# %% [markdown]
# Now, `true_walls` is a 21 x 21 array of 0s and 1s, where 1s indicate the presence of a wall.

# %% [markdown]
# ## Plotting
#
# In order to understand the problem, our code, and for debugging, it is useful to have visualizations.
# Hence we define the following functions to plot the map.

# %%
js_code = Plot.Import(
    source="""
export const wallsToCenters = (walls) => {
   // walls is a 2d array of 0s and 1s 
   // we want to return a list of [x1, y1, x2, y2] for each 1 in the array
   // we can do this by iterating over the array and collecting the coordinates
   // of each 1
   const centers = [];
   for (let i = 0; i < walls.length; i++) {
      for (let j = 0; j < walls[i].length; j++) {
        if (walls[i][j] > 0) {
            centers.push([i, j, walls[i][j]]);
        }
      }
   }
   return centers;
}
""",
    refer=["wallsToCenters"],
)

plot_walls = (
    Plot.rect(
        Plot.js("wallsToCenters($state.true_walls)"),
        x1=Plot.js("([x, y, value]) => x - 0.5"), 
        x2=Plot.js("([x, y, value]) => x + 0.5"),
        y1=Plot.js("([x, y, value]) => y - 0.5"),
        y2=Plot.js("([x, y, value]) => y + 0.5"),
        stroke=Plot.constantly("ground truth"),
        strokeWidth=2,
        fillOpacity=Plot.js("([x, y, value]) => value"),
    )
    + Plot.domain([0, GRID_SIZE], [0, GRID_SIZE])
    + Plot.aspectRatio(1)
    + Plot.width(500)
)

def make_plot(true_walls):
    return js_code & Plot.initial_state({"true_walls": true_walls}, sync=True) & Plot.new(plot_walls, Plot.color_legend())


# %%
make_plot(true_walls)


# %% [markdown]
# ## Interactive map

# %%
def on_click(widget, event):
    x, y = round(event["x"]), round(event["y"])
    true_walls = jnp.array(widget.state.true_walls)
    true_walls = true_walls.at[x, y].set(1)
    widget.state.update({"true_walls": true_walls})

walls = jnp.zeros((GRID_SIZE, GRID_SIZE))

interactive_walls = Plot.events(
    onClick=on_click, onDraw=on_click
)

def make_plot(true_walls, extra_plot=None, interactive=False):
    true_walls = true_walls.astype(jnp.float32)
    map_plot = plot_walls + extra_plot
    if interactive:
        map_plot = map_plot + interactive_walls
    plot = js_code & Plot.initial_state({"true_walls": true_walls}, sync=True) & map_plot
    if interactive:
        plot = plot | [
            "div.bg-blue-500.text-white.p-3.rounded-sm",
            {
                "onClick": lambda widget, _event: widget.state.update(
                    {"true_walls": jnp.zeros((GRID_SIZE, GRID_SIZE))}
                )
            },
            "Clear walls",
        ]
        plot = plot.display_as("widget")
    return plot

make_plot(true_walls, interactive=True)


# %% [markdown]
# ## Prior
#
# For a Bayesian model, we first need to specify our prior belief about the map.
# We will keep it very simple here and assume that each pixel is a wall with probability 0.5.

# %%
def walls_prior(prior_wall_prob):
    return genjax.flip.repeat(n=GRID_SIZE).repeat(n=GRID_SIZE)(prior_wall_prob)


# %% [markdown]
# In order to get a better understanding, let us sample from the prior.
# To do this, JAX requires a random seed, called "key", which we pass to the `.simulate` method of the generative function.
# The result is a **trace**, which contains the return value of the function, the score, and the choice ("sampled values during execution").
# The score is the log probability of the choices.

# %%
key = jrand.key(0)
sample_prior_jitted = jax.jit(walls_prior(prior_wall_prob=0.1).simulate)
tr = sample_prior_jitted(key, ())
display(tr.get_score())
display(tr.get_choices())
display(tr.get_retval())

# %% [markdown]
# Let's visualize the map of pixels that was sampled from the prior.

# %%
walls = tr.get_retval()
make_plot(walls)

# %% [markdown]
# We can see that, in fact, about half the pixels are walls. So the prior seems to work as intended.
#
# POSSIBLE EXTENSION: However, these maps don't look like real-world maps. The prior could be refined in various ways:
# * Probably, less than 50% of the pixels are walls, so the probability could be lowered.
# * "Wall pixels" are likely connected, so we could generate line segments in the prior rather than individual pixels.

# %% [markdown]
# ## Exact sensor model
#
# As mentioned at the start, the robot located at the origin has a sensor to measure the distance to the nearest wall in several directions.
# We can model this as follows.
# The math in the next cell is standard geometry, but not relevant for the overall understanding of modeling and inference in GenJAX, so feel free to skip the details.

# %%
OFFSETS = jnp.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
])

def line_segments_of_pixel(pixel: IntArray):
    vertices = pixel[None, :] + OFFSETS
    return jnp.stack([vertices, jnp.roll(vertices, 1, axis=0)], axis=1)

line_segments = jax.vmap(line_segments_of_pixel, in_axes=0)

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


def distance(pos, dir, seg):
    """
    Computes the distance from `pos` to a segment `seg`, in a given direction `dir`.

    Args:
    - pos: The position: `[pos_x, pos_y]`.
    - dir: The direction: `[dir_x, dir_y]`.
    - seg: The Segment object: `[[start_x, start_y], [end_x, end_y]]`.

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    a = solve_lines(pos, dir, seg[0], seg[1] - seg[0])
    return jnp.where(
        (a[0] >= 0.0) & (a[1] >= 0.0) & (a[1] <= 1.0),
        a[0],
        jnp.inf,
    )

def unit_dir(angle):
    """Unit vector in the direction of `angle`."""
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


# %% [markdown]
# The robot's sensor can measure the distance in `NUM_DIRECTIONS` equidistant directions.
#
# The following code computes the sensor readings.
# It makes use of the JAX functions `jax.lax.cond` (instead of `if`) and `jax.vmap` (instead of a for-loop) because we want JAX to JIT-compile and vectorize these operations.

# %%
def angles(num_angles):
    return jnp.arange(0, 1, 1 / num_angles) * 2 * jnp.pi

NUM_DIRECTIONS = 100
MAX_DISTANCE = GRID_SIZE * 2 # cap on distances to avoid infinities

def distance_to_pixel(pos, dir, coord, wall):
    """Distance from the origin to the pixel at `coord` if the ray in direction `dir` hits the pixel and it is a wall."""
    segs = line_segments_of_pixel(coord)
    dists = jax.vmap(lambda seg: distance(pos, dir, seg))(segs)
    return jax.lax.cond(
        wall > 0, # is there a wall?
        lambda: jnp.min(dists, axis=0),
        lambda: jnp.inf
    )

def distance_to_pixels(pos, dir, walls):
    """Distance from the origin to the nearest wall among `pixels` in direction `dir`.
    
    The distance is capped at `MAX_DISTANCE` to avoid infinities in the calculations."""
    return jnp.minimum(
        jnp.min(
            jax.vmap(
                lambda i, row: jax.vmap(
                    lambda j, is_wall: distance_to_pixel(pos, dir, jnp.array([i, j]), is_wall)
                )(jnp.arange(walls.shape[1]), row)
            )(jnp.arange(walls.shape[0]), walls)
        ),
        MAX_DISTANCE
    )

def sensor_distances(pos, pixels, angles):
    """Sensor distances in all directions (as specified by `angles`) for the map given by `pixels`."""
    return jax.vmap(lambda angle: distance_to_pixels(pos, unit_dir(angle), pixels))(angles)


# %% [markdown]
# As before, it is useful to have visualizations.

# %%
def plot_sensors(pos, readings, angles):
    """Plot the sensor readings."""
    unit_vecs = jax.vmap(unit_dir, in_axes=0)(angles)
    ray_endpoints = unit_vecs * readings[:, None]
    return [
        Plot.line([pos, pos + endpoint], stroke=Plot.constantly("sensor rays"))
        for endpoint in ray_endpoints
    ] + [
        Plot.ellipse([pos + endpoint], r=0.1, fill=Plot.constantly("sensor readings"))
        for endpoint in ray_endpoints
    ] + Plot.ellipse([pos], r=0.2, fill="red")

true_readings = sensor_distances(CENTER, true_walls, angles(NUM_DIRECTIONS))

make_plot(true_walls, plot_sensors(CENTER, true_readings, angles(NUM_DIRECTIONS)))


# %% [markdown]
# ## Noisy sensor model
#
# A real sensor is going to be noisy, which we model with a normal distribution.
# As before, we specify the model for a single direction/angle first, and then `vmap` it over all directions.

# %%
@genjax.gen
def sensor_model_single(pos, pixels, sensor_noise, angle):
    exact_distance = distance_to_pixels(pos, unit_dir(angle), pixels)
    return genjax.normal(exact_distance, sensor_noise) @ ()

sensor_model = sensor_model_single.vmap(in_axes=(None, None, None, 0))


# %% [markdown]
# Let's sample from the sensor model to see what data the robot receives.
# We can see that the sensor readings are no longer exact, but contain quite a bit of noise.
# The noise level can be controlled by changing the `NOISE` variable above.

# %%
@pz.pytree_dataclass
class ModelParams(genjax.Pytree):
    prior_wall_prob: float = genjax.Pytree.static()
    sensor_noise: float = genjax.Pytree.static()
    num_angles: int = genjax.Pytree.static()

DEFAULT_PARAMS = ModelParams(prior_wall_prob=0.5, sensor_noise=0.2, num_angles=NUM_DIRECTIONS)

# %%
key = jrand.key(0)
trace = sensor_model.simulate(key, (CENTER, true_walls, DEFAULT_PARAMS.sensor_noise, angles(NUM_DIRECTIONS)))
observed_readings = trace.get_retval()
make_plot(true_walls, plot_sensors(CENTER, observed_readings, angles(NUM_DIRECTIONS)))

# %% [markdown]
# To get an idea of what the robot sees, let's remove the walls. How can we infer the walls?

# %%
make_plot(jnp.zeros((GRID_SIZE, GRID_SIZE)), plot_sensors(CENTER, observed_readings, angles(NUM_DIRECTIONS)))


# %% [markdown]
# ## Full model
#
# Now, we can put the pieces together by combining the prior and the sensor model.

# %%
@genjax.gen
def full_model(pos, model_params):
    walls = walls_prior(model_params.prior_wall_prob) @ "walls"
    readings = sensor_model(pos, walls, model_params.sensor_noise, angles(model_params.num_angles)) @ "readings"
    return (walls, readings)


# %% [markdown]
# This model samples a map from the prior and then samples sensor readings from the sensor model.
# The samples are given the addresses `"walls"` and `"readings"`.
# When inspecting the trace, we can see that the sampled values are stored under these addresses.

# %%
key = jrand.key(1)
trace = full_model.simulate(key, (CENTER, DEFAULT_PARAMS))
walls, readings = trace.get_retval()
make_plot(walls, plot_sensors(CENTER, readings, angles(NUM_DIRECTIONS)))
trace.get_choices()


# %% [markdown]
# How can we infer the walls given some observed readings?

# %% [markdown]
# ## Importance sampling (self-normalized importance sampling)
#
# One of the simplest inference methods is importance sampling.
# To do this, we constrain our model to the observed data.
# For samples corresponding to observations (i.e. the sensor readings), we don't actually sample, but instead use the observed value and record the likelihood of seeing that observation.
# GenJAX provides a method for this: `model.importance(key, constraints, args)` runs the model with the random seed `key` and arguments `args`, but constrains some sampled values according to the `constraints` (the observations, i.e. the readings).
# It returns a trace with the sampled values ($walls$) and a log weight $\log p(observations \mid walls)$.
#
# If we run `.importance` several times, some traces (with a better prior sample) will have a higher weight and others (with a worse prior choice) will have lower weight.
# Suppose these weighted samples are $(walls_i, \log(w_i))$ for $i = 1 \ldots N$.
# We can **normalize** them as follows: $w'_i := \frac{w_i}{\sum_{i} w_i}$.
# Then posterior expected values can be approximated using the weighted samples: $\mathbb{E}_{walls \sim p(walls \mid observations)} [f(walls)] \approx \sum_{i=1}^N w'_i f(walls_i)$.
#
# We can also obtain unweighted samples approximating the posterior by **resampling**: sampling from a categorical distribution where each category $walls_i$ has $w'_i$.
#
# Let's try this.

# %%
def importance_sampling(key, args, observed_readings, N = 100):
    """Very naive MAP estimation. Just try N random samples and take the one with the highest weight."""
    model_importance = jax.jit(full_model.importance)
    keys = jrand.split(key, N)
    constraints = C["readings"].set(observed_readings)
    traces, log_weights = jax.vmap(lambda key: model_importance(key, constraints, args))(keys)
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    return traces, log_weights

key, subkey = jrand.split(jrand.key(0))
traces, log_weights = importance_sampling(subkey, (CENTER, DEFAULT_PARAMS), observed_readings, N=100000)
subkeys = jrand.split(subkey, 100)
resampled_indices = jax.vmap(lambda i: jrand.categorical(subkeys[i], log_weights))(jnp.arange(100))
resampled_indices

# %% [markdown]
# If we look at the resampled indices, we see that they are all the same.
# What this means is that one trace had such a high weight that none of the other traces had a chance to be resampled.
# This makes sense, because it is extremely unlikely to guess a "good" map just by sampling from the prior.
# So one (bad) sample will get lucky and have a high weight by chance and be selected every time.
# We can confirm this by looking at a histogram of the weights.

# %%
import matplotlib.pyplot as plt
plt.hist(log_weights, bins=100, range=(-10000, 0))
plt.show()


# %% [markdown]
# As we can see, the next best trace has a **log** weight that is orders of magnitude worse than the best sample.
#
# This is clearly bad. We need better inference methods.

# %% [markdown]
# ## Basic Gibbs sampling
#
# As a better inference algorithm, we turn to Gibbs sampling.
# Gibbs sampling is a simple algorithm to sample from a joint distribution $p(x_1, \dots, x_n)$, assuming one can sample from the conditional distributions $p(x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n)$.
# A Gibbs starts with initial values $x_1^{(0)}, \dots, x_n^{(0)}$ and then iteratively updates each $x_i$ from the conditional distribution $p(x_i \mid x_1^{(t)}, \dots, x_{i-1}^{(t)}, x_{i+1}^{(t - 1)}, \dots, x_n^{(t - 1)})$ until convergence.
#
# In our case, the $x_i$'s are the pixels.
# We can sample from $p(x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = \frac{p(x_1, \dots, x_n)}{p(x_1, \dots, x_{i-1}, 0, x_{i+1}, \dots, x_n) + p(x_1, \dots, x_{i-1}, 1, x_{i+1}, \dots, x_n)}$ by enumeration: evaluating the joint density at $x_i = 0$ and $x_i = 1$ (free space or wall).
# To evaluate the density, GenJAX provides the method `model.assess(choice_map, args)`, which runs the model with all samples constrained to `choice_map` and returns the likelihood.

# %%
full_model_assess = jax.jit(full_model.assess)

def gibbs_update_pixel(key, args, readings, walls, i, j):
    is_wall_false = walls.at[i, j].set(0)
    is_wall_true = walls.at[i, j].set(1)
    chm_false = C["readings"].set(readings) | C["walls"].set(is_wall_false)
    (false_weight, _) = full_model_assess(chm_false, args)
    chm_true = C["readings"].set(readings) | C["walls"].set(is_wall_true)
    (true_weight, _) = full_model_assess(chm_true, args)
    # categorical automatically normalizes the weights
    pixel_is_wall = genjax.categorical.sample(key, jnp.array([false_weight, true_weight])).astype(jnp.float32)
    return pixel_is_wall

def simple_gibbs_sweep(key, args, readings, walls):
    subkeys = jrand.split(key, walls.shape)
    walls = jax.lax.fori_loop(
        0,
        walls.shape[0],
        lambda i, walls: jax.lax.fori_loop(
            0,
            walls.shape[1],
            lambda j, walls: walls.at[i, j].set(gibbs_update_pixel(subkeys[i, j], args, readings, walls, i, j)),
            walls
        ),
        walls
    )
    return walls


# %% [markdown]
# Starting from any initial sample, applying Gibbs sweeps will (under certain conditions) yield approximate samples from the posterior distribution.
# Let's try this out.

# %%
def run_gibbs_chain(key, gibbs_update, args, readings, num_samples=100):
    walls = jnp.zeros((GRID_SIZE, GRID_SIZE))
    key, *subkeys = jrand.split(key, num_samples + 1)
    subkeys = jnp.array(subkeys)

    def inner():
        _, gibbs_chain = jax.lax.scan(lambda walls, key: (gibbs_update(key, args, readings, walls), walls), walls, subkeys)
        return gibbs_chain

    return jax.jit(inner)()

gibbs_chain = run_gibbs_chain(jrand.key(0), simple_gibbs_sweep, (CENTER, DEFAULT_PARAMS), observed_readings)
gibbs_chain[:10]


# %% [markdown]
# ## Incremental trace updates
#
# We can also write this more concisely in GenJAX by incrementally updating the current trace. Usually this should be faster, but it turns out to be slower in our case. This may be due to the overhead of the trace infrastructure involved.

# %%
def gibbs_update_pixel_incremental(key, trace: genjax.Trace, i, j):
    walls = trace.get_choices()["walls"]
    # IndexRequest should be faster but isn't. Too much overhead?
    # request = genjax.StaticRequest({"walls": genjax.IndexRequest(jnp.array(i), genjax.IndexRequest(jnp.array(j), genjax.Update(C.v(1.0 - walls[i, j]))))})
    request = genjax.StaticRequest({"walls": genjax.Update(C.v(walls.at[i, j].set(1.0 - walls[i, j])))})
    new_tr, inc_weight, _retdiff, _bwd_request = trace.edit(key, request, None)
    return jax.lax.cond(
        genjax.bernoulli.sample(key, inc_weight), # i.e. with probability e^inc_weight / (1 + e^inc_weight)
        lambda: new_tr, # use the updated trace
        lambda: trace, # otherwise keep the old trace
    )

def simple_gibbs_sweep_incremental(key, trace):
    shape = trace.get_choices()["walls"].shape
    subkeys = jrand.split(key, shape)
    trace = jax.lax.fori_loop(
        0,
        shape[0],
        lambda i, trace: jax.lax.fori_loop(
            0,
            shape[1],
            lambda j, trace: gibbs_update_pixel_incremental(subkeys[i][j], trace, i, j),
            trace
        ),
        trace
    )
    return trace

def run_gibbs_chain_incremental(key, gibbs_update, args, readings, num_samples=100):
    walls = jnp.zeros((GRID_SIZE, GRID_SIZE))
    constraints = C["walls"].set(walls) | C["readings"].set(readings)
    trace, _ = full_model.importance(key, constraints, args)
    key, *subkeys = jrand.split(key, num_samples + 1)
    subkeys = jnp.array(subkeys)

    def inner():
        _, gibbs_chain = jax.lax.scan(
            lambda trace, key: (gibbs_update(key, trace), trace.get_choices()["walls"]),
            trace,
            subkeys
        )
        return gibbs_chain
    
    return jax.jit(inner)()

gibbs_chain_incremental = jax.jit(lambda: run_gibbs_chain_incremental(jrand.key(0), simple_gibbs_sweep_incremental, (CENTER, DEFAULT_PARAMS), observed_readings))()
gibbs_chain_incremental[:10]


# %% [markdown]
# Due to the performance decrease, we will stick with the first version.

# %% [markdown]
# ## Plotting the inferred walls

# %%
def plot_inferred_walls(walls):
    return Plot.rect(
        Plot.js("wallsToCenters(%1)", walls),
        x1=Plot.js("([x, y, value]) => x - 0.5"), 
        x2=Plot.js("([x, y, value]) => x + 0.5"),
        y1=Plot.js("([x, y, value]) => y - 0.5"),
        y2=Plot.js("([x, y, value]) => y + 0.5"),
        fill=Plot.constantly("inferred walls"),
        fillOpacity=Plot.js("([x, y, value]) => value"),
    )

def make_plot(true_walls, pos=None, sensor_readings=None, angles=None, inferred_walls=None, interactive=False):
    true_walls = true_walls.astype(jnp.float32)
    map_plot = Plot.new()
    if inferred_walls is not None:
        inferred_walls = inferred_walls.astype(jnp.float32)
        map_plot += plot_inferred_walls(inferred_walls)
    map_plot += plot_walls
    if sensor_readings is not None and angles is not None:
        map_plot += plot_sensors(pos, sensor_readings, angles)
    if interactive:
        map_plot += interactive_walls
    plot = js_code & Plot.initial_state({"true_walls": true_walls}, sync=True) & Plot.new(map_plot, Plot.color_legend())
    if interactive:
        plot = plot | [
            "div.bg-blue-500.text-white.p-3.rounded-sm",
            {
                "onClick": lambda widget, _event: widget.state.update(
                    {"true_walls": jnp.zeros((GRID_SIZE, GRID_SIZE))}
                )
            },
            "Clear walls",
        ]
        plot = plot.display_as("widget")
    return plot

make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=true_walls)

# %% [markdown]
# Use the slider at the bottom of the first plot to visualize the iterations of the Gibbs sampler.
# The second plot averages over the chain and gives an idea of how likely each pixel is to be a wall.

# %%
num_frames = 100
thinning = gibbs_chain.shape[0] // num_frames
animation = Plot.Frames([
    make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=sample)
    for sample in gibbs_chain[::thinning]
])
display(animation)
gibbs_mean = jnp.mean(gibbs_chain, axis=0)
plot = make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=gibbs_mean)
display(plot)

# %% [markdown]
# We can see that it takes about 10 iterations before all the walls at the bottom are removed.

# %% [markdown]
# ## Gibbs sampling, part 2: better variable ordering
#
# Gibbs sampling does not require the variables to be updated in a specific order. They order can even be random (and that sometimes improves convergence).
#
# We can make Gibbs sampling converge faster by updating pixels in a different order.
# Note that if a pixel near the origin is a wall, it casts a "shadow" and all the pixels in its shadow are irrelevant (and thus sampled from the prior as fair coin flips).
# If the pixel near the origin flips and becomes free space, all the pixels in its shadow are suddenly "visible", but will have been randomly initialized before.
#
# For this reason it is better to update the pixels from the sensor position outwards.
# This way, the pixels near the sensor are updated **before** the pixels in their shadow, so the pixels in the shadow can be updated in the same sweep.
# Specifically, we first update the pixels according to their distance to the sensor position.
# First, the pixel at which the sensor is placed itself, then the "diamond" around it, then the next layer etc., like this:

# %%
jax.vmap(lambda dist: jax.vmap(lambda i: jax.vmap(lambda j: jnp.abs(i - 5) + jnp.abs(j - 5) == dist)(jnp.arange(11)))(jnp.arange(11)))(jnp.arange(10))


# %%
def smarter_gibbs_update_distance(key, args, readings, walls, distance):
    pos, _params = args
    subkeys = jrand.split(key, (GRID_SIZE, GRID_SIZE))
    updated_walls = jax.lax.fori_loop(
        0,
        GRID_SIZE,
        lambda i, walls: jax.lax.fori_loop(
            0,
            GRID_SIZE,
            lambda j, walls: walls.at[i, j].set(
                jax.lax.cond(
                    jnp.sum(jnp.floor(jnp.abs(jnp.array([i, j]) - pos))) == distance,
                    lambda: gibbs_update_pixel(subkeys[i, j], args, readings, walls, i, j),
                    lambda: walls[i, j]
                )
            ),
            walls
        ),
        walls
    )
    return updated_walls

def smarter_gibbs_update(key, args, readings, walls):
    subkeys = jrand.split(key, 2 * GRID_SIZE)
    walls = jax.lax.fori_loop(
        0,
        2 * GRID_SIZE,
        lambda distance, walls: smarter_gibbs_update_distance(subkeys[distance], args, readings, walls, distance),
        walls
    )
    return walls


# %%
gibbs_chain = run_gibbs_chain(jrand.key(0), smarter_gibbs_update, (CENTER, DEFAULT_PARAMS), observed_readings)
gibbs_chain[:10]

# %%
animation = Plot.Frames([
    make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=sample)
    for sample in gibbs_chain
])
display(animation)
gibbs_mean = jnp.mean(gibbs_chain, axis=0)
plot = make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=gibbs_mean)
display(plot)

# %% [markdown]
# Here it only takes 3 iterations before all the walls at the bottom are (correctly) removed.
#
# **Problem**: This approach only works well if the prior wall probability is near 0.5. Let's see what happens for 0.1.

# %%
modified_params = ModelParams(prior_wall_prob=0.05, sensor_noise=DEFAULT_PARAMS.sensor_noise, num_angles=DEFAULT_PARAMS.num_angles)
gibbs_chain = run_gibbs_chain(jrand.key(0), smarter_gibbs_update, (CENTER, modified_params), observed_readings)
gibbs_chain[:10]

# %%
animation = Plot.Frames([
    make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=sample)
    for sample in gibbs_chain
])
display(animation)
gibbs_mean = jnp.mean(gibbs_chain, axis=0)
plot = make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=gibbs_mean)
display(plot)


# %% [markdown]
# We see that Gibbs does not seem to converge at all. Why is that?
#
# Due to the low prior probability, there are very few walls on the map. Let's say we do a Gibbs update on a pixel that's currently a wall. If we remove the wall, that means the affected sensor rays will often not hit anything else and the sensor reading will change drastically. This means a sudden drop in likelihood, so the Gibbs update will leave the pixel unchanged. In other words, the pixel and its neighborhood are **highly correlated**, and Gibbs struggles in such situation.

# %% [markdown]
# ## Gibbs sampling, part 3: block Gibbs
#
# One way of dealing with correlated variables in Gibbs sampling is to update not a single variable at once, but a set of variables. In particular, we will update 2x2 blocks of pixels at once. For the Gibbs update of each 2x2 block, we will consider all 16 possible options for the placement of walls in that block. Intuitively, this helps because often we want to do something like move a wall "back" by one pixel, which is less likely to happen in independent Gibbs updates.
#
# Let's first generate all possible values for an `n x n` block

# %%
def generate_all_possible_blocks(block_size):
    """Returns all possible squares of size `block_size` x `block_size`, filled with 0s and 1s.
    
    Returned shape: 2^(block_size^2) x block_size x block_size"""
    values = jnp.meshgrid(*(jnp.arange(2) for _ in range(block_size * block_size)))
    values = [value.flatten() for value in values]
    values = jnp.stack(values, axis=1)
    return values.reshape(-1, block_size, block_size)

generate_all_possible_blocks(2)

# %% [markdown]
# The Gibbs update now works similar to the first version, but updates 2x2 blocks at once. Note that we don't use the clever ordering of the updates in Gibbs sweep. Instead, we go back to the first version of Gibbs that updates the pixels in the order of their indices. This still works well enough.
#
# Note that we consider all 2x2 blocks on the map, which may overlap (e.g. the one at (0, 0) and the one at (1, 1)). This does not a problem – the Gibbs update is still valid.

# %%
BLOCK_SIZE = 2

def gibbs_update_block(key, args, readings, walls, i, j):
    blocks = generate_all_possible_blocks(BLOCK_SIZE) # shape = (2^(BLOCK_SIZE^2), BLOCK_SIZE, BLOCK_SIZE)
    chm = C["readings"].set(readings) | C["walls"].set(walls)
    (weights, (walls_changed, _)) = jax.vmap(
        lambda block: full_model_assess(
            chm.at["walls"].set(jax.lax.dynamic_update_slice(walls, block.astype(jnp.float32), (i, j))),
            args
        )
    )(blocks)
    # categorical automatically normalizes the weights
    idx = genjax.categorical.sample(key, weights)
    return walls_changed[idx]

def block_gibbs_sweep(key, args, readings, walls):
    subkeys = jrand.split(key, walls.shape)
    walls = jax.lax.fori_loop(
        0,
        walls.shape[0] - BLOCK_SIZE + 1,
        lambda i, walls: jax.lax.fori_loop(
            0,
            walls.shape[1] - BLOCK_SIZE + 1,
            lambda j, walls: gibbs_update_block(subkeys[i, j], args, readings, walls, i, j),
            walls
        ),
        walls
    )
    return walls



# %%
gibbs_chain = run_gibbs_chain(jrand.key(0), block_gibbs_sweep, (CENTER, modified_params), observed_readings)
gibbs_chain

# %%
animation = Plot.Frames([
    make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=sample)
    for sample in gibbs_chain
])
display(animation)
gibbs_mean = jnp.mean(gibbs_chain, axis=0)
plot = make_plot(true_walls, pos=CENTER, sensor_readings=observed_readings, angles=angles(NUM_DIRECTIONS), inferred_walls=gibbs_mean)
display(plot)


# %% [markdown]
# One can see that each Gibbs sweep "pushes" the walls that are too close to the center outwards, until they match the observations better.

# %% [markdown]
# ## Fully interactive version
#
# You can play around with the setup and the block Gibbs sampler using the interactive widget below. Usage instructions are at the bottom. 

# %%
def on_click(widget, event):
    x, y = round(event["x"]), round(event["y"])
    pos = widget.state.position
    if (event["x"] - pos[0])**2 + (event["y"] - pos[1])**2 < 0.25:
        # don't draw a wall on the sensor
        widget.state.update({"wall_mode": False})
    else:
        widget.state.update({"wall_mode": True})
        x, y = round(event["x"]), round(event["y"])
        pos = widget.state.position
        true_walls = jnp.array(widget.state.true_walls)
        true_walls = true_walls.at[x, y].set(1 - true_walls[x, y])
        widget.state.update({"true_walls": true_walls})

def on_drag(widget, event):
    if widget.state.wall_mode:
        x, y = round(event["x"]), round(event["y"])
        pos = widget.state.position
        true_walls = jnp.array(widget.state.true_walls)
        true_walls = true_walls.at[x, y].set(1)
        widget.state.update({"true_walls": true_walls})

interactive_walls = Plot.events(
    onClick=on_click, onDraw=on_drag
)

plot_inferred_walls_interactive = Plot.rect(
    Plot.js("wallsToCenters($state.inferred_walls)", walls),
    x1=Plot.js("([x, y, value]) => x - 0.5"), 
    x2=Plot.js("([x, y, value]) => x + 0.5"),
    y1=Plot.js("([x, y, value]) => y - 0.5"),
    y2=Plot.js("([x, y, value]) => y + 0.5"),
    fill=Plot.constantly("inferred walls"),
    fillOpacity=Plot.js("([x, y, value]) => value"),
)

sensor_js_code = Plot.Import("""
export function rayEndpoints(pos, sensor_readings, num_angles) {
    let endpoints = [];
    let angleStep = (2 * Math.PI) / num_angles; // Divide full circle into equal parts

    for (let i = 0; i < num_angles; i++) {
        let angle = i * angleStep;
        let distance = sensor_readings[i];
        let x = pos[0] + distance * Math.cos(angle);
        let y = pos[1] + distance * Math.sin(angle);
        endpoints.push([x, y]);
    }
    return endpoints;
}
""", refer=["rayEndpoints"])

def on_position_drag(widget, event):
    widget.state.update({"wall_mode": False, "position": jnp.array([event["x"], event["y"]])})

plot_interactive_sensors = sensor_js_code & (
    Plot.line(Plot.js("rayEndpoints($state.position, $state.sensor_readings, $state.num_angles).flatMap(p => [p, $state.position])"), stroke=Plot.constantly("sensor rays"))
    + Plot.ellipse(Plot.js("rayEndpoints($state.position, $state.sensor_readings, $state.num_angles)"), r=0.1, fill=Plot.constantly("sensor readings"))
    + Plot.ellipse(
        [Plot.js("$state.position")],
        r=0.3,
        fill=Plot.constantly("sensor position"),
        render=Plot.renderChildEvents(onDrag=on_position_drag)
    )
)

def on_change(widget, _event):
    update_state(widget.state)

def update_state(state):
    true_walls = state.true_walls
    params = ModelParams(
        prior_wall_prob=float(state.prior_wall_prob),
        sensor_noise=float(state.sensor_noise),
        num_angles=int(state.num_angles),
    )
    pos = state.position
    key = jrand.key(0)
    readings = sensor_model.simulate(key, (pos, true_walls, params.sensor_noise, angles(params.num_angles))).get_retval()
    state.sensor_readings = readings
    update_inferred_walls(state)

def update_inferred_walls(state):
    if state.show_mean:
        state.inferred_walls = state.mean_chain if state.mean_chain is not None else jnp.zeros(true_walls.shape)
    else:
        state.inferred_walls = state.chain[state.chain_idx] if state.chain is not None else jnp.zeros(true_walls.shape)

def do_inference(state):
    update_state(state)
    pos = state.position
    readings = state.sensor_readings
    params = ModelParams(
        prior_wall_prob=float(state.prior_wall_prob),
        sensor_noise=float(state.sensor_noise),
        num_angles=int(state.num_angles),
    )
    key = jrand.key(0)
    state.chain = run_gibbs_chain(key, block_gibbs_sweep, (pos, params), readings)
    state.mean_chain = jnp.mean(state.chain, axis=0)
    update_inferred_walls(state)

def make_interactive_plot(true_walls):
    true_walls = true_walls.astype(jnp.float32)
    map_plot = Plot.new()
    map_plot += plot_inferred_walls_interactive
    map_plot += plot_walls
    map_plot += plot_interactive_sensors
    map_plot += interactive_walls
    buttons = (Plot.html([
        "div.bg-blue-500.text-white.p-3.rounded-sm",
        {
            "onClick": lambda widget, _event: do_inference(widget.state)
        },
        "Run inference",
    ]) & Plot.html([
        "div.bg-blue-500.text-white.p-3.rounded-sm",
        {
            "onClick": lambda widget, _event: widget.state.update(
                {"true_walls": jnp.zeros((GRID_SIZE, GRID_SIZE))}
            )
        },
        "Clear walls",
    ]) & Plot.html([
        "div.bg-blue-500.text-white.p-3.rounded-sm",
        {
            "onClick": lambda widget, _event: widget.state.update(
                {"inferred_walls": jnp.zeros((GRID_SIZE, GRID_SIZE)), "chain": None, "mean_chain": None}
            )
        },
        "Clear inferred walls",
    ]))
    sliders = (
        (Plot.html([
            "label",
            {"class": "flex items-center gap-2 cursor-pointer"},
            [
                "input",
                {
                    "type": "checkbox",
                    "checked": Plot.js("$state.show_mean"),
                    "onChange": Plot.js("(e) => $state.show_mean = e.target.checked")
                }
            ],
            "show average over chain"
        ]) & Plot.Slider(
            key="chain_idx",
            range=(0, 100),
            step=1,
            label=Plot.js("$state.show_mean ? `Deselect 'show average' to view individual samples` : `Sample no.: ${$state.chain_idx}`"),
        ))
        | (Plot.Slider(
            key="prior_wall_prob",
            range=(0, 1),
            step=0.005,
            label=Plot.js("`Prior wall probability: ${$state.prior_wall_prob}`"),
        ))
        & (Plot.Slider(
            key="sensor_noise",
            range=(0, 5),
            step=0.05,
            label=Plot.js("`Sensor noise: ${$state.sensor_noise}`"),
        ))
        & (Plot.Slider(
            key="num_angles",
            range=(10, 1000),
            step=10,
            label=Plot.js("`Number of angles: ${$state.num_angles}`"),

        ))
    )
    initial_pos = CENTER
    initial_noise = 0.2
    plot = (
        js_code & sensor_js_code
        & Plot.initial_state({
                "true_walls": true_walls,
                "position": initial_pos,
                "sensor_noise": initial_noise,
                "prior_wall_prob": 0.5,
                "num_angles": NUM_DIRECTIONS,
                "sensor_readings": sensor_model.simulate(key, (initial_pos, true_walls, initial_noise, angles(NUM_DIRECTIONS))).get_retval(),
                "inferred_walls": jnp.zeros(true_walls.shape),
                "show_mean": True,
                "chain_idx": 0,
                "chain": None,
                "mean_chain": None,
                "wall_mode": False,
            },
            sync=True)
        & Plot.new(map_plot, Plot.color_legend())
    ) | buttons | sliders | Plot.onChange({
        "prior_wall_prob": on_change,
        "sensor_noise": on_change,
        "num_angles": on_change,
        "true_walls": on_change,
        "position": on_change,
        "chain_idx": on_change,
        "show_mean": on_change,
    })
    plot = plot.display_as("widget")
    return plot

make_interactive_plot(true_walls)

# %% [markdown]
# ## Usage instructions
#
# * draw walls (ground truth) by clicking pixels or clicking and dragging
# * remove individual walls by clicking the pixel again
# * remove all walls by clicking "Clear walls"
# * clear the inference result by clicking "Clear inferred walls"
# * use the sliders to update the parameters of the model
# * click and drag the little circle to change the position of the sensor
# * run block Gibbs with the current parameter settings by clicking "Run inference" (This may take a few seconds, please be patient.)
# * inspect the estimated posterior probability of a pixel being a wall by selecting "show average over chain"
# * inspect individual Gibbs samples by deselecting "show average over chain" and using the slider to its right
#
# Note: inference does not update automatically due to the time it takes. You have to click "Run inference" to update the inference results.
#
