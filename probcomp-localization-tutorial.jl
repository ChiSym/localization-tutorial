# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# %% [markdown]
# # ProbComp Localization Tutorial
#
# This notebook aims to give an introduction to probabilistic computation (ProbComp).  This term refers to a way of expressing probabilistic constructs in a computational paradigm, made precise by a probablistic programming language (PPL).  The programmer can thus encode their probabilistic intuition for solving a problem into an algorithm.  Back-end language work automates the routine but error-prone derivations.

# %%
# Global setup code

# Install dependencies listed in Project.toml
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# The dependencies consist of the following Julia packages.
using Dates: now, value
using JSON: parsefile
using Plots
using Gen

# Ensure a location for image generation.
mkpath("imgs");

# %% [markdown]
# ## Sensing a robot's location on a map
#
# ### The map
#
# The tutorial will revolve around modeling the activity of a robot within some space.  A large simplifying assumption, which could be lifted with more effort, is that we have been given a *map* of the space, to which the robot will have access.
#
# The code below loads such a map, along with other data for later use.  Generally speaking, we keep general code and specific examples in separate cells, as signposted here.

# %%
# General code here

struct Segment
    p1 :: Vector{Float64}
    p2 :: Vector{Float64}
    # The quantity `p2-p1` is called upon in hot loops, so we cache it.
    dp :: Vector{Float64}
    Segment(p1 :: Vector{Float64}, p2 :: Vector{Float64}) = new(p1, p2, p2-p1)
end
Base.show(io :: IO, s :: Segment) = Base.show(io, "Segment($(s.p1), $(s.p2))")

function create_segments(verts; loop_around=false)
    segs = [Segment(p1, p2) for (p1, p2) in zip(verts[1:end-1], verts[2:end])]
    if loop_around; push!(segs, Segment(verts[end], verts[1])) end
    return segs
end

function make_world(walls_vec, clutters_vec; args...)
    walls = create_segments(walls_vec; args...)
    clutters = [create_segments(clutter; args...) for clutter in clutters_vec]
    all_points = [walls_vec ; clutters_vec...]
    x_min, x_max = extrema(first, all_points)
    y_min, y_max = extrema(last, all_points)
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
    return (; walls, clutters, bounding_box, box_size, center_point)
end

function load_world(file_name; args...)
    data = parsefile(file_name)
    walls_vec = Vector{Vector{Float64}}(data["wall_verts"])
    clutters_vec = Vector{Vector{Vector{Float64}}}(data["clutter_vert_groups"])
    return make_world(walls_vec, clutters_vec; args...)
end;

# %%
# Specific example code here

world = load_world("world.json");

# %% [markdown]
# ### Plotting
#
# It is crucial to picture what we are doing at all times, so we develop plotting code early and often.

# %%
unit_circle_xs = [cos(t) for t in LinRange(0., 2pi, 500)]
unit_circle_ys = [sin(t) for t in LinRange(0., 2pi, 500)]
make_circle(p, r) = (p[1] .+ r * unit_circle_xs, p[2] .+ r * unit_circle_ys)

function plot_list!(list; label=nothing, args...)
    if !isempty(list)
        plt = plot!(list[1]; label=label, args...)
        for item in list[2:end]; plot!(item; label=nothing, args...) end
        return plt
    end
end

Plots.plot!(seg :: Segment; args...) = plot!([seg.p1[1], seg.p2[1]], [seg.p1[2], seg.p2[2]]; args...)
Plots.plot!(segs :: Vector{Segment}; args...) = plot_list!(segs; args...)
Plots.plot!(seg_groups :: Vector{Vector{Segment}}; args...) = plot_list!(seg_groups; args...)

function plot_world(world, title; show=())
    border = world.box_size * (3.)/19.
    the_plot = plot(
        size         = (500, 500),
        aspect_ratio = :equal,
        grid         = false,
        xlim         = (world.bounding_box[1]-border, world.bounding_box[2]+border),
        ylim         = (world.bounding_box[3]-border, world.bounding_box[4]+border),
        title        = title,
        legend       = :bottomleft)
    (walls_label, clutter_label) = :label in show ? ("walls", "clutters") : (nothing, nothing)
    plot!(world.walls; c=:black, label=walls_label)
    if :clutters in show; plot!(world.clutters; c=:magenta, label=clutter_label) end
    return the_plot
end;

# %% [markdown]
# Following this initial display of the given data, we *suppress the clutters* until much later in the notebook.

# %%
plot_world(world, "Given data", show=(:label, :clutters))

# %% [markdown]
# POSSIBLE VIZ GOAL: user-editable map, clutters, etc.

# %% [markdown]
# ### Robot poses
#
# We will model the robot's physical state as a *pose* (or mathematically speaking a ray), defined to be a *position* (2D point relative to the map) plus a *heading* (angle from -$\pi$ to $\pi$).
#
# These will be visualized using arrows whose tip is at the position, and whose direction indicates the heading.

# %%
struct Pose
    p  :: Vector{Float64}
    hd :: Float64
    # The quantity `[cos(hd), sin(hd)]` is called upon in hot loops, so we cache it.
    dp :: Vector{Float64}
    Pose(p :: Vector{Float64}, hd :: Float64) = new(p, rem2pi(hd, RoundNearest), [cos(hd), sin(hd)])
end
Base.show(io :: IO, p :: Pose) = Base.show(io, "Pose($(p.p), $(p.hd))")


Plots.plot!(p :: Pose; r=0.5, args...) = plot!(Segment(p.p, p.p + r * p.dp); arrow=true, args...)
Plots.plot!(ps :: Vector{Pose}; args...) = plot_list!(ps; args...);

# %%
some_poses = [Pose([uniform(world.bounding_box[1], world.bounding_box[2]),
                    uniform(world.bounding_box[3], world.bounding_box[4])],
                   uniform(-pi,pi))
              for _ in 1:20]

plot_world(world, "Given data")
plot!(Pose([1., 2.], 0.); color=:green3, label="a pose")
plot!(Pose([2., 3.], pi/2.); color=:green4, label="another pose")
plot!(some_poses; color=:brown, label="some poses")

# %% [markdown]
# POSSIBLE VIZ GOAL: user can manipulate a pose.  (Unconstrained vs. map for now.)

# %% [markdown]
# ### Ideal sensors
#
# The robot will need to reason about its location on the map, on the basis of LIDAR-like sensor data.
#
# An "ideal" sensor reports the exact distance cast to a wall.  (It is capped off at a max value in case of error.)

# %%
# A general algorithm to find the interection of a ray and a line segment.

det2(u, v) = u[1] * v[2] - u[2] * v[1]

function distance(p :: Pose, seg :: Segment; PARALLEL_TOL=1.0e-10)
    # Check if pose is parallel to segment.
    det = det2(p.dp, seg.dp)
    if abs(det) < PARALLEL_TOL; return Inf end

    # Return unique s, t such that p.p + s * p.dp == seg.p1 + t * seg.dp.
    pq = (p.p[1] - seg.p1[1], p.p[2] - seg.p1[2])
    s = det2(seg.dp, pq) / det
    t = det2(p.dp, pq) / det

    # Pose is oriented towards from segment iff s >= 0.
    # Point of intersection lies on segment (as opposed to the infinite line) iff 0 <= t <= 1.
    return (s >= 0. && 0. <= t <= 1.) ? s : Inf
end;

# %%
function sensor_distance(pose, walls, box_size)
    d = minimum(distance(pose, seg) for seg in walls)
    # Capping to a finite value avoids issues below.
    return isinf(d) ? 2. * box_size : d
end;

sensor_angle(sensor_settings, j) =
    sensor_settings.fov * (j - (sensor_settings.num_angles - 1) / 2.) / (sensor_settings.num_angles - 1)

function ideal_sensor(pose, walls, sensor_settings)
    readings = Vector{Float64}(undef, sensor_settings.num_angles)
    for j in 1:sensor_settings.num_angles
        sensor_pose = Pose(pose.p, pose.hd + sensor_angle(sensor_settings, j))
        readings[j] = sensor_distance(sensor_pose, walls, sensor_settings.box_size)
    end
    return readings
end;

# %%
project_sensor(pose, angle, s) = let rotated = Pose(pose.p, pose.hd + angle); rotated.p + s * rotated.dp end

function plot_sensors!(pose, color, readings, label, sensor_settings)
    plot!([pose.p[1]], [pose.p[2]]; color=color, label=nothing, seriestype=:scatter, markersize=3, markerstrokewidth=0)
    projections = [project_sensor(pose, sensor_angle(sensor_settings, j), s) for (j, s) in enumerate(readings)]
    plot!(first.(projections), last.(projections);
            color=:blue, label=label, seriestype=:scatter, markersize=3, markerstrokewidth=1, alpha=0.25)
    plot!([Segment(pose.p, pr) for pr in projections]; color=:blue, label=nothing, alpha=0.25)
end

function frame_from_sensors(world, title, poses, poses_color, poses_label, pose, readings, readings_label, sensor_settings; show=())
    the_plot = plot_world(world, title; show=show)
    plot!(poses; color=poses_color, label=poses_label)
    plot_sensors!(pose, poses_color, readings, readings_label, sensor_settings)
    return the_plot
end;

# %%
sensor_settings = (fov = 2π*(2/3), num_angles = 41, box_size = world.box_size)

ani = Animation()
for pose in some_poses
    frame_plot = frame_from_sensors(
        world, "Ideal sensor distances",
        pose, :green2, "robot pose",
        pose, ideal_sensor(pose, world.walls, sensor_settings), "ideal sensors",
        sensor_settings)
    frame(ani, frame_plot)
end
gif(ani, "imgs/ideal_distances.gif", fps=1)

# %% [markdown]
# POSSIBLE VIZ GOAL: as user manipulates pose, sensors get updated.

# %% [markdown]
# ## First steps in modeling uncertainty using Gen
#
# The robot will need to reason about its possible location on the map using incomplete information—in a pun, it must nagivate the uncertainty.  The `Gen` system facilitates programming the required probabilistic logic.  We will introduce the features of Gen, starting with some simple features now, and bringing in more complex ones later.
#
# Each piece of the model is declared as a *generative function* (GF).  The `Gen` library provides two DSLs for constructing GFs: the dynamic DSL using the decorator `@gen` on a function declaration, and the static DSL similarly decorated with `@gen (static)`.  The dynamic DSL allows a rather wide class of program structures, whereas the static DSL only allows those for which a certain static analysis may be performed.
#
# The library offers primitive *distributions* such as "Bernoulli" and "normal", and these two DLSs offer the *sampling operator* `~`.  GFs may sample from distributions and, recursively, other GFs using `~`.  A generative function embodies the *joint distribution* over the latent choices indicated by the sampling operations.

# %% [markdown]
# ### Creating noisy measurements using `Gen.propose`
#
# We have on hand two kinds of things to model: the robot's pose (and possibly its motion), and its sensor data.  We tackle the sensor model first because it is simpler.
#
# Here is its declarative model in `Gen`:

# %%
@gen function sensor_model(pose, walls, sensor_settings)
    for j in 1:sensor_settings.num_angles
        sensor_pose = Pose(pose.p, pose.hd + sensor_angle(sensor_settings, j))
        {j => :distance} ~ normal(sensor_distance(sensor_pose, walls, sensor_settings.box_size), sensor_settings.s_noise)
    end
end;

# %% [markdown]
# This model differs from `ideal_sensor` in the following ways.  The ideal sensor measurements themselves are no longer stored into an array, but are instead used as the means of Gaussian distributions (representing our uncertainty about them).  *Sampling* from these distributions, using the `~` operator, occurs at the addresses `j => :distance`.
#
# Moreover, the function returns no explicit value.  But there is no loss of information here: the model can be run with `Gen.propose` semantics, which performs the required draws from the sampling operations and organizes them according to their address, returning the corresponding *choice map* data structure.  The method is called with the GF plus a tuple of arguments.

# %%
sensor_settings = (sensor_settings..., s_noise = 0.10)
cm, w = propose(sensor_model, (Pose([1., 1.], pi/2.), world.walls, sensor_settings))
cm

# %%
# For brevity, show just a subset of the choice map's addresses.
get_selected(cm, select((1:5)...))

# %% [markdown]
# With a little wrapping, one gets a function of the same type as `ideal_sensor`.

# %%
function noisy_sensor(pose, walls, sensor_settings)
    cm, _ = propose(sensor_model, (pose, walls, sensor_settings))
    return [cm[j => :distance] for j in 1:sensor_settings.num_angles]
end;

# %% [markdown]
# Let's get a picture of the distances returned by the model:

# %%
ani = Animation()
for pose in some_poses
    frame_plot = frame_from_sensors(
        world, "Sensor model (samples)",
        pose, :green2, "robot pose",
        pose, noisy_sensor(pose, world.walls, sensor_settings), "noisy sensors",
        sensor_settings)
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances.gif", fps=1)

# %% [markdown]
# POSSIBLE VIZ GOAL: same sensor interactive as before, now with noisy sensors.

# %% [markdown]
# ### Weighing data with `Gen.assess`
#
# The mathematical picture is as follows.  Given the parameters of a pose $z$, walls $w$, and settings $\nu$, one gets a distribution $\text{sensor}(z, w, \nu)$ over certain choice maps.  The supporting choice maps are identified with vectors $o = o^{(1:J)} = (o^{(1)}, o^{(2)}, \ldots, o^{(J)})$, where $J := \nu_\text{num\_angles}$, each $o^{(j)}$ independently following a certain normal distribution (depending, notably, on a distance to a wall).  Thus the density of $o$ factors into a product of the form
# $$
# P_\text{sensor}(o) = \prod\nolimits_{j=1}^J P_\text{normal}(o^{(j)})
# $$
# where we begin a habit of omitting the parameters to distributions that are implied by the code.
#
# As `propose` draws a sample, it simultaneously computes this density or *score* and returns its logarithm:

# %%
exp(w)

# %% [markdown]
# There are many scenarios where one has on hand a full set of data, perhaps via observation, and seeks their score according to the model.  One could write a program by hand to do this—but one would simply recapitulate the code for `noisy_sensor`.  The difference is that the sampling operations would be replaced with density computations, and instead of storing them in a choice map it would compute their log product.
#
# The construction of a log density function is automated by the `Gen.assess` semantics for generative functions.  This method is passed the GF, a tuple of arguments, and a choice map, and returns the log weight plus the return value.

# %%
exp(assess(sensor_model, (Pose([1., 1.], pi/2.), world.walls, sensor_settings), cm)[1])

# %% [markdown]
# ## First steps in probabilistic reasoning
#
# Let pick some measured noisy distance data
#
# > ***LOAD THEM FROM FILE***
#
# and try to reason about where the robot could have taken them from.

# %% [markdown]
# POSSIBLE VIZ GOAL: User can start from the loaded data, or move around to grab some noisy sensors.  Then, user can move around a separate candiate-match pose, and `assess` the data against it with updated result somehow.

# %% [markdown]
# What we are exploring here is in Bayesian parlance the *likelihood* of the varying pose.  One gets a sense that certain poses were somehow more likely than others, and the modeling of this intuitive sense is called *inference*.
#
# The above exploration points at a strategy of finding the pose (parameter) that optimizes the likelihood (some statistic), a ubiquitous process called *variational inference*.
#
# A subtle but crucial matter must be dealt with.  This act of variational inference silently adopts assumptions having highly nontrivial consequences for our inferences, having to do with the issue of *prior* over the parameter.
#
# First we must acknowledge at all, that our reasoning always approaches the question "Where is the robot?" already having some idea of where it is possible for the robot to be.  We interpret new information in terms of these assumptions, and they definitely influence the inferences we make.  For example, if we were utterly sure the robot were near the center of the map, we would only really consider how the sensor data around such poses, even if there were better fits elsewhere that we did not expect the robot to be.
#
# These assumptions are modeled by a distribution over poses called the *prior*.  Then, according to Bayes's Law, the key quantity to examine is not the likelihood density $P_\text{sensor}(o;z)$ but rather the *posterior* density
# $$
# P_\text{posterior}(z|o) = P_\text{sensor}(o;z) \cdot P_\text{prior}(z) / Z
# $$
# where $Z > 0$ is a normalizing constant.   Likelihood optimization amounts to assuming a prior having $P_\text{prior}(z) \equiv 1$, a so-called "uniform" prior over the parameter space.
#
# The uniform prior may appear to be a natural expression of "complete ignorance", not preferencing any parameter over another.  The other thing to acknowledge is that this is not the case: the parameterization of the latents itself embodies preferences among parameter values.  Different parametramizations of the latents lead to different "uniform" distributions over them.  For example, parameterizing the spread of a univariate normal distribution by its standard deviation and its variance lead to different "uniform" priors over the parameter space, the square map being nonlinear.  Thus likelihood optimization's second tacit assumption is a particular parametric representation of the latents space, according to which uniformity occurs.
#
# Summarizing, likelihood optimization does not lead to *intrinsic* inference conclusions, because it relies on a prior that in turn is not intrinsic, but rather depends on how the parameters are presented.  Intrinsic conclusions are instead drawn by specifying the prior as a distribution, which has a consistent meaning across parameterizations.
#
# So let us be upfront that we choose the uniform prior relative to the conventional meaning of the pose parameters.  Here is a view onto the posterior distribution over poses, given a set of sensor measurements.

# %% [markdown]
# POSSIBLE VIZ GOAL: Gather the preceding viz into one view: alpha blend all candidate-match poses by likelihood, so only plausible things appear, with the mode highlighted.

# %% [markdown]
# PUT HERE: expanded discussion of single-pose inference problem.
#
# From optimization/VI to sampling techniques.  Reasons:
# * Note *how much information we are throwing away* when passing from the distribution to a single statistic.  Something must be afoot.
# * Later inferences depend on the *whole distribution* of parameters.
#   * Reducing to (Dirac measures on) the modes breaks compositional validity!
# * The modes might not even be *representative* of the posterior:
#   * The mode might not even be where any mass actually accumulates, as in a high-dimensional Gaussian!
#   * Mass might be distributed among multiple near-tied modes, unnaturally preferencing one of them.
# * The posterior requires clearly specifying a prior, which (as mentioned above) prevents ambiguities of parameterization.
#
# Replace `argmax` with a resampling operation (SIR).  Grid vs. free choice.
#
# Compare to a NN approach.

# %% [markdown]
# ## Modeling robot motion
#
# As said initially, we are uncertain about the true initial position and subsequent motion of the robot.  In order to reason about these, we now specify a model using `Gen`.
#
# Each piece of the model is declared as a *generative function* (GF).  The `Gen` library provides two DSLs for constructing GFs: the dynamic DSL using the decorator `@gen` on a function declaration, and the static DSL similarly decorated with `@gen (static)`.  The dynamic DSL allows a rather wide class of program structures, whereas the static DSL only allows those for which a certain static analysis may be performed.
#
# The library offers two basic constructs for use within these DSLs: primitive *distributions* such as "Bernoulli" and "normal", and the sampling operator `~`.  Recursively, GFs may sample from other GFs using `~`.

# %% [markdown]
# ### Robot programs
#
# We also assume given a description of a robot's movement via
# * an estimated initial pose (= position + heading), and
# * a program of controls (= advance distance, followed by rotate heading).

# %%
# A value `c :: Control` corresponds to the robot *first* advancing in its present direction by `c.ds`, *then* rotating by `c.dhd`.
struct Control
    ds  :: Float64
    dhd :: Float64
end

function load_program(file_name)
    data = parsefile(file_name)
    start = Pose(Vector{Float64}(data["start_pose"]["p"]), Float64(data["start_pose"]["hd"]))
    controls = Vector{Control}([Control(control["ds"], control["dhd"]) for control in data["program_controls"]])
    return (; start, controls), length(controls)
end;

# %%
robot_inputs, T = load_program("robot_program.json");

# %% [markdown]
# Before we can visualize such a program, we will need to model robot motion.

# %% [markdown]
# POSSIBLE VIZ GOAL: user can manipulate a pose, and independently a control (vecor-like relative to it), with new pose in shadow.

# %% [markdown]
# ### Integrate a path from a starting pose and controls
#
# If the motion of the robot is determined in an ideal manner by the controls, then we may simply integrate to determine the resulting path.  Naïvely, this results in the following.

# %%
function integrate_controls_unphysical(robot_inputs)
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = robot_inputs.start
    for t in 1:length(robot_inputs.controls)
        p = path[t].p + robot_inputs.controls[t].ds * path[t].dp
        hd = path[t].hd + robot_inputs.controls[t].dhd
        path[t+1] = Pose(p, hd)
    end
    return path
end;

# %% [markdown]
# POSSIBLE VIZ GOAL: user can manipulate a whole path, still ignoring walls.

# %% [markdown]
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance from the point of contact in the normal direction to the wall.

# %%
norm(v) = sqrt(sum(v.^2))

function physical_step(p1, p2, hd, world_inputs)
    p21 = (p2[1] - p1[1], p2[2] - p1[2])
    step_pose = Pose(p1, atan(p21[2], p21[1]))
    s, i = findmin(w -> distance(step_pose, w), world_inputs.walls)
    if s > norm(p21)
        # Step succeeds without contact with walls.
        return Pose(p2, hd)
    else
        contact_point = p1 + s * step_pose.dp
        unit_tangent = world_inputs.walls[i].dp / norm(world_inputs.walls[i].dp)
        unit_normal = [-unit_tangent[2], unit_tangent[1]]
        # Sign of 2D cross product determines orientation of bounce.
        if det2(step_pose.dp, world_inputs.walls[i].dp) < 0.
            unit_normal = -unit_normal
        end
        return Pose(contact_point + world_inputs.bounce * unit_normal, hd)
    end
end

function integrate_controls(robot_inputs, world_inputs)
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = robot_inputs.start
    for t in 1:length(robot_inputs.controls)
        p = path[t].p + robot_inputs.controls[t].ds * path[t].dp
        hd = path[t].hd + robot_inputs.controls[t].dhd
        path[t+1] = physical_step(path[t].p, p, hd, world_inputs)
    end
    return path
end;

# %%
# How bouncy the walls are in this world.
world_inputs = (walls = world.walls, bounce = 0.1)

path_integrated = integrate_controls(robot_inputs, world_inputs);

# %%
plot_world(world, "Given data", show=(:label,))
plot!(robot_inputs.start; color=:green3, label="given start pose")
plot!([pose.p[1] for pose in path_integrated], [pose.p[2] for pose in path_integrated];
      color=:green2, label="path from integrating controls", seriestype=:scatter, markersize=3, markerstrokewidth=0)

# %% [markdown]
# We can also visualize the behavior of the model of physical motion:
#
# ![](imgs_stable/physical_motion.gif)

# %% [markdown]
# POSSIBLE VIZ GOAL: user can manipulate a whole path, now obeying walls.

# %% [markdown]
# ### Components of the motion model
#
# We start with the two building blocks: the starting pose and individual steps of motion.

# %%
@gen (static) function start_pose_prior(start, motion_settings)
    p ~ mvnormal(start.p, motion_settings.p_noise^2 * [1 0 ; 0 1])
    hd ~ normal(start.hd, motion_settings.hd_noise)
    return Pose(p, hd)
end

@gen (static) function step_model(start, c, world_inputs, motion_settings)
    p ~ mvnormal(start.p + c.ds * start.dp, motion_settings.p_noise^2 * [1 0 ; 0 1])
    hd ~ normal(start.hd + c.dhd, motion_settings.hd_noise)
    return physical_step(start.p, p, hd, world_inputs)
end;

# %% [markdown]
# Returning to the code, we can call a GF like a normal function and it will just run stochastically:

# %%
motion_settings = (p_noise = 0.5, hd_noise = 2π / 360)

N_samples = 50
pose_samples = [start_pose_prior(robot_inputs.start, motion_settings) for _ in 1:N_samples]

std_devs_radius = 2.5 * motion_settings.p_noise

plot_world(world, "Start pose prior (samples)")
plot!(make_circle(robot_inputs.start.p, std_devs_radius);
      color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.25)
plot!(pose_samples; color=:red, label="start pose samples")

# %%
N_samples = 50
noiseless_step = robot_inputs.start.p + robot_inputs.controls[1].ds * robot_inputs.start.dp
step_samples = [step_model(robot_inputs.start, robot_inputs.controls[1], world_inputs, motion_settings) for _ in 1:N_samples]

plot_world(world, "Motion step model model (samples)")
plot!(robot_inputs.start; color=:black, label="step from here")
plot!(make_circle(noiseless_step, std_devs_radius);
      color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.25)
plot!(step_samples; color=:red, label="step samples")

# %% [markdown]
# ### Traces: choice maps
#
# We can also perform *traced execution* of a generative function using the construct `Gen.simulate`.  This means that certain information is recorded during execution and packaged into a *trace*, and this trace is returned instead of the bare return value sample.
#
# The foremost information stored in the trace is the *choice map*, which is an associative array from labels to the labeled stochastic choices, i.e. occurrences of the `~` operator, that were encountered.  It is accessed by `Gen.get_choices`:

# %%
# `simulate` takes the GF plus a tuple of args to pass to it.
trace = simulate(start_pose_prior, (robot_inputs.start, motion_settings))
get_choices(trace)

# %% [markdown]
# The choice map being the point of focus of the trace in most discussions, we often abusively just speak of a *trace* when we really mean its *choice map*.

# %% [markdown]
# ### Gen.jl API for traces
#
# One can access the primitive choices in a trace using the bracket syntax `trace[address]`.  One can access from a trace the GF that produced it using `Gen.get_gen_fn`, along with with arguments that were supplied using `Gen.get_args`, and the return value sample of the GF using `Gen.get_retval`.  See below the fold for examples of all these.

# %%
trace[:p]

# %%
trace[:hd]

# %%
get_gen_fn(trace)

# %%
get_args(trace)

# %%
get_retval(trace)

# %% [markdown]
# ### Traces: scores/weights/densities
#
# Traced execution of a generative function also produces a particular kind of score/weight/density.  It is very important to be clear about which score/weight/density value is to be expected, and why.  Consider the following generative function
# ```
# p = 0.25
# @gen function g(x,y)
#   flip ~ bernoulli(p)
#   return flip ? x : y
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
get_score(trace)

# %% [markdown]
# #### Subscores/subweights/subdensities
#
# Instead of (the log of) the product of all the primitive choices made in a trace, one can take the product over just a subset using `Gen.project`.  See below the fold for examples.

# %%
project(trace, select())

# %%
project(trace, select(:p))

# %%
project(trace, select(:hd))

# %%
project(trace, select(:p, :hd)) == get_score(trace)

# %% [markdown]
# ### Modeling a full path
#
# The model contains all information in its trace, rendering its return value redundant.  The noisy path integration will just be a wrapper around its functionality, extracting what it needs from the trace.
#
# (It is worth acknowledging two strange things in the code below: the extra text "`_loop`" in the function name, and the seemingly redundant new parameter `T`.  Both will be addressed shortly, along with the aforementioned wrapper.)

# %%
@gen function path_model_loop(T, robot_inputs, world_inputs, motion_settings)
    pose = {:initial => :pose} ~ start_pose_prior(robot_inputs.start, motion_settings)

    for t in 1:T
        pose = {:steps => t => :pose} ~ step_model(pose, robot_inputs.controls[t], world_inputs, motion_settings)
    end
end

prefix_address(t, rest) = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest)
get_path(trace) = [trace[prefix_address(t, :pose)] for t in 1:(get_args(trace)[1]+1)];

# %% [markdown]
# A trace of `path_model_loop` is more interesting than the ones for `start_pose_prior` and `step_model`.  Let's take a look.  (To reduce notebook clutter, we just show the start pose plus the initial 5 timesteps.)

# %%
trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))
get_selected(get_choices(trace), select((prefix_address(t, :pose) for t in 1:6)...))

# %% [markdown]
# We find that a choicemap is a tree structure rather than a flat associative array.  Addresses are actually root-to-node paths, which are constructed using the `=>` operator.
#
# Moreover, when the source code of a GF applies the `~` operator not to a primitive distribution but to some other generative function, the latter's choice map is included as a subtree rooted at the corresponding node.  That is, the choice map captures the recursive structure of the stochastic locations of the control flow.

# %% [markdown]
# The corresponding mathematical picture is as follows.  We write $x_{a:b} = (x_a, x_{a+1}, \ldots, x_b)$ to gather items $x_t$ into a vector.
#
# In addition to the previous data, we are given an estimated start pose $r_0$ and controls $r_t = (s_t, \eta_t)$ for $t=1,\ldots,T$.  Then `path_model` corresponds to a distribution over traces denoted $\text{path}$; these traces are identified with vectors, namely, $z_{0:T} \sim \text{path}(r_{0:T}, w, \nu)$ is the same as $z_0 \sim \text{start}(r_0, \nu)$ and $z_t \sim \text{step}(z_{t-1}, r_t, w, \nu)$ for $t=1,\ldots,T$.  Here and henceforth we use the shorthand $\text{step}(z, \ldots) := \text{step}(\text{retval}(z), \ldots)$.  The density function is
# $$
# P_\text{path}(z_{0:T}; r_{0:T}, w, \nu)
# = P_\text{start}(z_0; r_0, \nu) \cdot \prod\nolimits_{t=1}^T P_\text{step}(z_t; z_{t-1}, r_t, w, \nu)
# $$
# where each term, in turn, factors into a product of two (multivariate) normal densities as described above.

# %% [markdown]
# As our truncation of the example trace above might suggest, visualization is an essential practice in ProbComp.  We could very well pass the output of the above `integrate_controls_noisy` to the `plot!` function to have a look at it.  However, we want to get started early in this notebook on a good habit: writing interpretive code for GFs in terms of their traces rather than their return values.  This enables the programmer include the parameters of the model in the display for clarity.

# %%
function frames_from_motion_trace(world, title, trace; show=())
    T = get_args(trace)[1]
    robot_inputs = get_args(trace)[2]
    poses = get_path(trace)
    noiseless_steps = [robot_inputs.start.p, [pose.p + c.ds * pose.dp for (pose, c) in zip(poses, robot_inputs.controls)]...]
    motion_settings = get_args(trace)[4]
    std_devs_radius = 2.5 * motion_settings.p_noise
    plots = Vector{Plots.Plot}(undef, T+1)
    for t in 1:(T+1)
        frame_plot = plot_world(world, title; show=show)
        plot!(poses[1:t-1]; color=:black, label="past poses")
        plot!(make_circle(noiseless_steps[t], std_devs_radius);
              color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.25)
        plot!(Pose(trace[prefix_address(t, :pose => :p)], poses[t].hd); color=:red, label="sampled next step")
        plots[t] = frame_plot
    end
    return plots
end;

# %% [markdown]
# Here is what a step through the code looks like:
#
# ![](imgs_stable/path_model_with_trace.gif)

# %%
N_samples = 5

ani = Animation()
for n in 1:N_samples
    trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))
    frames = frames_from_motion_trace(world, "Motion model (samples)", trace)
    for frame_plot in frames; frame(ani, frame_plot) end
end
gif(ani, "imgs/motion.gif", fps=2)

# %% [markdown]
# ### Modfying traces
#
# The metaprogramming approach of Gen affords the opportunity to explore alternate stochastic execution histories.  Namely, `Gen.update` takes as inputs a trace, together with modifications to its arguments and primitive choice values, and returns an accordingly modified trace.  It also returns (the log of) the ratio of the updated trace's density to the original trace's density, together with a precise record of the resulting modifications that played out.

# %% [markdown]
# One could, for instance, consider just the placement of the first pose, and replace its stochastic choice of heading with a specific value.

# %%
trace = simulate(start_pose_prior, (robot_inputs.start, motion_settings))
rotated_trace, rotated_trace_weight_diff, _, _ =
    update(trace, (robot_inputs.start, motion_settings), (NoChange(), NoChange()), choicemap((:hd, π/2.)))
plot_world(world, "Modifying a heading")
plot!(get_retval(trace); color=:green, label="some pose")
plot!(get_retval(rotated_trace); color=:red, label="with heading modified")

# %% [markdown]
# The original trace was typical under the pose prior model, whereas the modified one is rather less likely.  This is the log of how much unlikelier:

# %%
rotated_trace_weight_diff

# %% [markdown]
# It is worth carefully thinking through a tricker instance of this.  Suppose instead, within the full path, we replaced the $t = 1$ step's stochastic choice of heading with some specific value.

# %%
trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))
rotated_first_step, rotated_first_step_weight_diff, _, _ =
    update(trace,
           (T, robot_inputs, world_inputs, motion_settings), (NoChange(), NoChange(), NoChange(), NoChange()),
           choicemap((:steps => 1 => :pose => :hd, π/2.)))
plot_world(world, "Modifying another heading")
plot!(get_path(trace); color=:green, label="some path")
plot!(get_path(rotated_first_step); color=:red, label="with heading at first step modified")

# %% [markdown]
# In the above picture, the green path is apparently missing, having been near-completely overdrawn by the red path.  This is because in the execution of the model, the only change in the stochastic choices took place where we specified.  In particular, the stochastic choice of pose at the second step was left unchanged.  This choice was typical relative to the first step's heading in the old trace, and while it is not impossible relative to the first step's heading in the new trace, it is *far unlikelier* under the mulitvariate normal distribution supporting it:

# %%
rotated_first_step_weight_diff

# %% [markdown]
# Another capability of `Gen.update` is to modify the *arguments* to the generative function used to produce the trace.  In our example, we might have on hand a very long list of controls, and we wish to explore the space of paths incrementally in the timestep:

# %%
change_only_T = (UnknownChange(), NoChange(), NoChange(), NoChange())

trace = simulate(path_model_loop, (0, robot_inputs, world_inputs, motion_settings))
for t in 1:T
    trace, _, _, _ = update(trace, (t, robot_inputs, world_inputs, motion_settings), change_only_T, choicemap())
    # ...
    # Do something with the trace of the partial path up to time t.
    # ...
    @assert has_value(get_choices(trace), :steps => t => :pose => :p)
    @assert !has_value(get_choices(trace), :steps => (t+1) => :pose => :p)
end

println("Success");

# %% [markdown]
# Because performing such updates to traces occur frequently, and they seemingly require re-running the entire model, computational complexity considerations become important.  We detour next through an important speedup.

# %% [markdown]
# ### Improving performance using the static DSL and combinators
#
# Because the dynamic DSL does not understand the loop inside `path_model_loop`, calling `Gen.update` with the new value of `T` requires re-execution of the whole loop.  This means that the update requires $O(T)$ time, and the above code requires $O(T^2)$ time.
#
# But we humans understand that incrementing the argument `T` simply requires running the loop body once more.  This operation runs in $O(1)$ time, so the outer loop should require only $O(T)$ time.  Gen can intelligently work this way if we encode the structure of Markov chain in this model using a *combinator* for the static DSL, as follows.

# %%
@gen (static) function motion_path_kernel(t, state, robot_inputs, world_inputs, motion_settings)
    return {:pose} ~ step_model(state, robot_inputs.controls[t], world_inputs, motion_settings)
end
motion_path_chain = Unfold(motion_path_kernel)

@gen (static) function path_model(T, robot_inputs, world_inputs, motion_settings)
    initial = {:initial => :pose} ~ start_pose_prior(robot_inputs.start, motion_settings)
    {:steps} ~ motion_path_chain(T, initial, robot_inputs, world_inputs, motion_settings)
end;

# %% [markdown]
# The models `path_model_loop` and `path_model` have been arranged to produce identically structured traces with the same frequencies and return values, and to correspond to identical distributions over traces in the mathematical picture, thereby yielding the same weights.  They give rise to identical computations under `Gen.simulate`, whereas the new model is sometimes more efficient under `Gen.update`.  Here we illustrate the efficiency gain.
#
# (The noise in the graph is an artifact of Julia's garbage collection.)
#
# ![](imgs_stable/dynamic_static_comparison.png)

# %% [markdown]
# Owing to the efficiency comparison, we eschew `path_model_loop` in favor of `path_model` in what follows.  Thus we finally write our noisy path integration wrapper.

# %%
function integrate_controls_noisy(robot_inputs, world_inputs, motion_settings)
    return get_path(simulate(path_model, (length(robot_inputs.controls), robot_inputs, world_inputs, motion_settings)))
end;

# %% [markdown]
# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", whose traces describe simulations of the entire robot situation as we have described it.

# %%
@gen (static) function full_model_initial(robot_inputs, walls, full_settings)
    pose ~ start_pose_prior(robot_inputs.start, full_settings.motion_settings)
    {:sensor} ~ sensor_model(pose, walls, full_settings.sensor_settings)
    return pose
end

@gen (static) function full_model_kernel(t, state, robot_inputs, world_inputs, full_settings)
    pose ~ step_model(state, robot_inputs.controls[t], world_inputs, full_settings.motion_settings)
    {:sensor} ~ sensor_model(pose, world_inputs.walls, full_settings.sensor_settings)
    return pose
end
full_model_chain = Unfold(full_model_kernel)

@gen (static) function full_model(T, robot_inputs, world_inputs, full_settings)
    initial ~ full_model_initial(robot_inputs, world_inputs.walls, full_settings)
    steps ~ full_model_chain(T, initial, robot_inputs, world_inputs, full_settings)
end

get_sensors(trace) =
    [[trace[prefix_address(t, :sensor => j => :distance)]
      for j in 1:get_args(trace)[4].sensor_settings.num_angles]
     for t in 1:(get_args(trace)[1]+1)];

# %% [markdown]
# Again, the trace of the full model contains many choices, so we just show a subset of them: the initial pose plus 2 timesteps, and 5 sensor readings from each.

# %%
full_settings = (motion_settings=motion_settings, sensor_settings=sensor_settings)
full_model_args = (robot_inputs, world_inputs, full_settings)

trace = simulate(full_model, (T, full_model_args...))
selection = select((prefix_address(t, :pose) for t in 1:3)..., (prefix_address(t, :sensor => j) for t in 1:3, j in 1:5)...)
get_selected(get_choices(trace), selection)

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
function frame_from_sensors_trace(world, title, poses, poses_color, poses_label, pose, trace; show=())
    readings = [trace[j => :distance] for j in 1:sensor_settings.num_angles]
    return frame_from_sensors(world, title, poses, poses_color, poses_label, pose,
                             readings, "trace sensors", get_args(trace)[3]; show=show)
end

function frames_from_full_trace(world, title, trace; show=())
    T = get_args(trace)[1]
    robot_inputs = get_args(trace)[2]
    poses = get_path(trace)
    noiseless_steps = [robot_inputs.start.p, [pose.p + c.ds * pose.dp for (pose, c) in zip(poses, robot_inputs.controls)]...]
    settings = get_args(trace)[4]
    std_devs_radius = 2.5 * settings.motion_settings.p_noise
    sensor_readings = get_sensors(trace)
    plots = Vector{Plots.Plot}(undef, 2*(T+1))
    for t in 1:(T+1)
        frame_plot = plot_world(world, title; show=show)
        plot!(poses[1:t-1]; color=:black, label="past poses")
        plot!(make_circle(noiseless_steps[t], std_devs_radius);
              color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.25)
        plot!(Pose(trace[prefix_address(t, :pose => :p)], poses[t].hd); color=:red, label="sampled next step")
        plots[2*t-1] = frame_plot
        plots[2*t] = frame_from_sensors(
            world, title,
            poses[1:t], :black, nothing,
            poses[t], sensor_readings[t], "sampled sensors",
            settings.sensor_settings; show=show)
    end
    return plots
end;

# %% [markdown]
# Here is a stepping through the code:
#
# ![](imgs_stable/full_model_with_trace.gif)

# %%
N_samples = 5

ani = Animation()
for n in 1:N_samples
    trace = simulate(full_model, (T, full_model_args...))
    frames = frames_from_full_trace(world, "Full model (samples)", trace)
    for frame_plot in frames; frame(ani, frame_plot) end
end
gif(ani, "imgs/full_1.gif", fps=2)

# %% [markdown]
# ## The data
#
# Let us generate some fixed synthetic motion data that, for pedagogical purposes, we will work with as if it were the actual path of the robot.  We will generate two versions, one each with low or high motion deviation.

# %%
motion_settings_low_deviation = (p_noise = 0.05, hd_noise = (1/10.) * 2π / 360)
trace_low_deviation = simulate(full_model, (T, robot_inputs, world_inputs, (full_settings..., motion_settings=motion_settings_low_deviation)))

motion_settings_high_deviation = (p_noise = 0.25, hd_noise = 2π / 360)
trace_high_deviation = simulate(full_model, (T, robot_inputs, world_inputs, (full_settings..., motion_settings=motion_settings_high_deviation)))

frames_low = frames_from_full_trace(world, "Low motion deviation", trace_low_deviation)
frames_high = frames_from_full_trace(world, "High motion deviation", trace_high_deviation)
ani = Animation()
for (low, high) in zip(frames_low, frames_high)
    frame_plot = plot(low, high; size=(1000,500), plot_title="Two synthetic data sets")
    frame(ani, frame_plot)
end
gif(ani, "imgs/the_data.gif", fps=2)

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
constraint_from_sensors(t, readings) =
    choicemap(( (prefix_address(t, :sensor => j => :distance), reading) for (j, reading) in enumerate(readings) )...)

constraints_low_deviation = [constraint_from_sensors(o...) for o in enumerate(observations_low_deviation)]
constraints_high_deviation = [constraint_from_sensors(o...) for o in enumerate(observations_high_deviation)]
merged_constraints_low_deviation = merge(constraints_low_deviation...)
merged_constraints_high_deviation = merge(constraints_high_deviation...);

# %% [markdown]
# We summarize the information available to the robot to determine its location. On the one hand, one has to produce a guess of the start pose plus some controls, which one might integrate to produce an idealized guess of path. On the other hand, one has the sensor data.

# %%
function plot_bare_sensors(world, title, readings, label, sensor_settings)
    border = world.box_size * (3.)/19.
    the_plot = plot(
        size         = (500, 500),
        aspect_ratio = :equal,
        grid         = false,
        xlim         = (world.bounding_box[1]-border, world.bounding_box[2]+border),
        ylim         = (world.bounding_box[3]-border, world.bounding_box[4]+border),
        title        = title,
        legend       = :bottomleft)
    plot_sensors!(Pose(world.center_point, 0.), :black, readings, label, sensor_settings)
    return the_plot
end;

# %%
short_control(c) = "ds = $(round(c.ds, digits=2)), dhd = $(round(c.dhd, digits=2))"

ani = Animation()
for (t, (pose, readings_low, readings_high)) in enumerate(zip(path_integrated, observations_low_deviation, observations_high_deviation))
    # plot_integrated = plot_world(world, "Startup data")
    # plot!(path_integrated[1]; color=:green, label="start guess")
    # if t > 1; annotate!(5, 2.5, "Control $(t-1):\n$(short_control(robot_inputs.controls[t-1]))") end

    plot_low = plot_bare_sensors(world, "Low motion deviation", readings_low, "synthetic sensor data", sensor_settings)
    plot!(Pose(world.center_point, 0.0); color=:black, label=nothing)

    plot_high = plot_bare_sensors(world, "High motion deviation", readings_high, "synthetic sensor data", sensor_settings)
    plot!(Pose(world.center_point, 0.0); color=:black, label=nothing)

    the_frame = plot(plot_low, plot_high; size=(1000,500), layout=grid(1,2), plot_title="Observations available to robot")
    frame(ani, the_frame)
end
gif(ani, "imgs/robot_can_see.gif", fps=2)

# %% [markdown]
# ## Inference

# %% [markdown]
# ### Why we need inference: in a picture
#
# The path obtained by integrating the controls serves as a proposal for the true path, but it is unsatisfactory, especially in the high motion deviation case.  The picture gives an intuitive sense of the fit:

# %%
ani = Animation()
for (pose, readings_low, readings_high) in zip(path_integrated, observations_low_deviation, observations_high_deviation)
    low_plot = frame_from_sensors(
        world, "Low motion deviation",
        path_integrated, :green2, "path from integrating controls",
        pose, readings_low, "fixed sensor data",
        sensor_settings)
    high_plot = frame_from_sensors(
        world, "High motion deviation",
        path_integrated, :green2, "path from integrating controls",
        pose, readings_high, "fixed sensor data",
        sensor_settings)
    frame_plot = plot(low_plot, high_plot; size=(1000,500), plot_title="Integrated path as explanation of sensor data")
    frame(ani, frame_plot)
end
gif(ani, "imgs/need.gif", fps=1)

# %% [markdown]
# It would seem that the fit is reasonable in low motion deviation, but really breaks down in high motion deviation.
#
# We are not limited to visual judgments here: the model can quantitatively assess how good a fit the integrated path is for the data.  In order to do this, we detour to explain how to produce samples from our model that agree with the fixed observation data.

# %% [markdown]
# ### Producing samples with constraints
#
# We have seen how `Gen.simulate` performs traced execution of a generative function: as the program runs, it draws stochastic choices from all required primitive distributions, and records them in a choice map.
#
# Given a choice map of *constraints* that declare fixed values of some of the primitive choices, the operation `Gen.generate` proposes traces of the generative function that are consistent with these constraints.

# %%
trace, log_weight = generate(full_model, (T, full_model_args...), merged_constraints_low_deviation)

all(trace[prefix_address(i, :sensor => j => :distance)] == merged_constraints_low_deviation[prefix_address(i, :sensor => j => :distance)]
    for i in 1:(T+1) for j in 1:sensor_settings.num_angles)

# %% [markdown]
# A trace resulting from a call to `Gen.generate` is structurally indistinguishable from one drawn from `Gen.simulate`.  But there is a key situational difference: while `Gen.get_score` always returns the frequency with which `Gen.simulate` stochastically produces the trace, this value is **no longer equal to** the frequency with which the trace is stochastically produced by `Gen.generate`.  This is both true in an obvious and less relevant sense, as well as true in a more subtle and extremely germane sense.
#
# On the superficial level, since all traces produced by `Gen.generate` are consistent with the constraints, those traces that are inconsistent with the constraints do not occur at all, and in aggregate the traces that are consistent with the constraints are more common.
#
# More deeply and importantly, the stochastic choice of the *constraints* under a run of `Gen.simulate` might have any density, perhaps very low.  This constraints density contributes as always to the `Gen.get_score`, whereas it does not influence the frequency of producing this trace under `Gen.generate`.
#
# The ratio of the `Gen.get_score` of a trace to the probability density that `Gen.generate` would produce it with the given constraints, is called the *importance weight*.  For convenience, (the log of) this quantity is returned by `Gen.generate` along with the trace.
#
# We stress the basic invariant:
# $$
# \text{get\_score}(\text{trace})
# =
# (\text{importance weight from Gen.generate})
# \cdot
# (\text{frequency Gen.generate creates this trace}).
# $$

# %% [markdown]
# The preceding comments apply to generative functions in wide generality.  We can say even more about our present examples, because further assumptions hold.
# 1. There is no untraced randomness.  Given a full choice map for constraints, everything else is deterministic.  In particular, the importance weight is the `get_score`.
# 2. The generative function was constructed using Gen's DSLs and primitive distributions.  Ancestral sampling; `Gen.generate` with empty constraints reduces to `Gen.simulate` with importance weight $1$.
# 3. Combined, the importance weight is directly computed as the `Gen.project` of the trace upon the choice map addresses that were constrained in the call to `Gen.generate`.
#
#   In our running example, the projection in question is $\prod_{t=0}^T P_\text{sensor}(o_t)$.

# %%
log_weight - project(trace, select([prefix_address(i, :sensor) for i in 1:(T+1)]...))

# %% [markdown]
# ### Why we need inference: in numbers
#
# We return to how the model offers a numerical benchmark for how good a fit the integrated path is.
#
# In words, the data are incongruously unlikely for the integrated path.  The (log) density of the measurement data, given the integrated path...

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

# %% [markdown]
# ...more closely resembles the density of these data back-fitted onto any other typical (random) paths of the model...

# %%
N_samples = 200

traces_generated_low_deviation = [generate(full_model, (T, full_model_args...), merged_constraints_low_deviation)[1] for _ in 1:N_samples]
log_likelihoods_low_deviation = [project(trace, selection) for trace in traces_generated_low_deviation]
hist_low_deviation = histogram(log_likelihoods_low_deviation; label=nothing, bins=20, title="low dev data, typical paths")

traces_generated_high_deviation = [generate(full_model, (T, full_model_args...), merged_constraints_high_deviation)[1] for _ in 1:N_samples]
log_likelihoods_high_deviation = [project(trace, selection) for trace in traces_generated_high_deviation]
hist_high_deviation = histogram(log_likelihoods_high_deviation; label=nothing, bins=20, title="high dev data, typical paths")

plot(hist_low_deviation, hist_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="Log density of observations under the model")

# %% [markdown]
# ...than the log densities of data typically produced by the complete model run in its natural manner (*compare the scale at the bottom*):

# %%
traces_typical = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
log_likelihoods_typical = [project(trace, selection) for trace in traces_typical]
histogram(log_likelihoods_typical; label=nothing, bins=20, title="Log density of observations under the model\ntypical traces")

# %% [markdown]
# ### Inference: demonstration
#
# In the viewpoint of ProbComp, the goal of *inference* is to produce *likely* traces of a full model, given the observed data.  In the language of probability theory, as generative functions induce distributions on traces, and if we view the full model as a program embodying a *prior*, then applying an inference metaprogram to it (together with the observed data) produces a new program that embodies the *posterior*.

# %% [markdown]
# Let's show what we mean with a picture, keeping the code black-boxed until we explain it later.

# %%
# Load function `black_box_inference(constraints)`.

include("black_box.jl")

# %%
# Visualize distributions over traces.

function frame_from_traces(world, title, path, path_label, traces, trace_label; show=())
    the_plot = plot_world(world, title; show=show)
    if !isnothing(path); plot!(path; label=path_label, color=:brown) end
    for trace in traces
        poses = get_path(trace)
        plot!([p.p[1] for p in poses], [p.p[2] for p in poses]; label=nothing, color=:green, alpha=0.3)
        plot!([Segment(p1.p, p2.p) for (p1, p2) in zip(poses[1:end-1], poses[2:end])];
              label=trace_label, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=0.3)
        trace_label = nothing
    end
    return the_plot
end;

# %%
N_samples = 10

traces = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
prior_plot = frame_from_traces(world, "Prior on robot paths", nothing, nothing, traces, "prior samples")

t1 = now()
traces = [BlackBox.black_box_inference(full_model, full_model_args, T, constraints_low_deviation) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "posterior samples")

t1 = now()
traces = [BlackBox.black_box_inference(full_model, full_model_args, T, constraints_high_deviation) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "posterior samples")

plot(prior_plot, posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="Prior vs. approximate posteriors")

# %% [markdown]
# All of the traces thus produced have observations constrained to the data.  The log densities of the observations under their typical samples show some improvement:

# %%
N_samples = 100

traces_posterior_low_deviation = [BlackBox.black_box_inference(full_model, full_model_args, T, constraints_low_deviation) for _ in 1:N_samples]
log_likelihoods_low_deviation = [project(trace, selection) for trace in traces_posterior_low_deviation]
hist_low_deviation = histogram(log_likelihoods_low_deviation; label=nothing, bins=20, title="typical data under posterior: low dev data")

traces_posterior_high_deviation = [BlackBox.black_box_inference(full_model, full_model_args, T, constraints_high_deviation) for _ in 1:N_samples]
log_likelihoods_high_deviation = [project(trace, selection) for trace in traces_posterior_high_deviation]
hist_high_deviation = histogram(log_likelihoods_high_deviation; label=nothing, bins=20, title="typical data under posterior: high dev data")

plot(hist_low_deviation, hist_high_deviation; size=(1500,500), layout=grid(1,2), plot_title="Log likelihood of observations")

# %% [markdown]
# ## Generic strategies for inference
#
# We now spell out some generic strategies for conditioning the ouputs of a model towards some observed data.  The word "generic" indicates that they make no special intelligent use of the model structure, and their convergence is guaranteed by theorems of a similar nature.  In terms to be defined shortly, they simply take a pair $(Q,f)$ of a proposal and a weight function that implement importance sampling with target $P$.
#
# There is no free lunch in this game: generic inference recipies are inefficient, for example, converging very slowly or needing vast counts of particles, especially in high-dimensional settings.  One of the root problems is that proposals $Q$ may provide arbitrarily bad samples relative to our target $P$; if $Q$ still supports all samples of $P$ with microscopic but nonzero density, then the generic algorithm will converge in the limit, however astronomically slowly.
#
# Rather, efficiency will become possible when we do the *opposite* of generic: exploit what we actually know about the problem in our design of the inference strategy to propose better traces towards our target.  Gen's aim is to provide the right entry points to enact this exploitation.

# %% [markdown]
# ### The posterior distribution, `Gen.generate`, and importance sampling
#
# Mathematically, the passage from the prior to the posterior is the operation of conditioning distributions.
#
# Intuitively, the conditional distribution $\text{full}(\cdot | o_{0:T})$ is just the restriction of the joint distribution $\text{full}(z_{0:T}, o_{0:T})$ to where the parameter $o_{0:T}$ is constant, letting $z_{0:T}$ continue to vary.  This restriction no longer has total density equal to $1$, so we must renormalize it.  The normalizing constant must be
# $$
# P_\text{marginal}(o_{0:T})
# := \int P_\text{full}(Z_{0:T}, o_{0:T}) \, dZ_{0:T}
#  = \mathbf{E}_{Z_{0:T} \sim \text{path}}\big[P_\text{full}(Z_{0:T}, o_{0:T})\big].
# $$
# By Fubini's Theorem, this function of $o_{0:T}$ is the density of a probability distribution over observations $o_{0:T}$, called the *marginal distribution*; but we will often have $o_{0:T}$ fixed, and consider it a constant.  Then, finally, the *conditional distribution* $\text{full}(\cdot | o_{0:T})$ is defined to have the normalized density
# $$
# P_\text{full}(z_{0:T} | o_{0:T}) := \frac{P_\text{full}(z_{0:T}, o_{0:T})}{P_\text{marginal}(o_{0:T})}.
# $$
#
# The goal of inference is to produce samples $\text{trace}_{0:T}$ distributed (approximately) according to $\text{full}(\cdot | o_{0:T})$.  The most immediately evident problem with doing inference is that the quantity $P_\text{marginal}(o_{0:T})$ is intractable!

# %% [markdown]
# Define the function $\hat f(z_{0:T})$ of sample values $z_{0:T}$ to be the ratio of probability densities between the posterior distribution $\text{full}(\cdot | o_{0:T})$ that we wish to sample from, and the prior distribution $\text{path}$ that we are presently able to produce samples from.  Manipulating it à la Bayes's Rule gives:
# $$
# \hat f(z_{0:T})
# :=
# \frac{P_\text{full}(z_{0:T} | o_{0:T})}{P_\text{path}(z_{0:T})}
# =
# \frac{P_\text{full}(z_{0:T}, o_{0:T})}{P_\text{marginal}(o_{0:T}) \cdot P_\text{path}(z_{0:T})}
# =
# \frac{\prod_{t=0}^T P_\text{sensor}(o_t)}{P_\text{marginal}(o_{0:T})}.
# $$
# Noting that the intractable quantity
# $$
# Z := P_\text{marginal}(o_{0:T})
# $$
# is constant in $z_{0:T}$, we define the explicitly computable quantity
# $$
# f(z_{0:T}) := Z \cdot \hat f(z_{0:T}) = \prod\nolimits_{t=0}^T P_\text{sensor}(o_t).
# $$
# The right hand side has been written sloppily, but we remind the reader that $P_\text{sensor}(o_t)$ is a product of densities of normal distributions that *does depend* on $z_t$ as well as "sensor" and "world" parameters.
#
# Compare to our previous description of calling `Gen.generate` on `full_model` with the observations $o_{0:T}$ as constraints: it produces a trace of the form $(z_{0:T}, o_{0:T})$ where $z_{0:T} \sim \text{path}$ has been drawn from $\text{path}$, together with the weight equal to none other than this $f(z_{0:T})$.

# %% [markdown]
# This reasoning involving `Gen.generate` is indicative of the general scenario with conditioning, and fits into the following shape.
#
# We have on hand two distributions, a *target* $P$ from which we would like to (approximately) generate samples, and a *proposal* $Q$ from which we are presently able to generate samples.  We must assume that the proposal is a suitable substitute for the target, in the sense that every possible event under $P$ occurs under $Q$ (mathematically, $P$ is absolutely continuous with respect to $Q$).
#
# Under these hypotheses, there is a well-defined density ratio function $\hat f$ between $P$ and $Q$ (mathematically, the Radon–Nikodym derivative).  If $z$ is a sample drawn from $Q$, then $\hat w = \hat f(z)$ is how much more or less likely $z$ would have been drawn from $P$.  We only require that we are able to compute the *unnormalized* density ratio, that is, some function of the form $f = Z \cdot \hat f$ where $Z > 0$ is constant.
#
# The pair $(Q,f)$ is said to implement *importance sampling* for $P$, and the values of $f$ are called *importance weights*.  Generic inference attempts to use knowledge of $f$ to correct for the difference in behavior between $P$ and $Q$, and thereby use $Q$ to produce samples from (approximately) $P$.
#
# So in our running example, the target $P$ is the posterior distribution on paths $\text{full}(\cdot | o_{0:T})$, the proposal $Q$ is the path prior $\text{path}$, and the importance weight $f$ is the product of the sensor model densities.  We seek a computational model of the first; the second and third are computationally modeled by calling `Gen.generate` on `full_model` constrained by the observations $o_{0:T}$.  (The computation of the second, on its own, simplifies to `path_prior`.)
#
# We represent this discussion in code for future use:

# %%
function importance_sample(model, args, merged_constraints, N_samples)
    traces = Vector{Trace}(undef, N_samples)
    log_weights = Vector{Float64}(undef, N_samples)

    for i in 1:N_samples
        traces[i], log_weights[i] = generate(model, args, merged_constraints)
    end

    return traces, log_weights
end;

# %% [markdown]
# ### Rejection sampling
#
# One approach to inference, called *rejection sampling*, is to interfere with the probabilities of samples by throttling them in the correct proportions.  In other words, we go ahead and generate samples from $Q$, but accept (return) only some of them while rejecting (discarding) others, the probability of acceptance depending on the sampled value.  If for each sampled value $z$ the probability of acceptance is a constant times $f(z)$ (or equivalently $\hat f(z)$), then the samples that make it through will be distributed according to $P$.
#
# More precisely, in deciding whether to accept or reject a sample $z$, we flip a weight-$p(z)$ coin where $p(z) = f(z)/C$ for some constant $C > 0$, accepting if heads and rejecting of tails.  In order for this to make sense, $p(z)$ must always lie in the interval $[0,1]$.  This is equivalent to having $f(z) \leq C$, that is, we need $C$ to be an *upper bound* on the outputs of the function $f$.  There is an optimal upper bound constant, namely the supremum value $C_\text{opt} = \max_z f(z)$.  If we only know *some* upper bound $C \geq C_\text{opt}$, then rejection sampling with this constant still provides a correct algorithm, but it is inefficient by drawing a factor of $C/C_\text{opt}$ too many samples on average.
#
# The first of many problems with rejection sampling is the tractability of determining an upper bound $C$!  (This is independent of the intractability of determining the constant $Z$ above relating $f$ back to $\hat f$.)
#
# We can try to make do anyway, using a number $C > 0$ that is *guess* at an upper bound.  If we encounter a sample $z$ with $f(z) > C$, we replace $C$ with this new larger quantity and keep going.  Earlier samples, with a too-low intitial value for $C$, may occur with too high absolute frequency.  But over time as $C$ appropriately increases, the behavior tends towards the true distribution.  We may consider some of this early phase to be an *exploration* or *burn-in period*, and accordingly draw samples but keep only the maximum of their weights, before moving on to the rejection sampling *per se*.

# %%
function rejection_sample(model, args, merged_constraints, N_burn_in, N_particles, MAX_attempts)
    C = maximum(generate(model, args, merged_constraints)[2] for _ in 1:N_burn_in; init=-Inf)

    particles = []
    for _ in 1:N_particles
        attempts = 0
        while attempts < MAX_attempts
            attempts += 1

            # The use of `generate` is as explained in the preceding section.
            particle, log_weight = generate(model, args, merged_constraints)
            if log_weight > C + log(rand())
                if log_weight > C; C = log_weight end
                push!(particles, particle)
                break
            end
        end
    end

    return particles
end;

# %% [markdown]
# In the following examples, compare the requested number of particles (`N_particles`) to the number of particles found within the compute budget (`MAX_attempts`), as reported in the graph.

# %%
T_short = 6

N_burn_in = 0 # omit burn-in to illustrate early behavior
N_particles = 20
MAX_attempts = 5000

t1 = now()
traces = rejection_sample(full_model, (T_short, full_model_args...), merged_constraints_low_deviation, N_burn_in, N_particles, MAX_attempts)
t2 = now()
println("Time elapsed per run (short path): $(value(t2 - t1) / N_particles) ms.")

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i of $(length(traces)))", path_low_deviation[1:(T_short+1)],
                                   "path to fit", traces[1:i], "RS samples")
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %%
N_burn_in = 100
N_particles = 20
MAX_attempts = 5000

t1 = now()
traces = rejection_sample(full_model, (T_short, full_model_args...), merged_constraints_low_deviation, N_burn_in, N_particles, MAX_attempts)
t2 = now()
println("Time elapsed per run (short path): $(value(t2 - t1) / N_particles) ms.")

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i of $(length(traces)))", path_low_deviation[1:(T_short+1)],
                                   "path to fit", traces[1:i], "RS samples")
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS_2.gif", fps=1)

# %%
N_burn_in = 1000
N_particles = 20
MAX_attempts = 5000

t1 = now()
traces = rejection_sample(full_model, (T_short, full_model_args...), merged_constraints_low_deviation, N_burn_in, N_particles, MAX_attempts)
t2 = now()
println("Time elapsed per run (short path): $(value(t2 - t1) / N_particles) ms.")

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i of $(length(traces)))", path_low_deviation[1:(T_short+1)],
                                   "path to fit", traces[1:i], "RS samples")
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS_3.gif", fps=1)

# %% [markdown]
# The performance dynamics of this algorithm is an instructive exercise to game out.
#
# In general, as $C$ increases, the algorithm is increasingly *wasteful*, rejecting more samples overall, and taking longer to find likely hits.
#
# So long as it indeed bounds above all values of $f(z)$ that we encounter, the algorithm isn't nonsense, but if the proposal $Q$ is unlikely to generate representative samples for the target $P$ at all, all we are doing is adjusting the shape of the noise.

# %% [markdown]
# ### Sampling / importance resampling
#
# We turn to inference strategies that require only our proposal $Q$ and unnormalized weight function $f$ for the target $P$, *without* forcing us to wrangle any intractable integrals or upper bounds.
#
# Suppose we are given a list of nonnegative numbers, not all zero: $w^1, w^2, \ldots, w^N$.  To *normalize* the numbers means computing $\hat w^i := w^i / \sum_{j=1}^N w^j$.  The normalized list $\hat w^1, \hat w^2, \ldots, \hat w^N$ determines a *categorical distribution* on the indices $1, \ldots, N$, wherein the index $i$ occurs with probability $\hat w^i$.
# Note that for any constant $Z > 0$, the scaled list $Zw^1, Zw^2, \ldots, Zw^N$ leads to the same normalized $\hat w^i$ as well as the same categorical distribution.
#
# When some list of data $z^1, z^2, \ldots, z^N$ have been associated with these respective numbers $w^1, w^2, \ldots, w^N$, then to *importance **re**sample* $M$ values from these data according to these weights means to independently sample indices $a^1, a^2, \ldots, a^M \sim \text{categorical}([\hat w^1, \hat w^2, \ldots, \hat w^N])$ and return the new list of data $z^{a^1}, z^{a^2}, \ldots, z^{a^M}$.  Compare to the function `resample` implemented in the code box below.
#
# The *sampling / importance resampling* (SIR) strategy for inference runs as follows.  Let counts $N > 0$ and $M > 0$ be given.
# 1. Importance sample:  Independently sample $N$ data $z^1, z^2, \ldots, z^N$ from the proposal $Q$, called *particles*.  Compute also their *importance weights* $w^i := f(z^i)$ for $i = 1, \ldots, N$.
# 2. Importance resample:  Independently sample $M$ indices $a^1, a^2, \ldots, a^M \sim \text{categorical}([\hat w^1, \hat w^2, \ldots, \hat w^N])$, where $\hat w^i = w^i / \sum_{j=1}^N w^j$, and return $z^{a^1}, z^{a^2}, \ldots, z^{a^M}$. These sampled particles all inherit the *average weight* $\sum_{j=1}^N w^j / N$.
#
# As $N \to \infty$ with $M$ fixed, the samples produced by this algorithm converge to $M$ independent samples drawn from the target $P$.  This strategy is computationally an improvement over rejection sampling: intead of indefinitely constructing and rejecting samples, we can guarantee to use at least some of them after a fixed time, and we are using the best guesses among these.

# %%
function resample(particles, log_weights; M=nothing)
    @assert length(particles) == length(log_weights)
    if isnothing(M); M = length(particles) end
    log_total_weight = logsumexp(log_weights)
    norm_weights = exp.(log_weights .- log_total_weight)
    return [particles[categorical(norm_weights)] for _ in 1:M],
           [log_total_weight - log(M)            for _ in 1:M]
end

sample(particles, log_weights) = resample(particles, log_weights; M=1)[1][1]

sampling_importance_resampling(model, args, merged_constraints, N_SIR) =
    sample(importance_sample(model, args, merged_constraints, N_SIR)...)

# These are generic algorithms, so there are the following library versions.
# By the way, the library version of SIR includes a constant-memory optimization:
# It is not necessary to store all particles and categorically select one at the end.  Mathematically
# it amounts to the same instead to store just one candidate selection, and stochastically replace it
# with each newly generated particle with odds the latter's weight relative to the sum of the
# preceding weights.
importance_sample_library(model, args, merged_constraints, N_samples) =
    Gen.importance_sampling(model, args, merged_constraints, N_samples)[[1, 2]]
sampling_importance_resampling_library(model, args, merged_constraints, N_SIR) =
    Gen.importance_resampling(model, args, merged_constraints, N_SIR)[1];

# %% [markdown]
# For a short path, SIR can improve from chaos to a somewhat coarse/noisy fit without too much effort.

# %%
T_short = 6
N_SIR = 500

N_samples = 10

t1 = now()
traces = [sampling_importance_resampling(full_model, (T_short, full_model_args...), merged_constraints_low_deviation, N_SIR) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (short path): $(value(t2 - t1) / N_samples) ms.")

frame_from_traces(world, "SIR (short path)", path_low_deviation[1:(T_short+1)], "path to fit", traces, "SIR samples")

# %% [markdown]
# There are still problems with SIR.  SIR already do not provide high-quality traces on short paths.  For longer paths, the difficulty only grows, as one blindly searches for a needle in a high-dimensional haystack.  And if the proposal $Q$ is unlikely to generate typical samples from the target $P$, one would need a massive number of particles to get a good approximation; in fact, the rate of convergence of SIR towards the target can be super-exponentially slow in $N \to \infty$!

# %%
N_samples = 10
N_SIR = 500

N_samples = 10

t1 = now()
traces = [sampling_importance_resampling(full_model, (T, full_model_args...), merged_constraints_low_deviation, N_SIR) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")

frame_from_traces(world, "SIR (low dev)", path_low_deviation, "path to fit", traces, "SIR samples")

# %% [markdown]
# ## Sequential Monte Carlo (SMC) techniques
#
# We now begin to exploit the structure of the problem in significant ways to construct good candidate traces for the posterior.  Especially, we use the Markov chain structure to construct these traces step-by-step.  While generic algorithms like SIR and rejection sampling must first construct full paths $\text{trace}_{0:T}$ and then sift among them using the observations $o_{0:T}$, we may instead generate one $\text{trace}_t$ at a time, taking into account the datum $o_t$. Since one is working with only a few dimensions at any one time step, more intelligent searches become computationally feasible.

# %% [markdown]
# ### Particle filter
#
# Above, the function `importance_sample` produced a family of particles, each particle being `generate`d all in one go, together with the density of the observations relative to that path.
#
# The following function `particle_filter` constructs an indistinguishable stochastic family of weighted particles, each trace built by `update`ing one timestep of path at a time, incorporating also the density of that timestep's observations.  (This comes at a small computational overhead: the static DSL combinator largely eliminates recomputation in performing the `update`s, but there is still extra logic, as well as the repeated allocations of the intermediary traces.)
#
# Then, `particle_filter` applies `sample`, making its behavior overall the same as that of `sampling_importance_resampling` above.

# %%
# For this algorithm and each of its variants to come,
# we will present first a straight version of the code,
# followed by a logged version for display purposes.

function particle_filter(model, T, args, constraints, N_particles)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        end
    end

    return sample(traces, log_weights)
end

function particle_filter_infos(model, T, args, constraints, N_particles)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    infos = []

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
            push!(infos, (type = :initialize, time = now(), t = t, label = "sample from start pose prior", traces = copy(traces), log_weights = copy(log_weights)))
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :update, time = now(), t = t, label = "update to next step", traces = copy(traces), log_weights = copy(log_weights)))
        end
    end

    traces, log_weights = resample(traces, log_weights; M=1)
    push!(infos, (type = :final_sample, time = now(), t = T, label = "final sample", traces = copy(traces), log_weights = copy(log_weights)))

    return infos
end;

# %% [markdown]
# This refactoring is called a *particle filter* because of how it spreads the reasoning out along the time axis.  It has the important effect of allowing the inference programmer to intervene, possibly modifying the particles at each time step.
#
# Let's begin by picturing the step-by-step nature of SMC:

# %%
function frame_from_weighted_traces(world, title, path, path_label, traces, log_weights, trace_label; show=(), min_alpha=0.03)
    the_plot = plot_world(world, title; show=show)

    if !isnothing(path)
        plot!(path; label=path_label, color=:brown)
        plot!(path[get_args(traces[1])[1]+1]; label=nothing, color=:black)
    end

    norm_weights = exp.(log_weights .- logsumexp(log_weights))
    for (trace, weight) in zip(traces, norm_weights)
        alpha = max(min_alpha, 0.6*sqrt(weight))
        poses = get_path(trace)
        plot!([p.p[1] for p in poses], [p.p[2] for p in poses]; label=trace_label, color=:green, alpha=alpha)
        plot!(poses[end]; color=:green, alpha=alpha, label=nothing)
        plot!([Segment(p1.p, p2.p) for (p1, p2) in zip(poses[1:end-1], poses[2:end])];
              label=nothing, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=alpha)
        trace_label = nothing
    end

    return the_plot
end

function frame_from_info(world, title, path, path_label, info, info_label; show=(), min_alpha=0.03)
    the_plot = frame_from_weighted_traces(world, title * "\nt=$(info.t)|" * info.label, path, path_label,
                    info.traces, info.log_weights, info_label; show=show, min_alpha=min_alpha)
    if haskey(info, :vizs)
        viz_label = haskey(info.vizs[1].params, :label) ? info.vizs[1].params.label : nothing
        for viz in info.vizs
            plot!(viz.objs...; viz.params..., label=viz_label)
            viz_label=nothing
        end
    end
    return the_plot
end;

# %%
N_particles = 10

infos = particle_filter_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles)

ani = Animation()
for info in infos
    frame_plot = frame_from_info(world, "Run of PF", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, frame_plot)
end
gif(ani, "imgs/pf_animation_low.gif", fps=1)

# %%
N_particles = 10

infos = particle_filter_infos(full_model, T, full_model_args, constraints_high_deviation, N_particles)

ani = Animation()
for info in infos
    frame_plot = frame_from_info(world, "Run of PF", path_high_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, frame_plot)
end
gif(ani, "imgs/pf_animation_high.gif", fps=1)

# %% [markdown]
# ### Bootstrap
#
# One simple observation is that most of the partially proposed paths are clearly bad candidates from an early time.  We may intervene by pruning them out, say, replacing them with copies of the better ones before further exploration.  In other words, one can perform importance resampling on the particles in between the `update` steps.  The resulting kind of incremental SIR is often called a *bootstrap* in the literature.

# %%
effective_sample_size(log_weights) =
    exp(-logsumexp(2. * (log_weights .- logsumexp(log_weights))))

function particle_filter_bootstrap(model, T, args, constraints, N_particles, ESS_threshold)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
        end
    end

    return sample(traces, log_weights)
end

function particle_filter_bootstrap_infos(model, T, args, constraints, N_particles, ESS_threshold)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    infos = []

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
            push!(infos, (type = :initialize, time = now(), t = t, label = "sample from start pose prior", traces = copy(traces), log_weights = copy(log_weights)))
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :update, time = now(), t = t, label = "update to next step", traces = copy(traces), log_weights = copy(log_weights)))
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
            push!(infos, (type = :resample, time = now(), t = t, label = "resample", traces = copy(traces), log_weights = copy(log_weights)))
        end
    end

    traces, log_weights = resample(traces, log_weights; M=1)
    push!(infos, (type = :final_sample, time = now(), t = T, label = "final sample", traces = copy(traces), log_weights = copy(log_weights)))

    return infos
end;

# %% [markdown]
# Let's walk through the effect of resampling:
#
# ![](imgs_stable/bootstrap_with_code.gif)
#
# Here is the aggregate behavior:

# %%
N_particles = 10
ESS_threshold = 0.1

N_samples = 10

t1 = now()
traces = [particle_filter_bootstrap(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_bootstrap(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF+Bootstrap")

# %% [markdown]
# The results are already more accurate than blind SIR for only a fraction of the work.

# %% [markdown]
# ### Rejuvenation
#
# After resampling, our particles are more concentrated on the more likely ones, but they have the defficiency of being redundant.  We may again intervene by independently modifying the particles, for example by adding noise to increase diversity, and possibly using our knowledge of the target distrubtion to better approximate it.  Such modification is called *rejuvenation*.  The general structure is as follows.

# %%
function particle_filter_rejuv(model, T, args, constraints, N_particles, ESS_threshold, rejuv_kernel, rejuv_args_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
        end

        for rejuv_args in rejuv_args_schedule
            for i in 1:N_particles
                traces[i], log_weights[i] = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
            end
        end
    end

    return sample(traces, log_weights)
end

function particle_filter_rejuv_infos(model, T, args, constraints, N_particles, ESS_threshold, rejuv_kernel, rejuv_args_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    infos = []

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
            push!(infos, (type = :initialize, time = now(), t = t, label = "sample from start pose prior", traces = copy(traces), log_weights = copy(log_weights)))
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :update, time = now(), t = t, label = "update to next step", traces = copy(traces), log_weights = copy(log_weights)))
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
            push!(infos, (type = :resample, time = now(), t = t, label = "resample", traces = copy(traces), log_weights = copy(log_weights)))
        end

        for rejuv_args in rejuv_args_schedule
            vizs_collected = []
            for i in 1:N_particles
                traces[i], log_weights[i], vizs = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
                append!(vizs_collected, vizs)
            end
            push!(infos, (type = :rejuvenate, time = now(), t = t, label = "rejuvenate", traces = copy(traces), log_weights = copy(log_weights), vizs = vizs_collected))
        end
    end

    traces, log_weights = resample(traces, log_weights; M=1)
    push!(infos, (type = :final_sample, time = now(), t = T, label = "final sample", traces = copy(traces), log_weights = copy(log_weights)))

    return infos
end;

# %% [markdown]
# ### Proper weighting in rejuvenation: SMCP<sup>3</sup>
#
# As particles get modified in rejuvenation, what should happen to the weights?  The idea is to maintain the property that they properly weight the samples, and that "sequential Monte Carlo with probabilistic program proposals" (SMCP<sup>3</sup>) gives a means of modifying the weights to ensure this.
#
# We will need to generalize our means of approximating some target distribution $P$ beyond the importance sampling setup from above.  Now there is an extended distribution $\~Q$ that samples *pairs* $(w,z)$ where the $w\text{s}$ belong to $\mathbf{R}_{\geq 0}$ and the $z\text{s}$ are of the type sampled by $P$.  We let $Q$ be the distribution on $z\text{s}$ provided by $\~Q$ upon forgetting the $w\text{s}$ (i.e., marginalizing out that component); complementarily, we define $f(z_0)$ to be the expected value of $w\text{s}$ produced by $\~Q$ conditionally on the value $z = z_0$.  We say that $\~Q$ is *properly weighted* for $P$ if these $(Q,f)$ implement importance sampling for $P$.
#
# If we already can compute $Q$ and $f$ as in importance sampling, then we immediately get such a $\~Q$ by sampling $z$ from $Q$ then returning $(f(z),z)$.  The point of a properly weighted sampler $\~Q$, however, is that we need not compute $f(z)$ directly, but only provide stochastic estimates of this quantity—yielding a far more general notion.
#
# The following question soon arises: supposing we have on hand some properly weighted sampler $\~Q$ for $P$, where the values $z$ have type $X$, as well as some other target distribution $P'$ on values $z'$ of type $X'$, how might we concoct a properly weighted sampler $\~Q'$ for $P'$?  We might start by writing a generative function $g$ from $X$ to $X'$ to transform values $z$ into values $z'$.  So sampling $(w,z) \sim \~Q$ then applying $g$ to $z$ determines a proposal $Q'$ on $X'$.  It clearly would behoove us to engineer $g$ so to bring $Q'$ as near as possible to $P'$ in the first place.  But to close the remaining gap, for what construction of an *incremental weight* $\~w$ is it true that, setting $w' := w \cdot \~w$, the resulting distribution $(w',z') \sim \~Q'$ is properly weighted for $P'$?  An answer to this incremental weight question, whose shape we now describe, is provided by *SMCP<sup>3</sup>*.
#
# As described early on, the generative function that stochastically transforms $z$ to $z'$ corresponds to the following mathematical data: a probability kernel $k := k_g \colon X \dashrightarrow U$, where $U := U_g$ can be any auxiliary space that contains the trace information, together with a return value function written $g \colon X \times U \to X'$.  For SMCP<sup>3</sup> we program designers must extend the GF $g$ with the following interlocking pieces.  We must specify another auxiliary space $U'$ and that captures all the information in $X \times U$ thrown away by the return value function $g$, so that we may augment it to a *bijection* $\~g \colon X \times U \to X' \times U'$.  (In order to get a bijection, it is sometimes necessary to enlarge $U$ while designing $U'$.)  And, we must specify a "backwards" probability kernel $\ell \colon X' \dashrightarrow U'$.  In effect, $(k,\~g)$ determines a GF from $X$ to $X' \times U'$, while $(\ell,\~g^{-1})$ determines a GF back from $X'$ to $X \times U$.
#
# Now we have our properly weighted sampler $\~Q$ for $P$, producing $(w,z) \sim \~Q$.  We then run $\~g$ on it by sampling $u \sim k$ and setting $(z',u') = \~g(z,u)$.  Along the way, we let $J$ be the absolute value of the Jacobian determinant of $\~g$ at $(z,u)$.  Then take for our incremental weight
# $$
# \~w := \frac{P'(z')}{P(z)} \cdot \frac{\ell_{z'}(u')}{k_z(u)} \cdot J,
# $$
# and again set $w' = w \cdot \~w$.  (There is no problem if we instead compute some positive constant multiple of this $\~w$, for instance, because we only know $P$ and $P'$ up to such constant multiples.)  The conclusion of SMCP<sup>3</sup> is that this total process $\~Q'$ that produces the pair $(w',z')$ is properly weighted for $P'$, under suitable hypotheses.
#
# The leeway of choice of backwards kernel $\ell$ may be less surprising when we are reminded of all the leeway $\~Q$ had in stochastically estimating $f(z)$ to begin with.  But the choice of $\ell$ is not at all arbitrary from a practical point of view.  Namely, the variance of $\~w$ is determined by how well $\ell$ approximates the following ideal: given a value $z'$, the samples $u' \sim \ell_{z'}$ guess the auxiliary data such that $(z',u') = \~g(z,u)$ where $(z,u)$ are *likely* for $z \sim Q$ and $u \sim k_z$ conditionally on $(z',u')$.  In turn, the variance of the approximate importance weight $w' = w \cdot \~w$ affects the quality of later-*resampled* particles.
#
# In this notebook we will only apply SMCP<sup>3</sup> to generative function transformations upon a single space $X = X'$ with a single target distribution $P = P'$.  We will try to engineer the transformation so that it *improves sample quality*, that is, so that the resulting new proposal distribution $Q'$ might better approximate $P$ than $Q$ did (and, correspondingly, the variance of the importance weights should decrease as one passes from $\~Q$ to $\~Q'$).  Moreover, in our case, the bijection $\~g$ will amount to a permutation of some tuple components, so the Jacobian factor will be $J = 1$.  Thus SMCP<sup>3</sup> boils down to take the following shape:

# %%
# The argument `fwd_proposal` corresponds to `(k,\~g)` above.
# Its input, a trace `t`, corresponds to `z` above.
# Its output has the form `(cm_t, cm_u, vizs)` where
# * `t2, _ = update(t, cm_t)` is the proposed new particle trace, corresponding to `z'` above,
# * `cm_u` is a choice map for `bwd_proposal`, corresponding to `u'` above, and
# * `vizs` is data we supply for use in visualization.
# The argument `bwd_proposal` works vice versa for `(\ell,\~g^{-1})` above, sans the `vizs`.
function smcp3_step(particle, log_weight, fwd_proposal, bwd_proposal, proposal_args)
    fwd_proposal_trace = simulate(fwd_proposal, (particle, proposal_args...))
    (fwd_model_update, bwd_proposal_choicemap, vizs) = get_retval(fwd_proposal_trace)
    fwd_proposal_weight = get_score(fwd_proposal_trace)
    # Gen shorthand for the above lines:
    # _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap, vizs) = propose(fwd_proposal, (particle, proposal_args...))

    proposed_particle, model_weight_diff, _, _ = update(particle, fwd_model_update)

    _, bwd_proposal_weight = generate(bwd_proposal, (proposed_particle, proposal_args...), bwd_proposal_choicemap)
    # One might also see the equivalent Gen code:
    # bwd_proposal_weight, _ = assess(bwd_proposal, (proposed_particle, proposal_args...), bwd_proposal_choicemap)

    proposed_log_weight = log_weight + model_weight_diff + bwd_proposal_weight - fwd_proposal_weight
    return proposed_particle, proposed_log_weight, vizs
end

smcp3_kernel(fwd_proposal, bwd_proposal) =
    (particle, log_weight, proposal_args) -> smcp3_step(particle, log_weight, fwd_proposal, bwd_proposal, proposal_args);

# %% [markdown]
# ### Grid search proposal
#
# Having set up the apparatus of rejuvenation of particles and adjustment of their weights, we now specify a particular strategy of doing so.
#
# A simple idea is to try searching near a given pose $z_t$, say, ranging through a grid of poses around it.  We can then replace $z_t$ with a grid member in proportion to the likelihood of the data $o_t$ at that pose.  In other words, we sample exactly from the restriction of the posterior to this finite set.
#
# Bringing this into the langauge of SMCP<sup>3</sup>, the forward kernel's auxiliary randomness $U$ records an index $j$ that addresses a grid member $z'_t$, and then $g(z_t,j) = z'_t$.  Reciprocally the grid around $z'_t$ contains $z_t$ at a unique "inverse" index $j'$ that can be computed from $j$.  Therefore, we can take $U'$ to record indices too, and set $\~g(z_t,j) = (z'_t,j')$ to get a bijection.
#
# This idea is implemented by `grid_fwd_proposal` below.  The (log) likelihoods of the poses in the grid are, up to a common constant, encoded in `pose_log_weights`.
#
# We illustrate the leeway in the design of the reverse kernel by providing two examples for this one forward kernel.  In both cases, given $z'_t$ we are to guess a reverse index $j'$ so that $\~g^{-1}(z'_t,j') = (z_t,j)$ where $z_t$ was likely the pose prior to rejuvenation.
#
# The optimal way, from the point of view of theoretically minimizing the variance of the incremental weight, would be to sample $z_t$ from (the restriction to the grid of) the path model conditioned on the information that the forward kernel used the data $o_t$ to send $z_t$ to $z'_t$.  This strategy is implemented by `grid_bwd_proposal_exact` below.  While this design is admirable, it is computationally resource-intensive: one must iterate over the backwards grid, and at each of these grid members interate over its forwards grid to compute a weight.
#
# There is a much faster and simpler way, at the cost of so little incremental weight variance that it has empirically negligible impact on inference performance: ignore the data $o_t$, and just draw $z_t$ from (the restriction to the inverse grid of) the path model.  This strategy is implemented by `grid_bwd_proposal` below.

# %%
function vector_grid(center, grid_n_points, grid_sizes)
    offset = center .- (grid_n_points .+ 1) .* grid_sizes ./ 2.
    return reshape(map(I -> [Tuple(I)...] .* grid_sizes .+ offset, CartesianIndices(Tuple(grid_n_points))), (:,))
end

inverse_grid_index(grid_n_points, j) =
    LinearIndices(Tuple(grid_n_points))[(grid_n_points .+ 1 .- [Tuple(CartesianIndices(Tuple(grid_n_points))[j])...])...]

# Sample from the posterior, restricted to the grid.
@gen function grid_fwd_proposal(trace, grid_n_points, grid_sizes)
    t = get_args(trace)[1]
    p = trace[prefix_address(t+1, :pose => :p)]
    hd = trace[prefix_address(t+1, :pose => :hd)]

    pose_grid = vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)
    choicemap_grid = [choicemap((prefix_address(t+1, :pose => :p), [x, y]),
                                (prefix_address(t+1, :pose => :hd), h))
                      for (x, y, h) in pose_grid]
    # Posterior densities, up to a common renormalization.
    pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    fwd_j ~ categorical(pose_norm_weights)
    bwd_j = inverse_grid_index(grid_n_points, fwd_j)

    vizs = [(objs = ([Pose([x, y], h) for (x, y, h) in pose_grid],),
             params = (color=:red, label="pose grid"))]

    return choicemap_grid[fwd_j], choicemap((:bwd_j, bwd_j)), vizs
end

# Sample from the prior, restricted to the inverse grid.
@gen function grid_bwd_proposal(trace, grid_n_points, grid_sizes)
    t = get_args(trace)[1]
    p = trace[prefix_address(t+1, :pose => :p)]
    hd = trace[prefix_address(t+1, :pose => :hd)]

    pose_grid = vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)
    choicemap_grid = [choicemap((prefix_address(t+1, :pose => :p), [x, y]),
                                (prefix_address(t+1, :pose => :hd), h))
                      for (x, y, h) in pose_grid]
    # Prior densities.
    pose_log_weights = [project(update(trace, cm)[1], select(prefix_address(t+1, :pose)))
                        for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    bwd_j ~ categorical(pose_norm_weights)
    fwd_j = inverse_grid_index(grid_n_points, bwd_j)

    return choicemap_grid[bwd_j], choicemap((:fwd_j, fwd_j))
end

grid_smcp3_kernel = smcp3_kernel(grid_fwd_proposal, grid_bwd_proposal);

# Sample from the prior, conditioned to the forward kernel sending it to the given trace.
@gen function grid_bwd_proposal_exact(trace, grid_n_points, grid_sizes)
    t = get_args(trace)[1]
    p = trace[prefix_address(t+1, :pose => :p)]
    hd = trace[prefix_address(t+1, :pose => :hd)]

    pose_grid = vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)
    choicemap_grid = [choicemap((prefix_address(t+1, :pose => :p), [x, y]),
                                (prefix_address(t+1, :pose => :hd), h))
                      for (x, y, h) in pose_grid]
    # Densities for the conditional distribution of inverse grid upon the information of the forward kernel.
    pose_log_weights = Vector{Float64}(undef, length(choicemap_grid))
    for (bwd_j, cm) in enumerate(choicemap_grid)
        bwd_trace, _, _, fwd_cm = update(trace, cm)
        pose_log_weights[bwd_j] =
            project(bwd_trace, select(prefix_address(t+1, :pose))) +
            assess(grid_fwd_proposal, (bwd_trace, grid_n_points, grid_sizes), choicemap((:fwd_j, inverse_grid_index(grid_n_points, bwd_j))))[1]
    end
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    bwd_j ~ categorical(pose_norm_weights)
    fwd_j = inverse_grid_index(grid_n_points, bwd_j)

    return choicemap_grid[bwd_j], choicemap((:fwd_j, fwd_j))
end

grid_smcp3_kernel_exact = smcp3_kernel(grid_fwd_proposal, grid_bwd_proposal_exact);

# %% [markdown]
# Here is a visual on a run of the algorithm:
#
# ![](imgs_stable/smcp3_with_code.gif)
#
# It is immediately important to visualize its operation.  Try experimenting with the parameters in the following block:

# %%
N_particles = 10

grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:3]

infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of PF + SMCP3/Grid", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_smcp3_grid.gif", fps=1)

# %% [markdown]
# And here is the aggregate behavior:

# %%
N_particles = 10

grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:3]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + SMCP3/Grid")

# %% [markdown]
# The speed of this approach is already perhaps an issue.  The performance is even worse (~15x slower) using the "exact" backwards kernel, with no discernible improvement in inference, as can be seen by uncommenting and running the code below.

# %%
# N_particles = 10

# grid_n_points = [3, 3, 3]
# grid_sizes = [.5, .5, π/10]
# grid_args_schedule = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:3]

# N_samples = 10

# t1 = now()
# traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel_exact, grid_args_schedule) for _ in 1:N_samples]
# t2 = now()
# println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
# posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

# t1 = now()
# traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, grid_smcp3_kernel_exact, grid_args_schedule) for _ in 1:N_samples]
# t2 = now()
# println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
# posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

# plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + SMCP3/Grid")

# %% [markdown]
# Because that our rejuvenation scheme improves sample quality, perhaps we do not even need to track many particles.  Let's try out *one* particle (and vacuous resampling):

# %%
N_particles = 1

grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:3]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + SMCP3/Grid (1pc)")

# %% [markdown]
# Here we see some degredation in the inference quality.  But since there is one particle, maybe we can spend a little more effort in the grid search.

# %%
N_particles = 1

grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule_harder = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:6]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule_harder) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule_harder) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + SMCP3/Grid (1pc)")

# %%
N_particles = 1

grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule_harder = [(grid_n_points, grid_sizes .* (2/3)^j) for j=0:6]

infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of PF + SMCP3/Grid (1pc)", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_smcp3_grid.gif", fps=1)

# %% [markdown]
# Similar quality, and ~5x faster!
#
# This example embodies one of the central attitudes of ProbComp: devote computational effort to constructing *better* particles instead of to constructing *more* particles.

# %% [markdown]
# ### MCMC rejuvenation / Gaussian drift proposal
#
# A faster rejuvenation strategy than a grid search is to simply jiggle the point.

# %%
@gen function drift_fwd_proposal(trace, drift_factor)
    t = get_args(trace)[1]
    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    undrift_p = trace[prefix_address(t+1, :pose => :p)]
    undrift_hd = trace[prefix_address(t+1, :pose => :hd)]

    drift_p ~ mvnormal(undrift_p, (drift_factor * p_noise)^2 * [1 0 ; 0 1])
    drift_hd ~ normal(undrift_hd, drift_factor * hd_noise)

    std_devs_radius = 2.5 * drift_factor * p_noise
    vizs = [(objs = make_circle(undrift_p, std_devs_radius),
             params = (color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.1))]

    return choicemap((prefix_address(t+1, :pose => :p), drift_p), (prefix_address(t+1, :pose => :hd), drift_hd)),
           choicemap((:undrift_p, undrift_p), (:undrift_hd, undrift_hd)),
           vizs
end

@gen function drift_bwd_proposal(trace, drift_factor)
    t = get_args(trace)[1]
    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    if t == 0
       start_pose = get_args(trace)[2].start
       noiseless_p = start_pose.p
       noiseless_hd = start_pose.hd
    else
       prev_pose = trace[prefix_address(t, :pose)]
       prev_control = get_args(trace)[2].controls[t]
       noiseless_p = prev_pose.p + prev_control.ds * prev_pose.dp
       noiseless_hd = prev_pose.hd + prev_control.dhd
    end

    drift_p = trace[prefix_address(t+1, :pose => :p)]
    drift_hd = trace[prefix_address(t+1, :pose => :hd)]

    # Optimal choice of `undrift_pose` is obtained from conditioning the motion model that noises `noiseless_pose` upon
    # the information that the forward drift then further moved it to `drift_pose`.
    # In this case there is a closed form, which happens to be a Gaussian drift with some other parameters.
    e = 1/drift_factor^2
    undrift_p ~ mvnormal((noiseless_p + e * drift_p)/(1+e), p_noise^2/(1+e) * [1 0 ; 0 1])
    undrift_hd ~ normal((noiseless_hd + e * drift_hd)/(1+e), hd_noise/sqrt(1+e))

    return choicemap((prefix_address(t+1, :pose => :p), undrift_p), (prefix_address(t+1, :pose => :hd), undrift_hd)),
           choicemap((:drift_p, drift_p), (:drift_hd, drift_hd))
end

drift_smcp3_kernel = smcp3_kernel(drift_fwd_proposal, drift_bwd_proposal);

# %% [markdown]
# It is worth considering the critique that this forward proposal does not improve the samples in any way towards the particular target distribution; all it does is jiggle them.  Any beneficial effect would result from the jiggling having been applied to already-resampled particles, amounting to a local search around a good candidate, upon which the next update and resample will be based.
#
# The result is, unsurprisingly, not much different from the plain bootstrap:

# %%
N_particles = 10

drift_args_schedule = [0.7^k for k=1:7]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_smcp3_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, drift_smcp3_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + SMCP3/Drift")

# %% [markdown]
# We can compromise between the grid search and jiggling.  The idea is to perform a mere two-element search that compares the given point with the random one, or rather to immediately resample one from the pair.  This would have a chance of improving sample quality, without spending much time searching scrupulously for the improvement.
#
# The resulting algorithm is conventionally called "Markov chain Monte carlo (MCMC) rejuvenation".  Our strategy to just resample between the pair amounts to the "Boltzmann acceptance rule".  An even more common decision strategy that slightly biases in favor of the jiggled sample is called the "Metropolis–Hastings acceptance rule".

# %%
function mcmc_step(particle, log_weight, mcmc_proposal, mcmc_args, mcmc_rule)
    proposed_particle, proposed_log_weight, vizs = mcmc_proposal(particle, log_weight, mcmc_args)
    return mcmc_rule([particle, proposed_particle], [log_weight, proposed_log_weight])..., vizs
end
mcmc_kernel(mcmc_proposal, mcmc_rule) =
    (particle, log_weight, mcmc_args) -> mcmc_step(particle, log_weight, mcmc_proposal, mcmc_args, mcmc_rule)

boltzmann_rule(particles, log_weights) = first.(resample(particles, log_weights; M=1))

# Assumes `particles` is ordered so that first item is the original and second item is the proposed.
# Notes:
# * If the proposed item has higher weight, it is accepted unconditionally.  There is an overall bias.
# * In all cases, the weight is unchanged.
function mh_rule(particles, log_weights)
    @assert length(particles) == length(log_weights) == 2
    acceptance_ratio = min(1., exp(log_weights[2] - log_weights[1]))
    return (bernoulli(acceptance_ratio) ? particles[2] : particles[1]), log_weights[1]
end;

# %%
drift_boltzmann_kernel = mcmc_kernel(drift_smcp3_kernel, boltzmann_rule)
drift_mh_kernel = mcmc_kernel(drift_smcp3_kernel, mh_rule);

# %% [markdown]
# Here are a detailed run, followed by the aggregate behavior, using the Boltzmann rule:

# %%
N_particles = 10

drift_args_schedule = [0.7^k for k=1:7]

infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_boltzmann_kernel, drift_args_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of PF + Boltzmann/Drift", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_boltzmann_drift.gif", fps=1)

# %%
N_particles = 10

drift_args_schedule = [0.7^k for k=1:7]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_boltzmann_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, drift_boltzmann_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + Boltzmann/Drift")

# %% [markdown]
# Similarly, here are a detailed run, followed by the aggregate behavior, using the MH rule:

# %%
N_particles = 10

drift_args_schedule = [0.7^k for k=1:7]

infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of PF + MH/Drift", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_mh_drift.gif", fps=1)

# %%
N_particles = 10

drift_args_schedule = [0.7^k for k=1:7]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + MH/Drift")

# %% [markdown]
# Thus we can recover most of inference performance to the grid search, at a fraction of the compute cost.
#
# Can we again replace particle multiplicity with more aggressive rejuvenation?  For drift rejuvenation on its own, there is a clearer penalty in inference quality.

# %%
N_particles = 1

drift_args_schedule = [2 * 0.9^k for k=1:27]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="PF + MH/Drift (1pc)")

# %% [markdown]
# ### Reusable components
#
# The ingredients of the particle filter programs we have written may certainly be abstracted, then reused with brevity.  Although we will not do so here, out of an intention to keep all the techniques explicit, we note that such abstractions are provided by the `GenParticleFilters` library.  For an example of its use, the reader is encouraged to peer into the file `black_box.jl` in this repo and compare the inference code there to the present state of our approach to the robot problem.

# %% [markdown]
# ### Adaptive inference
#
# We the inference programmers do not have to be stuck with some fixed amount of rejuvenation effort: we get to choose how much computing resource to devote to our particle population's sample quality.  To do so programmatically, we will assume given some numerical test for the fitness of each proposed new time step in the family of particles.  If the proposed new particles meet the criterion, we do no further work on them and move on to the next time step.  As long as they do not, we keep trying more interventions, for example, rounds of increasingly expensive rejuvenation.

# %%
function particle_filter_fitness(model, T, args, constraints, N_particles, ESS_threshold, fitness_test, rejuv_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[1])
            end
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
        end

        for (rejuv_kernel, rejuv_args_schedule) in rejuv_schedule
            if fitness_test(traces, t); break end
            for rejuv_args in rejuv_args_schedule
                for i in 1:N_particles
                    traces[i], log_weights[i] = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
                end
            end
        end
    end

    return sample(traces, log_weights)
end

function particle_filter_fitness_infos(model, T, args, constraints, N_particles, ESS_threshold, fitness_test, rejuv_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    infos = []

    for t in 0:T
        if t == 0
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[1])
            end
            push!(infos, (type = :initialize, time = now(), t = t, label = "sample from start pose prior", traces = copy(traces), log_weights = copy(log_weights)))
        else
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :update, time = now(), t = t, label = "update to next step", traces = copy(traces), log_weights = copy(log_weights)))
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
            push!(infos, (type = :resample, time = now(), t = t, label = "resample", traces = copy(traces), log_weights = copy(log_weights)))
        end

        for (r, (rejuv_kernel, rejuv_args_schedule)) in enumerate(rejuv_schedule)
            if fitness_test(traces, t); break end
            for rejuv_args in rejuv_args_schedule
                vizs_collected = []
                for i in 1:N_particles
                    traces[i], log_weights[i], vizs = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
                    append!(vizs_collected, vizs)
                end
                push!(infos, (type = :rejuvenate, time=now(), t = t, label = "rejuvenate #$r", traces = copy(traces), log_weights = copy(log_weights), vizs = vizs_collected))
            end
        end
    end

    traces, log_weights = resample(traces, log_weights; M=1)
    push!(infos, (type = :final_sample, time = now(), t = T, label = "final sample", traces = copy(traces), log_weights = copy(log_weights)))

    return infos
end;

# %% [markdown]
# What we particularly mean by a "fitness test" recollects some ideas from earlier in this tutorial, as follows.
#
# We are assessing the suitability of a family of weighted particles $(w_t^{(i)}, z_{0:t}^{(i)})_{i=1}^N$ constructed up to time $t$.  These particles were constructed either using `Gen.generate` at the first time step $t=0$, or otherwise extended from a family $(w_{t-1}^{(i)},z_{0:t-1}^{(i)})_{i=1}^N$ using `Gen.update`.  Both of these `Gen` operations also returned some kind (log) "weight" $\~w_t^{(i)}$, namely the importance weight $\~w_0^{(i)} := w_0^{(i)}$ in the first case, and the incremental weight $\~w_t^{(i)} := w_t^{(i)}/w_{t-1}^{(i)}$ in the second case.  Each of these weights $\~w_t^{(i)}$ is equal to the sensor model probability density $P_\text{sensor}(o_t^{(i)};z_t^{(i)},\ldots)$ of the observations $o_t^{(i)}$ at time step $t$.  Thus, on the one hand, one can recover these numbers in a unform manner using `Gen.project` as explained earlier, and on the other hand, they measure the fitness of the step $z_t^{(i)}$ as an extension of the particle to time $t$.
#
# Out of many possible design choices, we will limit ourselves to considering when a given function $h(w)$ of tuples $w = (w^{(i)})_{i=1}^N$ meets a given "allowance" bound when evaluated at $\~w_t := (\~w_t^{(i)})_{i=1}^N$.  Here are two choices of such functions $h$.  They have been normalized so that the allowances do not need to be adjusted as the number $N$ of particles is changed:
# * The average log weight, $h(w) = \frac1N \sum_{i=1}^N \log w^{(i)}$.
# * The log average weight, also known as the *log marginal likelihood estimate*, $h(w) = \log \big[ \frac1N \sum_{i=1}^n w^{(i)} \big]$.
#
# The first of these responds equally to changes in each weight, and therefore measures the fitness of all the particles, whereas the second is predominately determined by the largest weights, and therefore measures the fitness of the best-fitting particles.  Let's have a look at how an inference step is assessed by these rules.

# %%
# Fitness tests:

incremental_weight(trace, t) = project(trace, select(prefix_address(t+1, :sensor)))
make_fitness_test(func, allowances) =
    (traces, t) -> func([incremental_weight(trace, t) for trace in traces]) > allowances[t+1]

average_log_weight(log_weights) = sum(log_weights) / length(log_weights)
log_average_weight(log_weights) = logsumexp(log_weights) - log(length(log_weights))

fitness_test = make_fitness_test(log_average_weight, [-1e2 for _ in 1:(T+1)])

# Sequence of rejuvenation strategies:

# First try a quicker Gaussian drift.
drift_args_schedule = [0.7^k for k=1:7]

# Then try a more determined grid search.
grid_n_points = [3, 3, 3]
grid_sizes = [.5, .5, π/10]
grid_args_schedule = [(grid_n_points, grid_sizes * (2/3)^j) for j=0:3]
grid_args_schedule_harder = [(grid_n_points, grid_sizes * (2/3)^j) for j=0:6]

rejuv_schedule =
    [(drift_mh_kernel, drift_args_schedule),
     (grid_smcp3_kernel, grid_args_schedule)];

# %%
N_particles = 10

infos = particle_filter_fitness_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of Controlled PF", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_controlled.gif", fps=1)

# %%
N_particles = 10

N_samples = 10

t1 = now()
traces = [particle_filter_fitness(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_fitness(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="Controlled PF")

# %% [markdown]
# ### Backtracking
#
# Sometimes the particle filter finds itself invested in paths with no good extensions, and no amount of rejuvenation on the final step alone seems to help.  In such cases we can choose to essentially broaden our scope to rejuvenating the final several steps.  In other words, if we find ourselves in a dead end, we can back out and try again.
#
# Note carefully how breaking the time flow, alternating forwards and back, could lead to an inference process that is stuck in indecision.  This is especially a risk when given data that admit no good fits, or data whose good fits are exceedingly hard to find: when do we decide to keep working, versus accept what we have, versus quit?
#
# There are many specific ways the inference programmer might choose to respond to this circumstance; as you consider what policy you might prefer to adopt here in response to indecision, you are invited to notice how inference programming offers an expression of your human *reasoning*, just as modeling with generative functions offers an expression of quantifiable *beliefs*, in the presence of uncertainty and incomplete understanding.
#
# Our choice of design here is to limit the backtracking to only a fixed schedule of backwards steps, within which we do not perform any sub-backtracking but instead die off early, and after which a (possibly unsatisfactory) advance is settled upon.
#
# Note the appearance of `Gen.update`'s sibling `Gen.regenerate`: instead of hand-fixing certain stochastic choices in a trace, it simply redraws them from the immanent distribution; it may also modify functional parameters in same way.

# %%
function particle_filter_backtrack(model, T, args, constraints, N_particles, ESS_threshold, fitness_test, rejuv_schedule, backtrack_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    # The flag `backtrack_state` holds the value `0` when no backtracking is taking place.
    # Otherwise, `t_saved` marks the time from which we have backtracked a distance of
    # `backtrack_schedule[backtrack_state]`, and `candidates` holds traces constructed on
    # prior complete forward runs up to `t_saved`.
    backtrack_state, candidates, t_saved = 0, [], 0

    t = 0
    action = :initialize
    while !(t == T && action == :advance)
        if action == :initialize
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
        elseif action == :advance
            t = t + 1
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        elseif action == :backtrack
            backtrack_state += 1
            t_saved = t
            dt = min(backtrack_schedule[backtrack_state], t)
            t = t - dt
            for i in 1:N_particles
                traces[i], log_weight_increment, _ = regenerate(traces[i], (t, args...), change_only_T, select(prefix_address(t+1, :pose)))
                log_weights[i] += log_weight_increment
            end
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
        end

        for (rejuv_kernel, rejuv_args_schedule) in rejuv_schedule
            if fitness_test(traces, t); break end
            for rejuv_args in rejuv_args_schedule
                for i in 1:N_particles
                    traces[i], log_weights[i] = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
                end
            end
        end

        # Advance-or-backtrack logic.
        if fitness_test(traces, t)
            # The fitness criterion has been met, so advance to the next time step.
            # If we are at the end of a backtrack then clear the backtracking state,
            # including discarding the prior, unfit runs up to `t_saved`.
            if backtrack_state > 0 && t == t_saved
                backtrack_state, candidates = 0, []
            end
            action = :advance
        else
            # The fitness criterion has failed, so decide what to do.
            # If we are either not yet backtracking, or at the end of some backtrack, then save the particles.
            # Otherwise, we are in an incomplete backtrack; terminate the current backtrack without saving it.
            if backtrack_state == 0 || t == t_saved
                append!(candidates, zip(traces, log_weights))
            else
                t = t_saved
            end
            # If more backtracking is on the schedule then do it.
            # Otherwise, resample from the available saved candidates, clear the state, and move on.
            if backtrack_state < length(backtrack_schedule)
                action = :backtrack
            else
                traces, log_weights = resample(first.(candidates), last.(candidates); M=N_particles)
                backtrack_state, candidates = 0, []
                action = :advance
            end
        end
    end

    return sample(traces, log_weights)
end

function particle_filter_backtrack_infos(model, T, args, constraints, N_particles, ESS_threshold, fitness_test, rejuv_schedule, backtrack_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    infos = []

    backtrack_state, candidates, t_saved = 0, [], 0

    t = 0
    action = :initialize
    while !(t == T && action == :advance)
        if action == :initialize
            for i in 1:N_particles
                traces[i], log_weights[i] = generate(model, (t, args...), constraints[t+1])
            end
            push!(infos, (type = :initialize, time = now(), t = t, label = "initialize fresh particles", traces = copy(traces), log_weights = copy(log_weights)))
        elseif action == :advance
            t = t + 1
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :update, time=now(), t = t, label = "update to next step", traces = copy(traces), log_weights = copy(log_weights)))
        elseif action == :backtrack
            backtrack_state += 1
            t_saved = t
            dt = min(backtrack_schedule[backtrack_state], t)
            t = t - dt
            for i in 1:N_particles
                traces[i], log_weight_increment, _ = regenerate(traces[i], (t, args...), change_only_T, select(prefix_address(t+1, :pose)))
                log_weights[i] += log_weight_increment
            end
            push!(infos, (type = :backtrack, time=now(), t = t, label = "backtrack $dt steps", traces = copy(traces), log_weights = copy(log_weights)))
        end

        if effective_sample_size(log_weights) < (1. + ESS_threshold * length(log_weights))
            traces, log_weights = resample(traces, log_weights)
            push!(infos, (type = :resample, time = now(), t = t, label = "resample", traces = copy(traces), log_weights = copy(log_weights)))
        end

        for (r, (rejuv_kernel, rejuv_args_schedule)) in enumerate(rejuv_schedule)
            if fitness_test(traces, t); break end
            for rejuv_args in rejuv_args_schedule
                vizs_collected = []
                for i in 1:N_particles
                    traces[i], log_weights[i], vizs = rejuv_kernel(traces[i], log_weights[i], rejuv_args)
                    append!(vizs_collected, vizs)
                end
                push!(infos, (type = :rejuvenate, time=now(), t = t, label = "rejuvenate #$r", traces = copy(traces), log_weights = copy(log_weights), vizs = vizs_collected))
            end
        end

        if fitness_test(traces, t)
            if backtrack_state > 0 && t == t_saved
                backtrack_state, candidates = 0, []
            end
            action = :advance
        else
            if backtrack_state == 0 || t == t_saved
                append!(candidates, zip(traces, log_weights))
            else
                t = t_saved
            end
            if backtrack_state < length(backtrack_schedule)
                action = :backtrack
            else
                traces, log_weights = first.(candidates), last.(candidates)
                push!(infos, (type = :indecision1, time = now(), t = t, label = "indecision: all candidates", traces = copy(traces), log_weights = copy(log_weights)))
                traces, log_weights = resample(traces, log_weights; M=N_particles)
                push!(infos, (type = :indecision2, time = now(), t = t, label = "indecision: resample", traces = copy(traces), log_weights = copy(log_weights)))
                backtrack_state, candidates = 0, []
                action = :advance
            end
        end
    end

    traces, log_weights = resample(traces, log_weights; M=1)
    push!(infos, (type = :final_sample, time = now(), t = T, label = "final sample", traces = copy(traces), log_weights = copy(log_weights)))

    return infos
end;

# %% [markdown]
# Here is a stepping through:
#
# ![](imgs_stable/controlled_smcp3_with_code.gif)

# %%
backtrack_schedule = [2, 4, 8];

# %%
N_particles = 10

infos = particle_filter_backtrack_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule, backtrack_schedule)

ani = Animation()
for info in infos
    graph = frame_from_info(world, "Run of Backtracking PF", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, graph)
end
gif(ani, "imgs/PF_backtrack.gif", fps=1)

# %%
N_particles = 10

N_samples = 10

t1 = now()
traces = [particle_filter_backtrack(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule, backtrack_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [particle_filter_backtrack(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, fitness_test, rejuv_schedule, backtrack_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms.")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

plot(posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1000,500), layout=grid(1,2), plot_title="Backtracking PF")

# %% [markdown]
# ## Improving robustness

# %% [markdown]
# ### Unanticipated challenges
#
# We may have a reasonable means of tracking the robot's position, when we maintain near-idealized conditions.  But how well does our inference code work when it is run on observation data sets that break our hypotheses?
#
# For example, what happens when we start the robot facing the wrong way?

# %%
full_settings_low_dev = (full_settings..., motion_settings=motion_settings_low_deviation)

ensure_askew_start = choicemap((prefix_address(1, :pose => :hd), Float64(pi/5)))
trace_askew_start, _ = generate(full_model, (T, robot_inputs, world_inputs, full_settings_low_dev), ensure_askew_start)
path_askew_start = get_path(trace_askew_start)
observations_askew_start = get_sensors(trace_askew_start)
constraints_askew_start = [constraint_from_sensors(o...) for o in enumerate(observations_askew_start)]

ani = Animation()
frames_askew_start = frames_from_full_trace(world, "Askew start", trace_askew_start)
for frame_plot in frames_askew_start[2:2:end]
    frame(ani, frame_plot)
end
gif(ani, "imgs/askew_start.gif", fps=2)

# %% [markdown]
# Our model doesn't propose very realistic hypotheses for the inference strategy to work with:

# %%
N_particles = 10

drift_args_schedule = [0.8^k for k=1:10]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_askew_start, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (askew start): $(value(t2 - t1) / N_samples) ms.")
frame_from_traces(world, "Askew start", path_askew_start, "path to be fit", traces, "samples")

# %% [markdown]
# Or how about if we "kidnapped" the robot: partway through the journey, the robot is paused, moved to another room, then resumed?
#
# Constructing such data using a single trace from the current `full_model` would be a little tricky, because the poses appearing in the trace are only *attempted* step destinations, which are then subjected to wall-collision detection: just using `generate` with a constraint for a pose drawn in the next room would only run the robot into the wall.
#
# Since we only need a data set, rather than a bona fide trace, we proceed instead by splicing two trajectories (or rather their observations), the second of which has been validly steered into the wrong room.

# %%
trace_kidnapped_first = simulate(full_model, (T, robot_inputs, world_inputs, full_settings_low_dev))
path_kidnapped_first = get_path(trace_kidnapped_first)
observations_kidnapped_first = get_sensors(trace_kidnapped_first)

controls_kidnapping = [Control(1.8,0.), Control(0.,-pi/2.), Control(4.,0.), Control(0.,pi/2.)]
T_kidnap = length(controls_kidnapping)
controls_kidnapped = [controls_kidnapping..., robot_inputs.controls[(T_kidnap+1):end]...]
trace_kidnapped_second = simulate(full_model, (T, (robot_inputs..., controls=controls_kidnapped), world_inputs, full_settings_low_dev))
path_kidnapped_second = get_path(trace_kidnapped_second)
observations_kidnapped_second = get_sensors(trace_kidnapped_second)

path_kidnapped = [path_kidnapped_first[1:T_kidnap]..., path_kidnapped_second[(T_kidnap+1):end]...]
observations_kidnapped = [observations_kidnapped_first[1:T_kidnap]..., observations_kidnapped_second[(T_kidnap+1):end]...]
constraints_kidnapped = [constraint_from_sensors(o...) for o in enumerate(observations_kidnapped)]

ani = Animation()
for t in 1:(T+1)
    frame_plot = frame_from_sensors(
        world, "Kidnapped after t=4",
        path_kidnapped[1:t], :black, nothing,
        path_kidnapped[t], observations_kidnapped[t], "sampled sensors",
        full_settings.sensor_settings)
    frame(ani, frame_plot)
end
gif(ani, "imgs/kidnapped.gif", fps=2)

# %% [markdown]
# Again, inference only produces explanations that were plausible in terms of the given `full_model`:

# %%
N_particles = 10

drift_args_schedule = [0.8^k for k=1:10]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_kidnapped, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (backwards start): $(value(t2 - t1) / N_samples) ms.")
frame_from_traces(world, "Kidnapped after t=4", path_kidnapped, "path to be fit", traces, "samples")

# %% [markdown]
# For another challenge, what if our map were modestly inaccurate?
#
# At the beginning of the notebook, we illustrated the data of "clutters", or extra boxes left inside the environment.  These would impact the motion and the sensory *observation data* of a run of the robot, but are not accounted for in the above *model* when attempting to infer its path.  How well does the inference process work in the presence of such discrepancies?

# %%
world_inputs_cluttered = (world_inputs..., walls=[world.walls ; world.clutters...])
trace_cluttered = simulate(full_model, (T, robot_inputs, world_inputs_cluttered, full_settings_low_dev))
path_cluttered = get_path(trace_cluttered)
observations_cluttered = get_sensors(trace_cluttered)
constraints_cluttered = [constraint_from_sensors(o...) for o in enumerate(observations_cluttered)]

ani = Animation()
frames_cluttered = frames_from_full_trace(world, "Cluttered space", trace_cluttered; show=(:label, :clutters))
for frame_plot in frames_cluttered[2:2:end]
    frame(ani, frame_plot)
end
gif(ani, "imgs/cluttered.gif", fps=2)

# %% [markdown]
# TODO: arrange clutters to confuse the inference!  Then comment here on how it breaks down.

# %%
N_particles = 10

drift_args_schedule = [0.8^k for k=1:10]

N_samples = 10

t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints_cluttered, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (backwards start): $(value(t2 - t1) / N_samples) ms.")
frame_from_traces(world, "Cluttered space", path_cluttered, "path to be fit", traces, "samples"; show=(:clutters,))

# %% [markdown]
# We take up the task of accommodating a wider range of phenomena in our modeling and inference.
#
# We require a model that is flexible enough to accommodate what we encounter in the first place.  Some of the above scenarios involve data that occur with zero probability under the prior.  (The probability of the backwards start underflows to zero in the floating point arithmetic; the probability of the kidnapping is definitely zero.)  For this we will employ Bayesian *hierarchical models* that express the belief that rare discrepancies occur.  We will also adjust our inference code to work with the richer model.

# %% [markdown]
# ### Hierarchical modeling
#
# We would like to model the possibility that we might hold different states of belief about the correct modeling strategy.  For example, we might find different noise parameters appropriate at different times, or even range among very differently conceived models embodying a discrete choice of strategy.  Thus we will have a familiy of models parameterized by this family of states of belief, as well as a distribution over this family.  Such is the nature of a *hierarchical model* as is commonly found in the Bayesian world.
#
# The family of states of belief, together with the distribution representing our prior over them, make up the *hyperprior*.  A state of belief drawn from the hyperprior is called a *parameter*.  The instantiation of the model downstream of this choice of parameter remains simply the *prior* (albeit specialized to this parameter).
#
# We will rework each of the motion and sensor models hierarchically, in such a way that their parameters represent increasing degrees of uncertainty, thus leading the prior to produce more diverse samples.  This will interoperate with an inference strategy that decides when to spend more effort ranging over more uncertain samples.

# %% [markdown]
# For simplicity let us propose four increasing levels of uncertainty around the motion of the robot.  They are, in order, represented by the following distributions:
# 1. Identical to the path model as specified above.
# 2. The path model as above, but with the motion noise parameters tripled.
# 3. The path model as above, but with the motion noise parameters decupled.
# 4. Complete ignorance, embodied as a uniform distribution over the bounding box.
#
#

# %% [markdown]
# ## Goal inference
#
#

# %% [markdown]
# ### Goals

# %%
function load_goals(file_name)
    data = parsefile(file_name)
    tasks = Dict(
        task => (Vector{Float64}(info["p"]), Float64(info["r"]))
        for (task, info) in data["tasks"])
    goals = Dict(
        goal => Dict(
            task => Set{String}(dependencies)
            for (task, dependencies) in tasks)
        for (goal, tasks) in data["goals"])
    return tasks, goals
end;

# %%
tasks, goals = load_goals("goals.json");

# %%
the_plot = plot_world(world, "Tasks")
for (task, geom) in tasks
    plot!(make_circle(geom...); label=task, seriestype=:shape)
end
the_plot

# %%
@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]
@dist labeled_categorical(labels, probs) = labels[categorical(probs)]
normalize_logprobs(lps) = lps .- logsumexp(lps);

@gen goals_prior(goals) = {:goal} ~ labeled_uniform(goals)
get_choices(simulate(goals_prior, (collect(keys(goals)),)))

# %% [markdown]
# ### Discretization

# %%
sort2(a, b) = (a <= b) ? (a, b) : (b, a)

function load_discretization(file_name)
    data = parsefile(file_name)
    rooms = Dict{String, Vector{Vector{Float64}}}(data["rooms"])
    doorways = Dict{Tuple{String, String}, Vector{Float64}}(sort2(doorway["rooms"]...) => Vector{Float64}(doorway["p"]) for doorway in data["doorways"])
    return rooms, doorways
end;

# %%
rooms, doorways = load_discretization("world_coarse.json");

# %%
the_plot = plot_world(world, "Discretization: rooms and doorways")
for (i, (name, ps)) in enumerate(rooms)
    plot!(first.(ps), last.(ps); seriestype=:shape, color=(i+1), label=nothing, markersize=3, markerstrokewidth=1)
    midpoint = sum(ps)/length(ps)
    annotate!(midpoint[1], midpoint[2], ("$name", :black))
end
plot!(first.(values(doorways)), last.(values(doorways)); seriestype=:scatter, color=:red, label=nothing, markersize=5, markerstrokewidth=1)
the_plot

# %%
# Assumes the polygon is simple (no self-crossings).
# Assumes `polygon` is an ordered list of points with first point repeated at the end.
function point_in_polygon(point, polygon; PARALLEL_TOL=1.0e-10)
    # Cast a ray from `point` in x-direction, and return whether ray intersects polygon exactly once.
    crossed = 0
    for (s1, s2) in zip(polygon[1:end-1], polygon[2:end])
        det = s2[2] - s1[2]
        s1p = (s1[1] - point[1], s1[2] - point[2])
        s2p = (s2[1] - point[1], s2[2] - point[2])
        if abs(det) < PARALLEL_TOL
            # If segment is parallel to x-direction, check whether point lies on segment.
            if (s1p[1] * s2p[1] <= 0) && (s1p[2] * s2p[2] <= 0); return true end
        else
            # Otherwise, test whether ray meets segment,
            # and increment/decrement count according to crossing orientation.
            s = det2(s1p, s2p) / det
            t = -s1p[2] / det
            if s >= 0. && (0 <= t <= 1.); crossed += (det > 0.) ? 1 : -1 end
        end
    end
    return crossed != 0
end

function locate_discrete(p, rooms, doorways; DOORWAY_RADIUS=1.0)
    for (name, ps) in rooms
        if point_in_polygon(p, ps); return name end
    end
    distance, door = findmin(v -> norm(p - v), doorways)
    return distance < DOORWAY_RADIUS ? door : nothing
end;

# %%
DOORWAY_RADIUS=1.0

ani = Animation()
for pose in some_poses
    frame_plot = plot_world(world, "Location discretization")
    location = locate_discrete(pose.p, rooms, doorways; DOORWAY_RADIUS=DOORWAY_RADIUS)
    if isnothing(location)
        annotate!(world.center_point..., ("Not located", :red))
    elseif isa(location, String)
        plot!(first.(rooms[location]), last.(rooms[location]); seriestype=:shape, color=:green3, label="room $location", markersize=3, markerstrokewidth=1, alpha=0.25)
    else
        plot!(make_circle(doorways[location], DOORWAY_RADIUS); label="doorway between rooms $(location[1]) and $(location[2])", seriestype=:shape, alpha=0.25)
    end
    plot!(pose; label="pose", color=:green)
    frame(ani, frame_plot)
end
gif(ani, "imgs/discretization.gif", fps=1)

# %% [markdown]
# ### Coarse planning

# %%
task_locations = Dict(
    task => locate_discrete(p, rooms, doorways)
    for (task, (p, _)) in tasks)

# %%
function location_to_room(location, destination_room, rooms, doorways)
    if isnothing(location)
        return nothing
    elseif isa(location, String)
        # Breadth-first search for path
        paths = [[location]]
        visited = Set((location,))
        while !isempty(paths)
            found = findfirst(path -> path[end] == destination_room, paths)
            if !isnothing(found); return paths[found] end
            new_paths = []
            for new_room in filter(!in(visited), keys(rooms))
                for path in filter(path -> sort2(path[end], new_room) in keys(doorways), paths)
                    push!(visited, new_room)
                    push!(new_paths, vcat(path, new_room))
                end
            end
            paths = new_paths
        end
        return nothing
    else
        paths = [location_to_room(branch, destination_room, rooms, doorways) for branch in location]
        return isnothing(paths[1]) && isnothing(paths[2]) ? nothing :
               isnothing(paths[1]) ? vcat(location, paths[2]) :
               isnothing(paths[2]) ? vcat(location, paths[1]) :
               length(paths[1]) <= length(paths[2]) ?
                    vcat(location, paths[1]) : vcat(location, paths[2])
    end
end;

# %%
location_to_room(task_locations["task1"], task_locations["task2"], rooms, doorways)

# %%
ani = Animation()
for pose in some_poses
    location = locate_discrete(pose.p, rooms, doorways)
    if isnothing(location)
        frame_plot = plot_world(world, "Location discretization")
        annotate!(world.center_point..., ("Not located", :red))
        plot!(pose; label="pose", color=:green)
        frame(ani, frame_plot)
        else
        for task in keys(tasks)
            frame_plot = plot_world(world, "Location discretization")
            path = location_to_room(location, task_locations[task], rooms, doorways)
            if isnothing(path)
                annotate!(world.center_point..., ("Routing fail", :red))
            else
                # annotate!(world.center_point..., ("$path", :blue))
                for node in path
                    if isa(node, String)
                        plot!(first.(rooms[node]), last.(rooms[node]); seriestype=:shape, color=:green3, label="room $node", markersize=3, markerstrokewidth=1, alpha=0.25)
                    else
                        plot!([doorways[node][1]], [doorways[node][2]]; seriestype=:scatter, color=:red, label="doorway", markersize=5, markerstrokewidth=1)
                    end
                end
            end
            plot!(make_circle(tasks[task]...); label=task, seriestype=:shape)
            plot!(pose; label="pose", color=:green)
            frame(ani, frame_plot)
        end
    end
end
gif(ani, "imgs/discretization.gif", fps=1)

# %%
