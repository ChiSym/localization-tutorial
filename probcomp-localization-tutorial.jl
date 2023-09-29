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
# # TO DO
#
# * Get a sense of the various sources of "synthetic" data.  First path is hard-wired.  Other paths and/or sensor data?
# * Consolidate (3?) MH proposals.  PF w/o Rejuv.  Consolidate PF+MH-Rejuvs.  PF+SMCP3-Rejuv.  Else?
# * Hierarchical (sensor) model?
# * Fix global vars in rejection_sample etc.
#
# Rif comments:
# * Correct understanding of initial pose.

# %%
# Fix for Jay's Jupytext setup
if occursin("sharlaon", pwd()); cd("/Users/sharlaon/dev/probcomp-localization-tutorial") end

# %% [markdown]
# # ProbComp Localization Tutorial
#
# Some introductory text.

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

import JSON

norm(v :: Vector{Float64}) = sqrt(sum(v.^2))

struct Segment
    p1 :: Vector{Float64}
    p2 :: Vector{Float64}
    dp :: Vector{Float64}
    Segment(p1 :: Vector{Float64}, p2 :: Vector{Float64}) = new(p1, p2, p2-p1)
end
Segment(t :: Tuple) = Segment(t...)
Base.show(io :: IO, s :: Segment) = print(io, "Segment($(s.p1), $(s.p2))")

struct Pose
    p  :: Vector{Float64}
    hd :: Float64
    dp :: Vector{Float64}
    Pose(p :: Vector{Float64}, hd :: Float64) = new(p, rem2pi(hd, RoundNearest), [cos(hd), sin(hd)])
end
Pose(p :: Vector{Float64}, dp :: Vector{Float64}) = Pose(p, atan(dp[2], dp[1]))
Pose(t :: Tuple) = Pose(t...)
Base.show(io :: IO, p :: Pose) = print(io, "Pose($(p.p), $(p.hd))")

step_along_pose(p :: Pose, s :: Float64) :: Vector{Float64} = p.p + s * p.dp
rotate_pose(p :: Pose, a :: Float64) :: Pose = Pose(p.p, p.hd + a)

Segment(p1 :: Pose, p2 :: Pose) = Segment(p1.p, p2.p)

# A value `c :: Control` corresponds to the robot *first* advancing in its present direction by `c.ds`, *then* rotating by `c.dhd`.
struct Control
    ds  :: Float64
    dhd :: Float64
end
Control(t :: Tuple) = Control(t...)

function create_segments(verts :: Vector{Vector{Float64}}; loop_around=false) :: Vector{Segment}
    segs = Segment.(zip(verts[1:end-1], verts[2:end]))
    if loop_around; push!(segs, Segment(verts[end],verts[1])) end
    return segs
end

function load_world(file_name)
    data = JSON.parsefile(file_name)
    walls_vec = Vector{Vector{Float64}}(data["wall_verts"])
    walls = create_segments(walls_vec)
    clutters_vec = Vector{Vector{Vector{Float64}}}(data["clutter_vert_groups"])
    clutters = create_segments.(clutters_vec)
    start_guess = Pose(Vector{Float64}(data["start_pose_guess"]["p"]), Float64(data["start_pose_guess"]["hd"]))
    controls = Vector{Control}([Control(control["ds"], control["dhd"]) for control in data["program_controls"]])
    all_points = [walls_vec ; clutters_vec... ; [start_guess.p]]
    x_min = minimum(p[1] for p in all_points)
    x_max = maximum(p[1] for p in all_points)
    y_min = minimum(p[2] for p in all_points)
    y_max = maximum(p[2] for p in all_points)
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    T = length(controls)
    return ((walls=walls, clutters=clutters, bounding_box=bounding_box, box_size=box_size),
            (start_guess=start_guess, controls=controls),
            T)
end;

# %%
# Specific example code here

world, robot_inputs, T = load_world("example_20_program.json");

# %% [markdown]
# ### Integrate a path from a starting pose and controls
#
# If the motion of the robot is determined in an ideal manner by the controls, then we may simply integrate to determine the resulting path.  Naïvely, this results in the following.

# %%
"""
Assumes
* `robot_inputs` contains fields: `start_guess`, `controls`
"""
function integrate_controls_unphysical(robot_inputs :: NamedTuple) :: Vector{Pose}
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = robot_inputs.start_guess
    for t in 1:length(robot_inputs.controls)
        p = path[t].p + robot_inputs.controls[t].ds * path[t].dp
        hd = path[t].hd + robot_inputs.controls[t].dhd
        path[t+1] = Pose(p, hd)
    end
    return path
end;

# %% [markdown]
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance in the normal direction to the point of contact.

# %%
# Return unique s, t such that p + s*u == q + t*v.
function solve_lines(p :: Vector{Float64}, u :: Vector{Float64}, q :: Vector{Float64}, v :: Vector{Float64}; PARALLEL_TOL=1.0e-10)
    det = u[1] * v[2] - u[2] * v[1]
    if abs(det) < PARALLEL_TOL
        return nothing, nothing
    else
        s = (v[1] * (p[2]-q[2]) - v[2] * (p[1]-q[1])) / det
        t = (u[2] * (q[1]-p[1]) - u[1] * (q[2]-p[2])) / det
        return s, t
    end
end

function distance(p :: Pose, seg :: Segment) :: Float64
    s, t = solve_lines(p.p, p.dp, seg.p1, seg.dp)
    # Solving failed (including, by fiat, if pose is parallel to segment) iff isnothing(s).
    # Pose is oriented away from segment iff s < 0.
    # Point of intersection lies on segment (as opposed to the infinite line) iff 0 <= t <= 1.
    return (isnothing(s) || s < 0. || !(0. <= t <= 1.)) ? Inf : s
end

"""
Assumes
* `world_inputs` contains fields: `walls`, `bounce`
"""
function physical_step(p1 :: Vector{Float64}, p2 :: Vector{Float64}, hd :: Float64, world_inputs :: NamedTuple) :: Pose
    step_pose = Pose(p1, p2 - p1)
    (s, i) = findmin(w -> distance(step_pose, w), world_inputs.walls)
    if s > norm(p2 - p1)
        # Step succeeds without contact with walls.
        return Pose(p2, hd)
    else
        contact_point = p1 + s * step_pose.dp
        unit_tangent = world_inputs.walls[i].dp / norm(world_inputs.walls[i].dp)
        unit_normal = [-unit_tangent[2], unit_tangent[1]]
        # Sign of 2D cross product determines orientation of bounce.
        if step_pose.dp[1] * world_inputs.walls[i].dp[2] - step_pose.dp[2] * world_inputs.walls[i].dp[1] < 0.
            unit_normal = -unit_normal
        end
        return Pose(contact_point + world_inputs.bounce * unit_normal, hd)
    end
end

"""
Assumes
* `robot_inputs` contains fields: `start_guess`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
"""
function integrate_controls(robot_inputs :: NamedTuple, world_inputs :: NamedTuple)
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = robot_inputs.start_guess
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

path_ideal = integrate_controls(robot_inputs, world_inputs);

# %% [markdown]
# ### Plot such data

# %%
using Plots

function plot_list(list; label=nothing, args...)
    if isempty(list); return end
    plt = plot!(list[1]; label=label, args...)
    for item in list[2:end]; plot!(item; label=nothing, args...) end
    return plt
end

Plots.plot!(seg :: Segment; args...) = plot!([seg.p1[1], seg.p2[1]], [seg.p1[2], seg.p2[2]]; args...)
Plots.plot!(segs :: Vector{Segment}; args...) = plot_list(segs; args...)
Plots.plot!(seg_groups :: Vector{Vector{Segment}}; args...) = plot_list(seg_groups; args...)

Plots.plot!(p :: Pose; r=0.5, args...) = plot!(Segment(p.p, step_along_pose(p, r)); arrow=true, args...)
Plots.plot!(ps :: Vector{Pose}; args...) = plot_list(ps; args...)

function start_plot(world, title; label_world=false, show_clutters=true)
    border = world.box_size * (3.)/19.
    the_plot = plot(
        size         = (500, 500),
        aspect_ratio = :equal,
        grid         = false,
        xlim         = (world.bounding_box[1]-border, world.bounding_box[2]+border),
        ylim         = (world.bounding_box[3]-border, world.bounding_box[4]+border),
        title        = title,
        legend       = :bottomleft)
    (walls_label, clutter_label) = label_world ? ("walls", "clutter") : (nothing, nothing)
    plot!(world.walls; c=:black, label=walls_label)
    if show_clutters; plot!(world.clutters; c=:magenta, label=clutter_label) end
    return the_plot
end;

# %%
the_plot = start_plot(world, "Given data", label_world=true)
plot!(robot_inputs.start_guess; label="start pose guess", color=:green2)
plot!([pose.p[1] for pose in path_ideal], [pose.p[2] for pose in path_ideal];
      label="ideal program path", color=:green3, seriestype=:scatter, markersize=3, markerstrokewidth=0)
savefig("imgs/given_data")
the_plot

# %% [markdown]
# ## Modeling motion in Gen
#
# Now we specify a model.
#
# Each piece of the model is declared as a *generative function*, prefaced by the `@gen` construct.

# %%
using Gen;

# %% [markdown]
# ### Pose prior model
#
# To model the possible trajectories of the robot, we begin by modeling our uncertainty in where it started.

# %%
"""
Assumes
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
@gen (static) function pose_prior_model(start_guess :: Pose, motion_settings :: NamedTuple) :: Pose
    p ~ mvnormal(start_guess.p, motion_settings.p_noise * [1 0 ; 0 1])
    hd ~ normal(start_guess.hd, motion_settings.hd_noise)
    return Pose(p, hd)
end

# This call is required by Gen's static DSL in order to invoke `pose_prior_model` below.
load_generated_functions()

# %% [markdown]
# We can just call `pose_prior_model` like a normal function and it will just run stochastically.

# %%
motion_settings = (p_noise = 0.5, hd_noise = 2π / 360)

N_samples = 50
pose_samples = [pose_prior_model(robot_inputs.start_guess, motion_settings) for _ in 1:N_samples]

the_plot = start_plot(world, "Pose prior model (samples)"; show_clutters=false)
plot!(pose_samples; label=nothing, color=:red)
savefig("imgs/pose_prior")
the_plot

# %% [markdown]
# We can also perform *traced execution* using the construct `Gen.simulate`, and inspect the stochastic choices performed with the `~` operator.

# %%
trace = simulate(pose_prior_model, (robot_inputs.start_guess, motion_settings))
get_choices(trace)

# %% [markdown]
# ### Motion model
#
# The motion is modeled as updating the pose in response to controls of the program, but in effect only up to some error.

# %%
"""
Assumes
* `world_inputs` contains fields: `walls`, `bounce`
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
@gen (static) function motion_model(start :: Pose, c :: Control, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Pose
    p ~ mvnormal(start.p + c.ds * start.dp, motion_settings.p_noise * [1 0 ; 0 1])
    hd ~ normal(start.hd + c.dhd, motion_settings.hd_noise)
    return physical_step(start.p, p, hd, world_inputs)
end

@load_generated_functions()

"""
Assumes
* `robot_inputs` contains fields: `start_guess`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
function integrate_controls_noisy(robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Vector{Pose}
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = robot_inputs.start_guess
    for t in 1:length(robot_inputs.controls)
        path[t+1] = motion_model(path[t], robot_inputs.controls[t], world_inputs, motion_settings)
    end
    return path
end;

# %%
scaled_motion_settings(settings, x) = (p_noise = x * settings.p_noise, hd_noise = x * settings.hd_noise)

N_samples = 15

ani = Animation()
for n in 1:N_samples
    scale = 16. * (2.)^(n-N_samples)
    frame_plot = start_plot(world, "Motion model (samples)\nnoise factor $(round(scale, digits=3))"; show_clutters=false)
    sample_motion = integrate_controls_noisy(robot_inputs, world_inputs, scaled_motion_settings(motion_settings, scale))
    plot!(sample_motion; color=:brown)
    frame(ani, frame_plot)
end
gif(ani, "imgs/motion.gif", fps=1)

# %%
trace = simulate(motion_model, (robot_inputs.start_guess, robot_inputs.controls[1], world_inputs, motion_settings))
get_choices(trace)

# %% [markdown]
# ### Synthetic motion data
#
# Let us generate some fixed synthetic motion data that, for pedagogical purposes, we will work with as if it were the actual path of the robot.

# %%
# # Generate a path by adding noise:

motion_settings_synthetic = (p_noise = 0.05, hd_noise = 2π / 360)

start_actual = pose_prior_model(robot_inputs.start_guess, motion_settings_synthetic)
# path_actual = integrate_controls_noisy((robot_inputs..., start_guess=start_actual), world_inputs, motion_settings_synthetic)
# repr(path_actual) |> clipboard
# Here, we will use a hard-coded one we generated earlier that we selected to more clearly illustrate
# the main ideas in the notebook.
path_actual = Pose[Pose([1.8105055257302352, 16.95308477268976], 0.08768023894197674), Pose([3.80905621762144, 17.075619417709827], -0.5290211691806687), Pose([4.901118854352547, 16.374655088848304], -0.4554764850547685), Pose([6.308254748808569, 15.860770355551818], 0.05551953564181333), Pose([6.491438805390425, 15.493868458696895], -0.5802542842551736), Pose([7.447278355948555, 14.63103882275873], -1.315938749141227), Pose([7.434195388758904, 13.887476796022026], -1.515750524264586), Pose([7.045563974694356, 13.539511976225148], -1.3226432715239562), Pose([7.755917122113763, 12.118889998110918], -1.1875170980293068), Pose([8.031624143251104, 11.095208641644854], -0.38287120113753326), Pose([8.345690304200131, 10.843957790912832], -0.31488971003874827), Pose([8.971822052978622, 10.580306565768808], -0.0855234941283848), Pose([10.228980988810147, 10.430017431253829], -0.05160460191130738), Pose([11.337251889505731, 10.10090883752962], -0.025335824641921776), Pose([12.82024096259476, 9.81017583656567], 0.20336314833906002), Pose([13.658185429388778, 10.048753805232767], 1.4040405665068887), Pose([13.838175614976866, 10.788813324304678], 1.3842380063444915), Pose([14.384659102337947, 11.8750750875864], 0.9943086776465678), Pose([14.996345006995664, 12.681411208177314], 1.0223226390004532), Pose([15.226334529348852, 13.347705702094283], 1.017840325933929)]

the_plot = start_plot(world, "Actual motion deviates from ideal", label_world=false)
plot!(path_ideal; label="ideal path", color=:green2)
plot!(path_actual; label="\"actual\" robot path", color=:brown)
savefig("imgs/deviation")
the_plot

# %% [markdown]
# ## Observing with sensors
#
# We assume the robot is equipped with sensors that cast rays upon the environment at certain angles relative to the given pose, and return the distance to a hit.

# %% [markdown]
# ### Ideal sensors
#
# Ideally, there are true distances to the walls, to be sensed along the true path.
#
# Starting now, and for a while, we *suppress* the effects of clutter, which the robot is unaware of anyway, and assume the robot is in the environment ideally described by the map.

# %%
function sensor_distance(p :: Pose, walls :: Vector{Segment}, box_size :: Float64) :: Float64
    d = minimum(distance(p, s) for s in walls)
    # Capping to a finite value avoids issues below.
    return isinf(d) ? 2. * box_size : d
end;

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`
"""
sensor_angle(sensor_settings :: NamedTuple, j :: Int64) =
    sensor_settings.fov * (j - sensor_settings.num_angles) / (2. * sensor_settings.num_angles)

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
function ideal_sensor(p :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple) :: Vector{Float64}
    readings = Vector{Float64}(undef, 2 * sensor_settings.num_angles + 1)
    for j in 1:(2 * sensor_settings.num_angles + 1)
        sensor_pose = rotate_pose(p, sensor_angle(sensor_settings, j))
        readings[j] = sensor_distance(sensor_pose, walls, sensor_settings.box_size)
    end
    return readings
end

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
project_readings(p :: Pose, readings :: Vector{Float64}, sensor_settings :: NamedTuple) :: Vector{Vector{Float64}} =
    [step_along_pose(rotate_pose(p, sensor_angle(sensor_settings, j)), s) for (j, s) in enumerate(readings)]

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
function plot_sensors(world, title, path_or_paths, sensor_settings; show_clutters=true)
    the_plot = start_plot(world, title; show_clutters=show_clutters)
    paths = isa(path_or_paths, Tuple) ? [path_or_paths] : path_or_paths
    for (path, path_label, color, p, readings, readings_label) in paths
        plot!(path; label=path_label, color=color)
        if !isnothing(p)
            plot!([p.p[1]], [p.p[2]]; label=nothing, color=color, seriestype=:scatter, markersize=3, markerstrokewidth=0)
            projections = project_readings(p, readings, sensor_settings)
            plot!(first.(projections), last.(projections);
                  label=readings_label, color=:blue, seriestype=:scatter, markersize=3, markerstrokewidth=1, alpha=0.25)
            plot!([Segment(p.p, pr) for pr in projections]; label=nothing, color=:blue, alpha=0.25)
      end
    end
    return the_plot
end;

# %%
sensor_settings = (fov = 2π*(2/3), num_angles = 20, box_size = world.box_size)

ani = Animation()
for p in path_ideal
    frame_plot = plot_sensors(world, "Ideal sensor distances",
        (path_ideal, "ideal path", :green2, p, ideal_sensor(p, world.walls, sensor_settings), "ideal sensors"),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/ideal_distances.gif", fps=1)

# %% [markdown]
# ### Noisy sensors
#
# We may also assume the sensors only guarantee results within a certain *tolerance*, leading to noisy readings.

# %%
"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
function noisy_sensor(p :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple, tol :: Float64) :: Vector{Float64}
    readings = Vector{Float64}(undef, 2 * sensor_settings.num_angles + 1)
    for j in 1:(2 * sensor_settings.num_angles + 1)
        sensor_pose = rotate_pose(p, sensor_angle(sensor_settings, j))
        # Uniformly choose in the interval [distance - tol, distance + tol].
        readings[j] = sensor_distance(sensor_pose, walls, sensor_settings.box_size) + (2.0*tol*rand() - tol)
    end
    return readings
end;

# %% [markdown]
# Let us generate some fixed synthetic sensor data that, for pedagogical purposes, we will work with as if it were the actual sensor data of the robot.

# %%
tol = 0.2
observations = [noisy_sensor(p, world.walls, sensor_settings, tol) for p in path_actual]

ani = Animation()
for (p, readings) in zip(path_actual, observations)
    frame_plot = plot_sensors(world, "Noisy sensor distances",
        (path_actual, "actual path", :brown, p, readings, "noisy sensors"),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances.gif", fps=1)

# %% [markdown]
# ### The essential discrepancy
#
# The following animation summarizes the information available to the robot: its initial pose, its program, and the observed sensor readings.

# %%
ani = Animation()
for (p, readings) in zip(path_ideal, observations)
    frame_plot = plot_sensors(world, "Expected path vs. sensors",
        (path_ideal, "ideal path", :green2, p, readings, "sensors from \"actual\" path"),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/discrepancy.gif", fps=1)

# %% [markdown]
# ***The sensor data and the path visibly do not cohere!***  This is to be expected, because the robot's true path deviates from the ideal one integrated from the initial pose and program.

# %% [markdown]
# ## Modeling the sensors
#

# %% [markdown]
# ### Sensor model
#
# Suppose we do not know the details of the sensor, for example its tolerance behavior.  We simply model the readings as approximately correct, using Gaussian noise.

# %%
"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen function sensor_model_1(p :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple) :: Vector{Float64}
    readings = Vector{Float64}(undef, 2 * sensor_settings.num_angles + 1)
    for j in 1:(2 * sensor_settings.num_angles + 1)
        sensor_pose = rotate_pose(p, sensor_angle(sensor_settings, j))
        readings[j] = {j => :distance} ~ normal(sensor_distance(sensor_pose, walls, sensor_settings.box_size), sensor_settings.s_noise)
    end
    return readings
end;

# %% [markdown]
# The result does not appear much different from our noisy sensor distances, plotted over the path from which they were measured.

# %%
sensor_settings = (sensor_settings..., s_noise = 0.05)

ani = Animation()
for p in path_ideal
    frame_plot = plot_sensors(world, "Sensor model (samples)",
        (path_ideal, "ideal path", :green2, p, sensor_model_1(p, world.walls, sensor_settings), "synthetic sensor readings"),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/sensor_1.gif", fps=1)

# %% [markdown]
# The trace contains many choices corresponding to directions of sensor reading from the input pose.  To reduce notebook clutter, here we just show a subset of 5 of them:

# %%
trace = simulate(sensor_model_1, (robot_inputs.start_guess, world.walls, sensor_settings))

get_selected(get_choices(trace), select((1:5)...))

# %% [markdown]
# ### Full model
#
# We connect the pieces into a full model.  There are two ways of expressing this same functionality.  The first uses an explicit loop:

# %%
"""
Assumes
* `robot_inputs` contains fields: `start_guess`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen function full_model_1_loop(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, full_settings :: NamedTuple) :: Vector{Float64}
    sensor_readings = Vector{Vector{Float64}}(undef, length(robot_inputs.controls) + 1)

    pose = {:initial => :pose} ~ pose_prior_model(robot_inputs.start_guess, full_settings.motion_settings)
    sensor_readings[1] = {:initial => :sensor} ~ sensor_model_1(pose, world_inputs.walls, full_settings.sensor_settings)

    for t in 1:T
        pose = {:steps => t => :pose} ~ motion_model(pose, robot_inputs.controls[t], world_inputs, full_settings.motion_settings)
        sensor_readings[t+1] = {:steps => t => :sensor} ~ sensor_model_1(pose, world_inputs.walls, full_settings.sensor_settings)
    end

    return sensor_readings
end

# Handle asymmetry in trace addresses.
prefix_address(t :: Int, rest) :: Pair = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest);

# %% [markdown]
# This _generative function_ defines a probability distribution over _traces_.  Each _trace_ is a data structure containing a sequence of robot pose values, and a sequence of observations captured by the sensor.
#
# Notation:
# - $\textbf{z}_t$ = robot pose at time $t$
# - $\textbf{o}_t$ = observed robot sensor measurement at time $t$
# - $\textbf{z}_{0:T}$ = $[\textbf{z}_0, \textbf{z}_1, \dots, \textbf{z}_T]$
# - $\textbf{o}_{0:T}$ = $[\textbf{o}_0, \textbf{o}_1, \dots, \textbf{o}_T]$
#
# So a trace of this generative function is a data structure containing the pair $(\textbf{z}_{0:T}, \textbf{o}_{0:T})$.
#
# The probability distribution on this trace is defined using the following components:
# 1. $P_{\text{init}}(\textbf{z}_0)$ (`pose_prior_model`)
# 2. $P_{\text{step}}(\textbf{z}_t ; \textbf{z}_{t-1})$ (`motion_model`)
# 3. $P_{\text{obs}}(\textbf{o}_t ; \textbf{z}_t)$ (`sensor_model_1`)
#
# Let $P_\text{full}$ denote the distribution defined by `full_model_1_loop`.  The code above declares that the distribution is
# $$
# P_\text{full}(\textbf{z}_{0:T}, \textbf{o}_{0:T}) = P_{\text{init}}(\textbf{z}_0) P_{\text{obs}}(\textbf{o}_0 ; \textbf{z}_0) \prod_{t=1}^T{[P_{\text{step}}(\textbf{z}_t ; \textbf{z}_{t-1}) P_{\text{obs}}(\textbf{o}_t ; \textbf{z}_t)]}
# $$
#
# The `for` loop in the code implements the product in the math.

# %% [markdown]
# The second recognizes the model as a Markov chain, and accordingly invokes the `Unfold` combinator to capture this structure.

# %%
# The staging of `full_model_1` into these subfunctions is required for Gen's static DSL.

"""
Assumes
* `robot_inputs` contains fields: `start_guess`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function model_1_initial(robot_inputs :: NamedTuple, walls :: Vector{Segment}, full_settings :: NamedTuple)
    pose ~ pose_prior_model(robot_inputs.start_guess, full_settings.motion_settings)
    sensor ~ sensor_model_1(pose, walls, full_settings.sensor_settings)
    return (pose, sensor)
end

"""
Assumes
* `robot_inputs` contains fields: `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function model_1_kernel(t :: Int, state :: Tuple{Pose, Vector{Float64}}, robot_inputs :: NamedTuple,
                                      world_inputs :: NamedTuple, full_settings :: NamedTuple) :: Tuple{Pose, Vector{Float64}}
    pose ~ motion_model(state[1], robot_inputs.controls[t], world_inputs, full_settings.motion_settings)
    sensor ~ sensor_model_1(pose, world_inputs.walls, full_settings.sensor_settings)
    return (pose, sensor)
end
model_1_chain = Unfold(model_1_kernel)

combine_sensors(initial, steps) = [last(initial), last.(steps)...]

"""
Assumes
* `robot_inputs` contains fields: `start_guess`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function full_model_1(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, full_settings :: NamedTuple) :: Vector{Vector{Float64}}
    initial ~ model_1_initial(robot_inputs, world_inputs.walls, full_settings)
    steps ~ model_1_chain(T, initial, robot_inputs, world_inputs, full_settings)
    return combine_sensors(initial, steps)
end

@load_generated_functions()

# %% [markdown]
# We work with the `Unfold`-variant, because the static DSL can harness it for efficient evaluation in `Gen.update`.  The user can check that the explicit loop produces identical outputs.
#
# We note, by the way, the introduction of the parameter `T`, used to truncate the generation of steps to lengths less than `length(robot_inputs.controls)`, which will be helpful when writing SMC code below.

# %%
full_settings = (motion_settings=motion_settings, sensor_settings=sensor_settings)
scaled_full_settings(settings, x) = (settings..., motion_settings = scaled_motion_settings(settings.motion_settings, x))

N_samples = 10

ani = Animation()
for n in 1:N_samples
    scale = (2.)^(2+(n-N_samples))
    trace = simulate(full_model_1, (T, robot_inputs, world_inputs, scaled_full_settings(full_settings, scale)))
    poses = [trace[prefix_address(t, :pose)] for t in 1:(T+1)]
    for t in 1:(T+1)
        frame_plot = plot_sensors(world, "Full model (samples)\nnoise factor $(round(scale, digits=3))",
            [(path_actual, "actual path", :brown, nothing, nothing, nothing),
             (poses, "trace", :green, poses[t], trace[prefix_address(t, :sensor)], nothing)],
            sensor_settings; show_clutters=false)
        frame(ani, frame_plot)
    end
end
gif(ani, "imgs/full_1.gif", fps=5)

# %% [markdown]
# Again, the trace of the full model contains many choices, so we just show a subset of them: the initial pose plus 2 timesteps, and 5 sensor readings from each.

# %%
full_model_args = (robot_inputs, world_inputs, full_settings)

trace = simulate(full_model_1, (T, full_model_args...))
selection = select((prefix_address(t, :pose) for t in 1:3)..., (prefix_address(t, :sensor => j) for t in 1:3, j in 1:5)...)
get_selected(get_choices(trace), selection)

# %% [markdown]
# ## Inference: main idea
#
# In the viewpoint of ProbComp, the goal of *inference* is to produce *likely* traces of a full model, given the observed data.  In other words, *generative functions induce distributions on traces*, and if we view the full model as a program embodying a *prior*, then applying an inference metaprogram to it (together with the observed data) produces a new program that embodies the *posterior*.
#
# There is no free lunch in this game: generic inference recipies are inefficient.  Rather, efficiency becomes possible as we exploit what we actually know about the problem in our design of the inference strategy.  Gen's aim is to provide the right entry points to enact this exploitation.
#
# To picture this, here are the paths produced by the model in aggregate, to get an approximate sense of the prior as a distribution.  Next to it is a similar picture of what the posterior, our goal, looks like.  (The inference process is a *black box* for this moment, and will be approached through the rest of this notebook.)

# %%
poses_to_coords(poses :: Vector{Pose}) :: Vector{Vector{Float64}} = [[p.p[1] for p in poses], [p.p[2] for p in poses]]

function frame_from_traces(world, traces, T, title; show_clutters=true, path_actual=nothing)
    the_plot = start_plot(world, title; show_clutters=show_clutters)
    if !isnothing(path_actual); plot!(path_actual; label="actual path", color=:brown) end
    for trace in traces
        poses = [trace[prefix_address(t, :pose)] for t in 1:(T+1)]
        plot!(poses_to_coords(poses)...; label=nothing, color=:green, alpha=0.3)
        plot!(Segment.(zip(poses[1:end-1], poses[2:end]));
              label=nothing, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=0.3)
    end
    return the_plot
end;

# %%
# The code in this cell is the black box!

# Encode constraints into choice map.

function constraint_from_sensor_reading(cm :: ChoiceMap, t :: Int, sensor_reading :: Vector{Float64}) :: ChoiceMap
    for (j, reading) in enumerate(sensor_reading)
        cm[prefix_address(t, :sensor => j => :distance)] = reading
    end
    return cm
end

function constraints_from_sensor_readings(cm :: ChoiceMap, sensor_readings :: Vector{Vector{Float64}}) :: ChoiceMap
    for (t, sensor_reading) in enumerate(sensor_readings)
        constraint_from_sensor_reading(cm, t, sensor_reading)
    end
    return cm
end

# Propose a move for MH.

@gen function drift_proposal(trace, drift_step_factor)
    t = get_args(trace)[1] + 1

    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    {prefix_address(t, :pose => :p)} ~ mvnormal(p, drift_step_factor * p_noise * [1 0 ; 0 1])
    {prefix_address(t, :pose => :hd)} ~ normal(hd, hd_noise)
end

# PF with rejuvenation, using library code for the generic parts.

using GenParticleFilters

function particle_filter_rejuv_library(model, T, args, observations, N_particles, N_MH, MH_proposal, MH_proposal_args)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]
    state = pf_initialize(model, (0, args...), constraints[1], N_particles)
    for t in 1:T
        pf_resample!(state)
        pf_rejuvenate!(state, mh, (MH_proposal, MH_proposal_args), N_MH)
        pf_update!(state, (t, args...), (UnknownChange(),), constraints[t+1])
    end
    return state.traces, state.log_weights
end

# Run PF and return one of its particles.

function sample_from_posterior(model, T, args, observations; N_MH = 10, N_particles = 10)
    drift_step_factor = 1/3.
    traces, _ = particle_filter_rejuv_library(model, T, args, observations, N_particles, N_MH, drift_proposal, (drift_step_factor,))
    return traces[1]
end;

# %%
constraints = constraints_from_sensor_readings(choicemap(), observations);

# %%
N_samples = 10

traces = [simulate(full_model_1, (T, full_model_args...)) for _ in 1:N_samples]
prior_plot = frame_from_traces(world, traces, T, "Prior on robot paths";
                               path_actual=path_actual, show_clutters=false)

traces = [sample_from_posterior(full_model_1, T, full_model_args, observations; N_MH=10, N_particles=10) for _ in 1:N_samples]
posterior_plot = frame_from_traces(world, traces, T, "Posterior on robot paths";
                                   path_actual=path_actual, show_clutters=false)

the_plot = plot(prior_plot, posterior_plot, size=(1000,500))
savefig("imgs/prior_posterior")
the_plot

# %% [markdown]
# Mathematically, we describe inference as follows.
#
# Recall that our `model_full_1` generative function defined a probability distribution $P_\text{full}(\textbf{z}_{0:T}, \textbf{o}_{0:T})$.
#
# When the robot actually travels through the world, its sensors pick up a collection of measurements $\textbf{o}_{0:T}^*$.
#
# The goal of inference is to characterize the _posterior_ probability distribution
# $
# P_\text{full}(\textbf{z}_{0:T} | \textbf{o}^*_{0:T})
# $.
# This is the probability distribution defined such that
# $$
# P_\text{full}(\textbf{z}_{0:T} | \textbf{o}_{0:T}) := \frac{
#     P_\text{full}(\textbf{z}_{0:T}, \textbf{o}_{0:T})
# }{
#     P_\text{full}(\textbf{o}_{0:T})
# }
# $$
#
# The posterior distribution is a distribution over the space of latent trajectories $\textbf{z}_{0:T}$.  The posterior distribution answers the question: after observing observation sequence $\textbf{o}^*_{0:T}$, what would an optimal agent _believe_ about the latent trajectory?  The answer is: its belief is characterized by the distribution $P_\text{full}(\textbf{z}_{0:T} | \textbf{o}^*_{0:T})$.
#
# In the definition of the posterior distribution, we used the term $P_\text{full}(\textbf{o}_{0:T})$.  This is the _marginal distribution over the observations_.  This is defined by
# $$
# P_\text{full}(\textbf{o}_{0:T}) := \int P_\text{full}(\textbf{z}_{0:T}, \textbf{o}_{0:T}) d\textbf{z}_{0:T}
# $$
# It describes the agent's prior belief about how likely a given observation sequence is to occur, given the controls that are used.

# %% [markdown]
# ## Strategies for inference
#
# We now spell out some strategies for conditioning the ouputs of our model towards the observed sensor data.  These metaprograms employ the following richer aspects of generative functions.
#
# ### Deeper functionality of GFs
#
# Rif asks: maybe these discussions are too simplified?
#
# Traced execution of generative functions, via `Gen.simulate` as seen above, is a straightforward alternative semantic interpretation.  A more refined operation, `Gen.generate`, allows two deeper features.
#
# 1. It proposes traces satisfying optional *constraints*.  
# Note that the `get_choices(trace)` returns a tree with nodes at all invokations of `~` to sample from generative functions.  The *leaf* nodes correspond to *primitive choices* coming from honest *distributions*.  Presently, Gen only works with constraints upon these *primitive* choices.  The constraints are recorded in structures called *choice maps*, as modified in the black box code cell above.
#
# 2. It returns the proposed trace along with a *(log) weight*, which we then use in our inference strategies.  
# The meaning of this weight is complex:  The trace proposal code need not produce traces with frequencies according to the "actual" distribution represented by the function—it is allowed to do something simpler and more computationally efficient.  The weight represents any difference between the proposal frequencies and the "actual" frequencies, plus contributions arising from any imposed constraints.
#
# Along the same lines, given a trace, one may perform simple modifications of it using `Gen.update` without rerunning the entire GF, and it returns along with the new trace the difference in weights.

# %% [markdown]
# ### Sampling / importance resampling
#
# Let's try a classic generic inference strategy, *sampling / importance resampling (SIR)*.  The idea is to independently draw a number of samples with weights, called *particles*, in order to explore the space of the distribution, and then to select one of them in proportion to its weight as a representative.

# %% [markdown]
# **Sample generation.**
#
# Specifically, this algorithm will generate $N$ possible latent trajectories:
# $$\textbf{z}_{0:T}^i \sim P_\text{trajectory\_prior} \text{ for } i=1, 2, \dots, N$$
#
# Here, $P_\text{trajectory\_prior}(\textbf{z}_{0:T}) := P_{\text{pose}_0}(\textbf{z}_0) \prod_{t=1}^T{P_\text{step}(\textbf{z}_t ; \textbf{z}_{t-1})}$.
#
# Note that these trajectories are generated entirely without considering the robot's observations $\textbf{o}_{0:T}^*$.
#
# **Weight computation.**
#
# After generating $N$ trajectories, SIR computes the following _weight_, $w^i$, for each sample $\textbf{z}_{0:T}^i$:
# $$
# w^i := \frac{
# P_\text{full}(\textbf{z}^i_{0:T}, \textbf{o}_{0:T})
# }{
# P_\text{trajectory\_prior}(\textbf{z}^i_{0:T})
# }
# $$
#
# This $w^i$ will be large for samples $\textbf{z}^i_{0:T}$ which seem consistent with the observations $\textbf{o}_{0:T}$, since then $P_\text{full}(\textbf{z}^i_{0:T}, \textbf{o}_{0:T})$ will be large.
#
# **Weight normalization.**
#
# After computing the $w^i$, SIR computes _normalized_ weights,
# $$
# \hat{w}^i = \frac{w^i}{\sum_{j=1}^N{w^j}}
# $$
# Note that $[\hat{w}^1, \hat{w}^2, \dots, \hat{w}^N]$ is a probability distribution.
#
# **Resampling.**
#
# The last step of SIR is to _resample_ $M$ of the original $N$ particles.  That is, SIR will choose $M$ of the $N$ original samples $\textbf{z}_{0:T}^i$, which appear consistent with the observations.  It does this by being more likely to choose samples with high $w^i$ values.
#
# Specifically, resampling first chooses $M$ particle indices, $i_1, \dots, i_M$, according to
# $$
# \forall k, i_k \sim \text{categorical}([\hat{w}^1, \hat{w}^2, \dots, \hat{w}^N])
# $$
# Put another way, $P(i_k = j) = \hat{w}^j$.
#
# Finally, SIR outputs the collection of trajectories $\textbf{z}^{i_1}_{0:T}, \textbf{z}^{i_2}_{0:T}, \dots, \textbf{z}^{i_M}_{0:T}$.
#
# **Summary:** SIR generates possible samples without considering the observations $\textbf{o}_{0:T}^*$, but attempts to ultimately output a sub-collection of these randomly generated samples which are consistent with the observations.  It does this by computing the weights $w^i$.

# %%
function basic_SIR(model, args, observations, N_SIR)
    traces = Vector{Trace}(undef, N_SIR)
    log_weights = Vector{Float64}(undef, N_SIR)
    for i in 1:N_SIR
        traces[i], log_weights[i] = generate(model, args, observations)
    end
    weights = exp.(log_weights .- maximum(log_weights))
    weights = weights ./ sum(weights)
    index = categorical(weights)
    return traces[index], weights[index]
end

# This is a generic algorithm, so there is a library version.
# We will the library version use going forward, because it includes a constant-memory optimization.
# (It is not necessary to store all particles and categorically select one at the end.  Mathematically
# it amounts to the same instead to store just one candidate selection, and stochastically replace it
# with each newly generated particle with odds the latter's weight relative to the sum of the
# preceding weights.)
# To obtain the above from the library version, one would define:

basic_SIR_library(model, args, observations, N_SIR) = importance_resampling(model, args, observations, N_SIR);

# %% [markdown]
# Let us first consider a shorter robot path, but, to keep it interesting, allow a higher deviation from the ideal.

# %%
T_short = 4
robot_inputs_short = (start_guess = robot_inputs.start_guess, controls = robot_inputs.controls[1:T_short])
path_ideal_short = integrate_controls(robot_inputs_short, world_inputs)

start_actual_short = pose_prior_model(robot_inputs_short.start_guess, motion_settings_synthetic)
path_actual_short = integrate_controls_noisy((robot_inputs_short..., start_guess=start_actual_short), world_inputs, motion_settings_synthetic)

the_plot = start_plot(world, "Shorter path", label_world=false)
plot!(path_ideal_short; label="ideal path", color=:green2)
plot!(path_actual_short; label="shorter \"actual\" robot path", color=:brown)
savefig("imgs/deviation_short")
the_plot

# %%
tol = 0.2
observations_short = [noisy_sensor(p, world.walls, sensor_settings, tol) for p in path_actual_short]

ani = Animation()
for (p, readings) in zip(path_actual_short, observations_short)
    frame_plot = plot_sensors(world, "Noisy sensor distances",
        (path_actual_short, "shorter actual path", :brown, p, readings, nothing),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances_short.gif", fps=1)

# %%
ani = Animation()
for (p, readings) in zip(path_ideal_short, observations_short)
    frame_plot = plot_sensors(world, "Expected path vs. sensors",
        (path_ideal_short, "ideal path", :green2, p, readings, "sensors from shorter \"actual\" path"),
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/discrepancy_short.gif", fps=1)

# %% [markdown]
# For such a shorter path, SIR can find a somewhat noisy fit without too much effort.
#
# Rif asks
# > In `traces = ...` below, are you running SIR `N_SAMPLES` times and getting one sample each time? Why not run it once and get `N_SAMPLES`? Talk about this?

# %%
full_model_args_short = (robot_inputs_short, world_inputs, full_settings)
constraints_short = constraints_from_sensor_readings(choicemap(), observations_short)

N_samples = 10
N_SIR = 500
traces = [basic_SIR_library(full_model_1, (T_short, full_model_args_short...), constraints_short, N_SIR)[1] for _ in 1:N_samples]

the_plot = frame_from_traces(world, traces, T_short, "SIR (short path)";
                             show_clutters=false, path_actual=path_actual_short)
savefig("imgs/SIR_short")
the_plot

# %% [markdown]
# ### Rejection sampling
#
# Suppose we have a target distribution, and a stochastic program that generates samples plus weights that measure the *ratio* of their generated frequency to the target frequency.
#
# We may convert our program into a sampler for the target distribution via the metaprogram that draws samples and weights, stochastically accepts them with frequency equal to the reported ratio, and otherwise rejects them and tries again.  This metaprogram is called *rejection sampling*.
#
# Suppose that our stochastic program only reports *unnormalized* ratios of their generated frequency to the target frequency.  That is, there exists some constant $Z$ such that $Z$ times the correct ratio is reported.  If we knew $Z$, we could just correct the reported ratios by $Z$.  But suppose $Z$ itself is unavailable, and we only know a bound $C$ for $Z$, that is $Z < C$.  Then we can correct the ratios by $C$, obtaining an algorithm that is correct but inefficient by a factor of drawing $C/Z$ too many samples on average.  This metaprogram is called *approximate rejection sampling*.
#
# Finally, suppose we know that $Z$ exists, but we do not even know a bound $C$ for it.  Then we may proceed adaptively by tracking the largest weight encountered thus far, and using this number $C$ as above.  This metaprogram is called *adaptive approximate rejection sampling*.
#
# Earlier samples may occur with too high absolute frequency, but over time as $C$ appropriately increases, the behavior tends towards the true distribution.  We may consider some of this early phase to be an *exploration* or *burn-in period*, and accordingly draw samples but keep only the maximum of their weights, before moving on to the rejection sampling *per se*.

# %%
function rejection_sample(model, args, constraints, N_burn_in, N_particles, MAX_ITERS)
    C = (N_burn_in > 0) ? maximum(generate(full_model_1, (T, full_model_args...), constraints)[2] for _ in 1:N_burn_in) : -Inf
    println("C set to $C")

    n_iters = 0

    particles = []
    for i in 1:N_particles
        
        # Uncomment this for more information about the algorithm's
        # progression:
        # println("Particle $i ; n_iters = $n_iters | C = $C")
        
        reject = true
        compute = 0
        while reject && n_iters < MAX_ITERS
            n_iters += 1
            
            y, w = generate(model, args, constraints)
            
            if w > C
                reject = false
                prt, C = y, w
                push!(particles, prt)
            elseif w > C + log(rand())
                reject = false
                prt = y
                push!(particles, prt)
            end
        end
    end

    return particles
end;

# %%
T_RS = 9;

# %%
N_burn_in = 0 # omit burn-in to illustrate early behavior
N_particles = 20
compute_bound = 5_000
traces = rejection_sample(full_model_1, (T_RS, full_model_args...), constraints, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, traces[1:i], T_RS, "RS (particles 1 to $i)";
                                   path_actual=path_actual[1:T_RS])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %%
N_burn_in = 100
N_particles = 20
compute_bound = 5_000
traces = rejection_sample(full_model_1, (T_RS, full_model_args...), constraints, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, traces[1:i], T_RS, "RS (particles 1 to $i)";
                                   path_actual=path_actual[1:T_RS])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %%
N_burn_in = 1_000
N_particles = 20
compute_bound = 5_000
traces = rejection_sample(full_model_1, (T_RS, full_model_args...), constraints, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, traces[1:i], T_RS, "RS (particles 1 to $i)";
                                   path_actual=path_actual[1:T_RS])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %% [markdown]
# The runtime of this algorithm varies wildly! 

# %% [markdown]
# ## SIR and Adaptive Rejection Sampling scale poorly
#
# SIR does not scale because for longer paths, the search space is too large, and the results are only modestly closer to the posterior.
#
# Adaptive rejection sampling suffers from a similar issue.
#
# Below, we show SIR run on a long path to illustrate the type of poor inference results which arise from these algorithms.

# %%
N_samples = 10
N_SIR = 500
traces = [basic_SIR_library(full_model_1, (T, full_model_args...), constraints, N_SIR)[1] for _ in 1:N_samples]

the_plot = frame_from_traces(world, traces, T, "SIR (original path)";
                             show_clutters=false, path_actual=path_actual)
savefig("imgs/SIR")
the_plot

# %% [markdown]
# # Particle filter with MCMC Rejuvenation
#
# However, it is possible to use Gen to write more sophisticated inference algorithms which scale much better in the path lengths.
# These inference algorithms are implemented using the GFI methods (like `generate`, `propose`, `assess`, and `update`).
#
# (Gen also contains a library of built-in inference algorithm implementations; but we have spelled out the implementation below in terms of the GFI to illustrate the types of powerful inference algorithms one can develop with only a few dozen lines of code.)
#
# ## Key idea #1: resample after each timestep ('particle filtering')
#
# SIR and rejection sampling generate full trajectories $\textbf{z}_{0:T}$ from $P_{\text{trajectory\_prior}}$, and only consider the observations $\textbf{o}_{0:T}$ at the end.
#
# Instead, _particle filtering_ generates $\textbf{z}_t$ values one at a time, and considers the observed value $\textbf{o}_{0:T}$ one at a time.
#
# Specifically, particle filtering works as follows.
# 1. TIME 0 - Generate $N$ values $\textbf{z}_0^i \sim P_{\text{pose}_0}$ for $i = 1, \dots, N$.
# 2. TIME 0 - For each $i$, compute $w^i_0 := \frac{P_{\text{full}_0}(\textbf{z}_0^i, \textbf{o}_0*)}{P_{\text{pose}_0}}$.  We write $P_{\text{full}_0}$ to denote the model $P_\text{full}$ unrolled to just the initial timestep.
# 3. TIME 0 - Compute the normalized weights $\hat{w}^i_0 = \frac{w^i}{\sum_{j=1}^N{w^j}}$.
# 4. TIME 0 - Resample.  For $j = 1, \dots, N$, let $a^j_0 \sim \text{categorical}([\hat{w}^1_0, \hat{w}^2_0, \dots, \hat{w}^N_0])$.
# 5. TIME 1 - Generate $N$ values $\textbf{z}_1^i \sim P_\text{step}(\textbf{z}_1 ; \textbf{z}_0^{a^i_0})$.
# 6. TIME 1 - For each $i$, compute $w^i_0 := \frac{P_{\text{full}_1}([\textbf{z}_0^{a^i_0}, \textbf{z}_1^i], \textbf{o}_{0:1}*)}P_\text{step}(\textbf{z}_1 ; \textbf{z}_0^{a^i_0})$.
# 7. TIME 1 - Compute normalized weights $\hat{w}^i_1$.
# 8. TIME 1 - Resample. For $j = 1, \dots, N$, let $a^j_1 \sim \text{categorical}([\hat{w}^1_1, \hat{w}^2_1, \dots, \hat{w}^N_1])$.
# 9. TIME 2 - Generate $N$ values $\textbf{z}_2^i \sim P_\text{step}(\textbf{z}_2 ; \textbf{z}_1^{a^i_1})$.
# 10. ...
#
# The key idea is that the algorithm _no longer needs to get as lucky_.  In SIR and rejection sampling, the algorithm needs to generate a full trajectory where every $\textbf{z}_t$ value just happens to be consistent with the observation $\textbf{o}_t$.  In a particle filter, at each step $t$, the algorihtm only needs to generate _one_ $\textbf{z}_t$ which is consistent with $\textbf{o}_t$, not a full trajectory over $T + 1$ points where all the values are consistent.  Resampling ensures that before proceeding to the next $t$ value, the value of $\textbf{z}_{t-1}$ is consistent with the observations.
#
# ## Key idea #2: iteratively improve each latent hypothesis ('Markov Chain Monte Carlo rejuvenation')
#
# Particle filtering can be further improved by adding _Markov Chain Monte Carlo_ rejuvenation.
#
# This adds a step to the particle filter after resampling, which iteratively tweaks the values $\textbf{z}_t^{a^i_t}$ to make them more consistent observations.

# %%
function particle_filter_rejuv(model, T, args, observations, N_particles, N_MH, MH_proposal, MH_proposal_args)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]

    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end
    
    for t in 1:T
        weights = exp.(log_weights .- maximum(log_weights))
        weights = weights ./ sum(weights)
        for i in 1:N_particles
            resample_traces[i] = traces[categorical(weights)]
        end
        traces, resample_traces = resample_traces, traces

        for i in 1:N_particles
            for _ = 1:N_MH
                fwd_choices, fwd_weight, _ = propose(MH_proposal, (traces[i], MH_proposal_args...))
                propose_trace, propose_weight_diff, _, discard =
                    update(traces[i], get_args(traces[i]), map(_ -> NoChange(), get_args(traces[i])), fwd_choices)
                bwd_weight, _ = assess(MH_proposal, (propose_trace, MH_proposal_args...), discard)
                if log(rand()) < (propose_weight_diff - fwd_weight + bwd_weight)
                    traces[i] = propose_trace
                end
            end
        end

        for i in 1:N_particles
            traces[i], log_weights[i], _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
        end
    end

    return traces, log_weights
end
;
# # Alternatively, using library calls: `particle_filter_rejuv_library` from the black box above!

# %%
using Dates
dv = Dates.value

drift_step_factor = 1/3.

N_samples = 6
N_particles = 10
N_MH = 5
t1 = Dates.now()
traces = [particle_filter_rejuv(full_model_1, T, full_model_args, observations, N_particles,
                                N_MH, drift_proposal, (drift_step_factor,))[1][1] for _ in 1:N_samples]
t2 = Dates.now()
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")

the_plot = frame_from_traces(world, traces, T, "PF+Drift Rejuv";
                             path_actual=path_actual)
savefig("imgs/PF_rejuv")
the_plot

# %% [markdown]
# ## Grid Rejuvenation via MH

# %%
### UTILS for gridding

function sortperm_them!(vals, vecs...)
    perm = sortperm(vals)
    id   = 1:length(vals)
    for v in [vals, vecs...]
        v[id] = v[perm]
    end
end;

argdiffs(bs::Array{T,1}) where T <: Real = Tuple(map(b -> Bool(b) ? UnknownChange() : NoChange(), bs));

"""
Discretize into bins of diameter r, bin-centers lie 
at `z - k*r` for intergers `k`.
"""
quantize(x, r; zero=0) = Int.(floor.((x .+ r./2 .- zero)./r))

"""
    get_offset(v0, k, r)

Computes the offset to move the center 
of the grid to `v0`.
"""
function get_offset(v0, k, r)
    center = (r + k.*r)/2
    return v0 - center
end

function first_grid_vec(v0::Vector{Real}, k::Vector{Int}, r::Vector{Real})
    return r + get_offset(v0, k, r) 
end

"""
    vs, ls = vector_grid(v0, k, r)

Returns grid of vectors and their linear indices, given 
a grid center, numnber of grid points along each dimension and
the resolution along each dimension.
"""
function vector_grid(v0::Vector{Float64}, k::Vector{Int}, r::Vector{Float64})
    # Todo: Does it make sense to get a CUDA version of this?
    offset = get_offset(v0, k, r)
    
    shape = Tuple(k)
    cs = CartesianIndices(shape)
    ls = LinearIndices(shape)
    vs = map(I -> [Tuple(I)...].*r + offset, cs);
    return (vs=vs, linear_indices=ls)
end

function grid_index(x, v0, k, r; linear=false)
    I = quantize(x, r, zero=get_offset(v0, k, r));
    if linear
        if any(map(x -> x == 0, I))
            println("I = $I")
        end
        try
            shape = Tuple(k)
            I = LinearIndices(shape)[I...]
        catch e
            println("I = $I ; k = $k")
            error(e)
        end
    end
    return I
end;

# %%
@gen function grid_proposal(
        tr,
        n_steps, # (n_x_steps, n_y_steps, n_hd_steps),
        step_sizes # (x_step_size, y_step_size, hd_step_size)
    )
    t = get_args(tr)[1] + 1

    p_noise = get_args(tr)[4].motion_settings.p_noise
    hd_noise = get_args(tr)[4].motion_settings.hd_noise

    p = tr[prefix_address(t, :pose => :p)]
    hd = tr[prefix_address(t, :pose => :hd)]

    pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes).vs, (:,))
    
    # Collection of choicemaps which would update the trace to have each pose
    # in the grid
    chmap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]),
                            (prefix_address(t, :pose => :hd), h))
                  for (x, y, h) in pose_grid]
    
    # Get the score under the model for each grid point
    pose_scores = [Gen.update(tr, ch)[2] for ch in chmap_grid]
        
    pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
    j ~ categorical(pose_probs)
    new_p = pose_grid[j][1:2]
    new_hd = pose_grid[j][3]

    inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes; linear=true)

    return (j, chmap_grid[j], inverting_j)
end;

# %%
function grid_mh(tr, n_steps, step_sizes)
    (proposal_choicemap, fwd_proposal_logprob, (j, chmap, inv_j)) =
        Gen.propose(grid_proposal, (tr, n_steps, step_sizes))
    (new_tr, model_log_probratio, _, _) = Gen.update(tr, chmap)
    (bwd_proposal_logprob, (_, _, j2)) = Gen.assess(grid_proposal, (new_tr, n_steps, step_sizes), choicemap((:j, inv_j)))
    @assert j2 == j # Quick reversibility check
    log_acc_prob = model_log_probratio + bwd_proposal_logprob - fwd_proposal_logprob
    if log(rand()) <= log_acc_prob
        return new_tr
    else
        return tr
    end
end;

# %%
get_trajectory(trace) = [trace[prefix_address(t, :pose)] for t in 1:(get_args(trace)[1] + 1)]

function particle_filter_grid_rejuv_with_checkpoints(model, T, args, observations, N_particles, MH_arg_schedule)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]

    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (get_trajectory.(traces), copy(log_weights)))

    # for t in 1:T
        
    #     for i in 1:N_particles
    #         resample_traces[i] = traces[categorical(weights)]
    #     end
    #     traces, resample_traces = resample_traces, traces

    #     for i in 1:N_particles
    #         for proposal_args in MH_arg_schedule
    #             traces[i] = grid_mh(traces[i], proposal_args...)
    #         end
    #     end

    #     for i in 1:N_particles
    #         traces[i], log_weights[i], _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
    #     end
    # end
    
    for t in 1:T
        t % 5 == 0 && @info "t = $t"

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces
        end

        for i in 1:N_particles
            for proposal_args in MH_arg_schedule
                traces[i] = grid_mh(traces[i], proposal_args...)
            end
        end

        for i in 1:N_particles
            traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
            log_weights[i] += wt
        end

        push!(checkpoints, (get_trajectory.(traces), copy(log_weights)))
    end

    return checkpoints
end;

# %%
function frame_from_weighted_trajectories(world, trajectories, weights, T, title; show_clutters=false, path_actual=nothing, minalpha=.01, readings=nothing)
    the_plot = start_plot(world, title; show_clutters=show_clutters)
    if !isnothing(path_actual)
        plot!(path_actual; label="actual path", color=:brown)
        plot!(path_actual[length(first(trajectories))], color=:black)
    end

    normalized_weights = exp.(weights .- logsumexp(weights))
    
    for (traj, wt) in zip(trajectories, normalized_weights)
        al = max(minalpha, 0.6*sqrt(wt))
        
        plot!(poses_to_coords(traj)...;
              label=nothing, color=:green, alpha=al)
        plot!(traj[end]; color=:green, alpha=al, label=nothing)
        
        plot!(Segment.(zip(traj[1:end-1], traj[2:end]));
              label=nothing, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=al)
    end

    if !isnothing(path_actual) && !isnothing(readings)
        t = length(first(trajectories))
        projections = project_readings(path_actual[t], readings[t], sensor_settings)
        plot!(first.(projections), last.(projections);
          label=nothing, color=:blue, seriestype=:scatter, markersize=3, markerstrokewidth=1, alpha=0.25)
        plot!([Segment(path_actual[t].p, pr) for pr in projections];
          label=nothing, color=:blue, alpha=0.25)
    end
        
    return the_plot
end;

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/10]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10

t1 = Dates.now()
checkpointss =
    [particle_filter_grid_rejuv_with_checkpoints(
       #model,      T,   args,         observations, N_particles, MH_arg_schedule)
       full_model_1, T, full_model_args, observations, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = Dates.now();

# %%
merged_traj_list = []
merged_weight_list = []
for checkpoints in checkpointss
    (trajs, lwts) = checkpoints[end]
    merged_traj_list = [merged_traj_list..., trajs...]
    merged_weight_list = [merged_weight_list..., lwts...]
end
merged_weight_list = merged_weight_list .- log(length(checkpointss))
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list, merged_weight_list, T, "PF + Grid MH Rejuv"; path_actual, minalpha=0.03)

# %% [markdown]
# This is just a first step.  We'll improve it below by improving the quality of the particle weights (and, in turn, the resampling).

# %% [markdown]
# ## SMCP3

# %%
@gen function grid_proposal_smcp3_fwd(
        tr,
        n_steps, # (n_x_steps, n_y_steps, n_hd_steps),
        step_sizes # (x_step_size, y_step_size, hd_step_size)
    )
    t = get_args(tr)[1] + 1

    p = tr[prefix_address(t, :pose => :p)]
    hd = tr[prefix_address(t, :pose => :hd)]

    pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes).vs, (:,))
    
    # Collection of choicemaps which would update the trace to have each pose
    # in the grid
    chmap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]),
                            (prefix_address(t, :pose => :hd), h))
                  for (x, y, h) in pose_grid]
    
    # Get the score under the model for each grid point
    pose_scores = [Gen.update(tr, ch)[2] for ch in chmap_grid]
        
    pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
    j ~ categorical(pose_probs)
    new_p = pose_grid[j][1:2]
    new_hd = pose_grid[j][3]

    inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes; linear=true)

    return (j, chmap_grid[j], inverting_j)
end

@gen function grid_proposal_smcp3_bwd(
        updated_tr,
        n_steps, # (n_x_steps, n_y_steps, n_hd_steps),
        step_sizes # (x_step_size, y_step_size, hd_step_size)
    )
    t = get_args(updated_tr)[1] + 1

    new_p = updated_tr[prefix_address(t, :pose => :p)]
    new_hd = updated_tr[prefix_address(t, :pose => :hd)]

    pose_grid = reshape(vector_grid([new_p..., new_hd], n_steps, step_sizes).vs, (:,))
    
    # Collection of choicemaps which would update the trace to have each pose
    # in the grid
    chmap_grid = [choicemap((:p, [x, y]), (:hd, h)) for (x, y, h) in pose_grid]
    
    # Get the score under the model for each grid point
    _, robot_inputs, world_inputs, settings = get_args(updated_tr)
    if t > 1
        prev_p = updated_tr[prefix_address(t - 1, :pose => :p)]
        prev_hd = updated_tr[prefix_address(t - 1, :pose => :hd)]
        pose_scores = [
            Gen.assess(motion_model,
                       (Pose(prev_p, prev_hd), robot_inputs.controls[t - 1], world_inputs, settings.motion_settings),
                       ch)[1]
            for ch in chmap_grid]
    else
        pose_scores = [
            Gen.assess(pose_prior_model,
                       (robot_inputs.start_guess, settings.motion_settings),
                       ch)[1]
            for ch in chmap_grid]
    end
        
    pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
    j ~ categorical(pose_probs)
    old_p = pose_grid[j][1:2]
    old_hd = pose_grid[j][3]

    inverting_j = grid_index([new_p..., new_hd], [old_p..., old_hd], n_steps, step_sizes; linear=true)

    return (j, chmap_grid[j], inverting_j)
end;

# %%
function grid_smcp3(tr, n_steps, step_sizes)
    (proposal_choicemap, fwd_proposal_logprob, (j, chmap, inv_j)) =
        Gen.propose(grid_proposal_smcp3_fwd, (tr, n_steps, step_sizes))
    
    (new_tr, model_log_probratio, _, _) = Gen.update(tr, chmap)
    
    (bwd_proposal_logprob, (reinv_j, _, j2)) = Gen.assess(
        grid_proposal_smcp3_bwd,
        (new_tr, n_steps, step_sizes),
        choicemap((:j, inv_j)))
    
    @assert j2 == j # Quick reversibility check
    @assert reinv_j == inv_j
    
    log_weight_update = model_log_probratio + bwd_proposal_logprob - fwd_proposal_logprob
    
    return (new_tr, log_weight_update)
end;

# %%
function particle_filter_grid_smcp3_with_checkpoints(model, T, args, observations, N_particles, MH_arg_schedule)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]

    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (get_trajectory.(traces), copy(log_weights)))

    for t in 1:T
        if t % 5 == 1
            @info "t = $t"
        end

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles/10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces 
        end

        for i in 1:N_particles
            for proposal_args in MH_arg_schedule
                traces[i], wtupdate = grid_smcp3(traces[i], proposal_args...)
                log_weights[i] += wtupdate
            end
        end

        for i in 1:N_particles
            traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
            log_weights[i] += wt
        end

        push!(checkpoints, (get_trajectory.(traces), copy(log_weights)))
    end

    return checkpoints
end;

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/10]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10

t1 = Dates.now()
checkpointss2 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, MH_arg_schedule)
       full_model_1, T, full_model_args, observations, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = Dates.now()

merged_traj_list2 = []
merged_weight_list2 = []
for checkpoints in checkpointss2
    (trajs, lwts) = checkpoints[end]
    merged_traj_list2 = [merged_traj_list2..., trajs...]
    merged_weight_list2 = [merged_weight_list2..., lwts...]
end
merged_weight_list2 = merged_weight_list2 .- log(length(checkpointss2));

# %%
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list2, merged_weight_list2, T, "PF + Grid SMCP3 Rejuv"; path_actual, minalpha=0.03)

# %% [markdown]
# That's already better.  We'll improve this algorithm even further below.
#
# But first, I want to note that there is a major downside to this rejuvenation -- in some cases, we don't need it, and it takes a lot of computation time!

# %% [markdown]
# ### With low motion model noise, all this compute is overkill!
#
# Here, we generate a low noise trajectory, and show that the bootstrap particle filter (with no rejuvenation) is sufficient to perform good inferences.  (Low motion noise, moderate observation noise.)  Proposing from the prior is quite good!

# %%
obs_selector = select(:initial => :sensor, (:steps => t => :sensor  for t=1:100)...);
motion_settings_lownoise = (p_noise = 0.005, hd_noise = 1/50 * 2π / 360)
sensor_settings_noisy = (sensor_settings..., s_noise = 0.15)

tol2 = 0.10
path_actual_lownoise = integrate_controls_noisy((robot_inputs..., start_guess=start_actual), world_inputs, motion_settings_lownoise)
observations2 = [noisy_sensor(p, world.walls, sensor_settings, tol2) for p in path_actual_lownoise];

# %%
full_model_args_v2 = (robot_inputs, world_inputs, (
    motion_settings = motion_settings_lownoise,
    sensor_settings = sensor_settings_noisy));

# %%
ani = Animation()
for (p, readings) in zip(path_actual_lownoise, observations2)
    frame_plot = plot_sensors(world, "Data in low motion noise regie",
        [(path_ideal, "ideal path", :green2, nothing, nothing, nothing),
         (path_actual_lownoise, "actual path", :brown, p, readings, nothing)],
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances_lowmotionnoise.gif", fps=1)

# %%
N_samples = 6
N_particles = 10

t1 = Dates.now()
checkpointss4 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, grid)
       full_model_1, T, full_model_args_v2, observations2, N_particles, [])
     for _=1:N_samples]
t2 = Dates.now()

merged_traj_list4 = []
merged_weight_list4 = []
for checkpoints in checkpointss4
    (trajs, lwts) = checkpoints[end]
    merged_traj_list4 = [merged_traj_list4..., trajs...]
    merged_weight_list4 = [merged_weight_list4..., lwts...]
end
merged_weight_list4 = merged_weight_list4 .- log(length(checkpointss4));

# %%
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list4, merged_weight_list4, T, "Particle filter (no rejuv) - low motion noise"; path_actual=path_actual_lownoise, minalpha=0.03)


# %% [markdown]
# ### The issue is when motion noise is higher
#
# Now we'll generate a very high motion noise (low observation noise) trajectory.

# %%
obs_selector = select(:initial => :sensor, (:steps => t => :sensor  for t=1:100)...);
motion_settings_highnoise = (p_noise = 0.25, hd_noise = 1.5 * 2π / 360)

tol3 = .03
path_actual_highnoise = integrate_controls_noisy((robot_inputs...., start_guess=start_actual), world_inputs, motion_settings_highnoise)
observations3 = [noisy_sensor(p, world.walls, sensor_settings, tol3) for p in path_actual_highnoise];

ani = Animation()
for (p, readings) in zip(path_actual_highnoise, observations3)
    frame_plot = plot_sensors(world, "Data - high motion noise",
        [(path_ideal, "ideal path", :green2, nothing, nothing, nothing),
         (path_actual_highnoise, "actual path", :brown, p, readings, nothing)],
        sensor_settings; show_clutters=false)
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances_highmotionnoise.gif", fps=1)

# %% [markdown]
# If we try particle filtering with low-motion-noise settings and no rejuvenation, we have the issue that the particle filter basically just follows the integrated controls, ignoring the highly informative observations.

# %%
N_samples = 6
N_particles = 10

t1 = Dates.now()
checkpointss5 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, grid)
       full_model_1, T, full_model_args_v2, observations3, N_particles, [])
     for _=1:N_samples]
t2 = Dates.now()

merged_traj_list5 = []
merged_weight_list5 = []
for checkpoints in checkpointss5
    (trajs, lwts) = checkpoints[end]
    merged_traj_list5 = [merged_traj_list5..., trajs...]
    merged_weight_list5 = [merged_weight_list5..., lwts...]
end
merged_weight_list5 = merged_weight_list5 .- log(length(checkpointss5));

# %%
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list5, merged_weight_list5, T, "PF - motion noise:(model:low)(data:high)"; path_actual=path_actual_highnoise, minalpha=0.03)

# %% [markdown]
# Conversely, if we run a no-rejuvenation particle filter with the higher model noise parameters, the runs are inconsistent.

# %%
N_samples = 6
N_particles = 10

t1 = Dates.now()
checkpointss6 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, grid)
       full_model_1, T, full_model_args, observations3, N_particles, [])
     for _=1:N_samples]
t2 = Dates.now()

merged_traj_list6 = []
merged_weight_list6 = []
for checkpoints in checkpointss6
    (trajs, lwts) = checkpoints[end]
    merged_traj_list6 = [merged_traj_list6..., trajs...]
    merged_weight_list6 = [merged_weight_list6..., lwts...]
end
merged_weight_list6 = merged_weight_list6 .- log(length(checkpointss6));

# %%
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list6, merged_weight_list6, T, "PF - motion noise:(model:high)(data:high)"; path_actual=path_actual_highnoise, minalpha=0.03)

# %% [markdown]
# However, if we add back in SMCP3 rejuvenation, performance is a lot better!
#
# The only issue is that it is much slower.

# %%
N_samples = 6
N_particles = 10
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/10]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

t1 = Dates.now()
checkpointss7 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, grid)
       full_model_1, T, full_model_args, observations3, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = Dates.now()

merged_traj_list7 = []
merged_weight_list7 = []
for checkpoints in checkpointss7
    (trajs, lwts) = checkpoints[end]
    merged_traj_list7 = [merged_traj_list7..., trajs...]
    merged_weight_list7 = [merged_weight_list7..., lwts...]
end
merged_weight_list7 = merged_weight_list7 .- log(length(checkpointss7));

# %%
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, merged_traj_list7, merged_weight_list7, T, "PF + Grid SMCP3 Rejuv - motion noise:high"; path_actual=path_actual_highnoise, minalpha=0.03)

# %% [markdown]
# # Inference controller to automatically spend the right amount of compute for good accuracy
#
# Now we'll write an inference controller which decides when to run SMCP3 rejuvenation, and how much SMCP3 rejuvenation to run, based on thresholding the estimated marginal data likelihood.
#
# To make inference more robust, I have also written the controller so that if the inference results still seem poor after rejuvenation, the inference algorithm can re-propose particles from the previous timestep.  This helps avoid "dead ends" where the particle filter proposes only unlikely particles that rejuvenation cannot fix, at some timestep.
#
# With low-motion-noise settings, this will automatically realize there is no need to run rejuvenation, and will achieve very fast runtimes.
#
# With high-motion noise settings, this will automatically realize that rejuvenation is needed at some steps to alleviate artifacts.

# %%
function controlled_particle_filter_with_checkpoints(model, T, args, observations, N_particles::Int, og_arg_schedule)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]

    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (msg="init", t=0, traj=get_trajectory.(traces), wts=copy(log_weights)))
    prev_total_weight = 0.

    n_rejuv = 0
    for t in 1:T
        if t % 5 == 0
            @info "t = $t"
        end

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces
            push!(checkpoints, (msg="resample", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))
        end

        nr = 0
        arg_schedule = og_arg_schedule
        # CHECK the change in log marginal data likelihood estimate.
        # If below a (manually set) threshold, rejuvenate.  If this does
        # not improve the problem, modify the grid schedule slightly, and try
        # again.  Do this up to 3 times before giving up.

        
        while logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && nr < 3
            nr += 1
            for i in 1:N_particles
                for proposal_args in arg_schedule
                    tr, wtupdate = grid_smcp3(traces[i], proposal_args...)
                    if !isinf(wtupdate)
                        traces[i] = tr
                        log_weights[i] += wtupdate
                    end
                end
            end
            push!(checkpoints, (msg="rejuvenate (nr = $nr)", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))

            nsteps, sizes = arg_schedule[1]
            # increase the range and resolution of the grid search
            if nr % 1 == 0
                arg_schedule = [ (nsteps, sizes .* 0.75) for (nsteps, sizes) in arg_schedule ]
            else
                arg_schedule = [ (nsteps + 2, sizes) for (nsteps, sizes) in arg_schedule ]
            end
        end
        if nr > 0
            n_rejuv += 1
        end
        prev_total_weight = logsumexp(log_weights)

        

        for i in 1:N_particles
            traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
            log_weights[i] += wt
        end
        push!(checkpoints, (msg="update", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))
    end

    @info "Rejuvenated $n_rejuv of $T steps."
    return checkpoints
end;

# %%
function controlled_particle_filter_with_checkpoints_v2(model, T, args, observations, N_particles::Int, og_arg_schedule)
    constraints = [constraint_from_sensor_reading(choicemap(), t, sensor_reading) for (t, sensor_reading) in enumerate(observations)]

    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)
    prev_log_weights, prev_traces = [], []

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (msg="Initializing", t=0, traj=get_trajectory.(traces), wts=copy(log_weights)))
    prev_total_weight = 0.

    n_rejuv = 0
    for t in 1:T
        if t % 5 == 0
            @info "t = $t"
        end

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces
            push!(checkpoints, (msg="Resampling", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))
        end

        nr = 0
        arg_schedule = og_arg_schedule
        # CHECK the change in log marginal data likelihood estimate.
        # If below a (manually set) threshold, rejuvenate.  If this does
        # not improve the problem, modify the grid schedule slightly, and try
        # again.  Do this up to 3 times before giving up.

        MAX_REJUV = 3
        while logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && nr ≤ MAX_REJUV
            nr += 1
            for i in 1:N_particles
                for proposal_args in arg_schedule
                    tr, wtupdate = grid_smcp3(traces[i], proposal_args...)
                    if !isinf(wtupdate)
                        traces[i] = tr
                        log_weights[i] += wtupdate
                    end
                end
            end
            push!(checkpoints, (msg="Rejuvenating (repeats: $(nr))", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))

            # If it still looks bad, try re-generating from the previous timestep
            if logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && t > 1 && nr != MAX_REJUV
                traces = copy(prev_traces)
                log_weights = copy(prev_log_weights)

                push!(checkpoints, (msg="Reverting", t=t-1, traj=get_trajectory.(traces), wts=copy(log_weights)))
                
                for i in 1:N_particles
                    traces[i], wt, _, _ = update(traces[i], (t - 1, args...), (UnknownChange(),), constraints[t])
                    log_weights[i] += wt
                end

                lnormwts = log_weights .- logsumexp(log_weights)
                if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
                    weights = exp.(lnormwts)
                    for i in 1:N_particles
                        resample_traces[i] = traces[categorical(weights)]
                    end
                    log_weights .= logsumexp(log_weights) - log(N_particles)
                    traces, resample_traces = resample_traces, traces

                    push!(checkpoints, (msg="Resampling", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))
                end
            end

            nsteps, sizes = arg_schedule[1]
            # increase the range and resolution of the grid search
            if nr % 1 == 0
                arg_schedule = [(nsteps, sizes .* 0.75) for (nsteps, sizes) in arg_schedule]
            else
                arg_schedule = [(nsteps + 2, sizes) for (nsteps, sizes) in arg_schedule]
            end
        end
        if nr > 0
            n_rejuv += 1
        end
        prev_log_weights = copy(log_weights)
        prev_traces = copy(traces)
        prev_total_weight = logsumexp(log_weights)

        for i in 1:N_particles
            traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
            log_weights[i] += wt
        end
        push!(checkpoints, (msg="Extending", t=t, traj=get_trajectory.(traces), wts=copy(log_weights)))
    end

    @info "Rejuvenated $n_rejuv of $T steps."
    return checkpoints
end;

# %% [markdown]
# On the main trajectory we have been experimenting with, this controller visually achieves better results than SMCP3, at a comparable runtime.  The controller spends more computation at some steps (where it is needed), and makes up for it by spending less computation at other steps.

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/6]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10
checkpointss3 = []
t1 = Dates.now()
for _=1:N_samples
    push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         observations, N_particles, MH_arg_schedule)
        full_model_1, T, full_model_args, observations, N_particles, grid_schedule))
end
t2 = Dates.now();

# %%
merged_traj_list3 = []
merged_weight_list3 = []
for checkpoints in checkpointss3
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list3 = [merged_traj_list3..., trajs...]
    merged_weight_list3 = [merged_weight_list3..., lwts...]
end
merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, merged_traj_list3, merged_weight_list3, T, "Inference Controller (moderate noise)"; path_actual, minalpha=0.03)

# %% [markdown]
# **Animation showing the controller in action----**

# %%
ani = Animation()

checkpoints = checkpointss3[1]
for checkpoint in checkpoints
    frame_plot = frame_from_weighted_trajectories(world, checkpoint.traj, checkpoint.wts, T, "t = $(checkpoint.t) | operation = $(checkpoint.msg)"; path_actual, minalpha=0.08)
    frame(ani, frame_plot)
end
gif(ani, "imgs/controller_animation.gif", fps=1)

# %% [markdown]
# Slower version:

# %%
gif(ani, "imgs/controller_animation.gif", fps=1/3)

# %%
# let
#     nsteps = [3, 3, 3]
#     sizes1 = [.7, .7, π/6]
#     grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]
    
#     N_samples = 6
#     N_particles = 10
#     checkpointss3 = []
#     t1 = Dates.now()
#     for _=1:N_samples
#         push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
#             #model,      T,   args,         observations, N_particles, MH_arg_schedule)
#             full_model_1, T, full_model_args, observations, N_particles, grid_schedule))
#     end
#     t2 = Dates.now();
    
#     merged_traj_list3 = []
#     merged_weight_list3 = []
#     for checkpoints in checkpointss3
#         (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
#         merged_traj_list3 = [merged_traj_list3..., trajs...]
#         merged_weight_list3 = [merged_weight_list3..., lwts...]
#     end
#     merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
#     println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
#     frame_from_weighted_trajectories(world, merged_traj_list3, merged_weight_list3, T, "controlled grid rejuv"; path_actual, minalpha=0.03)
# end

# %% [markdown]
# ### Controller on LOW NOISE TRAJECTORY
#
# Here, the controller realizes it never needs to rejuvenate, and runtimes are very fast.

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/6]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10
checkpointss9 = []
t1 = Dates.now()
for _=1:N_samples
    push!(checkpointss9, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         observations, N_particles, MH_arg_schedule)
        full_model_1, T, full_model_args_v2, observations2, N_particles, grid_schedule))
end
t2 = Dates.now();

# %%
merged_traj_list9 = []
merged_weight_list9 = []
for checkpoints in checkpointss9
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list9 = [merged_traj_list9..., trajs...]
    merged_weight_list9 = [merged_weight_list9..., lwts...]
end
merged_weight_list9 = merged_weight_list9 .- log(length(checkpointss9));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, merged_traj_list9, merged_weight_list9, T, "Inference controller (low motion noise)"; path_actual=path_actual_lownoise, minalpha=0.03)

# %% [markdown]
# ### Controller on HIGH NOISE TRAJECTORY
#
# Here, the controller achieves similar accuracy to pure SMCP3, in slightly lower runtime, because at some steps it realizes there is no need to rejuvenate.

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/6]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10
checkpointss10 = []
t1 = Dates.now()
for _=1:N_samples
    push!(checkpointss10, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         observations, N_particles, MH_arg_schedule)
        full_model_1, T, full_model_args, observations3, N_particles, grid_schedule))
end
t2 = Dates.now();

# %%
merged_traj_list10 = []
merged_weight_list10 = []
for checkpoints in checkpointss10
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list10 = [merged_traj_list10..., trajs...]
    merged_weight_list10 = [merged_weight_list10..., lwts...]
end
merged_weight_list10 = merged_weight_list10 .- log(length(checkpointss10));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, merged_traj_list10, merged_weight_list10, T, "Inference controller (high motion noise)"; path_actual=path_actual_highnoise, minalpha=0.03)
