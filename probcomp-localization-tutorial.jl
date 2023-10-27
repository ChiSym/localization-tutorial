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
# # TODO
#
# * Explanation of `Gen.generate`, what frequencies/weights to expect, and why they are correct ones to use in context.
#
# * Comment on roles of fwd/bwd proposals in SMCP3.
# * Visualize grids somehow.
#
# * Color sensor reading picture lines via when unexpectedly low likelihood.
#
# * alternate world models for comparison with robot model.  Examples: variants of parameters
# * plotting multiple traces: sequencing vs. tiling vs. alpha-blending (in each case indicate weights differently)
# * label all (hyper)parameters in visualizations
# * fix docstrings, image filenames
# * Hierarchical (sensor) model?
#
# Rif comments:
# * Correct understanding of initial pose.

# %% [markdown]
# # ProbComp Localization Tutorial
#
# This notebook aims to give an introduction to probabilistic computation (ProbComp).  This term refers to a way of expressing probabilistic constructs in a computational paradigm, made precise by a probablistic programming language (PPL).  The programmer can thus encode their probabilistic intuition for solving a problem into an algorithm.  Back-end language work automates the routine but error-prone derivations.

# %%
# Global setup code

# The dependencies consist of the following Juila packages.
using Dates: now, value as dv
using JSON: parsefile
using Plots
using Gen
using GenParticleFilters

# Fix for Jay's Jupytext setup
if occursin("sharlaon", pwd()); cd("/Users/sharlaon/dev/probcomp-localization-tutorial") end

# Ensure a location for image generation.
mkpath("imgs");

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

norm(v :: Vector{Float64}) = sqrt(sum(v.^2))

struct Segment
    p1 :: Vector{Float64}
    p2 :: Vector{Float64}
    dp :: Vector{Float64}
    Segment(p1 :: Vector{Float64}, p2 :: Vector{Float64}) = new(p1, p2, p2-p1)
end
Segment(tuple :: Tuple) = Segment(tuple...)
Base.show(io :: IO, s :: Segment) = print(io, "Segment($(s.p1), $(s.p2))")

struct Pose
    p  :: Vector{Float64}
    hd :: Float64
    dp :: Vector{Float64}
    Pose(p :: Vector{Float64}, hd :: Float64) = new(p, rem2pi(hd, RoundNearest), [cos(hd), sin(hd)])
end
Pose(p :: Vector{Float64}, dp :: Vector{Float64}) = Pose(p, atan(dp[2], dp[1]))
Pose(tuple :: Tuple) = Pose(tuple...)
Base.show(io :: IO, p :: Pose) = print(io, "Pose($(p.p), $(p.hd))")

step_along_pose(p :: Pose, s :: Float64) :: Vector{Float64} = p.p + s * p.dp
rotate_pose(p :: Pose, a :: Float64) :: Pose = Pose(p.p, p.hd + a)

Segment(p1 :: Pose, p2 :: Pose) = Segment(p1.p, p2.p)

# A value `c :: Control` corresponds to the robot *first* advancing in its present direction by `c.ds`, *then* rotating by `c.dhd`.
struct Control
    ds  :: Float64
    dhd :: Float64
end
Control(tuple :: Tuple) = Control(tuple...)

function create_segments(verts :: Vector{Vector{Float64}}; loop_around=false) :: Vector{Segment}
    segs = Segment.(zip(verts[1:end-1], verts[2:end]))
    if loop_around; push!(segs, Segment(verts[end],verts[1])) end
    return segs
end

function load_world(file_name)
    data = parsefile(file_name)
    walls_vec = Vector{Vector{Float64}}(data["wall_verts"])
    walls = create_segments(walls_vec)
    clutters_vec = Vector{Vector{Vector{Float64}}}(data["clutter_vert_groups"])
    clutters = create_segments.(clutters_vec)
    walls_clutters = [walls ; clutters...]
    start = Pose(Vector{Float64}(data["start_pose"]["p"]), Float64(data["start_pose"]["hd"]))
    controls = Vector{Control}([Control(control["ds"], control["dhd"]) for control in data["program_controls"]])
    all_points = [walls_vec ; clutters_vec... ; [start.p]]
    x_min = minimum(p[1] for p in all_points)
    x_max = maximum(p[1] for p in all_points)
    y_min = minimum(p[2] for p in all_points)
    y_max = maximum(p[2] for p in all_points)
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
    T = length(controls)
    return ((walls=walls, clutters=clutters, walls_clutters=walls_clutters, bounding_box=bounding_box, box_size=box_size, center_point=center_point),
            (start=start, controls=controls),
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
* `robot_inputs` contains fields: `start`, `controls`
"""
function integrate_controls_unphysical(robot_inputs :: NamedTuple) :: Vector{Pose}
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
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance from the point of contact in the normal direction to the wall.

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
* `robot_inputs` contains fields: `start`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
"""
function integrate_controls(robot_inputs :: NamedTuple, world_inputs :: NamedTuple)
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

# %% [markdown]
# ### Plot such data

# %%
function plot_list!(list; label=nothing, args...)
    if isempty(list); return end
    plt = plot!(list[1]; label=label, args...)
    for item in list[2:end]; plot!(item; label=nothing, args...) end
    return plt
end

Plots.plot!(seg :: Segment; args...) = plot!([seg.p1[1], seg.p2[1]], [seg.p1[2], seg.p2[2]]; args...)
Plots.plot!(segs :: Vector{Segment}; args...) = plot_list!(segs; args...)
Plots.plot!(seg_groups :: Vector{Vector{Segment}}; args...) = plot_list!(seg_groups; args...)

Plots.plot!(p :: Pose; r=0.5, args...) = plot!(Segment(p.p, step_along_pose(p, r)); arrow=true, args...)
Plots.plot!(ps :: Vector{Pose}; args...) = plot_list!(ps; args...)

function plot_world(world, title; label_world=false, show_clutters=false)
    border = world.box_size * (3.)/19.
    the_plot = plot(
        size         = (500, 500),
        aspect_ratio = :equal,
        grid         = false,
        xlim         = (world.bounding_box[1]-border, world.bounding_box[2]+border),
        ylim         = (world.bounding_box[3]-border, world.bounding_box[4]+border),
        title        = title,
        legend       = :bottomleft)
    (walls_label, clutter_label) = label_world ? ("walls", "clutters") : (nothing, nothing)
    plot!(world.walls; c=:black, label=walls_label)
    if show_clutters; plot!(world.clutters; c=:magenta, label=clutter_label) end
    return the_plot
end;

# %% [markdown]
# Following this initial display of the given data, we *suppress the clutters* until much later in the notebook.

# %%
the_plot = plot_world(world, "Given data", label_world=true, show_clutters=true)
plot!(robot_inputs.start; color=:green3, label="given start pose")
plot!([pose.p[1] for pose in path_integrated], [pose.p[2] for pose in path_integrated];
      color=:green2, label="path from integrating controls", seriestype=:scatter, markersize=3, markerstrokewidth=0)
savefig("imgs/given_data")
the_plot

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
"""
Assumes
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
@gen (static) function start_pose_prior(start :: Pose, motion_settings :: NamedTuple) :: Pose
    p ~ mvnormal(start.p, motion_settings.p_noise^2 * [1 0 ; 0 1])
    hd ~ normal(start.hd, motion_settings.hd_noise)
    return Pose(p, hd)
end

"""
Assumes
* `world_inputs` contains fields: `walls`, `bounce`
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
@gen (static) function step_model(start :: Pose, c :: Control, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Pose
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

the_plot = plot_world(world, "Start pose prior (samples)")
plot!(pose_samples; color=:red, label=nothing)
savefig("imgs/start_prior")
the_plot

# %%
N_samples = 50
noiseless_step = robot_inputs.start.p + robot_inputs.controls[1].ds * robot_inputs.start.dp
step_samples = [step_model(robot_inputs.start, robot_inputs.controls[1], world_inputs, motion_settings) for _ in 1:N_samples]

std_devs_radius = 2

the_plot = plot_world(world, "Motion step model model (samples)")
plot!(robot_inputs.start; color=:black, label="step from here")
plot!([noiseless_step[1]], [noiseless_step[2]];
      color=:red, label="$(round(std_devs_radius, digits=2))σ region", seriestype=:scatter,
      markersize=(20. * std_devs_radius * motion_settings.p_noise), markerstrokewidth=0, alpha=0.25)
plot!(step_samples; color=:red, label="step samples")
savefig("imgs/motion_step")
the_plot

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
# Traced execution of a generative function also produces a particular kind of score/weight/density.  It is very important to be clear about which score/weight/density value is to be expected, and why.  Consider a generative function of two inputs $x,y$ that flips a coin with weight $p$, and accordingly returns $x$ or $y$.  When $x$ and $y$ are unequal, a sensible reporting of the score/weight/density in the sampling process would produce $p$ or $1-p$ accordingly.  If the user supplied equal values $x = y$, then which score/weight/density should be returned?
#
# One tempting view identifies a GF with a *distribution over its return values*.  In this view, the correct score/weight/density would be $1$.  Pursuing this approach for all GFs requires knowlege of all execution histories that might have produced any output, and then performing a sum over them.  For some small finite situations this may be fine, but this general problem of computing marginalizations is computationally impossible.  (More on this elsewhere, below a fold, or in an exercise?)  Therefore, this is ***not the viewpoint of Gen***, and the score/weight/density being introduced here is a ***different number***.
#
# The only thing a program can reasonably be expected to know is the score/weight/density of its arriving at its return value *via the particular stochastic computation path* that got it there, and the approach of Gen (ProbComp in general?!) is to report this number.  The corresponding mathematical picture imagines GFs as factored into *distributions over choice maps*, whose the score/weight/density is computable, together with deterministic functions on these data that produce the return value from them.  In the toy example, the data of the choice map consists of the sampled Boolean value, its correct score/weight/density is $p$ or $1-p$, accordingly, and its return value function chooses $x$ or $y$, regardless of whether they are equal.
#
# One may still be concerned with the distribution on return values.  This information arises in the aggregate of the sampled stochastic executions that lead to any return value, together with their weights.  (Check that this is true even in this simple example.)  In a sense, when we kick the can of marginalization down the road, we can proceed without difficulty.
#
# The common practice of confusing traces with their choice maps continues here, and we speak of a GF inducing a "distribution over traces".

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
# P_\text{init}(z; y, \nu)
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
# The model contains all information in its trace, rendering its return value redundant.  The the noisy path integration will just be a wrapper around its functionality, extracting what it needs from the trace.
#
# It is worth acknowledging two strange things in the code below: the extra text "`_loop`" in the function name, and the seemingly redundant new parameter `T`.  Both will be addressed in the next section, along with the aforementioned wrapper.

# %%
"""
Assumes
* `robot_inputs` contains fields: `start`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
@gen function path_model_loop(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Vector{Pose}
    pose = {:initial => :pose} ~ start_pose_prior(robot_inputs.start, motion_settings)

    for t in 1:T
        pose = {:steps => t => :pose} ~ step_model(pose, robot_inputs.controls[t], world_inputs, motion_settings)
    end
end

prefix_address(t :: Int, rest) :: Pair = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest)
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
# In addition to the previous data, we are given an estimated start pose $r_0$ and controls $r_t = (s_t, \eta_t)$ for $t=1,\ldots,T$.  Then `path_model` corresponds to a distribution over traces denoted $\text{path}$; these traces are identified with vectors, namely, $z_{0:T} \sim \text{path}(r_{0:T}, w, \nu)$ is the same as $z_0 \sim \text{init}(r_0, \nu)$ and $z_t \sim \text{step}(z_{t-1}, r_t, w, \nu)$ for $t=1,\ldots,T$.  Here and henceforth we use the shorthand $\text{step}(z, \ldots) := \text{step}(\text{retval}(z), \ldots)$.  The density function is
# $$
# P_\text{path}(z_{0:T}; r_{0:T}, w, \nu)
# = P_\text{init}(z_0; r_0, \nu) \cdot \prod\nolimits_{t=1}^T P_\text{step}(z_t; z_{t-1}, r_t, w, \nu)
# $$
# where each term, in turn, factors into a product of two (multivariate) normal densities as described above.

# %% [markdown]
# As our truncation of the example trace above might suggest, visualization is an essential practice in ProbComp.  We could very pass the output of the above `integrate_controls_noisy` to the `plot!` function to have a look at it.  However, we want to get started early in this notebook on a good habit: writing interpretive code for GFs in terms of their traces rather than their return values.  This enables the programmer include the parameters of the model in the display for clarity.

# %%
function frames_from_motion_trace(world, title, trace; show_clutters=false, std_devs_radius=2.)
    T = get_args(trace)[1]
    robot_inputs = get_args(trace)[2]
    poses = get_path(trace)
    noiseless_steps = [robot_inputs.start.p, [pose.p + c.ds * pose.dp for (pose, c) in zip(poses, robot_inputs.controls)]...]
    motion_settings = get_args(trace)[4]
    plots = Vector{Plots.Plot}(undef, T+1)
    for t in 1:(T+1)
        frame_plot = plot_world(world, title; show_clutters=show_clutters)
        plot!(poses[1:t-1]; color=:black, label="past poses")
        plot!([noiseless_steps[t][1]], [noiseless_steps[t][2]];
              color=:red, label="$(round(std_devs_radius, digits=2))σ region", seriestype=:scatter,
              markersize=(20. * std_devs_radius * motion_settings.p_noise), markerstrokewidth=0, alpha=0.25)
        plot!(Pose(trace[prefix_address(t, :pose => :p)], poses[t].hd); color=:red, label="sampled next step")
        plots[t] = frame_plot
    end
    return plots
end;

# %%
scaled_motion_settings(settings, x) = (p_noise = x * settings.p_noise, hd_noise = x * settings.hd_noise)

N_samples = 5

ani = Animation()
for n in 1:N_samples
    scale = 2. * (2.)^(n-N_samples)
    trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, scaled_motion_settings(motion_settings, scale)))
    frames = frames_from_motion_trace(world, "Motion model (samples)\nnoise factor $(round(scale, digits=3))", trace)
    for frame_plot in frames; frame(ani, frame_plot) end
end
gif(ani, "imgs/motion.gif", fps=2)

# %% [markdown]
# ### Updating traces, and improving performance using combinators.
#
# The metaprogramming approach of Gen affords the opportunity to explore alternate stochastic execution histories.  Namely, `Gen.update` takes as inputs a trace, together with modifications to its arguments and primitive choice values, and returns an accordingly modified trace.  It also returns (the log of) the ratio of the output trace's density to the input trace's density, together with a precise record of the resulting modifications to the trace.

# %% [markdown]
# In our example, one could, for instance, replace the first step's stochastic choice of heading with a specific value.

# %%
trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))
rotated_first_step, rotated_first_step_weight_diff, _, _ =
    update(trace,
           (T, robot_inputs, world_inputs, motion_settings), (NoChange(), NoChange(), NoChange(), NoChange()),
           choicemap((:steps => 1 => :pose => :hd, π/2.)))
the_plot = plot_world(world, "Modifying a heading")
plot!(get_path(trace); color=:green, label="Some path")
plot!(get_path(rotated_first_step); color=:red, label="With heading at first step modified")
savefig("imgs/modify_trace_1")
the_plot

# %% [markdown]
# In the above picture, the green path is apparently missing, having been near-completely overdrawn by the red path.  This is because in the execution of the model, the only change in the stochastic choices took place where we specified.  In particular, the stochastic choice of pose at the second step was left unchanged.  This choice was typical relative to the first step's heading in the old trace, and while it is not impossible relative to the first step's heading in the new trace, it is *far unlikelier* under the mulitvariate normal distribution supporting it.  This is the log of how much unlikelier:

# %%
rotated_first_step_weight_diff

# %% [markdown]
# One can also modify the arguments to the program.  In our example, we might have on hand a very long list of controls, and we wish to explore the space of paths incrementally in the timestep:

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
end;

# %% [markdown]
# This is a good opportunity to introduce some computational complexity considerations.
#
# Because the dynamic DSL does not understand the loop inside `path_model_loop`, calling `Gen.update` with the new value of `T` requires re-execution of the whole loop.  This means that the update requires $O(T)$ time, and the above code requires $O(T^2)$ time.
#
# But we humans understand that incrementing the argument `T` simply requires running the loop body once more.  This operation runs in $O(1)$ time, so the outer loop should require only $O(T)$ time.  Gen can intelligently work this way if we encode the structure of Markov chain in this model using a *combinator* for the static DSL, as follows.

# %%
@gen (static) function motion_path_kernel(t :: Int, state :: Pose, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Pose
    return {:pose} ~ step_model(state, robot_inputs.controls[t], world_inputs, motion_settings)
end
motion_path_chain = Unfold(motion_path_kernel)

@gen (static) function path_model(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Vector{Pose}
    initial = {:initial => :pose} ~ start_pose_prior(robot_inputs.start, motion_settings)
    {:steps} ~ motion_path_chain(T, initial, robot_inputs, world_inputs, motion_settings)
end;

# %% [markdown]
# The models `path_model_loop` and `path_model` have been arranged to produce identically structured traces with the same frequencies and return values, and to correspond to identical distributions over traces in the mathematical picture, thereby yielding the same weights.  They give rise to identical computations under `Gen.simulate`, whereas the new model is sometimes more efficient under `Gen.update`.  Here we illustrate the efficiency gain.
#
# (The following cell may need to be rerun to fix Julia garbage collection artifacts.)

# %%
N_repeats = 100
robot_inputs_long = (robot_inputs..., controls = reduce(vcat, [robot_inputs.controls for _ in 1:N_repeats]))

time_ends_loop = Vector(undef, T * N_repeats)
time_start = now()
trace = simulate(path_model_loop, (0, robot_inputs_long, world_inputs, motion_settings))
for t in 1:(T * N_repeats)
    trace, _, _, _ = update(trace, (t, robot_inputs_long, world_inputs, motion_settings), change_only_T, choicemap())
    time_ends_loop[t] = now()
end
time_diffs_loop = dv.(time_ends_loop - [time_start, time_ends_loop[1:end-1]...])
println("Explicit loop: $(dv(time_ends_loop[end]-time_start))ms")

time_ends_chain = Vector(undef, T * N_repeats)
time_start = now()
trace = simulate(path_model, (0, robot_inputs_long, world_inputs, motion_settings))
for t in 1:(T * N_repeats)
    trace, _, _, _ = update(trace, (t, robot_inputs_long, world_inputs, motion_settings), change_only_T, choicemap())
    time_ends_chain[t] = now()
end
time_diffs_chain = dv.(time_ends_chain - [time_start, time_ends_chain[1:end-1]...])
println("Markov chain combinator: $(dv(time_ends_chain[end]-time_start))ms")

the_plot = plot([range(1, T * N_repeats)...], time_diffs_loop; label="Explicit loop", title="Gen.update steps into trace", xlabel="t'th step", ylabel="time (ms)")
plot!([range(1, T * N_repeats)...], time_diffs_chain; label="Markov chain combinator")
savefig("imgs/dynamic_static_comparison")
the_plot

# %% [markdown]
# Owing to the efficiency comparison, we eschew `path_model_loop` in favor of `path_model` in what follows.  Thus we finally write our noisy path integration wrapper.

# %%
"""
Assumes
* `robot_inputs` contains fields: `start`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `motion_settings` contains fields: `p_noise`, `hd_noise`
"""
function integrate_controls_noisy(robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Vector{Pose}
    return get_path(simulate(path_model, (length(robot_inputs.controls), robot_inputs, world_inputs, motion_settings)))
end;

# %% [markdown]
# ### Ideal sensors
#
# We now, additionally, assume the robot is equipped with sensors that cast rays upon the environment at certain angles relative to the given pose, and return the distance to a hit.
#
# We first describe the ideal case, where the sensors return the true distances to the walls.

# %%
function sensor_distance(pose :: Pose, walls :: Vector{Segment}, box_size :: Float64) :: Float64
    d = minimum(distance(pose, seg) for seg in walls)
    # Capping to a finite value avoids issues below.
    return isinf(d) ? 2. * box_size : d
end;

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`
"""
sensor_angle(sensor_settings :: NamedTuple, j :: Int64) =
    sensor_settings.fov * (j - (sensor_settings.num_angles - 1) / 2.) / (sensor_settings.num_angles - 1)

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
function ideal_sensor(pose :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple) :: Vector{Float64}
    readings = Vector{Float64}(undef, sensor_settings.num_angles)
    for j in 1:sensor_settings.num_angles
        sensor_pose = rotate_pose(pose, sensor_angle(sensor_settings, j))
        readings[j] = sensor_distance(sensor_pose, walls, sensor_settings.box_size)
    end
    return readings
end;

# %%
# Plot sensor data.

"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`
"""
function plot_sensors!(pose, color, readings, label, sensor_settings)
    plot!([pose.p[1]], [pose.p[2]]; color=color, label=nothing, seriestype=:scatter, markersize=3, markerstrokewidth=0)
    projections = [step_along_pose(rotate_pose(pose, sensor_angle(sensor_settings, j)), s) for (j, s) in enumerate(readings)]
    plot!(first.(projections), last.(projections);
            color=:blue, label=label, seriestype=:scatter, markersize=3, markerstrokewidth=1, alpha=0.25)
    plot!([Segment(pose.p, pr) for pr in projections]; color=:blue, label=nothing, alpha=0.25)
end

function frame_from_sensors(world, title, poses, poses_color, poses_label, pose, readings, readings_label, sensor_settings; show_clutters=false)
    the_plot = plot_world(world, title; show_clutters=show_clutters)
    plot!(poses; color=poses_color, label=poses_label)
    plot_sensors!(pose, poses_color, readings, readings_label, sensor_settings)
    return the_plot
end;

# %%
sensor_settings = (fov = 2π*(2/3), num_angles = 41, box_size = world.box_size)

ani = Animation()
for pose in path_integrated
    frame_plot = frame_from_sensors(
        world, "Ideal sensor distances",
        path_integrated, :green2, "some path",
        pose, ideal_sensor(pose, world.walls, sensor_settings), "ideal sensors",
        sensor_settings)
    frame(ani, frame_plot)
end
gif(ani, "imgs/ideal_distances.gif", fps=1)

# %% [markdown]
# ### Noisy sensors
#
# We assume that the sensor readings are themselves uncertain, say, the distances only knowable up to some noise.  We model this as follows.  (We satisfy ourselves with writing a loop in the dynamic DSL because we will have no need for incremental recomputation within this model.)

# %%
"""
Assumes
* `sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen function sensor_model(pose :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple) :: Vector{Float64}
    for j in 1:sensor_settings.num_angles
        sensor_pose = rotate_pose(pose, sensor_angle(sensor_settings, j))
        {j => :distance} ~ normal(sensor_distance(sensor_pose, walls, sensor_settings.box_size), sensor_settings.s_noise)
    end
end

function noisy_sensor(pose :: Pose, walls :: Vector{Segment}, sensor_settings :: NamedTuple) :: Vector{Float64}
    trace = simulate(sensor_model, (pose, walls, sensor_settings))
    return [trace[j => :distance] for j in 1:sensor_settings.num_angles]
end;

# %% [markdown]
# The trace contains many choices corresponding to directions of sensor reading from the input pose.  To reduce notebook clutter, here we just show a subset of 5 of them:

# %%
sensor_settings = (sensor_settings..., s_noise = 0.10)

trace = simulate(sensor_model, (robot_inputs.start, world.walls, sensor_settings))
get_selected(get_choices(trace), select((1:5)...))

# %% [markdown]
# The mathematical picture is as follows.  Given the parameters of a pose $y$, walls $w$, and settings $\nu$, one gets a distribution $\text{sensor}(y, w, \nu)$ over the traces of `sensor_model`, and when $z$ is a motion model trace we set $\text{sensor}(z, w, \nu) := \text{sensor}(\text{retval}(z), w, \nu)$.  It samples are identified with vectors $o = (o^{(1)}, o^{(2)}, \ldots, o^{(J)})$, where $J := \nu_\text{num\_angles}$, each $o^{(j)}$ independently following a certain normal distribution (depending, notably, on the distance from the pose to the nearest wall).  Thus the density of $o$ factors into a product of the form
# $$
# P_\text{sensor}(o) = \prod\nolimits_{j=1}^J P_\text{normal}(o^{(j)})
# $$
# where we begin a habit of omitting the parameters to distributions that are implied by the code.
#
# Visualizing the traces of the model is probably more useful for orientation, so we do this now.

# %%
# TODO: Add settings/(hyper)params display code.
function frame_from_sensors_trace(world, title, poses, poses_color, poses_label, pose, trace; show_clutters=false)
    readings = [trace[j => :distance] for j in 1:sensor_settings.num_angles]
    return frame_from_sensors(world, title, poses, poses_color, poses_label, pose,
                             readings, "trace sensors", get_args(trace)[3];
                             show_clutters=show_clutters)
end;

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

# %% [markdown]
# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", which whose traces describe simulations of the entire robot situation as we have described it.

# %%
"""
Assumes
* `robot_inputs` contains fields: `start`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function full_model_initial(robot_inputs :: NamedTuple, walls :: Vector{Segment}, full_settings :: NamedTuple)  :: Pose
    pose ~ start_pose_prior(robot_inputs.start, full_settings.motion_settings)
    {:sensor} ~ sensor_model(pose, walls, full_settings.sensor_settings)
    return pose
end

"""
Assumes
* `robot_inputs` contains fields: `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function full_model_kernel(t :: Int, state :: Pose, robot_inputs :: NamedTuple, world_inputs :: NamedTuple,
                                      full_settings :: NamedTuple) :: Pose
    pose ~ step_model(state, robot_inputs.controls[t], world_inputs, full_settings.motion_settings)
    {:sensor} ~ sensor_model(pose, world_inputs.walls, full_settings.sensor_settings)
    return pose
end
full_model_chain = Unfold(full_model_kernel)

"""
Assumes
* `robot_inputs` contains fields: `start`, `controls`
* `world_inputs` contains fields: `walls`, `bounce`
* `full_settings` contains fields: `motion_settings`, `sensor_settings`
    * `full_settings.motion_settings` contains fields: `p_noise`, `hd_noise`
    * `full_settings.sensor_settings` contains fields: `fov`, `num_angles`, `box_size`, `s_noise`
"""
@gen (static) function full_model(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, full_settings :: NamedTuple) :: Nothing
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
# &= \big(P_\text{init}(z_0)\ P_\text{sensor}(o_0)\big)
#   \cdot \prod\nolimits_{t=1}^T \big(P_\text{step}(z_t)\ P_\text{sensor}(o_t)\big).
# \end{align*}$$
#
# By this point, visualization is essential.

# %%
function frames_from_full_trace(world, title, trace; show_clutters=false, std_devs_radius=2.)
    T = get_args(trace)[1]
    robot_inputs = get_args(trace)[2]
    poses = get_path(trace)
    noiseless_steps = [robot_inputs.start.p, [pose.p + c.ds * pose.dp for (pose, c) in zip(poses, robot_inputs.controls)]...]
    settings = get_args(trace)[4]
    sensor_readings = get_sensors(trace)
    plots = Vector{Plots.Plot}(undef, 2*(T+1))
    for t in 1:(T+1)
        frame_plot = plot_world(world, title; show_clutters=show_clutters)
        plot!(poses[1:t-1]; color=:black, label=nothing)
        plot!([noiseless_steps[t][1]], [noiseless_steps[t][2]];
              color=:red, label=nothing, seriestype=:scatter,
              markersize=(20. * std_devs_radius * settings.motion_settings.p_noise), markerstrokewidth=0, alpha=0.25)
        plot!(Pose(trace[prefix_address(t, :pose => :p)], poses[t].hd); color=:red, label=nothing)
        plots[2*t-1] = frame_plot
        plots[2*t] = frame_from_sensors(
            world, title,
            poses[1:t], :black, nothing,
            poses[t], sensor_readings[t], nothing,
            settings.sensor_settings; show_clutters=show_clutters)
    end
    return plots
end;

# %%
scaled_full_settings(settings, x) = (settings..., motion_settings=scaled_motion_settings(settings.motion_settings, x))

N_samples = 5

ani = Animation()
for n in 1:N_samples
    scale = 2. * (2.)^(n-N_samples)
    trace = simulate(full_model, (T, robot_inputs, world_inputs, scaled_full_settings(full_settings, scale)))
    frames = frames_from_full_trace(world, "Full model (samples)\nnoise factor $(round(scale, digits=3))", trace)
    for frame_plot in frames; frame(ani, frame_plot) end
end
gif(ani, "imgs/full_1.gif", fps=2)

# %% [markdown]
# ### Aside: abstracting essential features of generative functions, traces, and weights

# %% [markdown]
# #### Tree structure

# %%
@gen function example_sub_gf(p)
    foo ~ bernoulli(p)
    return foo ? "asdf" : -4
end

@gen function example_sup_gf(p)
    {:call} ~ example_sub_gf(p)
    {:bar} ~ normal(0., 1.)
    return 7
end

example_trace = simulate(example_sup_gf, (0.5,))
get_choices(example_trace)

# %% [markdown]
# #### Mathy picture
#
# TBD REVISE SIMPLIFY:
#
# For a particular GF, we call another function a *well-defined function of its traces* if it is a function of traces that invokes only valid choice addresses for every possible stochastic execution of the GF.
#
# [Some particular GFs, upon examination, may be determined to only produce traces satisfying certain constraints, so that certain functions of their traces are valid for all possible stochastic executions.  (For a conservative example, certain addresses might always exist, and the function might extract only these choices from the traces.)  We call the latter function a *well-defined function on traces* of the former GF.]
#
# A generative function in Gen can be mathematically modeled as specifying a probability distribution over the space of all traces, with some subset thereof as its support.  A well-defined function of its traces then corresponds to a random variable.
#
#
#
# Generally, if $P$ is a distribution depending on parameters $\theta$, then we write $P(z; \theta)$ for the probability density associated to the value $z$, and we write $z \sim P(\cdot\,; \theta)$ to declare that the value $z$ has been sampled according to these densities.  Thus the semicolon delimits general parameters.  When parameters are fixed and understood from context, we may delete them and write, say, $P(z)$ or $z \sim P(\cdot)$.
#
#

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
    frame_plot = plot(low, high; size=(1000,500), plot_title="Fixed synthetic data")
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
observations_high_deviation = get_sensors(trace_high_deviation);

# %% [markdown]
# We summarize the information available to the robot to determine its location.  On the one hand, one has guess of the start pose plus some controls, which one might integrate to produce an idealized guess of path.  On the other hand, one has the sensor data.

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
    plot_integrated = plot_world(world, "Startup data")
    plot!(path_integrated[1]; color=:green, label="start guess")
    if t > 1; annotate!(5, 2.5, "Control $(t-1):\n$(short_control(robot_inputs.controls[t-1]))") end

    plot_low = plot_bare_sensors(world, "Low motion deviation", readings_low, "fixed sensor data", sensor_settings)
    plot!(Pose(world.center_point, 0.0); color=:black, label=nothing)

    plot_high = plot_bare_sensors(world, "High motion deviation", readings_high, "fixed sensor data", sensor_settings)
    plot!(Pose(world.center_point, 0.0); color=:black, label=nothing)

    the_frame = plot(plot_integrated, plot_low, plot_high; size=(1500,500), layout=grid(1,3), plot_title="<— Data available to robot —>")
    frame(ani, the_frame)
end
gif(ani, "imgs/robot_can_see.gif", fps=2)

# %% [markdown]
# ## Why we need inference

# %% [markdown]
# ### In a picture
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
# We are not limited to intuitive judgments here: the model can quantitatively tell us how good a fit it is for the data.  Namely, we can compare the likelihoods of the observation data in typical samples produced by the model, to the likelihoods of our fixed observation data under samples from the model agreeing these data.
#
# In order to do this, we detour to explain how to produce samples from our model that agree with the fixed observation data.

# %% [markdown]
# ### Generating samples with constraints
#
# We have seen how `Gen.simulate` performs traced execution of a generative function: as the program runs, it draws stochastic choices from all required primitive distributions, and records them in a choice map.
#
# The operation `Gen.generate` performs a generalization of this process.  One also provides a choice map of *constraints* that declare fixed values for some of these primitive choices.  As the program runs, any primitive choice that has been named by the constraints is deterministically given the specified value, and otherwise it is drawn stochastically as in `Gen.simulate`.
#
# The trace resulting from a call to `Gen.generate` is indistinguishable from `Gen.simulate`, having the same kind of choice map, in turn having the same assignments of densities to its nodes according to the primitive distributions.  But there is a key situational difference: the total density is *no longer equal to* the frequency with which the trace stochastically occurs under the model.
#
# RATIO IS THE IMPORTANCE WEIGHT, EQUALS PROJECT OF CONSTRAINT ADDRESSES IN RESULTING TRACE, AND IS ALSO RETURNED.

# %% [markdown]
# ### Picturing generated samples

# %%
# Encode sensor readings into choice map.

constraint_from_sensors(t :: Int, readings :: Vector{Float64}) :: ChoiceMap =
    choicemap(( (prefix_address(t, :sensor => j => :distance), reading) for (j, reading) in enumerate(readings) )...)
constraint_from_sensors(tuple :: Tuple) = constraint_from_sensors(tuple...);

# %%
N_samples = 200

selection = select((prefix_address(i, :sensor => j => :distance) for i in 1:(T+1), j in 1:sensor_settings.num_angles)...)
traces_typical = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
log_likelihoods_typical = [project(trace, selection) for trace in traces_typical]
hist_typical = histogram(log_likelihoods_typical; label=nothing, bins=20, title="typical data")

constraints_low_deviation = constraint_from_sensors.(enumerate(observations_low_deviation))
merged_constraints_low_deviation = merge(constraints_low_deviation...)
traces_generated_low_deviation = [generate(full_model, (T, full_model_args...), merged_constraints_low_deviation)[1] for _ in 1:N_samples]
log_likelihoods_low_deviation = [project(trace, selection) for trace in traces_generated_low_deviation]
hist_low_deviation = histogram(log_likelihoods_low_deviation; label=nothing, bins=20, title="low dev data")

constraints_high_deviation = constraint_from_sensors.(enumerate(observations_high_deviation))
merged_constraints_high_deviation = merge(constraints_high_deviation...)
traces_generated_high_deviation = [generate(full_model, (T, full_model_args...), merged_constraints_high_deviation)[1] for _ in 1:N_samples]
log_likelihoods_high_deviation = [project(trace, selection) for trace in traces_generated_high_deviation]
hist_high_deviation = histogram(log_likelihoods_high_deviation; label=nothing, bins=20, title="high dev data")

the_plot = plot(hist_typical, hist_low_deviation, hist_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="Log likelihood of observations under the model")
savefig("imgs/likelihoods")
the_plot

# %% [markdown]
# Note the differences in scales along the bottom.

# %% [markdown]
# ## Inference: main idea
#
# In the viewpoint of ProbComp, the goal of *inference* is to produce *likely* traces of a full model, given the observed data.  In other words, as generative functions induce distributions on traces, and if we view the full model as a program embodying a *prior*, then applying an inference metaprogram to it (together with the observed data) produces a new program that embodies the *posterior*.

# %% [markdown]
# Mathematically, the passage from the prior to the posterior is the operation of conditioning distributions.  Namely, one defines first the *marginal distribution* over observations to have density
# $$
# P_\text{full}(o_{0:T})
# := \int P_\text{full}(Z_{0:T}, o_{0:T}) \, dZ_{0:T}
#  = \mathbf{E}_{Z_{0:T} \sim \text{path}}\big[P_\text{full}(Z_{0:T}, o_{0:T})\big],
# $$
# and then the *conditional distribution* has density
# $$
# P_\text{full}(z_{0:T} | o_{0:T}) := \frac{P_\text{full}(z_{0:T}, o_{0:T})}{P_\text{full}(o_{0:T})}.
# $$
# The goal of inference is to produce samples $\text{trace}_{0:T}$ distributed approximately according to the latter distribution.

# %% [markdown]
# Let's show what we mean with a picture.  The following short code, which we treat as a *black box* for the present purposes, very mildly exploits the model structure to bring the samples much closer to the true path.

# %%
# The code in this cell is the black box!

# Propose a move for MH.

@gen function drift_proposal(trace, drift_step_factor)
    t = get_args(trace)[1] + 1

    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    # Form expected by `mh` in library code, immediately following.
    fwd_p = {prefix_address(t, :pose => :p)} ~ mvnormal(p, drift_step_factor * p_noise^2 * [1 0 ; 0 1])
    fwd_hd = {prefix_address(t, :pose => :hd)} ~ normal(hd, hd_noise)

    # Form expected by `mh_step`, further below.
    return (choicemap((prefix_address(t, :pose => :p), fwd_p), (prefix_address(t, :pose => :hd), fwd_hd)),
            choicemap((prefix_address(t, :pose => :p), p), (prefix_address(t, :pose => :hd), hd)))
end

# PF with rejuvenation, using `GenParticleFilters` library code for the generic parts.

function particle_filter_rejuv_library(model, T, args, constraints, N_particles, N_MH, MH_proposal, MH_proposal_args)
    state = pf_initialize(model, (0, args...), constraints[1], N_particles)
    for t in 1:T
        pf_resample!(state)
        pf_rejuvenate!(state, mh, (MH_proposal, MH_proposal_args), N_MH)
        pf_update!(state, (t, args...), change_only_T, constraints[t+1])
    end
    return state.traces, state.log_weights
end

# Run PF and return one of its particles.

function sample(particles, log_weights)
    log_total_weight = logsumexp(log_weights)
    norm_weights = exp.(log_weights .- log_total_weight)
    index = categorical(norm_weights)
    return particles[index], log_weights[index]
end

function sample_from_posterior(model, T, args, constraints; N_MH = 10, N_particles = 10)
    drift_step_factor = 1/3.
    return sample(particle_filter_rejuv_library(model, T, args, constraints, N_particles, N_MH, drift_proposal, (drift_step_factor,))...)
end;

# %%
# Visualize distributions over traces.

function frame_from_traces(world, title, path_actual, traces, trace_label; show_clutters=false)
    the_plot = plot_world(world, title; show_clutters=show_clutters)
    if !isnothing(path_actual); plot!(path_actual; label="actual path", color=:brown) end
    for trace in traces
        poses = get_path(trace)
        plot!([p.p[1] for p in poses], [p.p[2] for p in poses]; label=nothing, color=:green, alpha=0.3)
        plot!(Segment.(zip(poses[1:end-1], poses[2:end]));
              label=trace_label, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=0.3)
        trace_label = nothing
    end
    return the_plot
end;

# %% [markdown]
# Here is a visual comparison.

# %%
N_samples = 10

traces = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
prior_plot = frame_from_traces(world, "Prior on robot paths", nothing, traces, "prior samples")

traces = [sample_from_posterior(full_model, T, full_model_args, constraints_low_deviation)[1] for _ in 1:N_samples]
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, traces, "posterior samples")

traces = [sample_from_posterior(full_model, T, full_model_args, constraints_high_deviation)[1] for _ in 1:N_samples]
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, traces, "posterior samples")

the_plot = plot(prior_plot, posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="Prior vs. posteriors")
savefig("imgs/prior_posterior")
the_plot

# %% [markdown]
# Numerical comparison

# %%
N_samples = 100

selection = select((prefix_address(i, :sensor => j => :distance) for i in 1:(T+1), j in 1:sensor_settings.num_angles)...)
traces_typical = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
log_likelihoods_typical = [project(trace, selection) for trace in traces_typical]
hist_typical = histogram(log_likelihoods_typical; label=nothing, bins=20, title="typical data under prior")

traces_posterior_low_deviation = [sample_from_posterior(full_model, T, full_model_args, constraints_low_deviation; N_MH=10, N_particles=10)[1] for _ in 1:N_samples]
log_likelihoods_low_deviation = [project(trace, selection) for trace in traces_posterior_low_deviation]
hist_low_deviation = histogram(log_likelihoods_low_deviation; label=nothing, bins=20, title="typical data under posterior: low dev data")

traces_posterior_high_deviation = [sample_from_posterior(full_model, T, full_model_args, constraints_high_deviation; N_MH=10, N_particles=10)[1] for _ in 1:N_samples]
log_likelihoods_high_deviation = [project(trace, selection) for trace in traces_posterior_high_deviation]
hist_high_deviation = histogram(log_likelihoods_high_deviation; label=nothing, bins=20, title="typical data under posterior: high dev data")

the_plot = plot(hist_typical, hist_low_deviation, hist_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="Log likelihood of observations")
savefig("imgs/likelihoods")
the_plot

# %% [markdown]
# ## Generic strategies for inference
#
# We now spell out some generic strategies for conditioning the ouputs of our model towards the observed sensor data.  The word "generic" indicates that they make no special intelligent use of the model structure, and their convergence is guaranteed by theorems of a similar nature.
#
# There is no free lunch in this game: generic inference recipies are inefficient, for example, converging very slowly or needing vast counts of particles, especially in high-dimensional settings.  Rather, efficiency will later become possible when we exploit what we actually know about the problem in our design of the inference strategy.  Gen's aim is to provide the right entry points to enact this exploitation.

# %% [markdown]
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
function basic_SIR(model, args, merged_constraints, N_SIR)
    traces = Vector{Trace}(undef, N_SIR)
    log_weights = Vector{Float64}(undef, N_SIR)
    for i in 1:N_SIR
        traces[i], log_weights[i] = generate(model, args, merged_constraints)
    end
    return sample(traces, log_weight)
end

# This is a generic algorithm, so there is a library version.
# We will the library version use going forward, because it includes a constant-memory optimization.
# (It is not necessary to store all particles and categorically select one at the end.  Mathematically
# it amounts to the same instead to store just one candidate selection, and stochastically replace it
# with each newly generated particle with odds the latter's weight relative to the sum of the
# preceding weights.)
# To obtain the above from the library version, one would define:

basic_SIR_library(model, args, merged_constraints, N_SIR) = importance_resampling(model, args, merged_constraints, N_SIR);

# %% [markdown]
# Let us first consider a shorter robot path, but, to keep it interesting, allow a higher deviation from the ideal.

# %%
T_short = 4

robot_inputs_short = (robot_inputs..., controls=robot_inputs.controls[1:T_short])
full_model_args_short = (robot_inputs_short, world_inputs, full_settings)

path_integrated_short = path_integrated[1:(T_short+1)]
path_actual_short = path_actual[1:(T_short+1)]
observations_short = observations[1:(T_short+1)]
constraints_short = constraints[1:(T_short+1)]

ani = Animation()
for (pose_actual, pose_integrated, readings) in zip(path_actual_short, path_integrated_short, observations_short)
    actual_plot = frame_from_sensors(
        world, "Actual data",
        path_actual_short, :brown, "actual path",
        pose_actual, readings, "actual sensors",
        sensor_settings)
    integrated_plot = frame_from_sensors(
        world, "Apparent data",
        path_integrated_short, :green2, "path from integrating controls",
        pose_integrated, readings, "actual sensors",
        sensor_settings)
    frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(shortened path)")
    frame(ani, frame_plot)
end
gif(ani, "imgs/discrepancy_short.gif", fps=1)

# %% [markdown]
# For such a shorter path, SIR can find a somewhat noisy fit without too much effort.
#
# Rif asks
# > In `traces = ...` below, are you running SIR `N_SAMPLES` times and getting one sample each time? Why not run it once and get `N_SAMPLES`? Talk about this?

# %%
N_samples = 10
N_SIR = 500
traces = [basic_SIR_library(full_model, (T_short, full_model_args_short...), constraints_short, N_SIR)[1] for _ in 1:N_samples]

the_plot = frame_from_traces(world, "SIR (short path)", path_actual_short, traces)
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
function rejection_sample(model, args, merged_constraints, N_burn_in, N_particles, MAX_attempts)
    particles = []
    C = maximum(generate(model, args, merged_constraints)[2] for _ in 1:N_burn_in; init=-Inf)

    for _ in 1:N_particles
        attempts = 0
        while attempts < MAX_attempts
            attempts += 1

            particle, weight = generate(model, args, merged_constraints)
            if weight > C
                C = weight
                push!(particles, particle)
                break
            elseif weight > C + log(rand())
                push!(particles, particle)
                break
            end
        end
    end

    return particles
end;

# %%
T_RS = 9
path_actual_RS = path_actual[1:(T_RS+1)]
constraints_RS = constraints[1:(T_RS+1)];

# %%
N_burn_in = 0 # omit burn-in to illustrate early behavior
N_particles = 20
compute_bound = 5000
traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %%
N_burn_in = 100
N_particles = 20
compute_bound = 5000
traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %%
N_burn_in = 1000
N_particles = 20
compute_bound = 5000
traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

ani = Animation()
for (i, trace) in enumerate(traces)
    frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
    frame(ani, frame_plot)
end
gif(ani, "imgs/RS.gif", fps=1)

# %% [markdown]
# The performance of this algorithm varies wildly!  Without the `MAX_attempts` way out, it may take a long time to run; and with, it may produce few samples.

# %% [markdown]
# ### SIR and Adaptive Rejection Sampling scale poorly
#
# SIR does not scale because for longer paths, the search space is too large, and the results are only modestly closer to the posterior.
#
# Adaptive rejection sampling suffers from a similar issue.
#
# Below, we show SIR run on a long path to illustrate the type of poor inference results which arise from these algorithms.

# %%
N_samples = 10
N_SIR = 500
traces = [basic_SIR_library(full_model, (T, full_model_args...), constraints, N_SIR)[1] for _ in 1:N_samples]

the_plot = frame_from_traces(world, "SIR (original path)", path_actual, traces)
savefig("imgs/SIR")
the_plot

# %% [markdown]
# ## Sequential Monte Carlo (SMC) techniques
#
# We now begin to exploit the structure of the problem in significant ways to construct good candidate traces for the posterior.  Especially, we use the Markov chain structure to construct these traces step-by-step.  While generic algorithms like SIR and rejection sampling must first construct full paths $\text{trace}_{0:T}$ and then sift among them using the observations $o_{0:T}$, we may instead generate one $\text{trace}_t$ at a time, taking into account the datum $o_t$.  Since then one is working with only a few dimensions any one time step, more intelligent searches become computationally feasible.

# %% [markdown]
# ### Particle filter
#
# One of the simplest manifestations of the preceding strategy is called a particle filter, which, roughly speaking, looks like a kind of incremental SIR.  One constructs a population of traces in parallel; upon constructing each new step of the traces, one assesses how well they fit the data, discarding the worse ones and keeping more copies of the better ones.
#
# More precisely:
#
# In the initial step, we draw $N$ samples $z_0^1, z_0^2, \ldots, z_0^N$ from the distribution $\text{start}$, which we call *particles*.
#
# There are iterative steps for $t = 1, \ldots, T$.  In the iterative step $t$, we have already constructed $N$ particles of the form $z_{0:{t-1}}^1, z_{0:t-1}^2, \ldots, z_{0:t-1}^N$.  First we *resample* them as follows.  Each particle is assigned a *weight*
# $$
# w^i := \frac{P_\text{full}(z_{0:t-1}^i, o_{0:t-1})}{P_\text{path}(z_{0:t-1}^i)}.
# $$
# The normalized weights $\hat w^i := w^i / \sum_{j=1}^n w^j$ define a categorical distribution on indices $i = 1, \ldots, N$, and for each index $i$ we *sample* a new index $a^i$ accordingly.  We *replace* the list of particles with the reindexed list $z_{0:t-1}^{a^1}, z_{0:t-1}^{a^2}, \ldots, z_{0:t-1}^{a^N}$.  Finally, having resampled thusly, we *extend* each particle $z_{0:t-1}^i$ to a particle of the form $z_{0:t}^i$ by drawing a sample $z_t^i$ from $\text{step}(z_{t-1}^i, \ldots)$.

# %% [markdown]
# WHY DOES `Gen.generate` GIVE THE SAME WEIGHTS AS ABOVE?

# %%
function resample!(particles, log_weights)
    log_total_weight = logsumexp(log_weights)
    norm_weights = exp.(log_weights .- log_total_weight)
    particles .= [particles[categorical(norm_weights)] for _ in particles]
    log_weights .= log_total_weight - log(length(log_weights))
end

function particle_filter(model, T, args, constraints, N_particles)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end
    
    for t in 1:T
        resample!(traces, log_weights)

        for i in 1:N_particles
            traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
            log_weights[i] += log_weight_increment
        end
    end

    return traces, log_weights
end;

# %% [markdown]
# Pictures and discussion of the drawbacks.

# %% [markdown]
# ### Prospects for improving accuracy, robustness, and efficiency
#
# One approach:
# * Improve accuracy with more particles.
# * Improve efficiency with smarter resamples (ESS, stratified...).
# * Hope robustness is good enough.
#
# Clearly not going to be fundamentally better than scaling a large NN, which is similar, just with offline training.
#
# ProbComp advocates instead:
# * A smart algorithm that fixes probable mistakes as it goes along.
# * One idea: fix mistakes by running MH on each particle.  If MH changes them, then mistakes were fixed.
#   * With generic Gaussian drift proposal.
#   * An improvement: grid MH.
# * Another idea: run SMCP3.
#   * Get correct weights —>
#     * algorithm has an estimate of its inference quality (math TBE: AIDE, EEVI papers)
#     * higher quality resampling
# * How good can we do, even with one particle?
#   * Controller

# %% [markdown]
# ### MCMC (MH) rejuvenation
#
# Two issues: particle diversity after resampling, and quality of these samples.

# %%
function mh_step(trace, proposal, proposal_args)
    _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap) = propose(proposal, (trace, proposal_args...))
    proposed_trace, model_weight_diff, _, _ = update(trace, fwd_model_update)
    bwd_proposal_weight, _ = assess(proposal, (proposed_trace, proposal_args...), bwd_proposal_choicemap)
    log_weight_increment = model_weight_diff + bwd_proposal_weight - fwd_proposal_logprob
    return (log(rand()) < log_weight_increment ? proposed_trace : trace), 0.
end
mh_kernel(proposal) =
    (trace, proposal_args) -> mh_step(trace, proposal, proposal_args);

# %% [markdown]
# Then PF+Rejuv code.

# %%
function resample!(particles, log_weights, ESS_threshold)
    log_total_weight = logsumexp(log_weights)
    log_norm_weights = log_weights .- log_total_weight
    if effective_sample_size(log_norm_weights) < ESS_threshold
        norm_weights = exp.(log_norm_weights)
        particles .= [particles[categorical(norm_weights)] for _ in particles]
        log_weights .= log_total_weight - log(length(log_weights))
    end
end


# Compare with the source code for the library calls used by `particle_filter_rejuv_library`!

function particle_filter_rejuv(model, T, args, constraints, N_particles, rejuv_kernel, rejuv_args_schedule, ESS_threshold=Inf)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    for t in 1:T
        resample!(traces, log_weights, ESS_threshold)

        for i in 1:N_particles
            for rejuv_args in rejuv_args_schedule
                traces[i], log_weight_increment = rejuv_kernel(traces[i], rejuv_args)
                log_weights[i] += log_weight_increment
            end
        end

        for i in 1:N_particles
            traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
            log_weights[i] += log_weight_increment
        end
    end

    return traces, log_weights
end;

# %% [markdown]
# Note usage with drift proposal:

# %%
ESS_threshold =  1. + N_particles / 10.

drift_step_factor = 1/3.
drift_proposal_args = (drift_step_factor,)
N_MH = 10
drift_args_schedule = [drift_proposal_args for _ in 1:N_MH]
drift_mh_kernel = mh_kernel(drift_proposal)
particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, drift_mh_kernel, drift_args_schedule; ESS_threshold=ESS_threshold)

# %% [markdown]
# VISUALIZE

# %% [markdown]
# More exploration with drift proposal?

# %% [markdown]
# ### Grid proposal for MH
#
# Instead of a random walk strategy to improve next steps, the search space is small enough that we very well could search a small nearby area for improvement.

# %%
function vector_grid(center :: Vector{Float64}, grid_n_points :: Vector{Int}, grid_sizes :: Vector{Float64}) :: Vector{Vector{Float64}}
    offset = center .- (grid_n_points .+ 1) .* grid_sizes ./ 2.
    return reshape(map(I -> [Tuple(I)...] .* grid_sizes .+ offset, CartesianIndices(Tuple(grid_n_points))), (:,))
end

inverse_grid_index(grid_n_points :: Vector{Int}, j :: Int) :: Int =
    LinearIndices(Tuple(grid_n_points))[(grid_n_points .+ 1 .- [Tuple(CartesianIndices(Tuple(grid_n_points))[j])...])...]

@gen function grid_proposal(trace, grid_n_points, grid_sizes)
    t = get_args(trace)[1] + 1
    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    choicemap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]), (prefix_address(t, :pose => :hd), h))
                      for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
    pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    j ~ categorical(pose_norm_weights)
    inv_j = inverse_grid_index(grid_n_points, j)

    return choicemap_grid[j], choicemap((:j, inv_j))
end;

# %% [markdown]
# Should be able to:

# %%
grid_args_schedule = ...
grid_mh_kernel = mh_kernel(grid_proposal)
particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_mh_kernel, grid_args_schedule; ESS_threshold=ESS_threshold)

# %% [markdown]
# ### Properly weighted samples
#
# Improve later resampling / end-to-end quality.

# %% [markdown]
# ### SMCP3 rejuvenation
#
# Takes the following shape:

# %%
function smcp3_step(trace, fwd_proposal, bwd_proposal, proposal_args)
    _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap) = propose(fwd_proposal, (trace, proposal_args...))
    proposed_trace, model_weight_diff, _, _ = update(trace, fwd_model_update)
    bwd_proposal_weight, _ = assess(bwd_proposal, (proposed_trace, proposal_args...), bwd_proposal_choicemap)
    log_weight_increment = model_weight_diff + bwd_proposal_weight - fwd_proposal_weight
    return proposed_trace, log_weight_increment
end
smcp3_kernel(fwd_proposal, bwd_proposal) =
    (trace, proposal_args) -> smcp3_step(trace, fwd_proposal, bwd_proposal, proposal_args);

# %% [markdown]
# Let us write the forward and backward transformations for the grid proposal.

# %%
@gen function grid_fwd_proposal(trace, grid_n_points, grid_sizes)
    t = get_args(trace)[1] + 1
    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    choicemap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]), (prefix_address(t, :pose => :hd), h))
                      for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
    pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    fwd_j ~ categorical(pose_norm_weights)
    bwd_j = inverse_grid_index(grid_n_points, fwd_j)

    return choicemap_grid[fwd_j], choicemap((:bwd_j, bwd_j))
end

@gen function grid_bwd_proposal(trace, grid_n_points, grid_sizes)
    prev_t, robot_inputs, world_inputs, settings = get_args(trace)
    t = prev_t + 1
    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    # TODO: Would be more intuitive if these same weights were obtained by restricting `trace` to `prev_t`,
    # then updating it back out to `t` with these steps.
    choicemap_grid = [choicemap((:p, [x, y]), (:hd, h))
                      for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
    if prev_t == 0
        assess_model = start_pose_prior
        assess_args = (robot_inputs.start, settings.motion_settings)
    else
        assess_model = step_model
        prev_p = trace[prefix_address(prev_t, :pose => :p)]
        prev_hd = trace[prefix_address(prev_t, :pose => :hd)]
        assess_args = (Pose(prev_p, prev_hd), robot_inputs.controls[prev_t], world_inputs, settings.motion_settings)
    end
    pose_log_weights = [assess(assess_model, assess_args, cm)[1] for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    bwd_j ~ categorical(pose_norm_weights)
    fwd_j = inverse_grid_index(grid_n_points, bwd_j)

    return choicemap_grid[bwd_j], choicemap((:fwd_j, fwd_j))
end;

# %%
grid_smcp3_kernel = smcp3_kernel(grid_fwd_proposal, grid_bwd_proposal)
particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_smcp3_kernel, grid_args_schedule; ESS_threshold=ESS_threshold)

# %% [markdown]
# ### Adaptive inference controller

# %%
function controlled_particle_filter_rejuv(model, T, args, constraints, N_particles, rejuv_kernel, rejuv_args_schedule, weight_change_bound, args_schedule_modifier;
                                          ESS_threshold=Inf, MAX_rejuv=3)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)

    prev_total_weight = 0.
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    for t in 1:T
        resample!(traces, log_weights, ESS_threshold)

        rejuv_count = 0
        temp_args_schedule = rejuv_args_schedule
        while logsumexp(log_weights) - prev_total_weight < weight_change_bound && rejuv_count <= MAX_rejuv

            for i in 1:N_particles
                for rejuv_args in rejuv_args_schedule
                    traces[i], log_weight_increment = rejuv_kernel(traces[i], rejuv_args)
                    log_weights[i] += log_weight_increment
                end
            end

            if logsumexp(log_weights) - prev_total_weight < weight_change_bound && rejuv_count != MAX_rejuv
                for i in 1:N_particles
                    traces[i], log_weight_increment, _, _ = regenerate(traces[i], select(prefix_address(t-1, :pose)))
                    log_weights[i] += log_weight_increment
                end

                resample!(traces, log_weights, ESS_threshold)
            end

            rejuv_count += 1
            temp_args_schedule = args_schedule_modifier(temp_args_schedule, rejuv_count)
        end

        prev_total_weight = logsumexp(log_weights)
        for i in 1:N_particles
            traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
            log_weights[i] += log_weight_increment
        end
    end
end;

# %%
weight_change_bound = (-1. * 10^5)/20

grid_args_schedule_modifier(args_schedule, rejuv_count) =
    (rejuv_count % 1 == 0) ? [(nsteps, sizes .* 0.75) for (nsteps, sizes) in args_schedule]
                           : [(nsteps + 2, sizes)     for (nsteps, sizes) in args_schedule];

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
function particle_filter_rejuv(model, T, args, constraints, N_particles, N_MH, MH_proposal, MH_proposal_args)
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
            traces[i], log_weights[i], _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
        end
    end

    return traces, log_weights
end
;
# # Alternatively, using library calls: `particle_filter_rejuv_library` from the black box above!

# %%
drift_step_factor = 1/3.

N_samples = 6
N_particles = 10
N_MH = 5
t1 = now()
traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints, N_particles,
                                N_MH, drift_proposal, (drift_step_factor,))[1][1] for _ in 1:N_samples]
t2 = now()
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")

the_plot = frame_from_traces(world, "PF+Drift Rejuv", path_actual, traces)
savefig("imgs/PF_rejuv")
the_plot

# %% [markdown]
# ## Grid Rejuvenation via MH

# %%
"""
    vs = vector_grid(v0, k, r)

Returns grid of vectors, given a grid center, number of grid points
along each dimension and the resolution along each dimension.
"""
function vector_grid(v0::Vector{Float64}, k::Vector{Int}, r::Vector{Float64})
    offset = v0 - (r + k.*r)/2
    return map(I -> [Tuple(I)...].*r + offset, CartesianIndices(Tuple(k)))
end

function grid_index(x, v0, k, r)
    offset = v0 - (r + k.*r)/2
    I = Int.(floor.((x .+ r./2 .- offset)./r))
    return LinearIndices(Tuple(k))[I...]
end

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

    pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes), (:,))
    
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

    inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes)

    return (chmap_grid[j], choicemap((:j, inverting_j)))
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
function particle_filter_grid_rejuv_with_checkpoints(model, T, args, constraints, N_particles, MH_arg_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (get_path.(traces), copy(log_weights)))
    
    for t in 1:T
        # t % 5 == 0 && @info "t = $t"

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

        push!(checkpoints, (get_path.(traces), copy(log_weights)))
    end

    return checkpoints
end;

# %%
function frame_from_weighted_trajectories(world, title, path_actual, trajectories, weights; show_clutters=false, minalpha=0.03)
    t = length(first(trajectories))
    the_plot = plot_world(world, title; show_clutters=show_clutters)
    if !isnothing(path_actual)
        plot!(path_actual; label="actual path", color=:brown)
        plot!(path_actual[t]; label=nothing, color=:black)
    end

    normalized_weights = exp.(weights .- logsumexp(weights))
    
    for (traj, wt) in zip(trajectories, normalized_weights)
        al = max(minalpha, 0.6*sqrt(wt))
        
        plot!([p.p[1] for p in traj], [p.p[2] for p in traj];
              label=nothing, color=:green, alpha=al)
        plot!(traj[end]; color=:green, alpha=al, label=nothing)
        
        plot!(Segment.(zip(traj[1:end-1], traj[2:end]));
              label=nothing, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=al)
    end

    return the_plot
end;

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/10]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10

t1 = now()
checkpointss =
    [particle_filter_grid_rejuv_with_checkpoints(
       #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
       full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = now()

merged_traj_list = []
merged_weight_list = []
for checkpoints in checkpointss
    (trajs, lwts) = checkpoints[end]
    merged_traj_list = [merged_traj_list..., trajs...]
    merged_weight_list = [merged_weight_list..., lwts...]
end
merged_weight_list = merged_weight_list .- log(length(checkpointss))
println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "PF + Grid MH Rejuv", path_low_deviation, merged_traj_list, merged_weight_list)

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

    pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes), (:,))
    
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

    inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes)

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

    pose_grid = reshape(vector_grid([new_p..., new_hd], n_steps, step_sizes), (:,))
    
    # Collection of choicemaps which would update the trace to have each pose
    # in the grid
    chmap_grid = [choicemap((:p, [x, y]), (:hd, h)) for (x, y, h) in pose_grid]
    
    # Get the score under the model for each grid point
    _, robot_inputs, world_inputs, settings = get_args(updated_tr)
    if t > 1
        prev_p = updated_tr[prefix_address(t - 1, :pose => :p)]
        prev_hd = updated_tr[prefix_address(t - 1, :pose => :hd)]
        pose_scores = [
            Gen.assess(step_model,
                       (Pose(prev_p, prev_hd), robot_inputs.controls[t - 1], world_inputs, settings.motion_settings),
                       ch)[1]
            for ch in chmap_grid]
    else
        pose_scores = [
            Gen.assess(start_pose_prior,
                       (robot_inputs.start, settings.motion_settings),
                       ch)[1]
            for ch in chmap_grid]
    end
        
    pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
    j ~ categorical(pose_probs)
    old_p = pose_grid[j][1:2]
    old_hd = pose_grid[j][3]

    inverting_j = grid_index([new_p..., new_hd], [old_p..., old_hd], n_steps, step_sizes)

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
function particle_filter_grid_smcp3_with_checkpoints(model, T, args, constraints, N_particles, MH_arg_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (get_path.(traces), copy(log_weights)))

    for t in 1:T
        # if t % 5 == 1
        #     @info "t = $t"
        # end

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

        push!(checkpoints, (get_path.(traces), copy(log_weights)))
    end

    return checkpoints
end;

# %%
nsteps = [3, 3, 3]
sizes1 = [.7, .7, π/10]
grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

N_samples = 6
N_particles = 10

t1 = now()
checkpointss2 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
       full_model, T, full_model_args, constraints, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = now()

merged_traj_list2 = []
merged_weight_list2 = []
for checkpoints in checkpointss2
    (trajs, lwts) = checkpoints[end]
    merged_traj_list2 = [merged_traj_list2..., trajs...]
    merged_weight_list2 = [merged_weight_list2..., lwts...]
end
merged_weight_list2 = merged_weight_list2 .- log(length(checkpointss2))

println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "PF + Grid SMCP3 Rejuv", path_actual, merged_traj_list2, merged_weight_list2)

# %% [markdown]
# That's already better.  We'll improve this algorithm even further below.
#
# But first, I want to note that there is a major downside to this rejuvenation -- in some cases, we don't need it, and it takes a lot of computation time!

# %% [markdown]
# ### With low motion model noise, all this compute is overkill!
#
# Here, we generate a low noise trajectory, and show that the bootstrap particle filter (with no rejuvenation) is sufficient to perform good inferences.  (Low motion noise, moderate observation noise.)  Proposing from the prior is quite good!

# %%
ani = Animation()
for (pose_actual, pose_integrated, readings) in zip(path_actual_low_deviation, path_integrated, observations_low_deviation)
    actual_plot = frame_from_sensors(
        world, "Actual data",
        path_actual_low_deviation, :brown, "actual path",
        pose_actual, readings, "actual sensors",
        sensor_settings)
    integrated_plot = frame_from_sensors(
        world, "Apparent data",
        path_integrated, :green2, "path from integrating controls",
        pose_integrated, readings, "actual sensors",
        sensor_settings)
    frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(low motion noise)")
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances_lowmotionnoise.gif", fps=1)

# %%
N_samples = 6
N_particles = 10

t1 = now()
checkpointss4 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         constraints, N_particles, grid)
       full_model, T, full_model_args, constraints2, N_particles, [])
     for _=1:N_samples]
t2 = now()

merged_traj_list4 = []
merged_weight_list4 = []
for checkpoints in checkpointss4
    (trajs, lwts) = checkpoints[end]
    merged_traj_list4 = [merged_traj_list4..., trajs...]
    merged_weight_list4 = [merged_weight_list4..., lwts...]
end
merged_weight_list4 = merged_weight_list4 .- log(length(checkpointss4))

println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "Particle filter (no rejuv) - low motion noise", path_actual_low_deviation, merged_traj_list4, merged_weight_list4)

# %% [markdown]
# ### The issue is when motion noise is higher
#
# Now we'll generate a very high motion noise (low observation noise) trajectory.

# %%
ani = Animation()
for (pose_actual, pose_integrated, readings) in zip(path_actual_high_deviation, path_integrated, observations_high_deviation)
    actual_plot = frame_from_sensors(
        world, "Actual data",
        path_actual_high_deviation, :brown, "actual path",
        pose_actual, readings, "actual sensors",
        sensor_settings)
    integrated_plot = frame_from_sensors(
        world, "Apparent data",
        path_integrated, :green2, "path from integrating controls",
        pose_integrated, readings, "actual sensors",
        sensor_settings)
    frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(high motion noise)")
    frame(ani, frame_plot)
end
gif(ani, "imgs/noisy_distances_highmotionnoise.gif", fps=1)

# %% [markdown]
# If we try particle filtering with low-motion-noise settings and no rejuvenation, we have the issue that the particle filter basically just follows the integrated controls, ignoring the highly informative observations.

# %%
N_samples = 6
N_particles = 10

t1 = now()
checkpointss5 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         observations, N_particles, grid)
       full_model, T, full_model_args, constraints3, N_particles, [])
     for _=1:N_samples]
t2 = now()

merged_traj_list5 = []
merged_weight_list5 = []
for checkpoints in checkpointss5
    (trajs, lwts) = checkpoints[end]
    merged_traj_list5 = [merged_traj_list5..., trajs...]
    merged_weight_list5 = [merged_weight_list5..., lwts...]
end
merged_weight_list5 = merged_weight_list5 .- log(length(checkpointss5))

println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "PF - motion noise:(model:low)(data:high)", path_actual_high_deviation, merged_traj_list5, merged_weight_list5)

# %% [markdown]
# Conversely, if we run a no-rejuvenation particle filter with the higher model noise parameters, the runs are inconsistent.

# %%
N_samples = 6
N_particles = 10

t1 = now()
checkpointss6 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         constraints, N_particles, grid)
       full_model, T, full_model_args, constraints3, N_particles, [])
     for _=1:N_samples]
t2 = now()

merged_traj_list6 = []
merged_weight_list6 = []
for checkpoints in checkpointss6
    (trajs, lwts) = checkpoints[end]
    merged_traj_list6 = [merged_traj_list6..., trajs...]
    merged_weight_list6 = [merged_weight_list6..., lwts...]
end
merged_weight_list6 = merged_weight_list6 .- log(length(checkpointss6))

println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "PF - motion noise:(model:high)(data:high)", path_actual_high_deviation, merged_traj_list6, merged_weight_list6)

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

t1 = now()
checkpointss7 =
    [particle_filter_grid_smcp3_with_checkpoints(
       #model,      T,   args,         constraints, N_particles, grid)
       full_model, T, full_model_args, constraints3, N_particles, grid_schedule)
     for _=1:N_samples]
t2 = now()

merged_traj_list7 = []
merged_weight_list7 = []
for checkpoints in checkpointss7
    (trajs, lwts) = checkpoints[end]
    merged_traj_list7 = [merged_traj_list7..., trajs...]
    merged_weight_list7 = [merged_weight_list7..., lwts...]
end
merged_weight_list7 = merged_weight_list7 .- log(length(checkpointss7))

println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
frame_from_weighted_trajectories(world, "PF + Grid SMCP3 Rejuv - motion noise:high", path_actual_high_deviation, merged_traj_list7, merged_weight_list7)

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
function controlled_particle_filter_with_checkpoints(model, T, args, constraints, N_particles::Int, og_arg_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (msg="init", t=0, traj=get_path.(traces), wts=copy(log_weights)))
    prev_total_weight = 0.

    n_rejuv = 0
    for t in 1:T
        # if t % 5 == 0
        #     @info "t = $t"
        # end

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces
            push!(checkpoints, (msg="resample", t=t, traj=get_path.(traces), wts=copy(log_weights)))
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
            push!(checkpoints, (msg="rejuvenate (nr = $nr)", t=t, traj=get_path.(traces), wts=copy(log_weights)))

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
        push!(checkpoints, (msg="update", t=t, traj=get_path.(traces), wts=copy(log_weights)))
    end

    @info "Rejuvenated $n_rejuv of $T steps."
    return checkpoints
end;

# %%
function controlled_particle_filter_with_checkpoints_v2(model, T, args, constraints, N_particles::Int, og_arg_schedule)
    traces = Vector{Trace}(undef, N_particles)
    log_weights = Vector{Float64}(undef, N_particles)
    resample_traces = Vector{Trace}(undef, N_particles)
    prev_log_weights, prev_traces = [], []

    checkpoints = []
    
    for i in 1:N_particles
        traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
    end

    push!(checkpoints, (msg="Initializing", t=0, traj=get_path.(traces), wts=copy(log_weights)))
    prev_total_weight = 0.

    n_rejuv = 0
    for t in 1:T
        # if t % 5 == 0
        #     @info "t = $t"
        # end

        lnormwts = log_weights .- logsumexp(log_weights)
        if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
            weights = exp.(lnormwts)
            for i in 1:N_particles
                resample_traces[i] = traces[categorical(weights)]
            end
            log_weights .= logsumexp(log_weights) - log(N_particles)
            traces, resample_traces = resample_traces, traces
            push!(checkpoints, (msg="Resampling", t=t, traj=get_path.(traces), wts=copy(log_weights)))
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
            push!(checkpoints, (msg="Rejuvenating (repeats: $(nr))", t=t, traj=get_path.(traces), wts=copy(log_weights)))

            # If it still looks bad, try re-generating from the previous timestep
            if logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && t > 1 && nr != MAX_REJUV
                traces = copy(prev_traces)
                log_weights = copy(prev_log_weights)

                push!(checkpoints, (msg="Reverting", t=t-1, traj=get_path.(traces), wts=copy(log_weights)))
                
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

                    push!(checkpoints, (msg="Resampling", t=t, traj=get_path.(traces), wts=copy(log_weights)))
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
        push!(checkpoints, (msg="Extending", t=t, traj=get_path.(traces), wts=copy(log_weights)))
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
t1 = now()
for _=1:N_samples
    push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
        full_model, T, full_model_args, constraints, N_particles, grid_schedule))
end
t2 = now()

merged_traj_list3 = []
merged_weight_list3 = []
for checkpoints in checkpointss3
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list3 = [merged_traj_list3..., trajs...]
    merged_weight_list3 = [merged_weight_list3..., lwts...]
end
merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, "Inference Controller (moderate noise)", path_actual, merged_traj_list3, merged_weight_list3)

# %% [markdown]
# **Animation showing the controller in action----**

# %%
ani = Animation()

checkpoints = checkpointss3[1]
for checkpoint in checkpoints
    frame_plot = frame_from_weighted_trajectories(world, "t = $(checkpoint.t) | operation = $(checkpoint.msg)", path_actual, checkpoint.traj, checkpoint.wts; minalpha=0.08)
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
#     t1 = now()
#     for _=1:N_samples
#         push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
#             #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
#             full_model, T, full_model_args, constraints, N_particles, grid_schedule))
#     end
#     t2 = now();
    
#     merged_traj_list3 = []
#     merged_weight_list3 = []
#     for checkpoints in checkpointss3
#         (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
#         merged_traj_list3 = [merged_traj_list3..., trajs...]
#         merged_weight_list3 = [merged_weight_list3..., lwts...]
#     end
#     merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
#     println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
#     frame_from_weighted_trajectories(world, "controlled grid rejuv", path_actual, merged_traj_list3, merged_weight_list3; minalpha=0.03)
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
t1 = now()
for _=1:N_samples
    push!(checkpointss9, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
        full_model, T, full_model_args, constraints2, N_particles, grid_schedule))
end
t2 = now()

merged_traj_list9 = []
merged_weight_list9 = []
for checkpoints in checkpointss9
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list9 = [merged_traj_list9..., trajs...]
    merged_weight_list9 = [merged_weight_list9..., lwts...]
end
merged_weight_list9 = merged_weight_list9 .- log(length(checkpointss9));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, "Inference controller (low motion noise)", path_actual_low_deviation, merged_traj_list9, merged_weight_list9)

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
t1 = now()
for _=1:N_samples
    push!(checkpointss10, controlled_particle_filter_with_checkpoints_v2(
        #model,      T,   args,         constraints, N_particles, MH_arg_schedule)
        full_model, T, full_model_args, constraints3, N_particles, grid_schedule))
end
t2 = now()

merged_traj_list10 = []
merged_weight_list10 = []
for checkpoints in checkpointss10
    (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
    merged_traj_list10 = [merged_traj_list10..., trajs...]
    merged_weight_list10 = [merged_weight_list10..., lwts...]
end
merged_weight_list10 = merged_weight_list10 .- log(length(checkpointss10));
println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
frame_from_weighted_trajectories(world, "Inference controller (high motion noise)", path_actual_high_deviation, merged_traj_list10, merged_weight_list10)
