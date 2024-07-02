using JSON: parsefile
# using GLMakie
# GLMakie.activate!(inline=true)
using WGLMakie
WGLMakie.activate!(inline=true)
using Bonito
using Gen
using StaticArrays
using Printf

norm(v) = sqrt(sum(v.^2))
import Base.angle

struct Pose
    position::SVector{2,Float64}
    angle::Float64
    orientation::SVector{2,Float64} 
end

position(pose::Pose) = pose.position
angle(pose::Pose) = pose.angle
orientation(pose::Pose) = pose.orientation
Pose(position, angle::Float64) = Pose(position, rem2pi(angle, RoundNearest), SVector(cos(angle), sin(angle)))
Pose(position, orientation::AbstractVector{Float64}) = Pose(position, atan(orientation[2], orientation[1]))
Base.show(io::IO, p::Pose) = print(io, "Pose(", p.position, " ", p.angle, ")")

struct Control
    ds ::Float64
    dhd::Float64
end
include("world.jl")
include("plotting.jl")
world, robot_init, robot_inputs, T = load_world("example_20_program.json");

step_along_pose(pose::Pose, step_size::Float64) = position(pose) .+ step_size .* orientation(pose)
rotate_pose(pose::Pose, Δθ::Float64) = Pose(position(pose), angle(pose) + Δθ)

function integrate_controls_unphysical(start_pose::Pose, robot_inputs::Vector{Control})
    path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
    path[1] = start_pose
    for t in 1:length(robot_inputs)
        position_new = step_along_pose(path[t], robot_inputs[t])
        angle_new = path[t].hd + robot_inputs[t].dhd
        path[t+1] = Pose(position_new, angle_new)
    end
    return path
end

# %% [markdown]
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance from the point of contact in the normal direction to the wall.

# %%
# Return unique s, t such that p + s*u == q + t*v.
function solve_lines(p, u, q, v; PARALLEL_TOL=1.0e-10)
    det = u[1] * v[2] - u[2] * v[1]
    if abs(det) < PARALLEL_TOL
        return nothing, nothing
    else
        s = (v[1] * (p[2]-q[2]) - v[2] * (p[1]-q[1])) / det
        t = (u[2] * (q[1]-p[1]) - u[1] * (q[2]-p[2])) / det
        return s, t
    end
end

function distance(pose, segment)
    s, t = solve_lines(position(pose), orientation(pose), segment.p1, segment.dp)
    # Solving failed (including, by fiat, if pose is parallel to segment) iff isnothing(s).
    # Pose is oriented away from segment iff s < 0.
    # Point of intersection lies on segment (as opposed to the infinite line) iff 0 <= t <= 1.
    return (isnothing(s) || s < 0. || !(0. <= t <= 1.)) ? Inf : s
end

function physical_step(p1, p2, hd, world_inputs)
    step_pose = Pose(p1, p2 .- p1)
    (s, i) = findmin(w -> distance(step_pose, w), world_inputs.walls)
    if s > norm(p2 .- p1)
        # Step succeeds without contact with walls.
        return Pose(p2, hd)
    else
        dp = orientation(step_pose)
        contact_point = p1 .+ s .* dp
        unit_tangent = world_inputs.walls[i].dp / norm(world_inputs.walls[i].dp)
        unit_normal = SVector(-unit_tangent[2], unit_tangent[1])
        # Sign of 2D cross product determines orientation of bounce.
        if dp[1] * world_inputs.walls[i].dp[2] - dp[2] * world_inputs.walls[i].dp[1] < 0.
            unit_normal = -unit_normal
        end
        return Pose(contact_point + world_inputs.bounce * unit_normal, hd)
    end
end

function integrate_controls(robot_init::Pose, robot_inputs::Vector{Control}, world_inputs)
    path = Vector{Pose}(undef, length(robot_inputs) + 1)
    path[1] = robot_init
    for t in 1:length(robot_inputs)
        position_new = step_along_pose(path[t], robot_inputs[t].dhd)
        angle_new = angle(path[t]) + robot_inputs[t].dhd
        # Perform the physical correction
        path[t+1] = physical_step(position(path[t]), position_new, angle_new, world_inputs)
    end
    return path
end

# %%
# How bouncy the walls are in this world.
world_inputs = (walls = world.walls, bounce = 0.1)

path_integrated = integrate_controls(robot_init, robot_inputs, world_inputs)

# %% [markdown]
# Following this initial display of the given data, we *suppress the clutters* until much later in the notebook.

pose_prior_fig = let
    f = Figure()
    ax = Axis(f[1,1], aspect=DataAspect(), title="Start Pose Prior")
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :dragpan)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)
    worldmap!(f[1,1], world)
    gridbottom = GridLayout(f[2,1])
    sl_steps = Makie.Slider(gridbottom[1,2], range=1:20)
    N_particles = sl_steps.value
    n_particle_tex = lift(N->"# steps $(N[])", N_particles)
    Makie.Label(gridbottom[1,1], n_particle_tex)
    displayed_path = lift((N,path)->path[1:N], N_particles, path_integrated) 
    arrows!(f[1,1], displayed_path)
    f
end

# %% [markdown]
# We can also visualize the behavior of the model of physical motion:
#
# ![](imgs_stable/physical_motion.gif)

# %% [markdown]
# ## Gen basics
#
# %% [markdown]
# ### Components of the motion model
#
# We start with the two building blocks: the starting pose and individual steps of motion.

# %%
@gen (static) function start_pose_prior(start, motion_settings)
    p ~ mvnormal(position(start), motion_settings.p_noise^2 * [1 0 ; 0 1])
    hd ~ normal(angle(start), motion_settings.hd_noise)
    return Pose(p, hd)
end

@gen (static) function step_model(start, control, world_inputs, motion_settings)
    position_new = step_along_pose(start, control.ds)
    p ~ mvnormal(position_new, motion_settings.p_noise^2 * [1 0 ; 0 1])
    hd ~ normal(angle(start) + control.dhd, motion_settings.hd_noise)
    return physical_step(position(start), p, hd, world_inputs)
end

# Map visualization
one_step_fig = let
    motion_settings = (p_noise = 0.5, hd_noise = 2π / 360)

    N_samples = 50
    pose_samples = [start_pose_prior(robot_init, motion_settings) for _ in 1:N_samples]

    std_devs_radius = 2.5 * motion_settings.p_noise
    f = Figure()
    ax = Axis(f[1,1], aspect=DataAspect(), title="Start Pose Prior")
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :dragpan)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)

    worldmap!(f[1,1], world)

    # Sliders for motion settings
    gridbottom = GridLayout(f[2,1])
    upper = gridbottom[1,1]
    bottom = gridbottom[2,1]
    p_noise_slider = Makie.Slider(upper[1,2], range=range(0.01, 1.0, 100))
    hd_noise_slider = Makie.Slider(upper[2,2], range=range(0.01, 1.0, 100))
    motion_settings = lift(p_noise_slider.value, hd_noise_slider.value) do p_noise, hd_noise
        (p_noise = p_noise, hd_noise=hd_noise)
    end

    pose_samples = lift(motion_settings) do motion_settings
        [start_pose_prior(robot_init, motion_settings) for _ in 1:20]
    end
    # Sliders for samples
    reset_n_particles_btn = Makie.Button(bottom[1,1], label="Resample")
    n_particle_slider = Makie.Slider(bottom[1,3], range=1:20, startvalue=5)
    n_particles_obs = n_particle_slider.value

    on(reset_n_particles_btn.clicks) do n
        reset_n_particles_btn.clicks
        pose_samples[] = [start_pose_prior(robot_init, motion_settings[]) for _ in 1:N_samples]
        notify(pose_samples)
    end

    # Numerical labels
    p_noise_tex = lift(p_noise->"σ_p = $(p_noise[])", p_noise_slider.value)
    hd_noise_tex = lift(hd_noise->"σ_hd = $(hd_noise[])", hd_noise_slider.value)
    n_particle_tex = lift(N->"# of samples $(N[])", n_particles_obs)

    Label(upper[1,1], p_noise_tex)
    Label(upper[2,1], hd_noise_tex)
    Label(bottom[1,2], n_particle_tex)
    displayed_path = lift((N,path)->path[1:N], n_particles_obs, pose_samples) 
    arrows!(f[1,1], displayed_path)
    # Figure out how to get the correct units
    scatter!(f[1,1], position(robot_init)..., markersize=std_devs_radius*p_noise_slider.value[], color = Makie.RGBA(1.0, .121, .121, 0.2))
    f
end

# %%
let
    N_samples = 50
    motion_settings = (p_noise = 0.5, hd_noise = 2π / 360)
    noiseless_step = step_along_pose(robot_init, robot_inputs[1].ds)

    # Map Visualization
    f = Figure()
    ax = Axis(f[1,1], aspect=DataAspect(), title="Motion step model (samples)")
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :dragpan)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)
    worldmap!(f[1,1], world)

    # Sliders
    gridbottom = GridLayout(f[2,1])
    upper = gridbottom[1,1]
    bottom = gridbottom[2,1]

    ds_slider = Makie.Slider(upper[1,2], range=0:0.1:3.0, startvalue=1.6)
    dhd_slider = Makie.Slider(upper[2,2], range=range(0, 2π, 100))
    p_noise_slider = Makie.Slider(upper[3,2], range=range(0.01, 1.0, 100), startvalue=.3)
    hd_noise_slider = Makie.Slider(upper[4,2], range=range(0.01, 1.0, 100), startvalue=0.3)
    n_particle_slider = Makie.Slider(bottom[2,3], range=1:20, startvalue=4)
    reset_n_particles_btn = Makie.Button(bottom[2,1], label="Resample")

    # Slider Text
    ds_tex = Makie.lift(ds->@sprintf("ds = %.3f", ds[]), ds_slider.value)
    dhd_tex = Makie.lift(dhd->@sprintf("dhd = %.3f", dhd[]), dhd_slider.value)
    p_noise_tex = Makie.lift(p_noise->@sprintf("σ_p = %.3f", p_noise[]), p_noise_slider.value)
    hd_noise_tex = Makie.lift(hd_noise->@sprintf("σ_hd = %.3f", hd_noise[]), hd_noise_slider.value)
    n_particle_tex = Makie.lift(N->"samples = $(N[])", n_particle_slider.value)

    Label(upper[1,1], ds_tex)
    Label(upper[2,1], dhd_tex)
    Label(upper[3,1], p_noise_tex)
    Label(upper[4,1], hd_noise_tex)
    Label(bottom[2,2], n_particle_tex)

    # Model Makie Observables
    next_control = lift((ds, dhd)-> Control(ds,dhd), ds_slider.value, dhd_slider.value)

    motion_settings = lift(
        (p_noise, hd_noise)-> (p_noise = p_noise, hd_noise=hd_noise),
        p_noise_slider.value, 
        hd_noise_slider.value
        )
    n_particles = n_particle_slider.value

    step_samples = lift(next_control, motion_settings) do control, motion_settings
        [step_model(robot_init, control, world_inputs, motion_settings) for _ in 1:N_samples]
    end

    noiseless_step = lift(next_control) do control
        n = step_along_pose(robot_init, control.ds)
        Point2f(n)
    end

    # Visual Makie Observables
    step_samples_visible = lift(n_particle_slider.value, step_samples) do N, data
        return data[1:N]
    end

    on(reset_n_particles_btn.clicks) do n
        Makie.notify(next_control)
    end

    scatter!(f[1,1], noiseless_step)
    arrows!(f[1,1], robot_init, color=:blue)
    arrows!(f[1,1], step_samples_visible, color=:red)
    f
end

# %% [markdown]
# ### Noisy sensors
#
# We assume that the sensor readings are themselves uncertain, say, the distances only knowable up to some noise.  We model this as follows.  (We satisfy ourselves with writing a loop in the dynamic DSL because we will have no need for incremental recomputation within this model.)

# %%
"""
    sensor_model(pose::Pose, walls, sensor_settings)

Simulates the sensor model of the robot at `pose`, returning distance readings from the `walls` to sensors on the robot.

The `sensor_settings` controls the FoV, number of sensors, range of sensors, and the reading noise.
"""
sensor_angle(sensor_settings, j) =
    sensor_settings.fov * (j - (sensor_settings.sensor_count - 1) / 2) / (sensor_settings.sensor_count - 1)

function sensor_distance(pose, walls, box_size)
    d = minimum(distance(pose, seg) for seg in walls)
    # Capping to a finite value avoids issues below.
    return isinf(d) ? 2. * box_size : d
end;

@gen function sensor_model(pose::Pose, walls, sensor_settings)
    for j in 1:sensor_settings.sensor_count
        sensor_pose = rotate_pose(pose, sensor_angle(sensor_settings, j))
        {j => :distance} ~ normal(sensor_distance(sensor_pose, walls, sensor_settings.box_size), sensor_settings.s_noise)
    end
end

function noisy_sensor(pose, walls, sensor_settings)
    trace = simulate(sensor_model, (pose, walls, sensor_settings))
    return [trace[j => :distance] for j in 1:sensor_settings.sensor_count]
end

sensor_fig = let
    f = Figure()
    ax = Axis(f[1,1], aspect=DataAspect(), title="Motion step model (samples)")
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :dragpan)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)
    worldmap!(ax, world)

    bottom = GridLayout(f[2,1])

    sensor_noise_slider = Makie.Slider(bottom[1,2], range=range(0,0.5,101), startvalue=0.1)
    sensor_count_slider = Makie.Slider(bottom[2,2], range=1:20, startvalue=10)

    # Slider Text
    sensor_noise_tex = Makie.lift(noise->@sprintf("σ_sensor = %.3f", noise), sensor_noise_slider.value)
    sensor_count_tex = Makie.lift(N->"sensors = $(N[])", sensor_count_slider.value)

    Label(bottom[1,1], sensor_noise_tex)
    Label(bottom[2,1], sensor_count_tex)
    
    sensor_settings = lift(sensor_noise_slider.value, sensor_count_slider.value) do s_noise, N
        (fov = 2π*(2/3), sensor_count = N, box_size = world.box_size, s_noise=s_noise)
    end

    robot_pose = Observable(robot_init)
    readings = Makie.lift(robot_pose, sensor_settings) do pose, sensor_settings
        noisy_sensor(pose, world.walls, sensor_settings)
    end

    sensormap!(ax, robot_pose, sensor_settings, readings)
    arrows!(ax, robot_pose; arrowsize=5, lengthscale=0.5)
    register_interaction!(ax, :set_pose) do event::MouseEvent, axis
        if event.type === MouseEventTypes.leftclick
            old_pose = robot_pose[]
            new_pose = Pose(event.data, angle(old_pose))
            robot_pose[] = new_pose
            notify(robot_pose)
        end
    end
    f
end

server = Server("localhost", 8080)
app = App() do
    Bonito.Grid(
        Card(pose_prior_fig, padding="0px", margin="0px"),
        Card(one_step_fig, padding="0px", margin="0px"),
        Card(sensor_fig, padding="0px", margin="0px"),
    )
    # return DOM.div(DOM.h1("hello world"), js"""console.log('hello world')""")
end
route!(server, "/"=>app)


# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", whose traces describe simulations of the entire robot situation as we have described it.

# %%

@gen (static) function robot_initialization_prior(robot_init, walls, full_settings)
    pose ~ start_pose_prior(robot_init, full_settings.motion_settings)
    sensor ~ sensor_model(pose, walls, full_settings.sensor_settings)
    return pose
end

@gen (static) function dynamics_kernel(t, state, robot_inputs, world_inputs, full_settings)
    pose ~ step_model(state, robot_inputs[t], world_inputs, full_settings.motion_settings)
    sensor ~ sensor_model(pose, world_inputs.walls, full_settings.sensor_settings)
    return pose
end
dynamics_unfold = Unfold(dynamics_kernel)

@gen (static) function full_model(T, robot_init, robot_inputs, world_inputs, full_settings)
    initial ~ robot_initialization_prior(robot_init, world_inputs.walls, full_settings)
    steps ~ dynamics_unfold(T, initial, robot_inputs, world_inputs, full_settings)
end



# selection = select((prefix_address(t, :pose) for t in 1:3)..., (prefix_address(t, :sensor => j) for t in 1:3, j in 1:5)...)
# get_selected(get_choices(trace), selection)

prefix_address(t, rest) = (t == 1) ? (:initial => rest) : (:steps => (t-1) => rest)

get_path(trace) = [trace[prefix_address(t, :pose)] for t in 1:(get_args(trace)[1]+1)]
function get_sensors(trace)
    T = get_args(trace)[1]
    settings = get_args(trace)[5]
    [[trace[prefix_address(t, :sensor => j => :distance)] 
            for j in 1:settings.sensor_settings.sensor_count ]
     for t in 1:(T+1)]
end


let


    f = Figure() 
    ax = Axis(f[1,1], aspect=DataAspect(), title="Motion step model (samples)")
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :dragpan)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)
    worldmap!(ax, world)

    # Sliders
    gridbottom = GridLayout(f[2,1])
    # upper = gridbottom[1,1]
    # bottom = gridbottom[2,1]

    p_noise_slider = Makie.Slider(gridbottom[1,2], range=range(0.01, 1.0, 100), startvalue=0.5)
    hd_noise_slider = Makie.Slider(gridbottom[2,2], range=range(0.01, 1.0, 100), startvalue=2π / 360)
    sensor_noise_slider = Makie.Slider(gridbottom[3,2], range=range(0.01, 1.0, 100), startvalue=0.5)
    sensor_count_slider = Makie.Slider(gridbottom[4,2], range=1:20, startvalue=4)
    # n_particle_slider = Makie.Slider(bottom[2,3], range=1:20, startvalue=4)
    # reset_n_particles_btn = Makie.Button(bottom[2,1], label="Resample")

    # Slider Text
    p_noise_tex = Makie.lift(p_noise->@sprintf("σ_p = %.3f", p_noise[]), p_noise_slider.value)
    hd_noise_tex = Makie.lift(hd_noise->@sprintf("σ_hd = %.3f", hd_noise[]), hd_noise_slider.value)
    sensor_noise_tex = Makie.lift(ds->@sprintf("σ_sensor = %.3f", ds[]), sensor_noise_slider.value)
    sensor_count_tex = Makie.lift(N->"sensors = $(N[])", sensor_count_slider.value)
    # n_particle_tex = Makie.lift(N->"samples = $(N[])", n_particle_slider.value)

    Label(gridbottom[1,1], p_noise_tex)
    Label(gridbottom[2,1], hd_noise_tex)
    Label(gridbottom[3,1], sensor_noise_tex)
    Label(gridbottom[4,1], sensor_count_tex)
    # Label(bottom[2,2], n_particle_tex)
    
    # Robot's current position
    robot_pose = Observable(robot_init)
    arrows!(robot_pose)
    register_interaction!(ax, :set_pose) do event::MouseEvent, axis
        if event.type === MouseEventTypes.leftclick
            old_pose = robot_pose[]
            new_pose = Pose(event.data, angle(old_pose))
            robot_pose[] = new_pose
        end
    end
    
    robot_settings = lift(
        p_noise_slider.value, 
        hd_noise_slider.value, 
        sensor_noise_slider.value, 
        sensor_count_slider.value) do p_noise, hd_noise, sensor_noise, sensor_count
            
        motion_settings = (p_noise = p_noise, hd_noise = 2π / 360)
        sensor_settings = (fov = 2π*(2/3), sensor_count = sensor_count, box_size = world.box_size, s_noise=sensor_noise)
        robot_settings = (motion_settings=motion_settings, sensor_settings=sensor_settings)
        return robot_settings
    end
    
    trajectory_trace = lift(robot_pose) do pose
        motion_settings = (p_noise = 0.5, hd_noise = 2π / 360)
        sensor_settings = (fov = 2π*(2/3), sensor_count = 41, box_size = world.box_size, s_noise=0.10)
        robot_settings = (motion_settings=motion_settings, sensor_settings=sensor_settings)
        model_args = (T, robot_init, robot_inputs, world_inputs, robot_settings)
        trace = simulate(full_model, model_args);
    end

    f
end

# function frames_from_full_trace(world, title, trace; show_clutters=false)
#     T = get_args(trace)[1]
#     robot_init = get_args(trace)[2]
#     robot_inputs = get_args(trace)[3]
#     poses = get_path(trace)
#     noiseless_steps = [position(robot_init), [step_along_pose(pose, c.ds) for (pose, c) in zip(poses, robot_inputs)]...]
#     settings = get_args(trace)[5]
#     std_devs_radius = 2.5 * settings.motion_settings.p_noise
#     sensor_readings = get_sensors(trace)
#     plots = Vector{Plots.Plot}(undef, 2*(T+1))
#     for t in 1:(T+1)
#         frame_plot = plot_world(world, title; show_clutters=show_clutters)
#         plot!(poses[1:t-1]; color=:black, label="past poses")
#         plot!(make_circle(noiseless_steps[t], std_devs_radius);
#               color=:red, linecolor=:red, label="95% region", seriestype=:shape, alpha=0.25)
#         plot!(Pose(trace[prefix_address(t, :pose => :p)], angle(poses[t])); color=:red, label="sampled next step")
#         plots[2*t-1] = frame_plot
#         plots[2*t] = frame_from_sensors(
#             world, title,
#             poses[1:t], :black, nothing,
#             poses[t], sensor_readings[t], "sampled sensors",
#             settings.sensor_settings; show_clutters=show_clutters)
#     end
#     return plots
# end;

