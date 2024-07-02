
const World = @NamedTuple{walls::Vector{Segment}, clutters::Vector{Vector{Segment}}, walls_clutters::Vector{Segment}, bounding_box::NTuple{4, Float64}, box_size::Float64, center_point::Vector{Float64}}
const SensorSettings = @NamedTuple{fov::Float64, sensor_count::Int64, box_size::Float64, s_noise::Float64}

Makie.@recipe(WorldMap) do scene
    Attributes(
        labelworld = false,
        upcolor = :green,
    )
end

function Makie.plot!(wm::WorldMap{<:Tuple{World}})
    world = wm[1]
    # labelworld = wm.labelworld
    walls = world[].walls
    for w in walls
        Makie.lines!(wm, stack([first(w), last(w)]), color=:black)
    end
    wm
end

function Makie.convert_arguments(::Type{<:Makie.Arrows}, poses::Vector{Pose})
    x = Makie.Point2f.(position.(poses))
    theta = Makie.Point2f.(orientation.(poses))
    return (x,theta)
end

function Makie.convert_arguments(::Type{<:Makie.Arrows}, pose::Pose)
    x = Makie.Point2f(position(pose))
    theta = Makie.Point2f(orientation(pose))
    return ([x], [theta])
end

function Makie.convert_arguments(::Type{<:Makie.Scatter}, pose::Pose)
    return (Makie.Point2f(position(pose)),)
end

Makie.@recipe(SensorMap) do scene
    Attributes(
        color=:lightgreen
    )
end

function Makie.plot!(sm::SensorMap{<:Tuple{Pose, SensorSettings, Vector{Float64}}})
    # world = wm[1]
    pose = sm[1]
    sensor_settings = sm[2]
    readings = sm[3]

    start = Point2f(position(pose[]))
    segments = Observable(NTuple{2, Point2f}[])
    
    function update_plot(readings)
        projections = Point2f.([step_along_pose(rotate_pose(pose[], sensor_angle(sensor_settings[], j)), s) for (j, s) in enumerate(readings)])
        segments[] = [(start, proj) for proj in projections]
    end
    Makie.Observables.onany(update_plot, readings)
    # segments = Observable()
    update_plot(readings[])
    Makie.linesegments!(sm, segments, color=sm.color[])
    sm
end
