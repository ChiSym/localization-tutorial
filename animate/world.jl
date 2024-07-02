struct Segment
    p1 :: SVector{2, Float64}
    p2 :: SVector{2, Float64}
    dp :: SVector{2, Float64}
end
Segment(p1, p2) = Segment(p1, p2, p2 .- p1)
Base.first(segment::Segment) = segment.p1
Base.last(segment::Segment) = segment.p2
Base.show(io:: IO, s:: Segment) = print(io, "(", first(s), "->",  last(s), ")")

function create_segments(verts; loop_around=false)
    segs = [Segment(p1, p2) for (p1, p2) in zip(@view(verts[1:end-1]), @view(verts[2:end]))]
    if loop_around 
        push!(segs, Segment(verts[end], verts[1])) 
    end
    return segs
end

function make_world(wall_vertices, clutters_vec, start, controls; loop_around=false)
    walls = create_segments(wall_vertices; loop_around=loop_around)
    clutters = [create_segments(clutter; loop_around=loop_around) for clutter in clutters_vec]
    walls_clutters = vcat(walls, clutters...)
    all_points = [wall_vertices ; clutters_vec... ; [start.position]]
    x_min = minimum(p[1] for p in all_points)
    x_max = maximum(p[1] for p in all_points)
    y_min = minimum(p[2] for p in all_points)
    y_max = maximum(p[2] for p in all_points)
    bounding_box = (x_min, x_max, y_min, y_max)
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
    T = length(controls)
    return (walls=walls, clutters=clutters, walls_clutters=walls_clutters,
            bounding_box=bounding_box, box_size=box_size, center_point=center_point),
           start, controls,
           T
end

function load_world(file_name; loop_around=false)
    data = parsefile(file_name)
    walls_vec = data["wall_verts"]
    clutters_vec = Vector{Vector{Vector{Float64}}}(data["clutter_vert_groups"])
    start = Pose(Vector{Float64}(data["start_pose"]["p"]), Float64(data["start_pose"]["hd"]))
    controls = [Control(control["ds"], control["dhd"]) for control in data["program_controls"]]
    return make_world(walls_vec, clutters_vec, start, controls; loop_around=loop_around)
end;