function mh_step(trace, proposal, proposal_args)
    _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap, viz) = propose(proposal, (trace, proposal_args...))
    proposed_trace, model_weight_diff, _, _ = update(trace, fwd_model_update)
    bwd_proposal_weight, _ = assess(proposal, (proposed_trace, proposal_args...), bwd_proposal_choicemap)
    log_weight_increment = model_weight_diff + bwd_proposal_weight - fwd_proposal_weight
    return (log(rand()) < log_weight_increment ? proposed_trace : trace), 0., viz
end
mh_kernel(proposal) =
    (trace, proposal_args) -> mh_step(trace, proposal, proposal_args)


@gen function drift_proposal(trace, drift_step_factor)
    t = get_args(trace)[1] + 1

    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    p = trace[prefix_address(t, :pose => :p)]
    hd = trace[prefix_address(t, :pose => :hd)]

    # For later visualization.
    std_devs_radius = 2.
    viz = (objs = ([p[1]], [p[2]]),
           params = (color=:red, label="$(round(std_devs_radius, digits=2))σ region", seriestype=:scatter,
                     markersize=(20. * std_devs_radius * p_noise), markerstrokewidth=0, alpha=0.25))

    # Form expected by `mh` in library code, immediately following.
    fwd_p = {prefix_address(t, :pose => :p)} ~ mvnormal(p, drift_step_factor * p_noise^2 * [1 0 ; 0 1])
    fwd_hd = {prefix_address(t, :pose => :hd)} ~ normal(hd, hd_noise)

    # Form expected by `mh_step`, further below.
    return choicemap((prefix_address(t, :pose => :p), fwd_p), (prefix_address(t, :pose => :hd), fwd_hd)),
           choicemap((prefix_address(t, :pose => :p), p), (prefix_address(t, :pose => :hd), hd)),
           viz
end
drift_mh_kernel = mh_kernel(drift_proposal)


N_particles = 10
ESS_threshold =  1. + N_particles / 10.

drift_step_factor = 1/3.
drift_proposal_args = (drift_step_factor,)
N_MH = 10
drift_args_schedule = [drift_proposal_args for _ in 1:N_MH]

N_samples = 10

traces = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
prior_plot = frame_from_traces(world, "Prior on robot paths", nothing, nothing, traces, "prior samples")

t1 = now()
traces = [sample(particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule)...) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms. (Total: $(value(t2 - t1)) ms.)")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [sample(particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule)...) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms. (Total: $(value(t2 - t1)) ms.)")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

the_plot = plot(prior_plot, posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="PF + MH/Drift")
savefig("imgs/PF_MH_drift")
the_plot


infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, drift_mh_kernel, drift_args_schedule)

ani = Animation()
for info in infos
    frame_plot = frame_from_info(world, "Run of PF + MH/Drift", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, frame_plot)
end
gif(ani, "imgs/pf_mh_dift_animation.gif", fps=1)


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

    pose_grid = vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)
    choicemap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]), (prefix_address(t, :pose => :hd), h))
                      for (x, y, h) in pose_grid]
    pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
    pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

    j ~ categorical(pose_norm_weights)
    inv_j = inverse_grid_index(grid_n_points, j)

    viz = (objs = ([Pose([x, y], h) for (x, y, h) in pose_grid],),
           params = (color=:red, label="pose grid"))

    return choicemap_grid[j], choicemap((:j, inv_j)), viz
end

grid_mh_kernel = mh_kernel(grid_proposal)


grid_n_points_start = [3, 3, 3]
grid_sizes_start = [.7, .7, π/10]
grid_args_schedule = [(grid_n_points_start, grid_sizes_start .* (2/3)^(j-1)) for j=1:3]

traces = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
prior_plot = frame_from_traces(world, "Prior on robot paths", nothing, nothing, traces, "prior samples")

t1 = now()
traces = [sample(particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_mh_kernel, grid_args_schedule)...) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (low dev): $(value(t2 - t1) / N_samples) ms. (Total: $(value(t2 - t1)) ms.)")
posterior_plot_low_deviation = frame_from_traces(world, "Low dev observations", path_low_deviation, "path to be fit", traces, "samples")

t1 = now()
traces = [sample(particle_filter_rejuv(full_model, T, full_model_args, constraints_high_deviation, N_particles, ESS_threshold, grid_mh_kernel, grid_args_schedule)...) for _ in 1:N_samples]
t2 = now()
println("Time elapsed per run (high dev): $(value(t2 - t1) / N_samples) ms. (Total: $(value(t2 - t1)) ms.)")
posterior_plot_high_deviation = frame_from_traces(world, "High dev observations", path_high_deviation, "path to be fit", traces, "samples")

the_plot = plot(prior_plot, posterior_plot_low_deviation, posterior_plot_high_deviation; size=(1500,500), layout=grid(1,3), plot_title="PF + MH/Grid")
savefig("imgs/PF_MH_grid")
the_plot


infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_mh_kernel, grid_args_schedule)

ani = Animation()
for info in infos
    frame_plot = frame_from_info(world, "Run of PF + MH/Grid", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
    frame(ani, frame_plot)
end
gif(ani, "imgs/pf_mh_grid_animation.gif", fps=1)
