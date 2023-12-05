# Propose a move for MH.

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

# Use `GenParticleFilters` library code for the generic parts.

function particle_filter_MH_rejuv_library(model, T, args, constraints, N_particles, N_MH, MH_proposal, MH_proposal_args)
    state = pf_initialize(model, (0, args...), constraints[1], N_particles)
    for t in 1:T
        pf_resample!(state)
        pf_rejuvenate!(state, mh, (MH_proposal, MH_proposal_args), N_MH)
        pf_update!(state, (t, args...), change_only_T, constraints[t+1])
    end
    return state.traces, state.log_weights
end

# Some black box params
N_particles = 10
N_MH = 10
drift_step_factor = 1/3.

black_box_inference(constraints) =
    particle_filter_MH_rejuv_library(full_model, T, full_model_args, constraints, N_particles, N_MH, drift_proposal, (drift_step_factor,))
