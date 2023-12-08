# Some black box params
drift_step_factor = 1/3.
N_particles = 10
N_MH = 10

@gen function drift_mh_proposal(trace)
    t = get_args(trace)[1]
    p_noise = get_args(trace)[4].motion_settings.p_noise
    hd_noise = get_args(trace)[4].motion_settings.hd_noise

    p = trace[prefix_address(t+1, :pose => :p)]
    hd = trace[prefix_address(t+1, :pose => :hd)]

    {prefix_address(t+1, :pose => :p)} ~ mvnormal(p, drift_step_factor * p_noise^2 * [1 0 ; 0 1])
    {prefix_address(t+1, :pose => :hd)} ~ normal(hd, hd_noise)
end

# Use `GenParticleFilters` library code for the generic parts.
function black_box_inference(constraints)
    state = pf_initialize(full_model, (0, full_model_args...), constraints[1], N_particles)
    for t in 1:T
        pf_resample!(state)
        pf_rejuvenate!(state, mh, (drift_mh_proposal, ()), N_MH)
        pf_update!(state, (t, full_model_args...), change_only_T, constraints[t+1])
    end
    return state.traces[categorical(state.log_weights .- logsumexp(state.log_weights))]
end;