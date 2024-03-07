module Drawit

function split_first(str, substr)
    res = split(str, substr; limit=2)
    return length(res) == 2 ? res : (res[1], nothing)
end

# Breakage of `block` is triggered by
# * "!["
# * followed by an optional charcter 'c' or 'm' for "code" or "math" (as opposed to plain text)
# * followed by a (possibly empty) comma-separated sequence of Ints
# * followed by "["
# and terminated by "]]!".
function parse_highlights(block)
    pieces = []
    remainder = block
    while !isnothing(remainder) && remainder != ""
        (head, remainder) = split_first(remainder, "![")

        if head != ""; push!(pieces, (:none, [], head)) end

        if !isnothing(remainder)
            if remainder[1] == 'c'
                mode = :code
                remainder = remainder[2:end]
            elseif remainder[1] == 'm'
                mode = :math
                remainder = remainder[2:end]
            else
                mode = :text
            end

            (head, remainder) = split_first(remainder, "[")
            @assert !isnothing(remainder) "![ not followed by ["

            labels = []
            (num, head) = split_first(head, ",")
            while !isnothing(head)
                push!(labels, parse(Int, num))
                (num, head) = split_first(head, ",")
            end
            if num != ""; push!(labels, parse(Int, num)) end

            (head, remainder) = split_first(remainder, "]]!")
            @assert !isnothing(remainder) "![...[ not followed by ]]! (concluding exclamation char?)"

            push!(pieces, (mode, labels, head))
        end
    end
    return pieces
end

function process_piece(mode, piece)
    if mode == :code
        return replace(piece, "_" => "\\_")
    else
        return piece
    end
end

# The param `varwidth_frac` is roughly (not exactly) an aspect ratio,
# controls truncation on the right.
# Without `n_labels` param, numerical labels are ignored and all highlights occur in a single output.
# When `n_labels` param is supplied, a list of this many outputs produced,
# with only label `i` highlighted in the `i`th output.
function highlighted_versions(pieces, border, varwidth_frac; n_labels=nothing)
    file_start_1 = "\\documentclass[crop=true,border={"
    file_start_2 = "pt 0pt 0pt 0pt},varwidth="
    file_start_3 = """
\\linewidth]{standalone}
%\\usepackage{mdframed}
\\usepackage{listings}
\\usepackage{textcomp}
\\usepackage{xcolor}
\\usepackage{bm}
\\usepackage{soul}
\\definecolor{highlight}{HTML}{ffe70f}
\\newcommand{\\hlight}[1]{\\setlength{\\fboxsep}{0pt}\\colorbox{highlight}{#1}}
\\newcommand{\\hlightmath}[1]{\\mathchoice%
    {\\hlight{\$\\displaystyle #1\$}}%
    {\\hlight{\$\\textstyle #1\$}}%
    {\\hlight{\$\\scriptstyle #1\$}}%
    {\\hlight{\$\\scriptscriptstyle #1\$}}}
\\definecolor{boldkwcolor}{HTML}{00679e}
\\lstset{
    escapeinside={(*}{*)},
    basicstyle=\\ttfamily\\small,
    numbers=left,
    columns=fullflexible,
    keepspaces=true,
    literate={~} {\$\\sim\$}{1},
    %upquote=true,
    % Define . and % and @ as letters to include them in keywords.
    %alsoletter={\\.},%,,\\.,\\%,\\#, \\@, \\?, \\/, \\~, !},
    alsoletter={!?-,.@},
    % First type of keywords.
    % Use \\bfseries\\textcolor{OliveGreen} to get bolded text.
    morekeywords=[1]{function, if, else, end, while, for, begin, in, const, struct, return},
    keywordstyle=[1]\\textcolor{brown},
    % Second type of keywords.
    % Use \\bfseries\\textcolor{OliveGreen} to get bolded text.
    morekeywords=[2]{\\@gen, \\@trace,Gen\\.generate, Gen\\.simulate,Gen\\.update,Gen\\.metropolis_hastings,Gen\\.maybe_resample!},
    keywordstyle=[2]\\textcolor{boldkwcolor},
    % Add strings
    showstringspaces=False,
    %stringstyle=\\ttfamily\\color{NavyBlue},
    stringstyle=\\ttfamily\\bfseries\\color{red},
    morestring=[b]{"},
    morestring=[b]{'},
    % l is for line comment
    morecomment=[l]{\\#},
    commentstyle=\\color{gray}\\ttfamily,
}
\\usepackage{algpseudocode}
\\algdef{SE}[DOWHILE]{Do}{doWhile}{\\algorithmicdo}[1]{\\algorithmicwhile\\ #1}%
\\usepackage{amsmath,amssymb}
\\newcommand{\\white}[1]{\\setlength{\\fboxsep}{0pt}\\colorbox{white}{#1}}
\\begin{document}
"""
    file_end = "\\end{document}"
    highlight_start = (text = "\\hlight{", code = "(*\\hlight{", math = "\\hlightmath{")
    highlight_end = (text = "}", code = "}*)", math = "}")

    if isnothing(n_labels)
        version = file_start_1 * "$(border)" * file_start_2 * "$(varwidth_frac)" * file_start_3
        for (mode, _, piece) in pieces
            if mode != :none
                version = version * highlight_start[mode] * process_piece(mode, piece) * highlight_end[mode]
            else
                version = version * piece
            end
        end
        version = version * file_end
        return version
    else
        versions = []
        for i in 1:n_labels
            version = file_start_1 * "$(border)" * file_start_2 * "$(varwidth_frac)" * file_start_3
            for (mode, labels, piece) in pieces
                if mode != :none && i in labels
                    version = version * highlight_start[mode] * process_piece(mode, piece) * highlight_end[mode]
                else
                    version = version * piece
                end
            end
            version = version * file_end
            push!(versions, version)
        end
        return versions
    end
end

backup_stdout = stdout # just in case
function build_pic(file_text, file_name; silence=false)
    if silence
        saved_stdout = stdout
        redirect_stdout(devnull)
    end
    try
        xrap = tempname("./")
        run(`mkdir $xrap`)
        open(f -> write(f, file_text), "$xrap/$xrap.tex", "w")
        run(`pdflatex -interaction=nonstopmode -output-directory=$xrap $xrap/$xrap.tex`)
        run(`convert -colorspace RGB -density 500 -quality 100 -background white -alpha remove -alpha off $xrap/$xrap.pdf $file_name`)
        run(`rm -rf $xrap`)
    finally
        if silence
            redirect_stdout(saved_stdout)
        end
    end
end

function build_highlighted_pics(block, border, varwidth_frac, file_name_base; n_labels=nothing, silence=false)
    pieces = parse_highlights(block)

    if isnothing(n_labels)
        file_text = highlighted_versions(pieces, border, varwidth_frac; n_labels=n_labels)
        file_name = "$(file_name_base).png"
        build_pic(file_text, file_name; silence=silence)
        return file_name
    else
        versions = highlighted_versions(pieces, border, varwidth_frac; n_labels=n_labels)
        files = []
        for (i, file_text) in enumerate(versions)
            file_name = "$(file_name_base)_$i.png"
            build_pic(file_text, file_name; silence=silence)
            push!(files, file_name)
        end
        return files
    end
end

# test_math = """
# \$\$
# ![m1,2,3,4[P_\\text{path}(z_{0:T}; r_{0:T}, w, \\nu)]]!
# = ![m2[P_\\text{start}(z_0; r_0, \\nu)]]! \\cdot ![m3[\\prod\\nolimits_{t=1}^T]]! ![m4[P_\\text{step}(z_t; z_{t-1}, r_t, w, \\nu)]]!
# \$\$
# """
# print(test_math)

lstlisting(s) = "\\begin{lstlisting}\n" * s * (s[end] == '\n' ? "" : "\n") * "\\end{lstlisting}"
# # Note that the `\begin{lstlisting}...\end{lstlisting}` is included here in contrast to Marco.
# test_code = lstlisting(
# """
# @gen function ![c1,2,3,4[path_model_loop]]!(T :: Int, robot_inputs :: NamedTuple, world_inputs :: NamedTuple, motion_settings :: NamedTuple) :: Vector{Pose}
#     pose = {:initial => :pose} ~ ![c2[start_pose_prior(robot_inputs.start, motion_settings)]]!

#     ![c3[for t in 1:T]]!
#         pose = {:steps => t => :pose} ~ ![c4[step_model(pose, robot_inputs.controls[t], world_inputs, motion_settings)]]!
#     end
# end
# """)
# print(test_code)


# using Plots, Images
# mkpath("imgs")
# # Math tends to get `0` border...
# math_files = build_highlighted_pics(test_math, 4, 0, 0.75, "imgs/test_math")
# # ...while code gets `20` border to not truncate off the line numbers.
# code_files = build_highlighted_pics(test_code, 4, 20, 1.5, "imgs/test_code")
# math_imgs = [load(file) for file in math_files]
# code_imgs = [load(file) for file in code_files]
# ani = Animation()
# for (math_img, code_img) in zip(math_imgs, code_imgs)
#     frame(ani, plot(mosaicview(math_img, code_img; nrow=1, fillvalue=1.0), axis=([], false), size=(3000,1500)))
# end
# gif(ani, "imgs/test.gif", fps=1)



using Plots, Gen

function physical_motion()
    sw, sri, sT = make_world([[1.,-1.], [1.,1.], [2.,1.], [2.,-1.], [1.,-1.]], Vector{Vector{Float64}}[],
                            Pose([0.,0.],0.), [Control(1.5,pi/6.), Control(1.,2pi/6.), Control(0.75,0.)])
    swi = (walls = sw.walls, bounce = 0.1)
    spi = integrate_controls(sri, swi)

    plots = []
    frame_plot = plot_world(sw, "Deterministic path")
    plot!(sri.start; color=:green3, label="given start pose")
    push!(plots, frame_plot)
    # plot!([pose.p[1] for pose in path_integrated], [pose.p[2] for pose in path_integrated];
    #       color=:green2, label="path from integrating controls", seriestype=:scatter, markersize=3, markerstrokewidth=0)
    for i in 1:sT
        frame_plot = plot_world(sw, "Deterministic path")
        for j in 1:i
            plot!(spi[j]; color=:black, label=nothing)
        end
        attempt = Pose(spi[i].p + sri.controls[i].ds * spi[i].dp, spi[i].hd + sri.controls[i].dhd)
        plot!(attempt; color=:red, label="attempted step", size=(500,500))
        push!(plots, frame_plot)

        frame_plot = plot_world(sw, "Deterministic path")
        for j in 1:i
            plot!(spi[j]; color=:black, label=nothing)
        end
        physical = spi[i+1]
        plot!(physical; color=:red, label="physically constrained step")
        push!(plots, frame_plot)
    end

    code = """
    \\begin{lstlisting}
    function integrate_controls(robot_inputs :: NamedTuple,
                                world_inputs :: NamedTuple)
        path = Vector{Pose}(undef, length(robot_inputs.controls) + 1)
        ![c1[path[1]]]! = robot_inputs.start
        for t in 1:length(robot_inputs.controls)
            p = ![c2[path[t].p + robot_inputs.controls[t].ds * path[t].dp]]!
            hd = ![c2[path[t].hd + robot_inputs.controls[t].dhd]]!
            path[t+1] = ![c3[physical_step(path[t].p, p, hd, world_inputs)]]!
        end
        return path
    end
    \\end{lstlisting}
    """
    codes_norepeat = [plot(load(f); axis=([], false), size=(2000,700))
                    for f in build_highlighted_pics(code, 20, 0.9, "imgs/code"; n_labels=3, silence=true)]
    codes = [codes_norepeat[1]]
    for _ in 1:sT; codes = [codes..., codes_norepeat[2:end]...] end

    input = """
    \\phantom{asdfasdfasdfasdfasdf} \\newline
    Start: \\newline
    \\phantom{asdf} ![1[\$[0,0],\\ 0\$]]! \\newline
    Controls: \\newline
    \\phantom{asdf} 1. ![2[\$1.5,\\ \\pi/6\$]]! \\newline
    \\phantom{asdf} 2. ![3[\$1.0,\\ 2\\pi/6\$]]! \\newline
    \\phantom{asdf} 2. ![4[\$0.75,\\ 0\$]]! \\newline
    \\phantom{asdfasdfasdfasdfasdf} \\newline
    \\phantom{asdfasdfasdfasdfasdf}
    """
    inputs_norepeat = [plot(load(f); axis=([], false), size=(1000,200))
                    for f in build_highlighted_pics(input, 0, 0.7, "imgs/input"; n_labels=(sT+1), silence=true)]
    inputs = [inputs_norepeat[1]]
    for i in 2:(sT+1); inputs = [inputs..., inputs_norepeat[i], inputs_norepeat[i]] end
    l = @layout [[a ; b] c]

    ani = Animation()
    for i in 1:length(codes)
        frame(ani, plot(codes[i], inputs[i], plots[i]; layout=l))
    end
    gif(ani, "imgs/physical_motion.gif", fps=0.5)
end


function path_model_with_trace()
    trace = simulate(path_model_loop, (T, robot_inputs, world_inputs, motion_settings))

    trace_str = [s * "\n" for s in split(sprint(show, MIME("text/plain"), get_choices(trace)), "\n")[1:end-1]]

    initial_i = findfirst(s -> contains(s, ":initial"), trace_str)-1
    trace_strs = [splice!(trace_str, initial_i:(initial_i+7))]

    push!(trace_strs, splice!(trace_str, 1:2))

    for _ in 1:T
        push!(trace_strs, splice!(trace_str, 1:8))
    end
    sort!(view(trace_strs, 3:length(trace_strs)), by=s -> parse(Int, s[2][findfirst(r"[0-9]", s[2])[1]:end]))

    hlt(s) = lpad("![c1[" * strip(s) * "]]!\n", length(s)+7)
    stringify(l) = replace(string([string(s...) for s in l]...), '\u2502' => "|", '\u251C' => "|", '\u2500' => "-", '\u2514' => "|")

    slide_texts = []
    push!(slide_texts, stringify([map(hlt, trace_strs[1]), trace_strs[2:4]...]))
    push!(slide_texts, stringify([trace_strs[1], map(hlt, trace_strs[2]), trace_strs[3:4]...]))
    for i in 1:(T-1)
        push!(slide_texts, stringify([trace_strs[i:i+1]..., map(hlt, trace_strs[i+2]), trace_strs[i+3]]))
    end
    push!(slide_texts, stringify([trace_strs[T-1:T+1]..., map(hlt, trace_strs[T+2])]))

    path_trace_slide_files = [build_highlighted_pics(lstlisting(slide_text), 0, 1., "imgs/path_trace_slide_$i"; silence=true) for (i, slide_text) in enumerate(slide_texts)]

    path_model_loop_code = """
    @gen function path_model_loop(T :: Int, ...)
        pose = ![c1[{:initial => :pose} ~ start_pose_prior]]!(...)

        for ![c2[t in 1:T]]!
            pose = {![c2[:steps]]! => ![c3[t => :pose]]!} ~ ![c3[step_model]]!(...)
        end
    end
    """
    path_model_loop_files = build_highlighted_pics(lstlisting(path_model_loop_code), 20, 1., "imgs/path_model_loop"; n_labels=3, silence=true)

    graph_frames = frames_from_motion_trace(world, "Motion model (sample)", trace)

    partners =
        [(path_trace_slide_files[1], path_model_loop_files[1], graph_frames[1]),
        (path_trace_slide_files[2], path_model_loop_files[2], graph_frames[1]),
        [(f, path_model_loop_files[3], g) for (f, g) in zip(path_trace_slide_files[3:end], graph_frames[2:end])]...]

    l = @layout [a{0.15h} ; [b c]]
    ani = Animation()
    for (trace_file, code_file, graph) in partners
        code_plot = plot(load(code_file); axis=([], false), size=(2000,700))
        trace_plot = plot(load(trace_file); axis=([], false), size=(2000,700))
        frame(ani, plot(code_plot, trace_plot, graph; layout=l, size=(2000,2000)))
    end
    gif(ani, "imgs/path_model_with_trace.gif", fps=1)
end


function dynamic_static_comparision()
    N_repeats = 100
    robot_inputs_long = (robot_inputs..., controls = reduce(vcat, [robot_inputs.controls for _ in 1:N_repeats]))

    time_ends_loop = Vector(undef, T * N_repeats)
    time_start = now()
    trace = simulate(path_model_loop, (0, robot_inputs_long, world_inputs, motion_settings))
    for t in 1:(T * N_repeats)
        trace, _, _, _ = update(trace, (t, robot_inputs_long, world_inputs, motion_settings), change_only_T, choicemap())
        time_ends_loop[t] = now()
    end
    time_diffs_loop = value.(time_ends_loop - [time_start, time_ends_loop[1:end-1]...])
    println("Explicit loop: $(value(time_ends_loop[end]-time_start))ms")

    time_ends_chain = Vector(undef, T * N_repeats)
    time_start = now()
    trace = simulate(path_model, (0, robot_inputs_long, world_inputs, motion_settings))
    for t in 1:(T * N_repeats)
        trace, _, _, _ = update(trace, (t, robot_inputs_long, world_inputs, motion_settings), change_only_T, choicemap())
        time_ends_chain[t] = now()
    end
    time_diffs_chain = value.(time_ends_chain - [time_start, time_ends_chain[1:end-1]...])
    println("Markov chain combinator: $(value(time_ends_chain[end]-time_start))ms")

    the_plot = plot([range(1, T * N_repeats)...], time_diffs_loop; label="Explicit loop", title="Gen.update steps into trace", xlabel="t'th step", ylabel="time (ms)")
    plot!([range(1, T * N_repeats)...], time_diffs_chain; label="Markov chain combinator")
    savefig("SLIDES/dynamic_static_comparison")
    the_plot
end


function full_model_with_trace()
    trace_step_length = 4*sensor_settings.num_angles+10
    trace = simulate(full_model, (T, full_model_args...))

    trace_str = [s * "\n" for s in split(sprint(show, MIME("text/plain"), get_choices(trace)), "\n")[1:end-1]]

    function clean_step(lines; init=false)
        head = lines[1:2]
        pose_i = findfirst(s -> contains(s, ":pose"), lines)-1
        pose = lines[pose_i:(pose_i+5)]
        sensor_i = findfirst(s -> contains(s, ":sensor"), lines)-1
        sensor_label = lines[sensor_i:(sensor_i+1)]
        sensor = [lines[i:i+3] for i in (sensor_i+2):4:(sensor_i+trace_step_length-11)]
        sort!(sensor, by=s->parse(Int, s[2][findfirst(r"[0-9]", s[2])[1]:end]))
        sensor = [sensor[1], "...\n", sensor[end]]
        return [head..., pose..., sensor_label..., vcat(sensor...)...]
    end

    initial_i = findfirst(s -> contains(s, ":initial"), trace_str)-1
    trace_strs = [clean_step(splice!(trace_str, initial_i:(initial_i+trace_step_length-1)); init=true)]
    push!(trace_strs, splice!(trace_str, 1:2))
    for _ in 1:T
        push!(trace_strs, clean_step(splice!(trace_str, 1:trace_step_length)))
    end
    sort!(view(trace_strs, 3:length(trace_strs)), by=s -> parse(Int, s[2][15:end]))

    hlt(s) = lpad("![c1[" * strip(s) * "]]!\n", length(s)+7)
    hlt_pose(strs) = [map(hlt, strs[1:8])..., strs[9:end]...]
    hlt_sensor(strs) = [map(hlt, strs[1:2])..., strs[3:8]..., map(hlt, strs[9:end])...]
    stringify(l) = replace(string([string(s...) for s in l]...), '\u2502' => "|", '\u251C' => "|", '\u2500' => "-", '\u2514' => "|")

    slide_texts = []
    push!(slide_texts, stringify([hlt_pose(trace_strs[1]), trace_strs[2:4]...]))
    push!(slide_texts, stringify([hlt_sensor(trace_strs[1]), trace_strs[2:4]...]))
    push!(slide_texts, stringify([trace_strs[1:2]..., hlt_pose(trace_strs[3]), trace_strs[4]]))
    push!(slide_texts, stringify([trace_strs[1:2]..., hlt_sensor(trace_strs[3]), trace_strs[4]]))
    for i in 2:(T-1)
        push!(slide_texts, stringify([trace_strs[i+1], hlt_pose(trace_strs[i+2]), trace_strs[i+3]]))
        push!(slide_texts, stringify([trace_strs[i+1], hlt_sensor(trace_strs[i+2]), trace_strs[i+3]]))
    end
    push!(slide_texts, stringify([trace_strs[T:T+1]..., hlt_pose(trace_strs[T+2])]))
    push!(slide_texts, stringify([trace_strs[T:T+1]..., hlt_sensor(trace_strs[T+2])]))

    full_trace_slide_files = [build_highlighted_pics(lstlisting(slide_text), 0, 1., "imgs/path_trace_slide_$i"; silence=true) for (i, slide_text) in enumerate(slide_texts)]

    full_model_code = """
    @gen (static) function full_model_initial(...)
        ![c1[pose]]! ~ ![c1[start_pose_prior(robot_inputs.start, ...)]]!
        ![c2[{:sensor}]]! ~ ![c2[sensor_model(pose, ...)]]!
        return pose
    end

    @gen (static) function full_model_kernel(t :: Int, state :: Pose, ...)
        ![c3[pose]]! ~ ![c3[step_model(state, robot_inputs.controls[t], ...)]]!
        ![c4[{:sensor}]]! ~ ![c4[sensor_model(pose, ...)]]!
        return pose
    end
    full_model_chain = Unfold(full_model_kernel)

    @gen (static) function full_model(T :: Int, robot_inputs, ...)
        ![c1,2[initial]]! ~ ![c1,2[full_model_initial(robot_inputs, world_inputs, ...)]]!
        ![c3,4[steps]]! ~ ![c3,4[full_model_chain(T, initial, robot_inputs, ...)]]!
    end
    """
    full_model_code_files = build_highlighted_pics(lstlisting(full_model_code), 20, 1., "imgs/full_model"; n_labels=4, silence=true)

    graph_frames = frames_from_full_trace(world, "Full model (sample)", trace)

    partners =
        [(full_trace_slide_files[1], full_model_code_files[1], graph_frames[1]),
        (full_trace_slide_files[2], full_model_code_files[2], graph_frames[2]),
        [(f, full_model_code_files[isodd(i) ? 3 : 4], g) for (i, (f, g)) in enumerate(zip(path_trace_slide_files[3:end], graph_frames[3:end]))]...]

    l = @layout [[a ; b] c]
    ani = Animation()
    for (trace_file, code_file, graph) in partners
        code_plot = plot(load(code_file); axis=([], false), size=(2000,700))
        trace_plot = plot(load(trace_file); axis=([], false), size=(2000,700))
        frame(ani, plot(code_plot, graph, trace_plot; layout=l, size=(2000,2000)))
    end
    gif(ani, "imgs/full_model_with_trace.gif", fps=1)
end


function bootstrap_with_code()
    bootstrap_code = """
    function particle_filter_bootstrap(...)
        traces = Vector{Trace}(undef, N_particles)
        log_weights = Vector{Float64}(undef, N_particles)
        
        for i in 1:N_particles
            traces[i], log_weights[i] =
                ![c1[generate]]!(model, (0, args...), constraints[1])
        end
        
        for t in 1:T
            traces, log_weights =
                ![c2[resample]]!(traces, log_weights)

            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ =
                    ![c3[update]]!(traces[i], (t, args...), ...)
                log_weights[i] += log_weight_increment
            end
        end

        return traces, log_weights
    end
    """
    bootstrap_code_files = build_highlighted_pics(lstlisting(bootstrap_code), 20, 1., "imgs/bootstrap_code"; n_labels=3, silence=true)
    bootstrap_code_dict = Dict(zip((:initialize, :resample, :update),
                                [plot(load(f); axis=([], false), size=(2000,700)) for f in bootstrap_code_files]))

    N_particles = 10
    infos = particle_filter_bootstrap_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles)

    ani = Animation()
    for info in infos
        code_plot = bootstrap_code_dict[info.type]
        graph = frame_from_info(world, "Run of PF+Bootstrap", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
        frame(ani, plot(code_plot, graph; size=(2000,1000)))
    end
    gif(ani, "imgs/bootstrap_with_code.gif", fps=1)
end


function smcp3_with_code()
    rejuv_code = """
    function particle_filter_rejuv(...)
        traces = Vector{Trace}(undef, N_particles)
        log_weights = Vector{Float64}(undef, N_particles)
        
        for i in 1:N_particles
            traces[i], log_weights[i] =
                ![c1[generate]]!(model, (0, args...), constraints[1])
        end

        for t in 1:T
            traces, log_weights =
                ![c2[resample_ESS]]!(traces, log_weights, ESS_threshold)

            for i in 1:N_particles
                for rejuv_args in rejuv_args_schedule
                    traces[i], log_weights[i] =
                        ![c3[rejuv_kernel]]!(traces[i], log_weights[i], rejuv_args)
                end
            end

            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ =
                    ![c4[update]]!(traces[i], (t, args...), change_only_T, constraints[t+1])
                log_weights[i] += log_weight_increment
            end
        end

        return traces, log_weights
    end
    """
    rejuv_code_files = build_highlighted_pics(lstlisting(rejuv_code), 20, 1., "imgs/rejuv_code"; n_labels=4, silence=true)
    rejuv_code_plots = Dict(zip((:initialize, :resample, :rejuvenate, :update),
                                [plot(load(f); axis=([], false), size=(2000,700)) for f in rejuv_code_files]))

    N_particles = 10
    ESS_threshold =  1. + N_particles / 10.

    grid_n_points_start = [3, 3, 3]
    grid_sizes_start = [.7, .7, π/10]
    grid_args_schedule = [(grid_n_points_start, grid_sizes_start .* (2/3)^(j-1)) for j=1:3]

    infos = particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule)

    ani = Animation()
    for info in infos
        code_plot = rejuv_code_plots[info.type]
        graph = frame_from_info(world, "Run of PF+SMCP3/Grid", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
        frame(ani, plot(code_plot, graph; size=(2000,1000)))
    end
    gif(ani, "imgs/smcp3_with_code.gif", fps=1)
end


function controlled_smcp3_with_code()
    controlled_code = """
    function controlled_particle_filter_rejuv(...)
        traces = Vector{Trace}(undef, N_particles)
        log_weights = Vector{Float64}(undef, N_particles)

        prev_total_weight = 0.
        for i in 1:N_particles
            traces[i], log_weights[i] = ![c1[generate]]!(model, (0, args...), ...)
        end

        for t in 1:T
            traces, log_weights = ![c2[resample_ESS]]!(traces, log_weights, ...)

            rejuv_count = 0
            temp_args_schedule = rejuv_args_schedule
            while logsumexp(log_weights) - prev_total_weight <
                    weight_change_bound && rejuv_count <= MAX_rejuv
                for i in 1:N_particles
                    for rejuv_args in rejuv_args_schedule
                        traces[i], log_weights[i] =
                            ![c3[rejuv_kernel]]!(traces[i], log_weights[i], rejuv_args)
                    end
                end

                if logsumexp(log_weights) - prev_total_weight <
                        weight_change_bound && rejuv_count != MAX_rejuv && t > 1
                    for i in 1:N_particles
                        traces[i], log_weight_increment, _, _ =
                            ![c4[update]]!(traces[i], (t-2, args...), ...)
                        log_weights[i] += log_weight_increment
                    end
                    for i in 1:N_particles
                        traces[i], log_weight_increment, _, _ =
                            ![c5[update]]!(traces[i], (t-1, args...), ...)
                        log_weights[i] += log_weight_increment
                    end

                    traces, weights = [!c6[resample_ESS]]!(traces, log_weights, ...)
                end

                rejuv_count += 1
                temp_args_schedule = args_schedule_modifier(temp_args_schedule, rejuv_count)
            end

            prev_total_weight = logsumexp(log_weights)
            for i in 1:N_particles
                traces[i], log_weight_increment, _, _ =
                    ![c7[update]]!(traces[i], (t, args...), ...)
                log_weights[i] += log_weight_increment
            end
        end

        return traces, log_weights
    end
    """
    # controlled_code_files = build_highlighted_pics(lstlisting(controlled_code), 20, 1., "imgs/controlled_code"; n_labels=7, silence=true)
    # controlled_code_plots = Dict(zip((:initialize, :resample, :rejuvenate, :regenernate_bwd, :regenernate_fwd, :resample2, :update),
    #                                  [plot(load(f); axis=([], false), size=(2000,700)) for f in controlled_code_files]))

    N_particles = 10
    ESS_threshold =  1. + N_particles / 10.

    grid_n_points_start = [3, 3, 3]
    grid_sizes_start = [.7, .7, π/10]
    grid_args_schedule = [(grid_n_points_start, grid_sizes_start .* (2/3)^(j-1)) for j=1:3]

    weight_change_bound = (-1. * 10^5)/20

    # TODO: FIXME
    grid_args_schedule_modifier(args_schedule, rejuv_count) =
        (rejuv_count % 1 == 0) ?
            [(nsteps, sizes .* 0.75) for (nsteps, sizes) in args_schedule] :
            [(nsteps + 2, sizes)     for (nsteps, sizes) in args_schedule];

    traces = [simulate(full_model, (T, full_model_args...)) for _ in 1:N_samples]
    prior_plot = frame_from_traces(world, "Prior on robot paths", nothing, nothing, traces, "prior samples")

    infos = controlled_particle_filter_rejuv_infos(full_model, T, full_model_args, constraints_low_deviation, N_particles, ESS_threshold, grid_smcp3_kernel, grid_args_schedule, weight_change_bound, grid_args_schedule_modifier)

    ani = Animation()
    for info in infos
        code_plot = controlled_code_plots[info.type]
        graph = frame_from_info(world, "Run of Controlled PF+SMCP3/Grid", path_low_deviation, "path to fit", info, "particles"; min_alpha=0.08)
        frame(ani, plot(code_plot, graph; size=(2000,1000)))
    end
    gif(ani, "imgs/controlled_smcp3_with_code.gif", fps=1)
end

export physical_motion, path_model_with_trace, dynamic_static_comparision, full_model_with_trace, bootstrap_with_code, smcp3_with_code, controlled_smcp3_with_code

end
