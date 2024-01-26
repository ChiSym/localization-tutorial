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
    while remainder != ""
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
            labels = []
            (num, head) = split_first(head, ",")
            while !isnothing(head)
                push!(labels, parse(Int, num))
                (num, head) = split_first(head, ",")
            end
            if num != ""; push!(labels, parse(Int, num)) end

            (head, remainder) = split_first(remainder, "]]!")
            push!(pieces, (mode, labels, head))
            @assert !isnothing(remainder) "concluding exclamation char?"
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

