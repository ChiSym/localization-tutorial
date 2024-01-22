# Breakage of `block` is triggered by
# * "!["
# * followed by an optional charcter 'c' or 'm' for "code" or "math" (as opposed to plain text)
# * followed by a nonempty comma-separated sequence of Ints
# * followed by "["
# and terminated by "]]!".
function parse_highlights(block)
    pieces = []
    remainder = block
    while remainder != ""
        start_brackets = findfirst("![", remainder)
        if isnothing(start_brackets)
            push!(pieces, (:text, 0, remainder))
            remainder = ""
        else
            if start_brackets[1] > 1
                push!(pieces, (:text, 0, remainder[1:start_brackets[1]-1]))
            end
            remainder = remainder[start_brackets[1]+2:end]

            if remainder[1] == 'c'
                mode = :code
                remainder = remainder[2:end]
            elseif remainder[1] == 'm'
                mode = :math
                remainder = remainder[2:end]
            else
                mode = :text
            end

            past_digits = findfirst("[", remainder)
            labels = []
            labels_str = remainder[1:past_digits[1]-1]
            chunk = findfirst(",", labels_str)
            while !isnothing(chunk)
                push!(labels, parse(Int, labels_str[1:chunk[1]-1]))
                labels_str = labels_str[chunk[1]+1:end]
                chunk = findfirst(",", labels_str)
            end
            push!(labels, parse(Int, labels_str))
            remainder = remainder[past_digits[1]+1:end]

            past_brackets = findfirst("]]!", remainder)
            push!(pieces, (mode, labels, remainder[1:past_brackets[1]-1]))
            remainder = remainder[past_brackets[1]+3:end]
        end
    end
    return pieces
end

# The param `varwidth_frac` is roughly (not exactly) an aspect ratio, controls truncation on the right.
function highlighted_versions(pieces, n_labels, border, varwidth_frac)
    stuffs = (
        file_start_1 = "\\documentclass[crop=true,border={",
        file_start_2 = "pt 0pt 0pt 0pt},varwidth=",
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
""",
        file_end = "\\end{document}",
        highlight_start = (text = "\\hlight{", code = "(*\\hlight{", math = "\\hlightmath{"),
        highlight_end = (text = "}", code = "}*)", math = "}")
    )

    versions = []
    for i in 1:n_labels
        version = stuffs.file_start_1 * "$(border)" * stuffs.file_start_2 * "$(varwidth_frac)" * stuffs.file_start_3
        for (mode, labels, piece) in pieces
            if i in labels
                version = version * stuffs.highlight_start[mode] * piece * stuffs.highlight_end[mode]
            else
                version = version * piece
            end
        end
        version = version * stuffs.file_end
        push!(versions, version)
    end
    return versions
end

function build_pic(file_text, file_name)
    xrap = tempname("./")
    run(`mkdir $xrap`)
    open(f -> write(f, file_text), "$xrap/$xrap.tex", "w")
    run(`pdflatex -interaction=nonstopmode -output-directory=$xrap $xrap/$xrap.tex`)
    run(`convert -colorspace RGB -density 500 -quality 100 -background white -alpha remove -alpha off $xrap/$xrap.pdf $file_name`)
    run(`rm -rf $xrap`)
end

function build_highlighted_pics(block, n_labels, border, varwidth_frac, file_name_base)
    pieces = parse_highlights(block)
    versions = highlighted_versions(pieces, n_labels, border, varwidth_frac)
    files = []
    for (i, file_text) in enumerate(versions)
        file_name = "$(file_name_base)_$i.png"
        build_pic(file_text, file_name)
        push!(files, file_name)
    end
    return files
end

test_math = """
\$\$
![m1,2,3,4[P_\\text{path}(z_{0:T}; r_{0:T}, w, \\nu)]]!
= ![m2[P_\\text{start}(z_0; r_0, \\nu)]]! \\cdot ![m3[\\prod\\nolimits_{t=1}^T]]! ![m4[P_\\text{step}(z_t; z_{t-1}, r_t, w, \\nu)]]!
\$\$
"""
print(test_math)


test_code = """
\\begin{lstlisting}
@gen function ![1,2,3,4[path\\_model\\_loop]]!(T :: Int, robot\\_inputs :: NamedTuple, world\\_inputs :: NamedTuple, motion\\_settings :: NamedTuple) :: Vector{Pose}
    pose = {:initial => :pose} ~ ![2[start\\_pose\\_prior(robot\\_inputs.start, motion\\_settings)]]!

    ![3[for t in 1:T]]!
        pose = {:steps => t => :pose} ~ ![4[step\\_model(pose, robot\\_inputs.controls[t], world\\_inputs, motion\\_settings)]]!
    end
end
\\end{lstlisting}
"""
print(test_code)
