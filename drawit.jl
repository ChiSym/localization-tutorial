# Breakage of `block` is triggered by ("![", followed by a parsable Int, followed by "["),
# and terminated by "]]!".
function parse_highlights(block)
    pieces = []
    remainder = block
    while remainder != ""
        start_brackets = findfirst("![", remainder)
        if isnothing(start_brackets)
            push!(pieces, (0, remainder))
            remainder = ""
        else
            if start_brackets[1] > 1
                push!(pieces, (0, remainder[1:start_brackets[1]-1]))
            end
            remainder = remainder[start_brackets[1]+2:end]

            past_digits = findfirst("[", remainder)
            label = parse(Int, remainder[1:past_digits[1]-1])
            remainder = remainder[past_digits[1]+1:end]

            past_brackets = findfirst("]]!", remainder)
            push!(pieces, (label, remainder[1:past_brackets[1]-1]))
            remainder = remainder[past_brackets[1]+3:end]
        end
    end
    return pieces
end

function highlighted_versions(block, n_labels, stuffs, varwidth_frac)
    pieces = parse_highlights(block)
    versions = []
    for i in 1:n_labels
        version = stuffs.file_start_start * "$(varwidth_frac)" * stuffs.file_start
        for (j, piece) in pieces
            if j == i
                version = version * stuffs.highlight_start * piece * stuffs.highlight_end
            else
                version = version * piece
            end
        end
        version = stuffs.file_end
        push!(versions, version)
    end
    return versions
end

math_stuffs = (
file_start_start = "\\documentclass[crop=true,border={0pt 0pt 0pt 0pt},varwidth=",
file_start = """
\\linewidth]{standalone}
%\\usepackage{mdframed}
\\usepackage{listings}
\\usepackage{textcomp}
\\usepackage{xcolor}
\\usepackage{bm}
\\usepackage{soul}
\\definecolor{highlight}{HTML}{ffe70f}
\\newcommand{\\hlight}[1]{\\setlength{\\fboxsep}{0pt}\\colorbox{highlight}{#1}}
\\definecolor{boldkwcolor}{HTML}{00679e}
\\lstset{
    escapeinside={(*}{*)},
    basicstyle=\\ttfamily\\small,
    numbers=left,
    columns=fullflexible,
    keepspaces=true,
    literate={~} {$\\sim$}{1},
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
file_end = """
\\end{document}
""",
highlight_start = "\\hlight{",
highlight_end = "}"
)

code_stuffs = (
file_start_start = "\\documentclass[crop=true,border={20pt 0pt 0pt 0pt},varwidth=",
file_start = math_stuffs.file_start * """
\\begin{lstlisting}
""",
file_end = """
\\end{lstlisting}
""" * math_stuffs.file_end,
highlight_start = "(*" * math_stuffs.highlight_start,
highlight_end = math_stuffs.hilight_end * "*)"
)


test_math = """
The corresponding mathematical picture is as follows.  We write \$x\_{a:b} = (x_a, x_{a+1}, \\ldots, x_b)\$ to gather items \$x\_t\$ into a vector.

In addition to the previous data, we are given an estimated start pose \$r\_0\$ and controls \$r\_t = (s\_t, \\eta\_t)\$ for \$t=1,\\ldots,T\$.  Then `path\_model` corresponds to a distribution over traces denoted \$\\text{path}\$; these traces are identified with vectors, namely, \$z\_{0:T} \\sim \\text{path}(r\_{0:T}, w, \\nu)\$ is the same as \$z\_0 \\sim \\text{start}(r\_0, \\nu)\$ and \$z\_t \\sim \\text{step}(z\_{t-1}, r\_t, w, \\nu)\$ for \$t=1,\\ldots,T\$.  Here and henceforth we use the shorthand \$\\text{step}(z, \\ldots) := \\text{step}(\\text{retval}(z), \\ldots)\$.  The density function is
\$\$
P\_\\text{path}(z\_{0:T}; r\_{0:T}, w, \\nu)
= P\_\\text{start}(z\_0; r\_0, \\nu) \\cdot \\prod\\nolimits\_{t=1}^T P\_\\text{step}(z\_t; z\_{t-1}, r\_t, w, \\nu)
\$\$
where each term, in turn, factors into a product of two (multivariate) normal densities as described above.
"""


test_code = """
function mcmc\_step(particle, log\_weight, mcmc\_proposal, mcmc\_args, mcmc\_rule)
    proposed\_particle, proposed\_log_weight, viz = mcmc\_proposal(particle, log\_weight, mcmc\_args)
    return mcmc\_rule([particle, proposed\_particle], [log\_weight, proposed\_log_weight])..., viz
end
mcmc\_kernel(mcmc\_proposal, mcmc\_rule) =
    (particle, log\_weight, mcmc\_args) -> mcmc\_step(particle, log\_weight, mcmc\_proposal, mcmc\_args, mcmc\_rule)

boltzmann\_rule = sample

# Assumes `particles` is ordered so that first item is the original and second item is the proposed.
function mh\_rule(particles, log\_weights)
    @assert length(particles) == length(log\_weights) == 2
    acceptance_ratio = min(1., exp(log\_weights[2] - log\_weights[1]))
    return (bernoulli(acceptance\_ratio) ? particles[2] : particles[1]), log\_weights[1]
end;
"""
