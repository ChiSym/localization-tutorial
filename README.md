# ProbComp Localization Tutorial

The Julia file `probcomp-localization-tutorial.jl` describes a Jupyter `.ipynb` notebook using the Jupytext `percent` format.  This notebook requires Julia version >= 1.9.1.

Instructions/prerequisites for use:
* Install Jupyter: `pip install jupyterlab`
* Install Julia: `curl -fsSL https://install.julialang.org | sh`
* Add Julia kernel to Jupyter: `using Pkg; Pkg.add("IJulia")`
* Install Jupytext: `pip install jupytext`
* Option 1: Run Jupytext yourself (`jupytext --to ipynb probcomp-localization-tutorial.jl`) and play in JupyterLab.
* Option 2: Work in VS Code.
  * Install extensions: "Julia" (langauge support), "Jupyter", "Jupyter Keymap", "Jupytext for Notebooks (congyiwu)".
  * Upon opening the tutorial `.jl` file, a button will appear to "Open as a Jupyter Notebook"; hit it.
  * Resulting notebook file is managed/ephemeral.  Hitting save will push changes back to the `.jl` file, *disregarding output cells*.
  * Upon first run, you will have to associate the file with a kernel; choose a Julia version.
  * Upon first run, the first cell will cause Julia to download and precompile many things, taking a long time.

Specific instances of `.ipynb` files (generated for use for particular real-world presentations, from particular commits of the `.jl` file, and including particular computations and graphics) are to be stored *in branches other than `main` of the repo*, to conserve clean diffs.
