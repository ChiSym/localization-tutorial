# ProbComp Localization Tutorial

The Julia file `probcomp-localization-tutorial.jl` describes a Jupyter `.ipynb` notebook using the Jupytext `percent` format.  This notebook requires Julia version >= 1.9.1.

If using VS Code, we recommend using the extension "Jupytext for Notebooks (congyiwu)" to make the translation.

Specific instances of `.ipynb` files (generated for use for particular real-world presentations, from particular commits of the `.jl` file, and including particular computations and graphics) are to be stored *in branches other than `main` of the repo*, to conserve clean diffs.

# Python Edition

### Create a venv with VSCode

As described on the [Visual Studio Code Python environment
page](https://code.visualstudio.com/docs/python/environments#_creating-environments):

- open the command palette with `Shift-Cmd-P` and search for / select the
  `Python: Create Environment` command
- select the `venv` environment type, and some flavor of Python 3.11 as your
  interpreter type
- at the command palette (`Shift-Cmd-P`) select `Python: Create Terminal`

  This will open a terminal at the bottom of the editor configured with the
  environment that you just created.

### Install GenJAX

At the VSCode terminal, run the following commands to install GenJAX:

```bash
pip install --quiet keyring keyrings.google-artifactregistry-auth
pip install genjax==0.3.1.post146.dev0+717cf10f --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

Next, install `jaxlib` for your desired architecture. If you're on a machine
with no GPU, or on a Mac, run this command:

```bash
pip install "jax[cpu]==0.4.28"
```

On a Linux machine with a GPU, run the following command:

```sh
pip install "jax[cuda12]==0.4.28"
```

To check that everything is working, type `ipython` and, at the prompt, run

```python
import genjax
```

### Notebooks in VSCode

To use this environment from an `ipynb` file:

- open an `ipynb` file, like `Intro.ipynb` in this directory
- click the kernel picker at the top right of the notebook:

![ipynb kernel picker](https://code.visualstudio.com/assets/docs/datascience/jupyter/native-kernel-picker.png)

- pick the recommended `.venv` kernel

In a Python cell in the notebook, confirm that you've picked the correct kernel by running:

```python
import genjax
```