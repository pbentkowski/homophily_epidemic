# Impact of homophily in adherence to anti-epidemic measures on the spread of infectious diseases in social networks
Code for a paper: [Bentkowski P, Gubiec T. Impact of Homophily in Adherence to Anti-Epidemic Measures on the Spread of Infectious Diseases in Social Networks. Entropy. 2025; 27(10):1071. https://doi.org/10.3390/e27101071 ](https://www.mdpi.com/1099-4300/27/10/1071)

**The main file is the `simulation.py` that creates a bundle of simulations with fixed *η* and *δ* values spanning across of *a* values.**

### Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). After cloning the repository, run once:

```bash
uv sync
```

This installs Python 3.12 (if needed) and all dependencies into a local `.venv`, using the exact versions pinned in `uv.lock`.

### Running

To run the simulation:

```bash
uv run python simulation.py [run_tag]
```

The analysis scripts are run the same way, e.g. `uv run python analysis_from_csv.py`.



### Other files:

* `analysis_from_csv.py` - analysis of raw results from a single simulation bundle.
* `analysis_post.py` - visualisation of analysed results.
* `generate_dirs_and_scripts.py` - generates series of directories with varying *η* and *δ* params according to a list given in:
* `SBM_param_dirr_list.txt` - the list of *eta* and *delta* params, and the names of directories

### Dependencies

The main dependencies are `igraph` and `joblib`; the analysis also uses `numpy`, `pandas`, `matplotlib` and `seaborn`. The full list is in `pyproject.toml`, and `uv sync` installs all of them.
