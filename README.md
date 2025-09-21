# Impact of homophily in adherence to anti-epidemic measures on the spread of infectious diseases in social networks
Code for the paper: Impact of homophily in adherence to anti-epidemic measures on the spread of infectious diseases in social networks

**The simulation are in file `simulations.py` that creates a bundle of simulations with fixed *eta* and *delta* values.**

To run the simulation:

```bash
python simulation.py [run_tag]
```



### Other files:

* `analysis_from_csv.py` - analysis of raw results from a single simulation bundle.
* `analysis_post.py` - visualisation of analysed results.
* `generate_dirs_and_scripts.py` - generates series of directories with varying *eta* and *delta* params according to a list given in:
* `SBM_param_dirr_list.txt` - the list of *eta* and *delta* params, and the names of directories

### Dependencies

The main dependencies are `python-igraph` and `joblib`.
