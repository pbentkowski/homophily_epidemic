# Impact of homophily in adherence to anti-epidemic measures on the spread of infectious diseases in social networks
Code for a paper: [Bentkowski P, Gubiec T. Impact of Homophily in Adherence to Anti-Epidemic Measures on the Spread of Infectious Diseases in Social Networks. Entropy. 2025; 27(10):1071. https://doi.org/10.3390/e27101071 ](https://www.mdpi.com/1099-4300/27/10/1071)

**The main file is the `simulations.py` that creates a bundle of simulations with fixed *η* and *δ* values spanning across of *a* values.**

To run the simulation:

```bash
python simulation.py [run_tag]
```



### Other files:

* `analysis_from_csv.py` - analysis of raw results from a single simulation bundle.
* `analysis_post.py` - visualisation of analysed results.
* `generate_dirs_and_scripts.py` - generates series of directories with varying *η* and *δ* params according to a list given in:
* `SBM_param_dirr_list.txt` - the list of *eta* and *delta* params, and the names of directories

### Dependencies

The main dependencies are `python-igraph` and `joblib`.
