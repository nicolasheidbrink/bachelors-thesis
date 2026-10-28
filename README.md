# Exploring the Geometry of the Space of Copulas under the Copula-Constrained Wasserstein Distance

This repository contains code for my bachelor's thesis. 

In brief, 99 copulas were taken as a sample to represent the space of copulas. The necessary pairwise 
copula-constrained Wasserstein distances were calculated on the computing cluster HAICORE@KIT. Because 
each pairwise distance involves training neural networks and is therefore computationally expensive, 
the training was optimized for efficiency. Distance calculations that were redundant due to symmetry 
properties were also found and excluded from the calculation,bringing the total to about 1000 pairwise 
distances. 

With the distances, multidimensional scaling maps of the sample were created to visualize the space of copulas 
in fewer dimensions. The map was analyzed to determine what factors best characterize the geometry of the space.

## Repository Layout
```
├── distance_calculation_campaign/
│   ├── copula_families.py              copula families and constructions
│   ├── copula_constrained_wasserstein_velocity_field.py
│   │                                   learning the transport velocity field
│   ├── fade_distance.py                L_fade upper bound
│   ├── compute_distance.py             calculates the distance for one pair
│   ├── run_pair_queue.py               cluster worker used for the campaign
│   ├── unique_pairs.py                 definition of the endpoint pairs
│   └── unique_pairs.json               results for all 1045 pairs
├── distance_calculation_optimization/  optimization of distance calculation
│   ├── hyperparameter_grid_search_nn_training.ipynb
│   ├── hyperparameter_grid_search_nn_architecture.ipynb
│   └── jitter_comparison.ipynb
├── analysis/
│   ├── distance_comparison.ipynb       compares distance metrics for copulas
│   ├── mds_maps.ipynb                  MDS maps of the distance matrix
│   └── scatterplots.ipynb              sample scatterplots of the copulas
```
