# Feature Selection in Neural Networks

## [LassoNet website](http://louisabraham.github.io/lassonet)
The website contains links to the paper as well as documentation/slides.

### Tips
LassoNet sometimes require fine tuning. For optimal performance, consider
- making sure that the *initial* dense model (with ![](https://latex.codecogs.com/svg.latex?%5Clambda%20%3D%200)) has trained well, before starting the LassoNet  regularization path. This may involve hyper-parameter tuning, choosing the right optimizer, and so on. If the dense model is underperforming, it is likely that the sparser models will as well.
- making sure the stepsize over the ![](https://latex.codecogs.com/svg.latex?%5Clambda) path is not too large. By default, the stepsize runs over the logscale between two values ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D) and ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D).
