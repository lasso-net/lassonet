# Feature Selection in Neural Networks

### Tips
LassoNet sometimes require fine tuning. For optimal performance, consider
- checking that the dense model with ![](https://latex.codecogs.com/svg.latex?%5Clambda%20%3D%200) has acceptable accuracy before running over the entire regularization path
- making sure the stepsize over the ![](https://latex.codecogs.com/svg.latex?%5Clambda) path is not too large. By default, the stepsize runs over the logscale between two values ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D) and ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D).