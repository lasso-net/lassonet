# LassoNet

This project is about performing feature selection in neural networks.
At the moment, we support fully connected feed-forward neural networks.
LassoNet is based on the work presented in [this paper]() ([bibtex here for citation]()).
Here is a link to the promo video:


<a href="https://www.youtube.com/watch?v=bbqpUfxA_OA" target="_blank"><img src="https://raw.githubusercontent.com/lasso-net/lassonet/master/docs/images/video_screenshot.png" width="450" alt="Promo Video"/></a>

Our plan is to add more functionality that help users understand the important features in neural networks.


## [LassoNet website: www.lassonet.ml](http://lassonet.ml)
It contains links to the paper as well as documentation/slides.

### Tips
LassoNet sometimes require fine tuning. For optimal performance, consider
- making sure that the *initial* dense model (with ![](https://latex.codecogs.com/svg.latex?%5Clambda%20%3D%200)) has trained well, before starting the LassoNet  regularization path. This may involve hyper-parameter tuning, choosing the right optimizer, and so on. If the dense model is underperforming, it is likely that the sparser models will as well.
- making sure the stepsize over the ![](https://latex.codecogs.com/svg.latex?%5Clambda) path is not too large. By default, the stepsize runs over the logscale between two values ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D) and ![](https://latex.codecogs.com/svg.latex?%5Clambda_%7Bmin%7D).
