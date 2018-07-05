# prior-information
Bachelor Thesis about using prior information for sparse grid machine learning.

# Abstract
This thesis discusses different ways of imposing prior knowledge about datasets on
the sparse grid model for supervised learning. We introduce a Tikhonov regularization
method that uses information about the smoothness of the function we want to approximate.
We also present the sparsity-inducing penalties lasso, elastic net, and group
lasso. The different regularization approaches are compared with the standard ridge
regularization. Because some regularization penalties are not differentiable, we discuss
the fast iterative shrinkage-thresholding algorithm and show how it can be used in
conjunction with our added regularization methods. Furthermore, we modify the grid
generation procedure. The first discussed method is generalized sparse grids, which
allows us to control the granularity of the grid. The second method is interaction-term
aware sparse grids, which are used to construct smaller and more efficient grids for
image recognition problems. All methods were implemented with the SG++ library
and showed promising results for both artificial and real-world datasets.

# Source Code
The actual source code is not contained in this repository.
All changes were integrated into the **SG++** library which can be found at [SG++](http://sgpp.sparsegrids.org/).
