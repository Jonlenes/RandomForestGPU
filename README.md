# RandomForestGPU

This repository has code to perform data prediction using a Random Forest classifier trained with Sklearn (python) in C++. High performance using the GPU to make predictions in parallel.
Train your model with all python facilities,  use this code to export your model from Python, then you can load it in C++ code and perform the prediction using the GPU.


## Perfomance - GPU memory 16 GB	 
200.000 samples, 5 features, 100 estimators, 3 classes - Just predict time
* Serial version: 11.328s
* Parallel - Data parallelization: 0.823659s
* Parallel - Trees and data parallelization 0.785101s

## Versions
* V2: Only the samples are paralleled, ie, the trees were placed on the GPU one by one (sequentially). 
* V3: Tree and sample paralleled. With a powerful GPU, version 3 is amazing.
