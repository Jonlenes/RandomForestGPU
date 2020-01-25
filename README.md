# RandomForestGPU-

Perfomance - GPU memory 16 GB	 
200.000 samples, 5 features, 100 estimators, 3 classes - Just predict time
* Serial version: 11.328s
* Parallel - Samples: 0.823659s
* Parallel - All trees and all samples 0.785101s


# Description
First I loaded all the trees into memory. Second, I loaded all the data. Thirdly, I used 2d blocks, the first dimension is the number of trees (that is, one block for each tree), the second dimension is to data parallelism. So it works like this: 1 x blocsY x nThreads samples are predicted in parallel. Meanwhile, the same samples are predicted BlocosX times by changing the tree.

In the first step, I define the limit I want to use on my GPU in terms of numbers of block and numbers of thread. Next step I define how many blocks will be used for the trees (one block for each tree) and how much blocks are free to parallelize the samples (Block limit/number of trees). Our block is 2d, the first dimension being the number of trees and the second the remainder. So now we can use all threads to parallelize the samples and we have a block for each of the trees. All running in parallel. In my case, 100 xBlocks * 10 yBlocks * 1024 threads = 1.024.000 Threads. We have 200.000 samples * 100 trees = 20.000.000 samples to process, so 20M / 1M is about 20 samples by threads.

# Versions
In version two we only paralleled the samples, but the trees were placed on the GPU one by one (sequentially). With a powerful GPU, version 3 is amazing.
