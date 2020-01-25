#ifndef CUDAPROCESS_H
#define CUDAPROCESS_H

#include "DT.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdint>

// Max that I want to use in my GPU
// You can calculate this values on execution time
const static int MAX_N_BLOCK = 1024;
const static int MAX_N_THREADS = 1024;


// The real number of threads and blocks basead on # samples
static int nThreads;
static int nBlocksX;
static int nBlocksY;
static int nProcess;


namespace {

    /*
     * Compute the the number of threads, blocks and # samples
     * per thread based on MAX_N_BLOCK and MAX_N_THREADS
     */
    __host__ void computeBlocksThreads(int n, int ne) {
        nThreads = MAX_N_THREADS; // # max of threads
        nBlocksX = ne; // One block for each tree
        nBlocksY = int(MAX_N_BLOCK / nBlocksX); // Rest of blocks
        nProcess = 1;

        if (n < nThreads) {
            nBlocksY = 1;
            nThreads = n;
        } else if ( n < (nThreads * nBlocksY) )
            nBlocksY = n / nThreads + 1;
        else
            nProcess = n / (nThreads * nBlocksY) + 1;
    }



    /*
     * Return index of max value
     */
    template< typename T >
    __device__ int argMax(T *v, int size){
        int indexMax = 0;
        for (int i = 1; i < size; ++i) {
            if (v[i] > v[indexMax])
                indexMax = i;
        }

        return indexMax;
    }


    /*
     * Compute mode
     */
    __device__ int arrayMode(uint8_t *v, int size, int nClasses) {
        uint8_t *votes = new uint8_t[nClasses];
        for (int i = 0; i < nClasses; ++i)
            votes[i] = 0;
        for (int i = 0; i < size; ++i)
            votes[ v[i] ]++;
        int i = argMax(votes, nClasses);
        delete votes;
        return i;
    }


    /*
     * Walk in the to find the leaf
     */
    __global__
    void _ddt_computePredict(DecisionTree *trees, float *features, int nTrees, int nFeatures,
                             int nClasses, int nProcess, int nSamples, uint8_t *predict) {

        // Tree of this thread
        int treeIndex = blockIdx.x;
        DecisionTree &dt = trees[ treeIndex ];

        // Begin and end samples process
        int sampleBegin = (blockIdx.y * blockDim.x + threadIdx.x) * nProcess;
        int sampleEnd = sampleBegin + nProcess;

        // Check bound and # Samples
        if (sampleBegin >= nSamples) return;
        if (sampleEnd > nSamples) sampleEnd = nSamples;

        // Process nProcess by thread
        for ( int sampleIndex = sampleBegin; sampleIndex < sampleEnd; ++sampleIndex) {
            int index = 0;
            while (dt.d_nodes[index].left != -1) {
                if ( features[ sampleIndex * nFeatures + dt.d_nodes[index].featureIndex ] > dt.d_nodes[index].threshold) {
                    index = dt.d_nodes[index].right;
                } else {
                    index = dt.d_nodes[index].left;
                }
            }

            // Get class with this tree
            predict[ sampleIndex * nTrees + treeIndex] = argMax(dt.d_values + (index * nClasses), nClasses);;
        }

    }


    __global__
    void _ddt_computeVotes(int nTrees, int nClasses, int nProcess, int nSamples, uint8_t *predict, uint8_t *out) {
        // Begin and end samples process
        int sampleBegin = (blockIdx.x * blockDim.x + threadIdx.x) * nProcess;
        int sampleEnd = sampleBegin + nProcess;

        // Check bound and # Samples
        if (sampleBegin >= nSamples) return;
        if (sampleEnd > nSamples) sampleEnd = nSamples;

        // Compute vote for each sample in this thread
        for ( int sampleIndex = sampleBegin; sampleIndex < sampleEnd; ++sampleIndex)
            out[ sampleIndex ] = arrayMode(predict + (sampleIndex * nTrees), nTrees, nClasses);
    }


    /*
     * Allocate GPU memory and copy data (if not null).
     */
    void d_allocate(void **m, int size, void *data=nullptr) {
        cudaMalloc(m, size);
        if (data)
            cudaMemcpy(*m, data, size, cudaMemcpyHostToDevice);
    }


    /*
     * Compute the votes
     * DDT - Device Deciosion Tree
     */
    uint8_t *ddt_predict(DecisionTree *&D_trees, float *&D_features, int nEstimators, int nSamples, int nFeatures, uint8_t nClasses) {
        // Allocate and copy nodes
        uint8_t *d_classes = nullptr;
        uint8_t *d_pred = nullptr;

        // Allocate memory on GPU
        d_allocate((void**)&d_classes, nSamples * sizeof(uint8_t));
        d_allocate((void**)&d_pred, nEstimators * nSamples * sizeof(uint8_t));
        cudaMemset(d_pred, 0, nEstimators * nSamples * sizeof(uint8_t));

        // Compute # threads and # blocks
        computeBlocksThreads(nSamples, nEstimators);
        dim3 blks(nBlocksX, nBlocksY);

        // Predict on GPU (parallel)
        _ddt_computePredict<<< blks, nThreads >>> (D_trees, D_features, nEstimators, nFeatures, nClasses, nProcess, nSamples, d_pred);
        // Compute votes on GPU
        _ddt_computeVotes<<< nBlocksY, nThreads>>>(nEstimators, nClasses, nProcess, nSamples, d_pred, d_classes);

        //Copy result back
        uint8_t *classes = new uint8_t[nSamples];
        cudaMemcpy(classes, d_classes, nSamples * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        return classes;
    }
}

#endif // CUDAPROCESS_H
