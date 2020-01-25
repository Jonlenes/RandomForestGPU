#include "RandomForest.h"
#include "DProcess.cpp"

RandomForest::RandomForest(const string &path)
{
    // Load the file
    ifstream &file = *load_file(path);
    
    // Number of trees and classes
    file >> nEstimators;
    file >> nClasses;
    
    // Allocate array of trees on GPU
    d_allocate((void **)&d_trees, nEstimators * sizeof(DecisionTree));

    // Creat the trees
    for (int i = 0; i < nEstimators; ++i)
        loadDT(d_trees[i], file, nClasses);

    // Close file model
    close_file(&file);
    
}


RandomForest::~RandomForest()
{
    cudaFree(d_trees);
}


uint8_t *RandomForest::predict(Data &data) {
    // Alocate memory
    float *d_features = nullptr;
    
    // Copy all data the to GPU
    d_allocate((void**)&d_features, data.nFeatures * data.nSamples * sizeof(float), data.features);

    // Predict in all the trees on GPU
    uint8_t *classes = ddt_predict(d_trees, d_features, nEstimators, data.nSamples, data.nFeatures, nClasses);

    // Free memory
    cudaFree( d_features );
    
    return classes;
}


/*
 * Load Decisoin Tree model
 */
void RandomForest::loadDT(DecisionTree &d_dt, ifstream &file, int nClasses) {
    // Number of nodes of this tree
    int nodeCount;
    file >> nodeCount;
    
    // Read informations about the tree
    float *children_left = getArray(file, nodeCount);
    float *children_right = getArray(file, nodeCount);
    float *feature = getArray(file, nodeCount);
    float *threshold = getArray(file, nodeCount);

    // Allocate all nodes on CPU
    Node *nodes = new Node[nodeCount];
    float *values = new float[nodeCount * nClasses];
    
    // Fill nodes and build the tree 
    //     This loop is slow - Many possibilities for improvements here
    //          omp parallel? Have to solver file dependence - Easy, new tread to read
    //          read per line instead of single value? 
    for (int i = 0; i < nodeCount; ++i) {
        
        nodes[i].featureIndex = feature[i];
        nodes[i].threshold = threshold[i];
        nodes[i].left = children_left[i];
        nodes[i].right = children_right[i];
        
        float *valuesTemp = getArray(file, nClasses);
        for (int j = 0; j < nClasses; ++j)
            values[i * nClasses + j] = valuesTemp[j];
        delete [] valuesTemp;
    }
    
    // Copy nodes to GPU
    Node *d_nodes;
    cudaMalloc((void **)&d_nodes, nodeCount * sizeof(Node));
    cudaMemcpy(d_nodes, nodes, nodeCount * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_dt.d_nodes), &d_nodes, sizeof(Node *), cudaMemcpyHostToDevice);
    
    // Copy values to GPU
    float *d_values;
    cudaMalloc((void **)&d_values, nodeCount * nClasses * sizeof(float));
    cudaMemcpy(d_values, values, nodeCount * nClasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_dt.d_values), &d_values, sizeof(float *), cudaMemcpyHostToDevice);

    // Free memory
    delete [] children_left;
    delete [] children_right;
    delete [] feature;
    delete [] threshold;
}
