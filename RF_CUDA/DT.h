#ifndef NODE_H
#define NODE_H

// Node of the tree
struct Node {
    int featureIndex;
    int left;
    int right;
    float threshold;

    Node() {
        left = -1;
        right = -1;
    }
};

// Tree
struct DecisionTree {
    Node *d_nodes;
    float *d_values;
    
    DecisionTree() {
        d_nodes = nullptr;
        d_values = nullptr;
    }
};

#endif // NODE_H
