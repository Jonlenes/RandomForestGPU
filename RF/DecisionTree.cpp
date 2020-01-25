#include "DecisionTree.h"


DecisionTree::DecisionTree(ifstream &file, int nClasses) {
    // Number of nodes of this tree
	int nodeCount;
    file >> nodeCount;

	// Read informations about the tree
    vector<double> children_left = getVector(file, nodeCount);
    vector<double> children_right = getVector(file, nodeCount);
    vector<double> feature = getVector(file, nodeCount);
    vector<double> threshold = getVector(file, nodeCount);

    // Create all nodes
    vector< shared_ptr<Node> > nodes;
    for (int i = 0; i < nodeCount; ++i) {
        shared_ptr<Node> node(new Node());
        node->featureIndex = feature[i];
        node->threshold = threshold[i];
        node->value = getVector(file, nClasses);
        nodes.push_back( node );
    }

    // Build the tree
    for (int i = 0; i < nodeCount; ++i) {
        shared_ptr<Node> left = nullptr;
        shared_ptr<Node> right = nullptr;

        if (int(children_left[i]) != -1)
            left = nodes[ children_left[i] ];

        if (int(children_right[i]) != -1)
            right = nodes[ children_right[i] ];

        nodes[i]->left = left;
        nodes[i]->right = right;
    }

	// First node is the root
    this->root = nodes[ 0 ];

}

/*
* Walk in the to find the right leaf
*/
vector<double> DecisionTree::computePredict(int sampleIndex, Data &Data) {
    auto node = root;
    while (node->left) {
		// Check if the feature is bigger than threshold
        if (Data.getFeature(sampleIndex, node->featureIndex) > node->threshold) {
            node = node->right;
        } else {
            node = node->left;
        }
    }
    return node->value;
}


/*
* Compute the votes
*/
void DecisionTree::predict(Data &Data, vector<vector<double>> &results) {
    for (int i = 0; i < results.size(); i++) {
        vector<double> pred_tree = computePredict(i, Data);
        int cls = argMax(pred_tree);
        results[i][cls] += 1;
    }
}
