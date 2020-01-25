
#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "Data.h"
#include <memory>
#include <functional>
#include <utility>
#include <cmath>
#include <fstream>

using namespace std;


class DecisionTree {
private:
	// Node of the tree
    struct Node {
        int featureIndex;
        shared_ptr<Node> left;
        shared_ptr<Node> right;
        double threshold;
        vector<double> value;

        Node() {
            left = nullptr;
            right = nullptr;
        }
    };

    shared_ptr<Node> root;

public:
    DecisionTree(ifstream &file, int nClasses);

    vector<double> computePredict(int sampleIndex, Data &Data);
    void predict(Data &Data, vector<vector<double> > &results);
    void loadModel(const string &path);
};

#endif
