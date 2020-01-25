#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "DecisionTree.h"
#include "Data.h"
#include "Util.h"

class RandomForest {
private:
    vector<DecisionTree> decisionTrees;
    int nEstimators;
    int nClasses;

public:
    RandomForest(const string &path);
    vector<int> predict(Data &data);
};

#endif //RANDOMFOREST_H
