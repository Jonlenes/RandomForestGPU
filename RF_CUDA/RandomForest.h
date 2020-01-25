#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "DT.h"
#include "Data.h"
#include "Util.h"
#include <string.h>

class RandomForest {
public:
    DecisionTree *d_trees;
    int nEstimators;
    int nClasses;

public:
    RandomForest(const string &path);
    ~RandomForest();

    void loadDT(DecisionTree &dt, ifstream &file, int nClasses);
    uint8_t *predict(Data &data);
};

#endif //RANDOMFOREST_H
