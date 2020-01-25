#include "RandomForest.h"


RandomForest::RandomForest(const string &path)
{
	// Load the file
    ifstream &file = *load_file(path);
    
	// Number of trees and classes
    file >> nEstimators;
    file >> nClasses;

	// Creat the trees
    for (int i = 0; i < nEstimators; ++i) {
        DecisionTree tree(file, nClasses);
        decisionTrees.push_back(tree);
    }

    close_file(&file);
}


vector<int> RandomForest::predict(Data &data) {
    vector< vector<double> > all_pred(data.getSampleSize(), vector<double>(nClasses, 0));

	// Predict in all the trees
    for (int i = 0; i < nEstimators; i++) {
        decisionTrees[i].predict(data, all_pred);
    }
	
	// Compute votes
    vector<int> classes;
    for (int i = 0; i < data.getSampleSize(); ++i) {
        classes.push_back( argMax(all_pred[i]) );
    }

    return classes;
}
