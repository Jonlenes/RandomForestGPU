#include <iostream>
#include "RandomForest.h"

using namespace std;

int main()
{
	// Load RF from file
    printf("Loading the model...\n");
    RandomForest randomForest("./data/model.rf");

    // 1000 samples and 20 features
	printf("Loading the data...\n");
    Data data;
    data.read("./data/X_test.data");

	// Predict
    printf("Predict...\n");
    vector<int> classes = randomForest.predict(data);

	// Save result in file
    ofstream pred_file("./data/cpp.pred");
    ostream_iterator<int> p_iterator(pred_file, "\n");
    copy(classes.begin(), classes.end(), p_iterator);

    return 0;
}
