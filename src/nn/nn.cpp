#include <nn.h>

void nn::create() {
    uzi id = 0;
    for (uzi i : model) {
        std::vector<perceptron *> tempVec;

        for (uzi j = 0; j < i; j++) {
            auto *temp = new perceptron(id++, 0.f);
            tempVec.push_back(temp);
        }
        P.push_back(tempVec);
    }
}

void nn::connect() {
    for (uzi i = 0; i < model.size() - 1; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->connect(P[i + 1][k]->id);
            }
        }
    }
}
