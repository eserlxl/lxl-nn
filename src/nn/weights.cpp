#include "nn/neuralNetwork.h"

void NeuralNetwork::randWeights() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
#ifdef BP_USE_BIAS
            bias[i][j] = 0;
#endif
            for (uzi k = 0; k < model[i]; k++) {
                P[i][k]->Link[j]->weight = randomNumberExtended();
            }
        }
    }
}

#ifdef ADAPTIVE_LEARNING

void NeuralNetwork::saveWeights() {
    weightBackup.clear();
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
            weightBackup.push_back(bias[i][j]);
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                weightBackup.push_back(P[i][j]->Link[k]->weight);
                weightBackup.push_back(P[i][j]->Link[k]->deltaWeight);
            }
        }
    }
}

void NeuralNetwork::loadWeights() {
    if (weightBackup.empty()) {
        return;
    }
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
            bias[i][j] = weightBackup[h++];
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->Link[k]->weight = weightBackup[h++];
                P[i][j]->Link[k]->deltaWeight = weightBackup[h++];
            }
        }
    }
}

#endif
