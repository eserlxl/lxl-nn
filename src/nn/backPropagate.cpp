#include <nn.h>

void nn::backPropagate() {
    std::vector<std::vector<float>> errorBPV(network.size());

    for (uzi j = 0; j < model[outputIndex]; j++) {
        float pValue = P[outputIndex][j]->value;

        float targetValue = 1;
        if (network[outputIndex][j] < 0) {
            targetValue = 0;
        }
        float deltaCost_deltaO = 2 * (pValue - targetValue);
        float deltaO_deltaZ = pValue * (1.0f - pValue);

        errorBPV[outputIndex].push_back(deltaCost_deltaO * deltaO_deltaZ);
    }

    for (int i = outputIndex - 1; i > 0; i--) {
        for (uzi j = 0; j < model[i]; j++) {
            float tempSum = 0.f;
            for (uzi k = 0; k < model[i + 1]; k++) {
                tempSum += P[i][j]->Link[k]->weight * errorBPV[i + 1][k];
            }

            float pValue = P[i][j]->value;
            errorBPV[i].push_back((tempSum / model[i + 1]) * (pValue * (1.0 - pValue)));//
        }
    }

    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {

            //TODO: P[i][j], hiddenLayerNum < outputLayerNum size error
            deltaShadow[i][j] = eta * errorBPV[i + 1][j] * P[i][j]->value + alpha * deltaShadow[i][j];
            shadow[i][j] += 0.875 * deltaShadow[i][j];

            for (uzi k = 0; k < model[i]; k++) {
                P[i][k]->Link[j]->deltaWeight = eta * P[i][k]->value * errorBPV[i + 1][j]
                                                + alpha * P[i][k]->Link[j]->deltaWeight;

                P[i][k]->Link[j]->weight -=P[i][k]->Link[j]->deltaWeight;
            }
        }
    }
}
