#include <nn.h>

void nn::feedForward() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
            float tempSum = shadow[i][j];

            for (uzi k = 0; k < model[i]; k++) {
                tempSum += P[i][k]->value * P[i][k]->Link[j]->weight;
            }
            P[i + 1][j]->value = logic(tempSum / (float) (model[i] + 1), (float) 1);
        }
    }
}
