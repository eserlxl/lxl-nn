#include <nn.h>

void nn::feedForward() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
#ifdef BP_USE_BIAS
            float tempSum = bias[i][j];
#else
            float tempSum = 0;
#endif
            for (uzi k = 0; k < model[i]; k++) {
                tempSum += P[i][k]->value * P[i][k]->Link[j]->weight;
            }
            float x = tempSum / (float) (model[i] + 1);
#ifdef LOGIC_SIGMOID
            P[i + 1][j]->value = sigmoid(x);
#elif defined(LOGIC_TANH)
            P[i + 1][j]->value = std::tanh(x);
#elif defined(LOGIC_SWISH)
            P[i + 1][j]->value = std::max(0.f,std::min(x*sigmoid(x),1.f));
#elif defined(LOGIC_RELU)
            P[i + 1][j]->value = reLU(x);
#endif
        }
    }
}
