#include <nn.h>

void nn::backPropagateInit() {
    errorBP[outputIndex].clear();
    float maxOutput = -1;
    uzi maxIndex = 0;
    for (uzi t = 0; t < model[outputIndex]; t++) {
        if (P[outputIndex][t]->value > maxOutput) {
            maxOutput = P[outputIndex][t]->value;
            maxIndex = t;
        }
    }

    bool skipChecking = false;
    for (uzi j = 0; j < model[outputIndex]; j++) {
        float pValue = P[outputIndex][j]->value;

        float targetValue = 1;
        if (network[outputIndex][j] < 0) {
            targetValue = 0;
        }

        if (maxIndex != j && pValue > 0.25) // Bellman's optimality, assume that 0.25 is enough for learning 0
        {
            skipChecking = true;
        }

        if (!skipChecking && maxIndex == j &&
            pValue >= 0.75) { // Bellman's optimality, assume that 0.75 is enough for learning 1
            correctChoice++;
        }
        float error = pValue - targetValue;
#ifdef BP_BELLMAN_OPT
        if(maxIndex == j)error -= gamma*std::max(0.f,0.75f-pValue); // Bellman's optimality simulation
        else error -= gamma*std::max(0.f,pValue-0.25f); // Bellman's optimality equation simulation
#endif

#ifdef BP_USE_PID
        error *= 2 * pValue * (1.0f - pValue);
        errorSumBP[j] += error;
        float pidValue = pid_P * error + pid_I * errorSumBP[j] + pid_D * (error - prevError[j]);
        errorBP[outputIndex].push_back(pidValue);
        prevError[j] = error;
#else
        errorBP[outputIndex].push_back(2 * error * pValue * (1.0f - pValue));
#endif

    }
    rmsErrorBP = rms(errorBP[outputIndex]);
}

void nn::backPropagate() {

    backPropagateInit();

    for (int i = outputIndex - 1; i > 0; i--) {
        errorBP[i].clear();
        for (uzi j = 0; j < model[i]; j++) {
            float tempSum = 0.f;
            for (uzi k = 0; k < model[i + 1]; k++) {
                tempSum += P[i][j]->Link[k]->weight * errorBP[i + 1][k];
            }

            float pValue = P[i][j]->value;
            errorBP[i].push_back((tempSum / model[i + 1]) * (pValue * (1.0 - pValue)));//
        }
    }

    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {

            //TODO: P[i][j], hiddenLayerNum < outputLayerNum gives size error
            deltaShadow[i][j] = eta * errorBP[i + 1][j] * P[i][j]->value + alpha * deltaShadow[i][j];
            shadow[i][j] += zeta * deltaShadow[i][j];

            for (uzi k = 0; k < model[i]; k++) {
                P[i][k]->Link[j]->deltaWeight = eta * P[i][k]->value * errorBP[i + 1][j]
                                                + alpha * P[i][k]->Link[j]->deltaWeight;

                P[i][k]->Link[j]->weight -= P[i][k]->Link[j]->deltaWeight;
            }
        }
    }
}
