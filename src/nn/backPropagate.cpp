#include <nn.h>

void nn::backPropagateOutputLayer() {
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
        if (maxIndex == j)error -= gamma * std::max(0.f, 0.75f - pValue); // Bellman's optimality simulation
        else error -= gamma * std::max(0.f, pValue - 0.25f); // Bellman's optimality equation simulation
#endif

#ifdef BP_USE_PID
        error *= 2 * pValue * (1.0f - pValue);
        errorSumBP[j] += error;
        if(std::fabs(errorSumBP[j])>10)
        {
            errorSumBP[j]*=0.9;
        }
        float pidValue = (pid_P+1) * error + pid_I * errorSumBP[j] + pid_D * (error - prevError[j]);
        errorBP[outputIndex].push_back(pidValue);
        prevError[j] = error;
#else
        float dLogic = 1;
#ifdef LOGIC_SIGMOID
        dLogic = pValue * (1.0f - pValue);
#elif defined(LOGIC_TANH)
        dLogic = 1.f - pValue * pValue;
#elif defined(LOGIC_SWISH)
        dLogic = pValue + logit(pValue) * (pValue * (1.f - pValue)); // (x*sigmoid(x))' = x'*sigmoid(x)+x*sigmoid(x)', inv(sigmoid) = logit
#elif defined(LOGIC_RELU)
        dLogic = dReLU(pValue);
#endif
        errorBP[outputIndex].push_back(2 * error * dLogic);
#endif

    }
    rmsErrorBP = rms(errorBP[outputIndex]);
}

void nn::backPropagate() {

    backPropagateOutputLayer();

    for (int i = outputIndex - 1; i >= 0; i--) {
        errorBP[i].clear();
        for (uzi j = 0; j < model[i]; j++) {
            float tempSum = 0.f;
            for (uzi k = 0; k < model[i + 1]; k++) {
                tempSum += P[i][j]->Link[k]->weight * errorBP[i + 1][k];
            }

            float pValue = P[i][j]->value;
            float bPValue = tempSum / model[i + 1];
#ifdef LOGIC_SIGMOID
            bPValue *= pValue * (1.f - pValue);
#elif defined(LOGIC_TANH)
            bPValue *= (1.f - pValue * pValue);
#elif defined(LOGIC_SWISH)
           bPValue *= pValue + logit(pValue) * (pValue * (1.f - pValue)); // (x*sigmoid(x))' = x'*sigmoid(x)+x*sigmoid(x)', inv(sigmoid) = logit
#elif defined(LOGIC_RELU)
            bPValue *= dReLU(pValue);
#endif
            errorBP[i].push_back(bPValue);

            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->Link[k]->weight -= eta * pValue * errorBP[i + 1][k];
            }
        }
#ifdef BP_USE_BIAS
        for (uzi j = 0; j < model[i + 1]; j++) {
            bias[i][j] -= zeta * errorBP[i + 1][j];
        }
#endif
    }
}
