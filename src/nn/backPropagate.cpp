#include "nn/neuralNetwork.h"

void NeuralNetwork::backPropagateOutputLayer() {
    errorBP[outputIndex].clear();
#ifdef LOGIC_NETWORK
    std::vector<float> errorVec;
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

        if (maxIndex != j &&
            pValue > network1stQuarterValue) // Bellman's optimality, assume that 0.25 is enough for learning 0
        {
            skipChecking = true;
        }

        if (!skipChecking && maxIndex == j &&
            pValue >= network3rdQuarterValue) { // Bellman's optimality, assume that 0.75 is enough for learning 1
            correctChoice++;
        }
        float error = pValue - network[outputIndex][j];
        errorVec.push_back(error);
#ifdef BP_BELLMAN_OPT
        if (maxIndex == j)
            error -= learningMatrix[bellmanLearningRateIndex] *
                     std::max(0.f, network3rdQuarterValue - pValue); // Bellman's optimality simulation
        else
            error -= learningMatrix[bellmanLearningRateIndex] *
                     std::max(0.f, pValue - network1stQuarterValue); // Bellman's optimality equation simulation
#endif
#else
    for (uzi j = 0; j < model[outputIndex]; j++) {
        float pValue = P[outputIndex][j]->value;
        float error = pValue - network[outputIndex][j];
#ifdef BP_BELLMAN_OPT
        if (network[outputIndex][j] > networkMidValue) {
            error -= learningMatrix[bellmanLearningRateIndex] *
                     std::max(0.f, network3rdQuarterValue - pValue);
        } else {
            error -= learningMatrix[bellmanLearningRateIndex] *
                     std::max(0.f, pValue - network1stQuarterValue);
        }
#endif

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

    std::vector<float> tempVec;
    for (uzi j = 0; j < model[outputIndex]; j++) {
        tempVec.push_back(convertTargetDiffToOutputDiff(P[outputIndex][j]->value - network[outputIndex][j]));
    }

    rmsErrorBP = rms(tempVec);

#ifndef LOGIC_NETWORK
    for(uzi i=0;i<reqNormRMSE.size();i++)
    {
        correctChoice[i] += rmsErrorBP < reqNormRMSE[i] * outputMaxValue;
    }
#endif
}

void NeuralNetwork::backPropagate() {

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
                P[i][j]->Link[k]->weight -= learningMatrix[weightLearningRateIndex] * pValue * errorBP[i + 1][k];
            }
        }
#ifdef BP_USE_BIAS
        for (uzi j = 0; j < model[i + 1]; j++) {
            bias[i][j] -= learningMatrix[biasLearningRateIndex] * errorBP[i + 1][j];
        }
#endif
    }
}
