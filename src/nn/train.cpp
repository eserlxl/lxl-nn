#include <nn.h>

void nn::checkTestData(MNISTData *testData) {
    uzi correct = 0;
    for (uzi p = 0; p < testData->m_imageCount; p++) {

        for (uzi i = 0; i < model[0]; i++) {
            network[0][i] = testData->input[p][i];
            P[0][i]->value = testData->input[p][i];
        }
        feedForward();

        float maxOutput = -1;
        uzi maxIndex = 0;
        for (uzi t = 0; t < model[outputIndex]; t++) {
            if (P[outputIndex][t]->value > maxOutput) {
                maxOutput = P[outputIndex][t]->value;
                maxIndex = t;
            }
        }

        for (uzi t = 0; t < model[outputIndex]; t++) {
            if (testData->output[p][t] >= 0) {
                if (maxIndex == t) {
                    correct++;
                    break;
                }
            }
        }
    }

    std::cout << "Test Data:" << correct << " Error: " << 100.f * (1.f - (float) correct / testData->m_imageCount)
              << "%" << std::endl;
}

void nn::checkTrainingData() {
    std::vector<float> errorVec(sourceSize);

    uzi correct = 0;
    for (uzi p = 0; p < sourceSize; p++) {//sourceSize
        setIO(source[p], target[p]);
        feedForward();

        float maxOutput = -1;
        uzi maxIndex = 0;
        for (uzi t = 0; t < model[outputIndex]; t++) {
            if (P[outputIndex][t]->value > maxOutput) {
                maxOutput = P[outputIndex][t]->value;
                maxIndex = t;
            }
        }

        for (uzi t = 0; t < model[outputIndex]; t++) {
            if (target[p][t] >= 0) {
                if (maxIndex == t) {
                    correct++;
                    break;
                }
            }
        }
    }
    std::cout << "Training Data:" << correct << " Error: " << 100.f * (1.f - (float) correct / sourceSize) << "%"
              << std::endl;
}

void nn::train(uzi loopMax) {

    randWeights();

    for (uzi outerLoop = 0; outerLoop < loopMax; outerLoop++) {
#ifndef NO_RANDOMIZATION
        e2.seed(rd());
#endif
        for (uzi p = 0; p < sourceSize; p++) { //sourceSize

            uzi u = e2() % sourceSize;

            setIO(source[u], target[u]);

            feedForward();

            backPropagate();
        }
    }
}
