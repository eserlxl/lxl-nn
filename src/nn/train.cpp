#include <nn.h>

TestResult nn::checkTestData(MNISTData *testData) {
    chronometer->initTimer();

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
            if (maxIndex == t && testData->output[p][t] > networkMidValue) {
                correct++;
                break;
            }
        }
    }

    TestResult test{};
    test.correct = correct;
    test.errorPercentage = 100.f * (1.f - (float) correct / testData->m_imageCount);
    test.elapsedTime = chronometer->getElapsedTime();
    return test;
}

TestResult nn::checkTrainingData() {
    chronometer->initTimer();
    uzi correct = 0;
    for (uzi p = 0; p < sourceSize; p++) {
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
            if (maxIndex == t && target[p][t] > networkMidValue) {
                correct++;
                break;
            }
        }
    }

    TestResult ret{};
    ret.correct = correct;
    ret.errorPercentage = 100.f * (1.f - (float) correct / sourceSize);
    ret.elapsedTime = chronometer->getElapsedTime();
    return ret;
}

void nn::train(uzi loopMax, MNISTData *testData) {
    float minRMSError = 1000;
#ifdef ANALYSE_TRAINING
    double loopDuration, checkTrainingResultDuration, checkTestResultDuration, trainingDuration;
    double loopDurationSum = 0;
    double checkDataDurationSum = 0;
#endif
    randWeights();

    clock->initTimer();

#ifdef ADAPTIVE_LEARNING
    uzi learningSize = learningMatrix.size();
    float perturbationRatio = 1e-2f;
    std::vector<float> rmseHistory;
    std::vector<std::vector<float>> coeffHistory;
    std::vector<std::vector<float>> uHistory;
    Eigen::Matrix<float, 5, 5> aMatrix;
    Eigen::Matrix<float, 5, 1> rmseMatrix;

    std::vector<float> learningMatrixInitial;
    for (float &x : learningMatrix) {
        learningMatrixInitial.push_back(x);
    }
    std::vector<float> learningMatrixPrev;
    for (float &x : learningMatrix) {
        learningMatrixPrev.push_back(x);
    }
    float rateLimit = learningMatrix[learningSize - 1];
#endif
    for (uzi loop = 0; loop < loopMax; loop++) {

#ifdef ANALYSE_TRAINING
        chronometer->initTimer();
#endif
        std::vector<float> rmseVec;
        correctChoice = 0;
        for (uzi p = 0; p < sourceSize; p++) {

            uzi u = e2() % sourceSize;

            setIO(source[u], target[u]);

            feedForward();

            backPropagate();

            rmseVec.push_back(rmsErrorBP);
        }

        correctChoice -= 1;
        feedForward();

        backPropagateOutputLayer();

        rmseVec[rmseVec.size() - 1] = rmsErrorBP;

        float RMSE = rms(rmseVec);

        if (RMSE < minRMSError) {
            minRMSError = RMSE;
#ifdef ADAPTIVE_LEARNING
            saveWeights();
#endif
        }
#ifdef ADAPTIVE_LEARNING
        else if(RMSE > 1.0125*minRMSError)
        {
            smoothWeights(0.9875);

        }
#endif
#ifdef ANALYSE_TRAINING
        loopDuration = chronometer->getElapsedTime();
        loopDurationSum += loopDuration;

        TestResult trainingResult{};
        trainingResult.correct = correctChoice;
        trainingResult.errorPercentage = 100.f * (1.f - (float) correctChoice / sourceSize);

        chronometer->initTimer();
        TestResult testResult = checkTestData(testData);
        checkTestResultDuration = chronometer->getElapsedTime();

        checkDataDurationSum += checkTrainingResultDuration + checkTestResultDuration;

        uzi h = 0;
        std::cout << loop
                  << " > η: " << learningMatrix[h++]
                  #ifdef BP_USE_BIAS
                  << ", ζ: " << learningMatrix[h++]
                  #endif
                  #ifdef BP_BELLMAN_OPT
                  << ", γ: " << learningMatrix[h++]
                  #endif
                  #ifdef ADAPTIVE_LEARNING
                  << ", α: " << learningMatrix[h++]
                  << ", r: " << learningMatrix[h++]
                  #endif
                  << ", RMSE: " << RMSE << "/" << minRMSError
                  << ", Training => [ ✓: "
                  << trainingResult.correct << "/" << sourceSize << ", !: "
                  << trainingResult.errorPercentage << "% ]"
                  << ", Test => [ ✓: " << testResult.correct << "/" << testData->m_imageCount
                  << ", !: " << testResult.errorPercentage << "% ]"
                  << ", Time => [ Training: " << loopDuration
                  //<< ", ✓Training: " << checkTrainingResultDuration
                  << ", ✓Test: " << checkTestResultDuration << " ]"
                  << std::endl;
#endif

#ifdef ADAPTIVE_LEARNING
        rmseHistory.push_back(RMSE);
        std::vector<float> tempVec;

        for (uzi i = 0; i < learningSize; i++) {
            tempVec.push_back(learningMatrix[i] / learningMatrixInitial[i]);
        }

        coeffHistory.push_back(tempVec);

        if (loop >= learningSize - 1) {
            for (uzi i = 0; i < learningSize; i++) {
                for (uzi j = 0; j < learningSize; j++) {
                    aMatrix(i, j) = coeffHistory[loop - i][j];
                }
                rmseMatrix(i, 0) = rmseHistory[loop - i] / minRMSError;
            }

            Eigen::FullPivLU<Eigen::Matrix<float, 5, 5>> lu(aMatrix);
            Eigen::Matrix<float, 5, 1> u = lu.inverse() * rmseMatrix;

            // Rate limit
            for (uzi i = 0; i < learningSize; i++) {
                learningMatrix[i] *= 1 - u(i, 0);

                if (learningMatrix[i] > learningMatrixPrev[i] * (1 + rateLimit)) {
                    learningMatrix[i] = learningMatrixPrev[i] * (1 + rateLimit);
                } else if (learningMatrix[i] < learningMatrixPrev[i] * (1 - rateLimit)) {
                    learningMatrix[i] = learningMatrixPrev[i] * (1 - rateLimit);
                }
            }
        } else {
            for (float &x : learningMatrix) {
                x *= 1 + perturbationRatio * rateLimit * randomNumberExtended();
            }
        }

        // Max-Min Bounds
        for (uzi i = 0; i < learningSize; i++) {
            learningMatrix[i] = std::min(
                    learningMatrixUpperLimits[i] * (1 - perturbationRatio * rateLimit * randomNumber()),
                    std::max(learningMatrixLowerLimits[i] * (1 + perturbationRatio * rateLimit * randomNumber()),
                             learningMatrix[i]));
        }

        for (uzi i = 0; i < learningSize; i++) {
            learningMatrixPrev[i] = learningMatrix[i];
        }
#endif
    }
#ifdef ANALYSE_TRAINING
    trainingDuration = clock->getElapsedTime();
    std::cout << std::endl << "Total training time: " << loopDurationSum << " s" << std::endl;
    std::cout << "Time loss due to checking data: " << checkDataDurationSum << " s" << std::endl;
    std::cout << "Time loss due to measuring time: " << trainingDuration - loopDurationSum - checkDataDurationSum
              << " s" << std::endl;
    std::cout << "Final training time: " << trainingDuration << " s" << std::endl;
#endif
}
