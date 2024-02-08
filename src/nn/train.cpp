#include "nn/neuralNetwork.h"

float NeuralNetwork::checkBinaryOutputData() {

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

    return 1.f - (float) correct / sourceSize;
}

void NeuralNetwork::train(uzi loopMax) {

    float minRMSError = 1000;
#ifdef ANALYSE_TRAINING
    double loopDuration, trainingDuration;
    double loopDurationSum = 0;
    double checkDataDurationSum = 0;
#endif
    if (weightBackup.empty()) {
        randWeights();
    }

    clock->initTimer();

#ifdef ADAPTIVE_LEARNING
    uzi learningSize = learningMatrix.size();
    float perturbationRatio = 1e-2f;
    matrixFloat1D rmseHistory;
    matrixFloat2D coeffHistory;
    matrixFloat2D uHistory;
    Eigen::Matrix<float, 7, 7> aMatrix;
    Eigen::Matrix<float, 7, 1> rmseMatrix;
    float lastSavedRMSE=100;

    matrixFloat1D learningMatrixInitial;
    for (float &x : learningMatrix) {
        learningMatrixInitial.push_back(x);
    }
    matrixFloat1D learningMatrixPrev, learningMatrixBest;
    for (float &x : learningMatrix) {
        learningMatrixPrev.push_back(x);
        learningMatrixBest.push_back(x);
    }
    float rateLimit = learningMatrix[learningSize - 1];
#endif
    for (uzi loop = 0; loop < loopMax; loop++) {

#ifdef ANALYSE_TRAINING
        chronometer->initTimer();
#endif
        matrixFloat1D rmseVec;
        correctChoice.clear();
        correctChoice.resize(reqNormRMSE.size());
        for (uzi p = 0; p < sourceSize; p++) {

            uzi u = e2() % sourceSize;

            setIO(source[u], target[u]);

            feedForward();

            backPropagate();

            rmseVec.push_back(rmsErrorBP);
        }

        for (float &x : correctChoice)x--;

        feedForward();

        backPropagateOutputLayer();

        rmseVec[rmseVec.size() - 1] = rmsErrorBP;

        float RMSE = rms(rmseVec);

        if (RMSE < minRMSError) {
            minRMSError = RMSE;
#ifdef ADAPTIVE_LEARNING
            saveWeights();

            for (uzi i = 0; i < learningSize; i++) {
                learningMatrixBest[i] = learningMatrix[i];
            }

            if(!libFile.empty() && (RMSE < 0.9*lastSavedRMSE || RMSE / outputMaxValue<1e-2)){
                save(libFile);
                lastSavedRMSE = RMSE;
            }
#endif
        }
#ifdef ANALYSE_TRAINING
        loopDuration = chronometer->getElapsedTime();
        loopDurationSum += loopDuration;

        TestResult trainingResult(sourceSize);
        trainingResult.correct = correctChoice;
        trainingResult.calcCorrect();

#ifdef BINARY_OUTPUT_DATA_CHECK
        chronometer->initTimer();
        float binaryDataCheckErrorPercentage = checkBinaryOutputData()*100.f;
        checkDataDurationSum += chronometer->getElapsedTime();
#endif
        uzi h = 0;
        std::cout << loop
                  << " > η: " << learningMatrix[h++]
                  #ifdef BP_USE_BIAS
                  << ", ζ: " << learningMatrix[h++]
                  #endif
                  #ifdef BP_BELLMAN_OPT
                  << ", γ0: " << learningMatrix[h++]
                  << ", γ1: " << learningMatrix[h++]
                  << ", γ2: " << learningMatrix[h++]
                  #endif
                  #ifdef ADAPTIVE_LEARNING
                  << ", α: " << learningMatrix[h++]
                  << ", r: " << learningMatrix[h++]
                  #endif
                  << ", E(%): " << RMSE / outputMaxValue * 100.f << " / "
                  << minRMSError / outputMaxValue * 100.f
                  #ifdef BINARY_OUTPUT_DATA_CHECK
                  << ", E-01(%) => "<<binaryDataCheckErrorPercentage
                  #endif
                  << ", Training => "
                  << trainingResult.print()
                  << ", Time => [ Training: " << loopDuration
                  << " ]"
                  << std::endl;
#else
        std::cout<<"Training loop: "<<loop<<"/"<<loopMax<<"\r";
#endif

#ifdef ADAPTIVE_LEARNING
        rmseHistory.push_back(RMSE);
        matrixFloat1D tempVec;

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

            Eigen::FullPivLU<Eigen::Matrix<float, 7, 7>> lu(aMatrix);
            Eigen::Matrix<float, 7, 1> u = lu.inverse() * rmseMatrix;

            // Rate limit
            for (uzi i = 0; i < learningSize; i++) {
                learningMatrix[i] *= 1 - u(i, 0) + 1e-3 * rateLimit * randomNumber();

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
    loadWeights();
    for (uzi i = 0; i < learningSize; i++) {
        learningMatrix[i] = learningMatrixBest[i];
    }

#ifdef ANALYSE_TRAINING
#ifdef BINARY_OUTPUT_DATA
    chronometer->initTimer();
    binaryDataErrorPercentage = checkBinaryOutputData() * 100.f;
    checkDataDurationSum += chronometer->getElapsedTime();
#endif
    chronometer->initTimer();
    NRMSEPercentage = calcNormRMSEPercentage();
    checkDataDurationSum += chronometer->getElapsedTime();

    trainingDuration = clock->getElapsedTime();
    std::cout << std::endl << "Total training time: " << loopDurationSum << " s" << std::endl;
    std::cout << "Time loss due to checking data: " << checkDataDurationSum << " s" << std::endl;
    std::cout << "Time loss due to measuring time: " << trainingDuration - loopDurationSum - checkDataDurationSum
              << " s" << std::endl;
    std::cout << "Final training time: " << trainingDuration << " s" << std::endl;
#else
    std::cout<<"\n"<<std::endl;
#endif
}
