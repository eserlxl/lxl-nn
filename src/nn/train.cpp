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
            if (testData->output[p][t] >= 0) {
                if (maxIndex == t) {
                    correct++;
                    break;
                }
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
            if (target[p][t] >= 0) {
                if (maxIndex == t) {
                    correct++;
                    break;
                }
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
#ifdef ANALYSE_TRAINING
    double loopDuration, checkTrainingResultDuration, checkTestResultDuration, trainingDuration;
    double loopDurationSum = 0;
    double checkDataDurationSum = 0;
    float minRMSError = 1000;
#endif
    randWeights();

    clock->initTimer();

#ifdef ADAPTIVE_LEARNING
    float kEta = 1;
    float prevEta = 0;
    float minEta = 10.f;
    float maxEta = 25.f;
    float deltaEta;
#ifdef BP_USE_BIAS
    float kZeta = 1;
    float prevZeta = 0;
    float minZeta = 5.f;
    float maxZeta = 25.f;
    float deltaZeta;
#endif

#endif
    for (uzi loop = 0; loop < loopMax; loop++) {
#ifndef NO_RANDOMIZATION
        e2.seed(rd());
#endif

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

        rmseVec[sourceSize - 1] = rmsErrorBP;

        float RMSE = rms(rmseVec);

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

        if (RMSE < minRMSError) {
            minRMSError = RMSE;
        }

        std::cout << loop << " > η: " << eta
                  #ifdef BP_USE_BIAS
                  << ", ζ: " << zeta
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
        if (RMSE <= minRMSError) {
            minRMSError = RMSE;
#ifdef ADAPTIVE_LEARNING
            if ((prevEta <= eta && kEta < 0) || (prevEta >= eta && kEta > 0)) {
                kEta *= -1;
            }
#ifdef BP_USE_BIAS
            if ((prevZeta <= eta && kZeta < 0) || (prevZeta >= eta && kZeta > 0)) {
                kZeta *= -1;
            }
#endif
#endif
        }
#ifdef ADAPTIVE_LEARNING
        else {
            if ((prevEta <= eta && kEta > 0) || (prevEta >= eta && kEta < 0)) {
                kEta *= -1;
            }
#ifdef BP_USE_BIAS
            if ((prevZeta <= eta && kZeta > 0) || (prevZeta >= eta && kZeta < 0)) {
                kZeta *= -1;
            }
#endif
        }

        deltaEta = (1.f + RMSE * RMSE / minRMSError * (eta - prevEta));
        eta = std::min(maxEta, std::max(minEta, eta * (1.f + alpha * kEta * deltaEta)));
        prevEta = eta;

#ifdef BP_USE_BIAS
        deltaZeta=(1.f+RMSE*RMSE/minRMSError*(zeta-prevZeta));
        zeta = std::min(maxZeta, std::max(minZeta, zeta * (1.f +alpha*kZeta * deltaZeta)));
        prevZeta = zeta;
#endif
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
