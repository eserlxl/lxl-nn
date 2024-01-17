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
#ifdef ADAPTIVE_TRAINING
    float k = 1;
    float etaStable = eta;
    float prevEta = eta;
    float minEta = 0.1f;
    float deltaEta = 1e-2f;//std::min(std::max(1e-2f*5e-3f/minRMSError,5.f*minRMSError),1e-1f);//2.5e-2;
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

        backPropagateInit();

        rmseVec[sourceSize - 1] = rmsErrorBP;

        float rmse = rms(rmseVec);
#ifdef ADAPTIVE_TRAINING
        prevEta = eta;
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

        if (rmse < minRMSError) {
            minRMSError = rmse;
        }

        std::cout << loop << " > η: " << eta << ", RMSE: " << rmse << "/" << minRMSError << ", Training => [ ✓: "
                  << trainingResult.correct << "/" << sourceSize << ", !: "
                  << trainingResult.errorPercentage << "% ]"
                  << ", Test => [ ✓: " << testResult.correct << "/" << testData->m_imageCount
                  << ", !: " << testResult.errorPercentage << "% ]"
                  << ", Time => [ Training: " << loopDuration
                  //<< ", ✓Training: " << checkTrainingResultDuration
                  << ", ✓Test: " << checkTestResultDuration << " ]"
                  << std::endl;
#endif
        if (rmse <= minRMSError) {
            minRMSError = rmse;
#ifdef ADAPTIVE_TRAINING
            //saveWeights();
            /*if(eta<etaStable){
                etaStable = eta;
            }*/

            if((prevEta<eta && k<0)||(prevEta>eta && k>0)){
                k*=-1;
            }

            eta = std::min(12.5f,std::max(0.95f*etaStable,1.f+k*deltaEta));
#endif
        }
#ifdef ADAPTIVE_TRAINING
        else {

            if((prevEta<eta && k>0)||(prevEta>eta && k<0)){
                k*=-1;
            }

            eta = std::min(12.5f,std::max(0.95f*etaStable,1.f-k*deltaEta));

            /*if(std::fabs(eta-etaStable)/etaStable<0.1){
                loadWeights();
            }
            else {
                smoothWeights(0.5);
            }*/
        }
         etaStable=eta;
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
