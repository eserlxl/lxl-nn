#include <nn.h>

testData nn::checkTestData(MNISTData *testData) {
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

    struct testData test{};
    test.correct = correct;
    test.errorPercentage = 100.f * (1.f - (float) correct / testData->m_imageCount);
    test.elapsedTime = chronometer->getElapsedTime();

#ifdef ANALYSE_TRAINING
    std::cout<<std::endl<<"Test Data"<<std::endl<<"\t\tCorrect: "<<test.correct<<"/"<<testData->m_imageCount<<"\t\tError: "<<test.errorPercentage
             <<"%\t\tCheck Time: "<<test.elapsedTime
             <<std::endl;
#endif
    return test;
}

testData nn::checkTrainingData() {
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

    testData ret;
    ret.correct = correct;
    ret.errorPercentage = 100.f * (1.f - (float) correct / sourceSize);
    ret.elapsedTime = chronometer->getElapsedTime();
    return ret;
}

void nn::train(uzi loopMax) {
#ifdef ANALYSE_TRAINING
    double loopDuration, checkDataDuration, trainingDuration;
    double loopDurationSum = 0;
    double checkDataDurationSum = 0;
#endif
    randWeights();

    clock->initTimer();
    for (uzi loop = 0; loop < loopMax; loop++) {
#ifndef NO_RANDOMIZATION
        e2.seed(rd());
#endif
#ifdef USE_BP_BETA
        //errorSumBP.clear();
#endif
#ifdef ANALYSE_TRAINING
        chronometer->initTimer();
#endif
        for (uzi p = 0; p < sourceSize; p++) {

            uzi u = e2() % sourceSize;

            setIO(source[u], target[u]);

            feedForward();

            backPropagate();
        }
#ifdef ANALYSE_TRAINING
        loopDuration = chronometer->getElapsedTime();
        loopDurationSum+=loopDuration;
        chronometer->initTimer();
        testData trainingData = checkTrainingData();
        checkDataDuration = chronometer->getElapsedTime();
        checkDataDurationSum+=checkDataDuration;
        std::cout<<"Loop: "<<loop<<"\t\tCorrect: "<<trainingData.correct<<"/"<<sourceSize<<"\t\tError: "<<trainingData.errorPercentage
        <<"%\t\tTraining Time: "<<loopDuration
        <<"\t\tCheck Time: "<<checkDataDuration
        <<std::endl;
#endif
    }
#ifdef ANALYSE_TRAINING
    trainingDuration = clock->getElapsedTime();
    std::cout<<std::endl<<"Total training time: "<<loopDurationSum<<" s"<<std::endl;
    std::cout<<"Time loss due to checking data: "<<checkDataDurationSum<<" s"<<std::endl;
    std::cout<<"Time loss due to measuring time: "<<trainingDuration-loopDurationSum-checkDataDurationSum<<" s"<<std::endl;
    std::cout<<"Final training time: "<<trainingDuration<<" s"<<std::endl;
#endif
}
