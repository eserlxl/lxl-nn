#ifndef LXL_NN_TESTRESULT_H
#define LXL_NN_TESTRESULT_H

typedef struct TestResult {
    uzi domainSize;
    matrixFloat1D correct;
    matrixFloat1D errorPercentage;
    matrixFloat1D correctPercentage;
    double elapsedTime = 0;

    explicit TestResult(uzi domainSize) {
        this->domainSize = domainSize;
    }

    void calcCorrect() {
        correctPercentage.clear();
        for (auto x:correct) {
            correctPercentage.push_back(100.f * (std::max(0.f, x) / domainSize));
        }
    }

    void calcError() {
        errorPercentage.clear();
        for (auto x:correct) {
            errorPercentage.push_back(100.f * (1.f - std::max(0.f, x) / domainSize));
        }
    }

    std::string print() {
        std::string text = "{ ";
        for (uzi i = 0; i < correct.size(); i++) {
            //std::string errorPercentageText = roundStr(errorPercentage[i], 2);
            std::string correctPercentageText = roundStr(correctPercentage[i], 2);
            text += "L" + std::to_string(correct.size() - i) + " => [ ✓: " + std::to_string((int) (correct[i])) + "/" +
                    std::to_string(domainSize) + " " + correctPercentageText + "% ]";
            if (i < correct.size() - 1)text += ", ";
        }
        text += " }";

        return text;
    }
} TestResult;

#endif //LXL_NN_TESTRESULT_H
