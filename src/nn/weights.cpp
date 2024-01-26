#include <nn.h>

void nn::randWeights() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
#ifdef BP_USE_BIAS
            bias[i][j] = 0;
#endif
            for (uzi k = 0; k < model[i]; k++) {
                P[i][k]->Link[j]->weight = randomNumberExtended();
            }
        }
    }
}

#ifdef ADAPTIVE_LEARNING

void nn::saveNetwork() {
    pBackup.clear();
    for (auto &p : P) {
        std::vector<float> tempVec;
        for (uzi j = 0; j < p.size(); j++) {
            tempVec.push_back(p[j]->value);
        }
        pBackup.push_back(tempVec);
    }
}

void nn::loadNetwork() {
    if (pBackup.empty()) {
        return;
    }
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            P[i][j]->value = pBackup[i][j];
        }
    }
}

void nn::saveWeights() {
    lastWeightBackup.clear();
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
            lastWeightBackup.push_back(bias[i][j]);
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                lastWeightBackup.push_back(P[i][j]->Link[k]->weight);
                lastWeightBackup.push_back(P[i][j]->Link[k]->deltaWeight);
            }
        }
    }
}

void nn::saveWeights(uzi maxBackupCount) {

    std::vector<float> tempVec;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            tempVec.push_back(bias[i][j]);
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                tempVec.push_back(P[i][j]->Link[k]->weight);
                tempVec.push_back(P[i][j]->Link[k]->deltaWeight);
            }
        }
    }
    if (weightBackup.size() >= maxBackupCount) {
        weightBackup.erase(weightBackup.begin());
    }
    weightBackup.push_back(tempVec);
}

void nn::loadWeights() {
    if (lastWeightBackup.empty()) {
        return;
    }
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            bias[i][j] = lastWeightBackup[h++];
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->Link[k]->weight = lastWeightBackup[h++];
                P[i][j]->Link[k]->deltaWeight = lastWeightBackup[h++];
            }
        }
    }
}

void nn::loadWeights(uzi backupIndex) {
    if (weightBackup.empty() || weightBackup[backupIndex].empty()) {
        return;
    }
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            bias[i][j] = weightBackup[backupIndex][h++];
        }
    }
#endif
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->Link[k]->weight = weightBackup[backupIndex][h++];
                P[i][j]->Link[k]->deltaWeight = weightBackup[backupIndex][h++];
            }
        }
    }
}

void nn::predictWeights() {
    if (weightBackup.empty() || weightBackup.size() < 4) {
        return;
    }

    std::vector<float> tempVec;
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {

            tempVec.clear();
            for (uzi w = 0; w < weightBackup.size() - 1; w++) {
                tempVec.push_back(bias[i][j] - weightBackup[w][h]);
            }
            bias[i][j] -= (tempVec[0] - 2 * tempVec[1] + tempVec[2]);
            h++;
        }
    }
#endif
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                tempVec.clear();
                for (uzi w = 0; w < weightBackup.size() - 1; w++) {
                    tempVec.push_back(P[i][j]->Link[k]->weight - weightBackup[w][h]);
                }
                P[i][j]->Link[k]->weight -= (tempVec[0] - 2 * tempVec[1] + tempVec[2]);
                h++;

                tempVec.clear();
                for (uzi w = 0; w < weightBackup.size() - 1; w++) {
                    tempVec.push_back(P[i][j]->Link[k]->deltaWeight - weightBackup[w][h]);
                }
                P[i][j]->Link[k]->deltaWeight += (tempVec[0] - 2 * tempVec[1] + tempVec[2]);
                h++;
            }
        }
    }

};

void nn::smoothLastWeights() {
    if (weightBackup.empty() || weightBackup.size() < 3) {
        return;
    }
    uzi backupIndex1 = weightBackup.size() - 3;
    uzi backupIndex2 = weightBackup.size() - 2;
    uzi backupIndex3 = weightBackup.size() - 1;
    float backupRatio1 = 0.1;
    float backupRatio2 = 0.2;
    float backupRatio3 = 0.4;
    float newWeightRatio = 0.3;
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            bias[i][j] = backupRatio1 * weightBackup[backupIndex1][h] + backupRatio2 * weightBackup[backupIndex2][h] +
                         backupRatio3 * weightBackup[backupIndex3][h] + newWeightRatio * bias[i][j];
            h++;
        }
    }
#endif
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                P[i][j]->Link[k]->weight =
                        backupRatio1 * weightBackup[backupIndex1][h] + backupRatio2 * weightBackup[backupIndex2][h] +
                        backupRatio3 * weightBackup[backupIndex3][h] + newWeightRatio * P[i][j]->Link[k]->weight;
                h++;
                P[i][j]->Link[k]->deltaWeight =
                        backupRatio1 * weightBackup[backupIndex1][h] + backupRatio2 * weightBackup[backupIndex2][h] +
                        backupRatio3 * weightBackup[backupIndex3][h] + newWeightRatio * P[i][j]->Link[k]->deltaWeight;
                h++;
            }
        }
    }
}

void nn::smoothWeights(float backupRatio) {
    if (weightBackup.empty()) {
        return;
    }
    uzi backupIndex = 0;
    float newWeightRatio = 1.f - backupRatio;
    uzi h = 0;
#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            bias[i][j] = backupRatio * weightBackup[backupIndex][h++] + newWeightRatio * bias[i][j];
        }
    }
#endif
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                P[i][j]->Link[k]->weight =
                        backupRatio * weightBackup[backupIndex][h++] + newWeightRatio * P[i][j]->Link[k]->weight;
                P[i][j]->Link[k]->deltaWeight =
                        backupRatio * weightBackup[backupIndex][h++] + newWeightRatio * P[i][j]->Link[k]->deltaWeight;
            }
        }
    }
}

#endif