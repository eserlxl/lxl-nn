#include <nn.h>

void nn::randWeights() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            shadow[i][j] = 2.f * (e2() / (float) std::mt19937::max() - 0.5f);

            for (uzi k = 0; k < P[i].size(); k++) {
                P[i][k]->Link[j]->weight = 2.f * (e2() / (float) std::mt19937::max() - 0.5f);
            }
        }
    }
}

void nn::saveWeights() {
    weightBackup.clear();

    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            weightBackup.push_back(shadow[i][j]);
            weightBackup.push_back(deltaShadow[i][j]);
        }
    }

    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                weightBackup.push_back(P[i][j]->Link[k]->weight);
                weightBackup.push_back(P[i][j]->Link[k]->deltaWeight);
            }
        }
    }
}

void nn::loadWeights() {
    if (weightBackup.empty()) {
        return;
    }
    uzi h = 0;
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            shadow[i][j] = weightBackup[h++];
            deltaShadow[i][j] = weightBackup[h++];
        }
    }
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                P[i][j]->Link[k]->weight = weightBackup[h++];
                P[i][j]->Link[k]->deltaWeight = weightBackup[h++];
            }
        }
    }
}

void nn::smoothWeights(float backupRatio) {
    if (weightBackup.empty()) {
        return;
    }
    float newWeightRatio = 1 - backupRatio;
    uzi h = 0;
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            shadow[i][j] = backupRatio * weightBackup[h++] + newWeightRatio * shadow[i][j];
            deltaShadow[i][j] = backupRatio * weightBackup[h++] + newWeightRatio * deltaShadow[i][j];
        }
    }
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            for (uzi k = 0; k < P[i][j]->Link.size(); k++) {
                P[i][j]->Link[k]->weight = backupRatio * weightBackup[h++] + newWeightRatio * P[i][j]->Link[k]->weight;
                P[i][j]->Link[k]->deltaWeight =
                        backupRatio * weightBackup[h++] + newWeightRatio * P[i][j]->Link[k]->deltaWeight;
            }
        }
    }
}
