#include <nn.h>

void nn::randWeights() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            shadow[i][j] = e2()/(float)std::mt19937::max();

            for (uzi k = 0; k < P[i].size(); k++) {
                P[i][k]->Link[j]->weight = e2()/(float)std::mt19937::max();;
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
