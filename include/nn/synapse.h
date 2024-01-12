#ifndef lxl_nn_NN_SYNAPSE_H_
#define lxl_nn_NN_SYNAPSE_H_

typedef std::size_t uzi;
class synapse {
public:
    uzi inputID;
    uzi outputID;
    float weight;
    float deltaWeight;

    synapse(uzi inputID, uzi outputID) {

        this->inputID = inputID;
        this->outputID = outputID;
        weight = 0;
        deltaWeight = 0;
    }

    ~synapse() {
    }
};

#endif // lxl_nn_NN_SYNAPSE_H_
