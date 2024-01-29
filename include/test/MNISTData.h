#ifndef lxl_nn_TEST_MNISTDATA_H
#define lxl_nn_TEST_MNISTDATA_H

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>

// Derived from: https://github.com/yaroslavbohonos/Handwritten-Digit-Recognition

/* Used to change the byte order. E.g., if the first two uint32_t values are 0x01080000 or 0x03080000.
The Inline specifier improves program performance by reducing the overhead of function calls and increasing execution speed. */
inline uint32_t EndianSwap(uint32_t a) {
    uint32_t b = 8, c = 24, d = 0x00ff0000, e = 0x0000ff00;
    return (a < c) | ((a << b) & d) | ((a >> b) & e) | (a >> c);
}

class MNISTData {
public:
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> output;

    size_t m_imageCount;

    MNISTData() {
        m_labelData = nullptr;
        m_imageData = nullptr;
        m_imageCount = 0;
        m_labels = nullptr;
        m_pixels = nullptr;
    }

    bool load(bool training) {
        // Set the expected image count
        m_imageCount = training ? 60000 : 10000;

        // Read labels
        const char *labelsFileName = training ? "../data/train-labels.idx1-ubyte" : "../data/t10k-labels.idx1-ubyte";
        FILE *file = fopen(labelsFileName, "rb");
        if (!file) {
            printf("Could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8_t[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);

        // Read images
        const char *imagesFileName = training ? "../data/train-images.idx3-ubyte" : "../data/t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file) {
            printf("Could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8_t[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);

        /* Perform endian swapping on the label file's first two uint32 values if required,
        while the rest of the file contains uint8 values. */
        auto *data = (uint32_t *) m_labelData;
        if (data[0] == 0x01080000) {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        // Verifying if the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount) {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        m_labels = (uint8_t *) &(data[2]);

        /* If necessary, perform endian swapping on the image file's first 
        four uint32 values. The remaining data consists of uint8 values. */
        data = (uint32_t *) m_imageData;
        if (data[0] == 0x03080000) {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        // Verifying if the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28) {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        m_pixels = (uint8_t *) &(data[4]);

        // Convert the pixels from uint8 to float
        for (size_t j = 0; j < m_imageCount; j++) {
            std::vector<float> tempVec;
            for (size_t i = 0; i < 10; i++) {
                if (i == m_labels[j]) {
                    tempVec.push_back(1);
                } else {
                    tempVec.push_back(0);
                }
            }
            output.push_back(tempVec);
            tempVec.clear();
            for (size_t i = 0; i < 28 * 28; i++) {
                tempVec.push_back(float(m_pixels[i + j * 28 * 28]) / 255.0f);
            }
            input.push_back(tempVec);
        }

        return true;
    }

    ~MNISTData() {
        delete[] m_labelData;
        delete[] m_imageData;
    }

private:

    uint8_t *m_labelData;
    uint8_t *m_imageData;
    uint8_t *m_labels;
    uint8_t *m_pixels;
};

#endif // lxl_nn_TEST_MNISTDATA_H