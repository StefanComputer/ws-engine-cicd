#pragma once

#include "LibAiExportOs.h"
#include <array>
#include <stdio.h>
#include <string>

namespace WS
{
class LIB_AI_EXPORT BiquadEQ
{
public:
    BiquadEQ();
    BiquadEQ(float samplingRate);
    ~BiquadEQ();

    bool makeFilter(std::string const& type, float fc, float Q = 0.7, float gain = 0.0);

    float* getCoefficients();

    float processSample(float& sample);

private:
    enum FilterType
    {
        Highpass,
        Lowshelf,
        Peak,
        Highshelf,
        Lowpass,
        Pre_Lowpass
    };

    FilterType filterType;

    bool setFilterType(std::string const& type);

    void makePeak(float fc, float Q, float gain);
    void makeLowShelf(float fc, float Q, float gain);
    void makeHighShelf(float fc, float Q, float gain);
    void makeLowpass(float fc);
    void makeHighpass(float fc);

    void constrainParams(float& fc, float& Q, float& gain);
    void constrainParams(float& fc);

    float fs = 0.0F;
    float c0 = 0.0F;
    float d0 = 0.0F;

    std::array<float, 5> coeffs = {0.0F};

    float x_1 = 0.0F, x_2 = 0.0F;
    float y_1 = 0.0F, y_2 = 0.0F;
};
} // namespace WS