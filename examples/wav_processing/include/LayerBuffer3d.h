#ifndef _WS_LAYERBUFFER3D_H
#define _WS_LAYERBUFFER3D_H

#include "LayerBufferBase.h"
#include "LayerTimer.h"
#include <iostream>

namespace WS
{
template <int FILTER_CNT, int SAMPLE_CNT, int ZSIZE>
class LayerBuffer3d : public LayerBufferBase
{
public:
    LayerBuffer3d(std::string const& name, u32 uniqueId, OutputSize const& outputSize) : LayerBufferBase(name, uniqueId, outputSize, false), mParamsCnt{0} {}
    virtual ~LayerBuffer3d() {}

    /// LayerBufferBase overridables
    void getBufferInfo(int& filterCnt, int& sampleCnt, bool& biasValid, bool& weightsValid, size_t* thirdDimensionSize = nullptr,bool* recWeightsValid = nullptr) override
    {
        LayerBufferBase::getBufferInfo(filterCnt, sampleCnt, biasValid, weightsValid);
        filterCnt = FILTER_CNT;
        sampleCnt = SAMPLE_CNT;
        if(mBufferSmallerSizeFilterCnt != -1)
        {
            filterCnt = mBufferSmallerSizeFilterCnt;
        }
        if(mBufferSmallerSizeSampleCnt != -1)
        {
            sampleCnt = mBufferSmallerSizeSampleCnt;
        }

        // check for our 3rd dimension size and report it if requested.
        if(thirdDimensionSize != nullptr)
        {
            *thirdDimensionSize = ZSIZE;
        }
    }
    float* getBiasBuffer() override { return reinterpret_cast<float*>(&mBias); }
    float* getWeightsBuffer() override { return reinterpret_cast<float*>(&mWeights); }
    void setupEigenMatrices(int numBreakPoints, size_t paramsCnt) override
    {
        LayerBufferBase::setupEigenMatrices(numBreakPoints, paramsCnt);
        mParamsCnt = paramsCnt;
    }
    Eigen::MatrixXf& process(Eigen::MatrixXf& input, SaafActivationFct saafActivation, CustomProcessing = nullptr) override
    {
        mLayerOutput.setZero();
        LayerTimer processTiming;
        processTiming.startTimer();
        if(isValid())
        {
#ifdef USING_THREADS
#pragma omp parallel for schedule(static, 4) num_threads(N_TRHEADS)
#endif
            for(size_t filterIndex = 0; filterIndex < FILTER_CNT; filterIndex++)
            {
                auto w = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    &mWeights[filterIndex][0][0], ZSIZE, SAMPLE_CNT);
                auto bias = Eigen::Map<Eigen::MatrixXf>(&mBias[filterIndex][0], 1, SAMPLE_CNT);
                auto inputMaxPool = input.row(filterIndex);
                mLayerOutput.row(filterIndex) = inputMaxPool * w + bias;
            }
            if(saafActivation != nullptr)
            {
                mLayerOutput = saafActivation(mLayerOutput);
            }
        }
        processTiming.endTimer(getName());
        return mLayerOutput;
    }

    float mBias[FILTER_CNT][SAMPLE_CNT];
    float mWeights[FILTER_CNT][SAMPLE_CNT][ZSIZE];

    /// Cache information for the processing
    size_t mParamsCnt;
};

} // namespace WS

#endif