#ifndef _WS_LAYERBUFFERDEBUG_H
#define _WS_LAYERBUFFERDEBUG_H

#include "LayerBufferBase.h"

namespace WS
{
/// @brief Class of buffer that is used to override binary buffers input
/// with actual real bias/weigths as saved by the python scripts, which
/// are simple C arrays. The implementation of getBiasBuffer()/getWeigthsBuffer()
/// and process() are put in the test/ folder, in a cpp module.
class LayerBufferDebug : public LayerBufferBase
{
public:
    LayerBufferDebug(std::string const &name, u32 uniqueId, OutputSize const& outputSize, int filterCnt, int sampleCnt, bool bias, bool weights) : LayerBufferBase(name, uniqueId, outputSize, false), mFilterCnt{filterCnt}
    , mSampleCnt{sampleCnt}
    , mBias{bias}
    , mWeights{weights}
    #ifdef CRAfx
    , mRecWeights{recWeights}
    #endif
    {}
    virtual ~LayerBufferDebug() {}

    /// Returns the specs for this buffer.
    void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* = nullptr,bool* recWeightsValid = nullptr) override
    {
        LayerBufferBase::getBufferInfo(filterCnt, sampleCnt, biasValid, weightsValid);
        filterCnt = mFilterCnt;
        sampleCnt = mSampleCnt;
        biasValid = mBias;
        weightsValid = mWeights;
    }
    virtual float* getBiasBuffer() override;
    virtual float* getWeightsBuffer() override;

    virtual bool isReadOnly() const override { return true; }
    Eigen::MatrixXf& process(Eigen::MatrixXf& input, SaafActivationFct saafActivation, CustomProcessing = nullptr) override;

private:
    int mFilterCnt;
    int mSampleCnt;
    bool mBias;
    bool mWeights;    
    bool mRecWeights;
};

} // WS

#endif
