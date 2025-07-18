#ifndef _WS_LAYERBUFFERREFERENCE_H
#define _WS_LAYERBUFFERREFERENCE_H

#include "LayerBufferBase.h"

namespace WS
{
/// @brief Class of buffer that uses another buffer's weight and/or bias buffers as
/// input for their processing.
class LayerBufferReference : public LayerBufferBase
{
public:
    LayerBufferReference(std::string const &name, u32 uniqueId, LayerBufferBase* biasBufferRef, LayerBufferBase* weightsBufferRef) : 
        LayerBufferBase(name, uniqueId, OutputSize{0U, 0U}, false)
        , mBiasBufferRef{biasBufferRef != nullptr ? biasBufferRef->getBiasBuffer() : nullptr}
        , mWeightsBufferRef{weightsBufferRef != nullptr ? weightsBufferRef->getWeightsBuffer() : nullptr}
    {
    }
    virtual ~LayerBufferReference() {}

    /// Returns the specs for this buffer.
    void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* = nullptr,bool* recWeightsValid = nullptr) override
    {
        LayerBufferBase::getBufferInfo(filterCnt, sampleCnt, biasValid, weightsValid);
        filterCnt = 0;
        sampleCnt = 0;
    }
    virtual float* getBiasBuffer() override { return mBiasBufferRef; }
    virtual float* getWeightsBuffer() override { return mWeightsBufferRef; }

private:
    float* mBiasBufferRef;
    float* mWeightsBufferRef;
};
} // WS

#endif
