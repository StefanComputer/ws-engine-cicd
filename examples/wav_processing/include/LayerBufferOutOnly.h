#ifndef _WS_LAYERBUFFEROUTONLY_H
#define _WS_LAYERBUFFEROUTONLY_H

#include "LayerBufferBase.h"

namespace WS
{
/// @brief Class of buffer that contains empty input layers and only the output buffer
/// is properly setup. This class of buffers are not saved as binary.
class LayerBufferOutOnly : public LayerBufferBase
{
public:
    LayerBufferOutOnly(std::string const &name, u32 uniqueId, OutputSize const& outputSize, int outputFilterCnt) : LayerBufferBase(name, uniqueId, outputSize, false), mOutFilterCnt{outputFilterCnt} {}
    virtual ~LayerBufferOutOnly() {}

    /// Returns the specs for this buffer.
    void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* = nullptr,bool* recWeightsValid = nullptr) override
    {
        LayerBufferBase::getBufferInfo(filterCnt, sampleCnt, biasValid, weightsValid);
        filterCnt = mOutFilterCnt;
        sampleCnt = 0;
    }
    virtual float* getBiasBuffer() override { return nullptr; }
    virtual float* getWeightsBuffer() override { return nullptr; }
private:
    int mOutFilterCnt;
};
} // WS

#endif
