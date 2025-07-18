#ifndef _WS_LAYERBUFFER_H
#define _WS_LAYERBUFFER_H

#include "LayerBufferBase.h"

namespace WS
{
    template <int FILTER_CNT, int SAMPLE_CNT>
    class LayerBuffer : public LayerBufferBase
    {
    public:
        LayerBuffer(std::string const &name, u32 uniqueId, OutputSize const& outputSize, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, transposeWeightsM) {}
        LayerBuffer(std::string const &name, u32 uniqueId, OutputSize const& outputSize, EigenMappings eMapping, DynamicYResize dynMapping=DynamicYResize::NONE, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, eMapping, dynMapping, transposeWeightsM) {}
        virtual ~LayerBuffer() {}
    
        /// Returns the specs for this buffer.
        void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* thirdDimensionSize = nullptr, bool* recWeightsValid = nullptr) override
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
                *thirdDimensionSize = 0;
            }

            if(recWeightsValid != nullptr)
            {
                *recWeightsValid = true;
            }
        }
        virtual float* getBiasBuffer() override { return reinterpret_cast<float*>(&mBias); }
        virtual float* getWeightsBuffer() override { return reinterpret_cast<float*>(&mWeights); }
        virtual float* getRecWeightsBuffer() override { return reinterpret_cast<float*>(&mRecWeights); }

    
        float mBias[FILTER_CNT];

        float mWeights[FILTER_CNT][SAMPLE_CNT];

        float mRecWeights[FILTER_CNT][FILTER_CNT];
    };
} // WS

#endif
