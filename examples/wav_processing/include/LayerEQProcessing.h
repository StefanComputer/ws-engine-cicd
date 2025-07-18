#ifndef _WS_LAYEREQPROCESSING_H
#define _WS_LAYEREQPROCESSING_H

#include "LibAiExportOs.h"
#include "ModelTopo.h"
#include "LayerPrePostProcessing.h"
#include <Eigen/Dense>
#include <functional>
#include <locale>
#include <string>
#include <utility>
#include <vector>
#include "BiquadEQ.h"

namespace WS
{
using SaafActivationFct = Eigen::MatrixXf (*)(Eigen::MatrixXf const& mat);
using OutputSize = std::pair<u32, u32>;

class LIB_AI_EXPORT LayerEQProcessing : public LayerPrePostProcessing
{

public:
LayerEQProcessing(std::string const &name, u32 uniqueId, OutputSize const& outputSize) : LayerPrePostProcessing(name, uniqueId, outputSize) {}
virtual ~LayerEQProcessing() {}

/// @brief Loads the initial parameter values for the EQ processing layer
/// @param topo the model topo onject from AudioModel.imp, contains most values
/// @param sampleRate sampleRate object from AudioModel.imp for sample rate
inline void loadLayer(ModelTopo const& topo, int sampleRate){
    mNumEqFilters = topo.eqConfig.mNumEqFilters;
    mApplyEq = topo.eqConfig.mApplyEq;
    mApplyPreEq = topo.eqConfig.mApplyPreEq;

    std::cout << "Prepping layers" << std::endl;

    if(mNumEqFilters > 0)
    {
        // If model has EQ, filters must go in order, meaning for example that it can't
        // apply a lowpass filter at 5KHz and then apply a peak filter at 15KHz to boost
        // high frequencies. In this case, there will be no more high frequencies due
        // low pass filter applied. So all filters must be applied in order to have
        // something to be filtered.

        for(int filter = 0; filter < mNumEqFilters; filter++)
        {
            // check if we need to apply EQ at the input
            if(topo.eqConfig.mEQFilters[filter].mType == "Pre_Lowpass")
                mPreEQIndex = filter;
                //after this is post eq

            eq.push_back(WS::BiquadEQ(sampleRate));
            eq[filter].makeFilter(topo.eqConfig.mEQFilters[filter].mType,
                topo.eqConfig.mEQFilters[filter].mFrequency,
                topo.eqConfig.mEQFilters[filter].mQ,
                topo.eqConfig.mEQFilters[filter].mGain);
        }
    }
}

/// Override of the basic processing routine associated to pre/post layers for EQ.
virtual Eigen::MatrixXf& process(Eigen::MatrixXf& inputOutput, CustomProcessing customProcess = nullptr) override{


    if(getName()=="eq_pre_process"){     
        //float* mBuf{outMatrix.data()};
        float* inputOutputData{inputOutput.data()};

        //Apply pre EQ if necessary
        if(mApplyPreEq)
            filter(eq[mPreEQIndex], inputOutputData, mFrameLength);

    } else {  
        //float* mBuf{outMatrix.data()};
        float* inputOutputData{inputOutput.data()};

        if(mApplyEq)
        {
            for(int filt = 0; filt < mNumEqFilters; filt++)
            {
                if(mApplyPreEq && filt == mPreEQIndex)
                    continue;
                filter(eq[filt], inputOutputData, mFrameLength);
                }
            }

    }
    return inputOutput;

}

protected:

private:
void filter(WS::BiquadEQ& filter, float* signal, size_t length)
{
    for(size_t i = 0; i < length; i++)
        signal[i] = filter.processSample(signal[i]);
}
    // EQ related members moved from AudioModel.imp
    std::vector<WS::BiquadEQ> eq{};
    int mNumEqFilters = 0;
    bool mApplyEq = false;
    bool mApplyPreEq = false;
    int mPreEQIndex = 0;
    u32 mFrameLength = 0;

};
} // namespace WS

#endif
