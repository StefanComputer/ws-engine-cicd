#ifndef _WS_LAYERPOSTPROCESS_H
#define _WS_LAYERPOSTPROCESS_H

#include "LibAiExportOs.h"
#include "ModelTopo.h"
#include "LayerBase.h"
#include <Eigen/Dense>
#include <functional>
#include <locale>
#include <string>
#include <utility>
#include <vector>

namespace WS
{
using SaafActivationFct = Eigen::MatrixXf (*)(Eigen::MatrixXf const& mat);
using OutputSize = std::pair<u32, u32>;

class LIB_AI_EXPORT LayerPrePostProcessing : public LayerBase
{
public:
    static const int KERNEL_LENGTH{64};

    /// A custom processing functor can be provided to easily specialize computation of this layer.
    using CustomProcessing = std::function<void(LayerPrePostProcessing&, Eigen::MatrixXf& input, Eigen::MatrixXf& output)>;

    LayerPrePostProcessing(std::string const& name, u32 uniqueId, OutputSize const& outputSize);
    virtual ~LayerPrePostProcessing() {}

    virtual bool isReadOnly() const { return false; }

    // To be overriden
    // Loads initial parameters for the layer
    //virtual void loadLayer(ModelTopo const& topo) { /* default: maybe do nothing */ }

    /// Basic processing routine associated to "standard" layers.
    virtual Eigen::MatrixXf& process(Eigen::MatrixXf& input, CustomProcessing customProcess = nullptr);

    // Accessors
    inline Eigen::MatrixXf& getOutputMatrix() { return mLayerOutput; }

    // Data members
protected:

    /// Check if the actual buffer is smaller than its allocated storage.
    /// If storage and buyffer size are equal, this is set to -1, by default.
    /// Otherwise it overrides that actual sizes.
    int mBufferSmallerSizeSampleCnt;
    int mBufferSmallerSizeFilterCnt;

    /// Here is the output buffer
    Eigen::MatrixXf mLayerOutput;

private:
};
} // namespace WS

#endif
