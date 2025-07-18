#ifndef _WS_MODELTOPO_H
#define _WS_MODELTOPO_H

#include "LibAiExportOs.h"
#include "Scriptable.h"
#include <string>
#include <vector>
#include "DataBuffer.h"
#include "JsonConfigEQ.h"



namespace WS
{
    constexpr static char const* MT_CRAFX{"CRAFx"};
    constexpr static char const* MT_CAFX{"CAFx"};
    constexpr static char const* MT_PCAFXC{"PCAFx_C"};
    constexpr static char const* MT_UNICRAFX{"uni_CRAFx"};
class LIB_AI_EXPORT ModelTopo : public WS::Util::Scriptable
{
public:
    ModelTopo(size_t frLength);
    ModelTopo();

    inline void setParamSize(size_t paramSize)
    {
        modelParams.clear();
        modelParams.resize(paramSize);
    }

    void resetFrameLength(u32 const newFrameLength);

    /// @brief  Scriptable overrides
    /// @param  reader or writer, depending if reading or writing.
    /// @return true on success.
    bool readFromStream(WS::Util::JsonReader&) override;
    bool writeToStream(WS::Util::JsonWriter&) override;

    /// Data members
    u32 filterCnt;
    u32 frameLength;
    u32 hopLength;
    u32 denseLocalSize;
    u32 maxPoolingPoolSize;
    u32 breakPointsCnt;
    bool isCondDenseImplementation;

    /// Overall scaling factor for the model [0.0, 1.0]
    float modelFactor;

    std::vector<float> modelParams;
    

    /// @brief  Other relevent information
    std::string modelType;
    WS::Util::DataBuffer enc_modelType;
    std::string version;

    JsonConfigEQ eqConfig;
};
} // namespace WS

#endif
