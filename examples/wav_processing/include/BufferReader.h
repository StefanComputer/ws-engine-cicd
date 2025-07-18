#ifndef _WS_BUFFERREADER_H
#define _WS_BUFFERREADER_H

#include "LibAiExportOs.h"
#include "Logger.h"
#include "ModelBuffers.h"
#include "ModelTopo.h"
#include <string>
#include <vector>

namespace WS
{
class LIB_AI_EXPORT BufferReader
{
public:
    BufferReader(std::string const& fileName) : mLogger{nullptr}, mFileName{fileName} {}
    virtual ~BufferReader() {}

    /// @brief  Loads-in a model file (in whatever format) and allocates the appropriate ModelBuffers
    /// memory space and fills-in the ModelTopo structure with infromation read from the model.
    /// @param buffers returns pointer to allocated buffers
    /// @param topo is filled with information from the model
    /// @return true on success, false otherwise
    virtual bool loadModel(ModelBuffers*& buffers, ModelTopo& topo) = 0;

    // Accessors
    inline std::string const& getFileName() const { return mFileName; }
    inline void setLogger(WS::Logger* logger) { mLogger = logger; }

    /// Operations
    bool convertJsonToTopo(std::string const& jsonTopoStr, ModelTopo& topo);

protected:
    template <int BUFFER3D_YSIZE, int BUFFER3D_ZSIZE>
    ModelBuffers* instantiateModelBuffers(ModelTopo const& modelTopo)
    {
        WS::ModelBuffers* buffers = nullptr;
        if(modelTopo.filterCnt == 8)
        {
            buffers = new WS::ModelBuffersImp<8, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 16)
        {
            buffers = new WS::ModelBuffersImp<16, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 19)
        {
            buffers = new WS::ModelBuffersImp<19, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 32)
        {
            buffers = new WS::ModelBuffersImp<32, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 64)
        {
            buffers = new WS::ModelBuffersImp<64, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 96)
        {
            buffers = new WS::ModelBuffersImp<96, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }
        else if(modelTopo.filterCnt == 128)
        {
            buffers = new WS::ModelBuffersImp<128, BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(modelTopo);
        }

        std::cout << "size of buffers : " << modelTopo.filterCnt << std::endl;
        return buffers;
    }

    template <int BUFFER3D_YSIZE>
    ModelBuffers* instantiateModelBuffers(ModelTopo const& topo)
    {
        ModelBuffers* buffers{nullptr};
        u32 paramsCnt = static_cast<u32>(topo.modelParams.size());
        if(paramsCnt == 0)
        {
            constexpr int BUFFER3D_ZSIZE = LayerBufferBase::KERNEL_LENGTH + 0;
            buffers = instantiateModelBuffers<BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(topo);
        }
        else if(paramsCnt == 1)
        {
            constexpr int BUFFER3D_ZSIZE = LayerBufferBase::KERNEL_LENGTH + 1;
            buffers = instantiateModelBuffers<BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(topo);
        }
        else if(paramsCnt == 2)
        {
            constexpr int BUFFER3D_ZSIZE = LayerBufferBase::KERNEL_LENGTH + 2;
            buffers = instantiateModelBuffers<BUFFER3D_YSIZE, BUFFER3D_ZSIZE>(topo);
        }
        return buffers;
    }

    // Data members
protected:
    // Output some info to a logger if setup.
    WS::Logger* mLogger;

private:
    std::string mFileName;
};
} // namespace WS

#endif
