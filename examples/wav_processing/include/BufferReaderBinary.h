#ifndef _WS_BUFFERREADERBINARY_H
#define _WS_BUFFERREADERBINARY_H

#include <iostream>
#include <fstream>
#include "BufferReader.h"

namespace WS
{
class LayerBufferBase;

class LIB_AI_EXPORT BufferReaderBinary : public BufferReader
{
public:
    BufferReaderBinary(std::string const &fileName) : BufferReader(fileName) {}
    virtual ~BufferReaderBinary() {}

    // BufferReader overrides
    bool loadModel(ModelBuffers* &buffers, ModelTopo &topo) override;
    
    // Accessors

private:
    bool readHeader(ModelTopo &topo);
    bool loadFromBinary( std::ifstream& in, LayerBufferBase& layerBuffer, int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid);
    bool loadFromBinary3d( std::ifstream& in, LayerBufferBase& layerBuffer, size_t numParams, int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid);
    bool loadFromBinaryLSTM( std::ifstream& in, LayerBufferBase& layerBuffer, int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, bool &recWeightsValid);
    bool loadFromBinaryBirnn( std::ifstream& in, LayerBufferBase& layerBuffer, int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, bool &recWeightsValid);
    
};
} // WS

#endif
