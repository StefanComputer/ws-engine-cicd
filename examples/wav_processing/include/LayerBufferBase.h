#ifndef _WS_LAYERBUFFERBASE_H
#define _WS_LAYERBUFFERBASE_H

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
    static const std::int32_t MAX_NUMBER_OF_PARAMS = 8;
    static const std::int32_t NUM_SAAF = 13;

using SaafActivationFct = Eigen::MatrixXf (*)(Eigen::MatrixXf const& mat);
using OutputSize = std::pair<u32, u32>;
static constexpr OutputSize OUTPUT_SIZE_DYNAMIC{0U, 0U};

enum class EigenMappings
{
    NONE = 0,
    WEIGHT_MATRIX_ONLY,
    WEIGHT_MATRIX_ONLY_ROW_MAJOR,
    WEIGHT_MATRIX_ONLY_TRANSPOSE_XY,
    WEIGHT_BIAS_MATRICES,
    WEIGHT_BIAS_MATRICES_ROW_MAJOR,
    WEIGHT_BIAS_MATRICES_TRANSPOSE_XY
};

enum class DynamicYResize
{
    NONE, // no need to dynamic map Eigen matrix
    HALF_BREAKPOINT,
    KERNEL_LENGTH_PLUS_NUM_PARAMS
};

class LIB_AI_EXPORT LayerBufferBase : public LayerBase
{
public:
    static const int KERNEL_LENGTH{64};

    /// A custom processing functor can be provided to easily specialize computation of this layer.
    using CustomProcessing = std::function<void(LayerBufferBase&, Eigen::MatrixXf& input, Eigen::MatrixXf& output)>;

    LayerBufferBase(std::string const& name, u32 uniqueId, OutputSize const& outputSize, bool transposeWeightsM);
    LayerBufferBase(std::string const& name, u32 uniqueId, OutputSize const& outputSize, EigenMappings eMapping, DynamicYResize dynMapping, bool transposeWeightsM);
    virtual ~LayerBufferBase() {}

    /// To be overriden
    virtual void getBufferInfo(int& filterCnt, int& sampleCnt, bool& biasValid, bool& weightsValid, size_t* thirdDimensionSize = nullptr,bool* recWeightsValid = nullptr);

    /// Returns the local basic storage
    virtual float* getBiasBuffer();
    virtual float* getWeightsBuffer();

    virtual float* getRecWeightsBuffer();
    inline Eigen::MatrixXf const& getRecWeightsMatrix() const { return mRecWeightsMatrix; }
    inline void setRecWeightsValid() { mRecWeightsValid = true; }

    inline Eigen::MatrixXf Sigmoid(Eigen::MatrixXf const& mat)
    {
        Eigen::MatrixXf result(mat.rows(), mat.cols());
        for(int i = 0; i < mat.rows(); i++)
        {
            for(int j = 0; j < mat.cols(); j++)
            {

                    result(i, j) = 1 / (1 + std::exp(-mat(i, j)));
            }
        }
        return result;
    }
    
    virtual bool isReadOnly() const { return false; }

    /// Call to setup the output matrix with proper size based on frameLenght
    /// and trigger Eigen matrix mapping on top of basic local storage.
    /// Base does nothing.
    virtual void setupEigenMatrices(int numBreakPoints, size_t paramsCnt);

    /// Basic processing routine associated to "standard" layers.
    virtual Eigen::MatrixXf& process(Eigen::MatrixXf& input, SaafActivationFct saafActivation, CustomProcessing customProcess = nullptr);

    // Accessors
    //inline std::string const& getName() const { return mName; }
    //inline u32 getId() const { return mId; }
    //inline bool isValid() const { return mIsValid; }
    inline Eigen::MatrixXf const& getWeightsMatrix() const { return mWeightsMatrix; }
    inline Eigen::RowVectorXf const& getBiasVector() const { return mBiasVector; }
    inline Eigen::MatrixXf& getOutputMatrix() { return mLayerOutput; }
    inline Eigen::MatrixXf& getOutputPrevMatrix() { return mLayerOutputPrev; }
    inline Eigen::MatrixXf& getOutputNextMatrix() { return mLayerOutputNext; }
    
    inline void clearBufferProcessCounter(){mBufferToProcess = 0;}
    inline void increaseBufferProcessCounter(){mBufferToProcess++;}
    inline void firstProcessDone(){mFirstProcess = false;}
    inline bool isFirstProcess(){return mFirstProcess;}

    
    //inline void setValid(bool valid) { mIsValid = valid; }
    inline void setBiasValid() { mBiasValid = true; }
    inline void setWeightsValid() { mWeightsValid = true; }
    inline EigenMappings getEigenMapping() const { return mEigenMapping; }
    inline void clearValid()
    {
        LayerBase::clearValid();
        mBiasValid = false;
        mWeightsValid = false;
        //mIsValid = false;
        mRecWeightsValid = false;
    }
    inline void setSmallerSizes(int smallerSizeSampleCnt, int smallerSizeFilterCnt)
    {
        mBufferSmallerSizeSampleCnt = smallerSizeSampleCnt;
        mBufferSmallerSizeFilterCnt = smallerSizeFilterCnt;
    }
    inline bool const getTransposeWeights() const
    {
        return mTransposeWeightsM;
    }

    // Data members
protected:
    /// Set if this is a valid buffer
    //bool mIsValid;

    /// Set if there is a valid bias buffer present
    bool mBiasValid;

    /// Set if there is a valid weights buffer present
    bool mWeightsValid;

    bool mRecWeightsValid;

    /// Check if the actual buffer is smaller than its allocated storage.
    /// If storage and buyffer size are equal, this is set to -1, by default.
    /// Otherwise it overrides that actual sizes.
    int mBufferSmallerSizeSampleCnt;
    int mBufferSmallerSizeFilterCnt;

    /// Eigen buffers. This is the Eigne's equivalent buffer of the input,
    /// and it is mapped on thop of the LayerBuffe's containers.
    Eigen::MatrixXf mWeightsMatrix;
    Eigen::RowVectorXf mBiasVector;

    Eigen::MatrixXf mRecWeightsMatrix;

    /// Indicate if the weights matrix needs to be transposed before processing.
    bool mTransposeWeightsM;

    /// Here is the output buffer
    Eigen::MatrixXf mLayerOutput;
    Eigen::MatrixXf mLayerOutputPrev;
    Eigen::MatrixXf mLayerOutputNext;

    int mBufferToProcess = 0;
    bool mFirstProcess = true;

    /// @brief Holds the configuration information for the mapping of the Eigen matrices.
    EigenMappings mEigenMapping;
    DynamicYResize mDynYResize;

private:
    /// @brief The name of this layer buffer.
    //std::string mName;

    /// A unique identifier used to link the binary buffer file.
    //u32 mId;
};
} // namespace WS

#endif
