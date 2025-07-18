#ifndef _WS_MODELBUFFERS_H
#define _WS_MODELBUFFERS_H

#include "LayerBuffer.h"
#include "LayerBufferOutOnly.h"
#include "LayerEQProcessing.h"
#include "LayerBuffer3d.h"
#include "LayerBufferLSTM.h"
#include "LayerBufferBirnn.h"


#include "LayerBufferReference.h"
#include "LibAiExportOs.h"
#include <string>
#include <vector>

namespace WS
{
    constexpr static const u32 LAYER_CONV{0};
    constexpr static const u32 LAYER_CONV_SMOOTHING{1};
    constexpr static const u32 LAYER_DECONV{2};
    constexpr static const u32 LAYER_DENSE_LOCAL_IN{3};
    constexpr static const u32 LAYER_DENSE_IN{4};
    constexpr static const u32 LAYER_DENSE_OUT{5};
    constexpr static const u32 LAYER_SAAF1{6};
    constexpr static const u32 LAYER_DENSE_SAAF_H1{7};
    constexpr static const u32 LAYER_DENSE_SAAF_H2{8};
    constexpr static const u32 LAYER_DENSE_SAAF_H3{9};
    constexpr static const u32 LAYER_DENSE_SAAF_IN{10};
    constexpr static const u32 LAYER_DENSE_SAAF_OUT{11};
    constexpr static const u32 LAYER_SAAF_OUT{12};
    constexpr static const u32 LAYER_COND_DENSE{13};
    constexpr static const u32 LAYER_MAX_POOL_OUT{14};
    constexpr static const u32 LAYER_UP_SAMPLING_1D_OUT{15};
    constexpr static const u32 LAYER_CHANNEL_INPUT_PADDED{16};
    constexpr static const u32 LAYER_CONV_1D_OUTPUT_ABS{17};
    constexpr static const u32 LAYER_CONV_SMOOTHING_INPUT_PADDED{18};
    constexpr static const u32 LAYER_INPUT_PADDED{19};
    constexpr static const u32 LAYER_COMBINE_CHANNELS{20};
    constexpr static const u32 LAYER_LSTM1{21};
    constexpr static const u32 LAYER_LSTM2{22};
    constexpr static const u32 LAYER_LSTM3{23};
    constexpr static const u32 LAYER_BIRNN_IN{24};
    constexpr static const u32 LAYER_BIRNN1{25};
    constexpr static const u32 LAYER_BIRNN2{26};
    constexpr static const u32 LAYER_DN_SAAF_SE{27};
    constexpr static const u32 LAYER_SE_DENSE1{28};
    constexpr static const u32 LAYER_SE_DENSE2{29};
    constexpr static const u32 LAYER_EQ_PRE_PROCESS{100};
    constexpr static const u32 LAYER_EQ_POST_PROCESS{101};
    
class LIB_AI_EXPORT ModelBuffers
{
public:
    ModelBuffers();
    virtual ~ModelBuffers() = default;
    
    /// to be overriden
    LayerBufferBase *getBuffer(std::string const &name) const;
    LayerBufferBase* getBuffer(u32 bufId) const;
    LayerPrePostProcessing *getBufferProcess(std::string const &name) const;
    LayerPrePostProcessing* getBufferProcess(u32 bufId) const;

    inline std::vector<LayerBufferBase *>& getAllBuffers() { return mAll; }

    

protected:
    std::vector<LayerBufferBase *> mAll;
    std::vector<LayerPrePostProcessing *> mAllPrePostProcessing;
};

template <int FILTER_CNT, int MAX_POOLING_SIZE, int BUFFER3D_ZSIZE>
class ModelBuffersImp : public ModelBuffers
{
public:
    ModelBuffersImp(ModelTopo const& topo)
        : ModelBuffers()
        , mConv("conv", 0, OutputSize{FILTER_CNT, topo.frameLength}, EigenMappings::WEIGHT_MATRIX_ONLY_ROW_MAJOR)
        , mSmoothing("conv_smoothing", 1, OutputSize{topo.frameLength, FILTER_CNT})
        , mMaxPool("max_pool_out", 14, OutputSize{FILTER_CNT, topo.frameLength / topo.maxPoolingPoolSize}, FILTER_CNT)

        , mLSTM1("lstm_1", 21, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, 2 * FILTER_CNT})
        , mLSTM2("lstm_2", 22, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, FILTER_CNT})
        , mLSTM3("lstm_3", 23, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, FILTER_CNT})

        , mBirnnIn("birnn_in", 24, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, 4 * FILTER_CNT})
        , mBirnn1("birnn_1", 25, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, 2 * FILTER_CNT})
        , mBirnn2("birnn_2", 26, OutputSize{topo.frameLength / topo.maxPoolingPoolSize, FILTER_CNT})

        , mDnSaafSe("dn_saaf_se_1", 27, OutputSize{topo.frameLength, FILTER_CNT}, FILTER_CNT)
        , mSeDense1("se_dense1_1", 28, OutputSize{1, FILTER_CNT * 16}, EigenMappings::WEIGHT_BIAS_MATRICES_ROW_MAJOR, DynamicYResize::NONE, true)
        , mSeDense2("se_dense2_1", 29, OutputSize{1, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES_ROW_MAJOR, DynamicYResize::NONE, true)
        
        
        , mDeconvIn("deconv", 2, OUTPUT_SIZE_DYNAMIC)
        , mDenseLocalIn("dense_local_in", 3, OutputSize{FILTER_CNT, topo.denseLocalSize})
        , mDenseIn("dense_in", 4, OUTPUT_SIZE_DYNAMIC, EigenMappings::WEIGHT_BIAS_MATRICES_TRANSPOSE_XY, DynamicYResize::KERNEL_LENGTH_PLUS_NUM_PARAMS)
        , mDenseOut("dense_out", 5, OutputSize{FILTER_CNT, topo.denseLocalSize}, EigenMappings::WEIGHT_BIAS_MATRICES_ROW_MAJOR, DynamicYResize::NONE, true)
        , mSaaf1("saaf_1", 6, OUTPUT_SIZE_DYNAMIC)
        , mUpSampling1DOut("up_sampling_1d_out", 15, OutputSize{topo.frameLength, FILTER_CNT}, FILTER_CNT)
        , mDenseSaafH1("dense_saaf_h1", 7, OutputSize{topo.frameLength, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES_TRANSPOSE_XY)
        , mDenseSaafH2("dense_saaf_h2", 8, OutputSize{topo.frameLength, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES)
        , mDenseSaafH3("dense_saaf_h3", 9, OutputSize{topo.frameLength, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES)
        , mDenseSaafIn("dense_saaf_in", 10, OutputSize{topo.frameLength, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES)
        , mDenseSaafOut("dense_saaf_out", 11, OutputSize{topo.frameLength, FILTER_CNT}, EigenMappings::WEIGHT_BIAS_MATRICES_TRANSPOSE_XY)
        , mSaafOut("saaf_out", 12, OUTPUT_SIZE_DYNAMIC, EigenMappings::WEIGHT_MATRIX_ONLY_TRANSPOSE_XY, DynamicYResize::HALF_BREAKPOINT)
        , mCondDense("cond_dense", 13, OutputSize{FILTER_CNT, topo.frameLength / topo.maxPoolingPoolSize})
        , mChannelInputPadded("channel_input_padded", 16, OutputSize{topo.frameLength + LayerBufferBase::KERNEL_LENGTH, FILTER_CNT}, FILTER_CNT)
        , mConv1DOutputAbs("conv_1d_abs_out", 17, OutputSize{FILTER_CNT, topo.frameLength}, FILTER_CNT)
        , mConvSmoothingInputPadded("conv_smoothing_input_padded", 18, OutputSize{topo.frameLength + LayerBufferBase::KERNEL_LENGTH * 2 - 1, FILTER_CNT}, FILTER_CNT)
        , mInputPadded("input_padded", 19, OutputSize{topo.frameLength + LayerBufferBase::KERNEL_LENGTH, 1}, FILTER_CNT)
        , mCombineChannels("combine_channels", 20, &mDeconvIn, &mConv)
        , mEQPreProcess("eq_pre_process", 100, OutputSize{topo.frameLength, 1})
        , mEQPostProcess("eq_post_process", 101, OutputSize{topo.frameLength, 1})
    {
        // This is the order of processing the buffers.
        mAll.push_back(&mInputPadded);
        mAll.push_back(&mConv);
        mAll.push_back(&mSmoothing);
        mAll.push_back(&mMaxPool);

        mAll.push_back(&mLSTM1);
        mAll.push_back(&mLSTM2);
        mAll.push_back(&mLSTM3);
        
        mAll.push_back(&mBirnnIn);
        mAll.push_back(&mBirnn1);
        mAll.push_back(&mBirnn2);

        mAll.push_back(&mDnSaafSe);
        mAll.push_back(&mSeDense1);
        mAll.push_back(&mSeDense2);



        mAll.push_back(&mDeconvIn);
        mAll.push_back(&mDenseLocalIn);
        mAll.push_back(&mDenseIn);
        mAll.push_back(&mDenseOut);
        mAll.push_back(&mSaaf1);
        mAll.push_back(&mUpSampling1DOut);
        mAll.push_back(&mDenseSaafH1);
        mAll.push_back(&mDenseSaafH2);
        mAll.push_back(&mDenseSaafH3);
        mAll.push_back(&mDenseSaafIn);
        mAll.push_back(&mDenseSaafOut);
        mAll.push_back(&mSaafOut);
        mAll.push_back(&mCondDense);
        mAll.push_back(&mChannelInputPadded); // used to comput the final output!
        mAll.push_back(&mConv1DOutputAbs); // this is the mConv result, absolute values!
        mAll.push_back(&mConvSmoothingInputPadded); // this is the mConvSmoothing input padded!
        mAll.push_back(&mCombineChannels);
        mAllPrePostProcessing.push_back(&mEQPostProcess); // Pre processing EQ layer
        mAllPrePostProcessing.push_back(&mEQPreProcess); // Post processing EQ layer
    }
    
private:
    //Kernel Length is always 64, and sometimes its multiplied by 2.
    LayerBuffer<FILTER_CNT, 64> mConv;

    LayerBuffer<FILTER_CNT, 128> mSmoothing;

    LayerBufferOutOnly mMaxPool;

    LayerBufferLSTM<2 * FILTER_CNT, 2 * FILTER_CNT> mLSTM1;
    LayerBufferLSTM<2 * FILTER_CNT, FILTER_CNT> mLSTM2;
    LayerBufferLSTM<FILTER_CNT, FILTER_CNT> mLSTM3;

    LayerBufferBirnn<3 * FILTER_CNT, 2 * FILTER_CNT> mBirnnIn;
    LayerBufferBirnn<4 * FILTER_CNT, FILTER_CNT> mBirnn1;
    LayerBufferBirnn<2 * FILTER_CNT, FILTER_CNT / 2> mBirnn2;

    LayerBuffer<FILTER_CNT, FILTER_CNT * 16> mSeDense2;
    LayerBuffer<FILTER_CNT * 16, FILTER_CNT> mSeDense1;
    LayerBufferOutOnly mDnSaafSe;


    LayerBuffer<1, 1> mDeconvIn; //Is only one float

    // Here, since we are using a specialized 3d buffer, the 3rd dimension is LayerBufferBase::KERNEL_LENGTH + MAX_NUMBER_OF_PARAMS.
    LayerBuffer3d<FILTER_CNT, MAX_POOLING_SIZE, BUFFER3D_ZSIZE> mDenseLocalIn; 

    // 8 is random for now (max number of params)
    LayerBuffer<MAX_POOLING_SIZE, LayerBufferBase::KERNEL_LENGTH + MAX_NUMBER_OF_PARAMS> mDenseIn;
    //LayerBuffer<FILTER_CNT / 2, LayerBufferBase::KERNEL_LENGTH + MAX_NUMBER_OF_PARAMS> mDenseIn;

    LayerBuffer<MAX_POOLING_SIZE, MAX_POOLING_SIZE> mDenseOut;
    //LayerBuffer<LayerBufferBase::KERNEL_LENGTH, LayerBufferBase::KERNEL_LENGTH> mDenseOut;

    LayerBuffer<FILTER_CNT, NUM_SAAF> mSaaf1;

    LayerBufferOutOnly mUpSampling1DOut;

    LayerBuffer<FILTER_CNT / 2, FILTER_CNT> mDenseSaafH1;

    LayerBuffer<FILTER_CNT / 2, FILTER_CNT / 2> mDenseSaafH2;

    LayerBuffer<FILTER_CNT / 2, FILTER_CNT / 2> mDenseSaafH3;

    LayerBuffer<FILTER_CNT, FILTER_CNT> mDenseSaafIn;

    LayerBuffer<FILTER_CNT, FILTER_CNT / 2> mDenseSaafOut;

    LayerBuffer<FILTER_CNT, NUM_SAAF> mSaafOut;
    
    LayerBuffer<1, MAX_NUMBER_OF_PARAMS> mCondDense;

    LayerBufferOutOnly mChannelInputPadded;
    
    LayerBufferOutOnly mConv1DOutputAbs;

    LayerBufferOutOnly mConvSmoothingInputPadded;

    LayerBufferOutOnly mInputPadded;

    LayerBufferReference mCombineChannels;

    LayerEQProcessing mEQPreProcess;

    LayerEQProcessing mEQPostProcess;
   
};
} // WS

#endif
