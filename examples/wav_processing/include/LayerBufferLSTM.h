#ifndef _WS_LAYERBUFFERLSTM_H
#define _WS_LAYERBUFFERLSTM_H

#include "LayerBufferBase.h"


namespace WS
{

    template <int SAMPLE_CNT, int FILTER_CNT>
    class LayerBufferLSTM : public LayerBufferBase
    {
    public:
        LayerBufferLSTM(std::string const &name, u32 uniqueId, OutputSize const& outputSize, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, transposeWeightsM) {}
        LayerBufferLSTM(std::string const &name, u32 uniqueId, OutputSize const& outputSize, EigenMappings eMapping, DynamicYResize dynMapping=DynamicYResize::NONE, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, eMapping, dynMapping, transposeWeightsM) {}
        virtual ~LayerBufferLSTM() {}

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
        
        /// Returns the specs for this buffer.
        void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* thirdDimensionSize = nullptr,bool* recWeightsValid = nullptr) override
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
                *recWeightsValid = mRecWeightsValid;
            }
        }
        virtual float* getBiasBuffer() override { return reinterpret_cast<float*>(&mBias); }
        virtual float* getWeightsBuffer() override { return reinterpret_cast<float*>(&mWeights); }
    
        virtual float* getRecWeightsBuffer() override { return reinterpret_cast<float*>(&mRecWeights); }
        void setupEigenMatrices(int numBreakPoints, size_t paramsCnt) override
        {
            int filterCnt{0};
            int sampleCnt{0};
            bool biasValid{false};
            bool weightsValid{false};
            bool recWeightsValid{false};


            
            getBufferInfo(filterCnt, sampleCnt, biasValid, weightsValid, nullptr, &recWeightsValid);
            mWeightsMatrix = Eigen::Map<Eigen::MatrixXf>(getWeightsBuffer(), SAMPLE_CNT, 4 * FILTER_CNT);
            mRecWeightsMatrix = Eigen::Map<Eigen::MatrixXf>(getRecWeightsBuffer(), FILTER_CNT, 4 * FILTER_CNT);
            mBiasVector = Eigen::Map<Eigen::RowVectorXf>(getBiasBuffer(), 4 * FILTER_CNT);

        }

        Eigen::MatrixXf& process(Eigen::MatrixXf& inMatrix, SaafActivationFct saafActivation, CustomProcessing customProcess = nullptr) override
        {
            mLayerOutput.setZero();

            Eigen::MatrixXf inMatrixTransposed = inMatrix.transpose();
            Eigen::MatrixXf* outMatrix = new Eigen::MatrixXf();
            outMatrix->resize(inMatrixTransposed.rows(),FILTER_CNT);
            
            Eigen::Index cols = FILTER_CNT;

            // Extract weights for each gate
            Eigen::MatrixXf mWeightsInput = mWeightsMatrix.leftCols(cols);                      // Input gate (i)
            Eigen::MatrixXf mWeightsForget = mWeightsMatrix.middleCols(cols, cols);             // Forget gate (f)
            Eigen::MatrixXf mWeightsCandidate = mWeightsMatrix.middleCols(2*cols, cols);        // Candidate (c)
            Eigen::MatrixXf mWeightsOutput = mWeightsMatrix.rightCols(cols);

            Eigen::MatrixXf mRecWeightsInput = mRecWeightsMatrix.leftCols(cols);                // Input gate (i)
            Eigen::MatrixXf mRecWeightsForget = mRecWeightsMatrix.middleCols(cols, cols);       // Forget gate (f)
            Eigen::MatrixXf mRecWeightsCandidate = mRecWeightsMatrix.middleCols(2*cols, cols);  // Candidate (c)
            Eigen::MatrixXf mRecWeightsOutput = mRecWeightsMatrix.rightCols(cols);

            Eigen::RowVectorXf mBiasInput = mBiasVector.segment(0, cols);                          // Input gate (i)
            Eigen::RowVectorXf mBiasForget = mBiasVector.segment(cols, cols);                      // Forget gate (f)
            Eigen::RowVectorXf mBiasCandidate = mBiasVector.segment(2*cols, cols);                // Candidate (c)
            Eigen::RowVectorXf mBiasOutput = mBiasVector.segment(3*cols, cols);
            
            Eigen::MatrixXf hPrev = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf cPrev = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
        
            Eigen::MatrixXf mInput = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mForget = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mOutput = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mCandidate = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();

            hPrev.setZero();
            cPrev.setZero();

            mInput.setZero();
            mForget.setZero();
            mOutput.setZero();
            mCandidate.setZero();
            
            for (int x_idx = 0; x_idx < inMatrixTransposed.rows(); x_idx++)
            {
                Eigen::MatrixXf x = inMatrixTransposed.row(x_idx);
                
                /* i = sigmoid(np.dot(hPrev, U_i) + np.dot(x, W_i) + b_i) */
                mInput = Sigmoid(hPrev * mRecWeightsInput + x * mWeightsInput + mBiasInput);
                
                /* f = sigmoid(np.dot(hPrev, U_f) + np.dot(x, W_f) +  b_f) */
                mForget = Sigmoid(hPrev * mRecWeightsForget + x * mWeightsForget + mBiasForget);

                /* o = sigmoid(np.dot(hPrev, U_o) + np.dot(x, W_o) + b_o) */
                mOutput = Sigmoid(hPrev * mRecWeightsOutput + x * mWeightsOutput + mBiasOutput);

                mCandidate = hPrev * mRecWeightsCandidate + x * mWeightsCandidate + mBiasCandidate;
                if(saafActivation != nullptr)
                {
                    mCandidate = saafActivation(mCandidate);
                }

                /* mCandidate = np.tanh(np.dot(hPrev, U_c) + np.dot(x, W_c) + b_c) */
        
                /* c = np.multiply(mCandidate, i) + np.multiply(cPrev, f) */
                cPrev = mCandidate.cwiseProduct(mInput) + cPrev.cwiseProduct(mForget);

                if(saafActivation != nullptr)
                    hPrev = mOutput.cwiseProduct(saafActivation(cPrev));
                else
                    hPrev = mOutput.cwiseProduct(cPrev);
                /* h_t = np.multiply(o, np.tanh(c)) */
                

                outMatrix->row(x_idx) = hPrev;
        
            }

            mLayerOutput = outMatrix->transpose();  // 3x2

            outMatrix->resize(mLayerOutput.rows(), mLayerOutput.cols());  // 3x2
            
            *outMatrix = mLayerOutput;
            return *outMatrix;
        }
        float mBias[4 * FILTER_CNT];
        float mWeights[4 * FILTER_CNT][SAMPLE_CNT];

        float mRecWeights[4 * FILTER_CNT][FILTER_CNT];
    };
} // WS

#endif
