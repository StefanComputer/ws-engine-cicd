#ifndef _WS_LAYERBUFFERBIRNN_H
#define _WS_LAYERBUFFERBIRNN_H

#include "LayerBufferBase.h"


namespace WS
{

    template <int SAMPLE_CNT, int FILTER_CNT>
    class LayerBufferBirnn : public LayerBufferBase
    {
    public:
        LayerBufferBirnn(std::string const &name, u32 uniqueId, OutputSize const& outputSize, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, transposeWeightsM) {}
        LayerBufferBirnn(std::string const &name, u32 uniqueId, OutputSize const& outputSize, EigenMappings eMapping, DynamicYResize dynMapping=DynamicYResize::NONE, bool transposeWeightsM=false) : LayerBufferBase(name, uniqueId, outputSize, eMapping, dynMapping, transposeWeightsM) {}
        virtual ~LayerBufferBirnn() {}

        inline Eigen::MatrixXf sigmoid(Eigen::MatrixXf const& mat)
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
        void getBufferInfo(int &filterCnt, int &sampleCnt, bool &biasValid, bool &weightsValid, size_t* thirdDimensionSize = nullptr ,bool* recWeightsValid = nullptr) override
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
            mWeightsMatrix = Eigen::Map<Eigen::MatrixXf>(getWeightsBuffer(), SAMPLE_CNT, 8 * FILTER_CNT);
            mRecWeightsMatrix = Eigen::Map<Eigen::MatrixXf>(getRecWeightsBuffer(), FILTER_CNT, 8 * FILTER_CNT);
            mBiasVector = Eigen::Map<Eigen::RowVectorXf>(getBiasBuffer(), 8 * FILTER_CNT);

        }

        Eigen::MatrixXf& process(Eigen::MatrixXf& inMatrix, SaafActivationFct saafActivation, CustomProcessing customProcess = nullptr) override
        {
            mLayerOutput.setZero();

            Eigen::MatrixXf inMatrixTransposed = inMatrix.transpose();
            Eigen::MatrixXf* outMatrix = new Eigen::MatrixXf();
            outMatrix->resize(inMatrixTransposed.rows(),2 * FILTER_CNT);
            Eigen::MatrixXf outMatrixForward = Eigen::MatrixXf::Zero(inMatrixTransposed.rows(), FILTER_CNT);
            Eigen::MatrixXf outMatrixBackward = Eigen::MatrixXf::Zero(inMatrixTransposed.rows(), FILTER_CNT);
            
            Eigen::Index cols = FILTER_CNT;

            // Extract weights for each gate
            Eigen::MatrixXf mWeightsInputForward = mWeightsMatrix.leftCols(cols);                      // Input gate (i)
            Eigen::MatrixXf mWeightsForgetForward = mWeightsMatrix.middleCols(cols, cols);             // Forget gate (f)
            Eigen::MatrixXf mWeightsCandidateForward = mWeightsMatrix.middleCols(2*cols, cols);        // Candidate (c)
            Eigen::MatrixXf mWeightsOutputForward = mWeightsMatrix.middleCols(3*cols,cols);

            Eigen::MatrixXf mRecWeightsInputForward = mRecWeightsMatrix.leftCols(cols);                // Input gate (i)
            Eigen::MatrixXf mRecWeightsForgetForward = mRecWeightsMatrix.middleCols(cols, cols);       // Forget gate (f)
            Eigen::MatrixXf mRecWeightsCandidateForward = mRecWeightsMatrix.middleCols(2*cols, cols);  // Candidate (c)
            Eigen::MatrixXf mRecWeightsOutputForward = mRecWeightsMatrix.middleCols(3*cols,cols);

            Eigen::RowVectorXf mBiasInputForward = mBiasVector.segment(0, cols);                          // Input gate (i)
            Eigen::RowVectorXf mBiasForgetForward = mBiasVector.segment(cols, cols);                      // Forget gate (f)
            Eigen::RowVectorXf mBiasCandidateForward = mBiasVector.segment(2*cols, cols);                // Candidate (c)
            Eigen::RowVectorXf mBiasOutputForward = mBiasVector.segment(3*cols, cols);
            
            Eigen::MatrixXf hPrevForward = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf cPrevForward = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
        
            Eigen::MatrixXf mWeightsInputBackward = mWeightsMatrix.middleCols(4*cols,cols);                      // Input gate (i)
            Eigen::MatrixXf mWeightsForgetBackward = mWeightsMatrix.middleCols(5*cols, cols);             // Forget gate (f)
            Eigen::MatrixXf mWeightsCandidateBackward = mWeightsMatrix.middleCols(6*cols, cols);        // Candidate (c)
            Eigen::MatrixXf mWeightsOutputBackward = mWeightsMatrix.rightCols(cols);

            Eigen::MatrixXf mRecWeightsInputBackward = mRecWeightsMatrix.middleCols(4*cols,cols);                // Input gate (i)
            Eigen::MatrixXf mRecWeightsForgetBackward = mRecWeightsMatrix.middleCols(5*cols, cols);       // Forget gate (f)
            Eigen::MatrixXf mRecWeightsCandidateBackward = mRecWeightsMatrix.middleCols(6*cols, cols);  // Candidate (c)
            Eigen::MatrixXf mRecWeightsOutputBackward = mRecWeightsMatrix.rightCols(cols);

            Eigen::RowVectorXf mBiasInputBackward = mBiasVector.segment(4*cols, cols);                          // Input gate (i)
            Eigen::RowVectorXf mBiasForgetBackward = mBiasVector.segment(5*cols, cols);                      // Forget gate (f)
            Eigen::RowVectorXf mBiasCandidateBackward = mBiasVector.segment(6*cols, cols);                // Candidate (c)
            Eigen::RowVectorXf mBiasOutputBackward = mBiasVector.segment(7*cols, cols);
            
            Eigen::MatrixXf hPrevBackward = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf cPrevBackward = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
        
            Eigen::MatrixXf mInput = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mForget = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mOutput = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            Eigen::MatrixXf mCandidate = Eigen::Matrix<float,1,FILTER_CNT,Eigen::RowMajor>();
            
            hPrevForward.setZero();
            cPrevForward.setZero();
            hPrevBackward.setZero();
            cPrevBackward.setZero();

            mInput.setZero();
            mForget.setZero();
            mOutput.setZero();
            mCandidate.setZero();
            
            for (int x_idx = 0; x_idx < inMatrixTransposed.rows(); x_idx++)
            {
                Eigen::MatrixXf x = inMatrixTransposed.row(x_idx);

                mInput = sigmoid(hPrevForward * mRecWeightsInputForward + x * mWeightsInputForward + mBiasInputForward);

                mForget = sigmoid(hPrevForward * mRecWeightsForgetForward + x * mWeightsForgetForward + mBiasForgetForward);

                mOutput = sigmoid(hPrevForward * mRecWeightsOutputForward + x * mWeightsOutputForward + mBiasOutputForward);

                mCandidate = hPrevForward * mRecWeightsCandidateForward + x * mWeightsCandidateForward + mBiasCandidateForward;
                if(saafActivation != nullptr)
                {
                    mCandidate = saafActivation(mCandidate);
                }

                cPrevForward = mCandidate.cwiseProduct(mInput) + cPrevForward.cwiseProduct(mForget);
                if(saafActivation != nullptr)
                    hPrevForward = mOutput.cwiseProduct(saafActivation(cPrevForward));
                else
                    hPrevForward = mOutput.cwiseProduct(cPrevForward);
                
                outMatrixForward.row(x_idx) = hPrevForward;
            }
            for (int x_idx = inMatrixTransposed.rows()-1; x_idx >= 0 ; x_idx--)
            {
                Eigen::MatrixXf x = inMatrixTransposed.row(x_idx);

                mInput = sigmoid(hPrevBackward * mRecWeightsInputBackward + x * mWeightsInputBackward + mBiasInputBackward);

                mForget = sigmoid(hPrevBackward * mRecWeightsForgetBackward + x * mWeightsForgetBackward + mBiasForgetBackward);

                mOutput = sigmoid(hPrevBackward * mRecWeightsOutputBackward + x * mWeightsOutputBackward + mBiasOutputBackward);

                mCandidate = hPrevBackward * mRecWeightsCandidateBackward + x * mWeightsCandidateBackward + mBiasCandidateBackward;
                if(saafActivation != nullptr)
                {
                    mCandidate = saafActivation(mCandidate);
                }

                cPrevBackward = mCandidate.cwiseProduct(mInput) + cPrevBackward.cwiseProduct(mForget);
                if(saafActivation != nullptr)
                    hPrevBackward = mOutput.cwiseProduct(saafActivation(cPrevBackward));
                else
                    hPrevBackward = mOutput.cwiseProduct(cPrevBackward);
                outMatrixBackward.row(x_idx) = hPrevBackward;
            }

            *outMatrix << outMatrixForward, outMatrixBackward;
            mLayerOutput = outMatrix->transpose();  // 3x2

            outMatrix->resize(mLayerOutput.rows(), mLayerOutput.cols());  // 3x2
            
            *outMatrix = mLayerOutput;
            return *outMatrix;
        }

        float mBias[8 * FILTER_CNT];
        float mWeights[8 * FILTER_CNT][SAMPLE_CNT];
        float mRecWeights[8 * FILTER_CNT][FILTER_CNT];
    };
} // WS

#endif
