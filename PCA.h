#ifndef __PCA__PCA__
#define __PCA__PCA__

#include "LoadMatrix.h"
#include <cmath>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Dense>

class PCA
{
public:
    PCA(std::string add, const int cols, const int rows);
    virtual ~PCA(){};
    
    void first_k_ONB(const int k);
    void sole_k_ONB(const int k);
    
    VectorXf Proj2LowDim(VectorXf OriginalVector);
    VectorXf Reconstruction();
    MatrixXf Whitening();
    
    VectorXf GetOriginalVector(const int k);
    void writeProj(std::string add) const;
    
private:
    
    MatrixXf OriginalData;
    MatrixXf ONB;
    
    VectorXf EigenVector;
    
    VectorXf Average;
    VectorXf proj;

    const int n_Samples;
    const int n_Dimensions;
};

#endif
