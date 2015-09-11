/*******************************************************************
 *  Copyright(c) 2015
 *  All rights reserved.
 *
 *  Name: Principle Component Analysis
 *  Description: A method mapping the data from high dimension to the low
 *
 *  Date: 2015-9-7
 *  Author: Yang
 *  Instruction: The Original matrix consists n d-dimensional vectors
 *  where i-th row represents (x_i)^T
 
 ******************************************************************/
#include "PCA.h"

PCA::PCA(std::string add, const int cols, const int rows):
        n_Samples(rows), n_Dimensions(cols)
{
    /*
     *  Description:
     *  A constructed function to load the original data and make
     *  the mean to be zero
     *
     *  @param add: string address to load .csv file
     *  @param cols: data matrix cols
     *  @param rows: data matrix rows
     *
     */
    
    OriginalData = Eigen::MatrixXf(rows, cols);
    LoadMatrix(OriginalData, add, cols, rows);
    
    /* Make the vector mean to be zero */
    /* For natural image, there is no need to make the covariance diagonal*/
    Average = OriginalData.row(0);
    for (int i = 1; i < rows; i++)
    {
        Average += OriginalData.row(i);
    }
    Average /= rows;
    
    VectorXf ones(rows);
    ones.setOnes();
    OriginalData -= ones * Average.transpose();
}

void PCA::first_k_ONB(const int k)
{
    /*
     *  Description:
     *  Find the first k largest eigen values and the corresponding
     *  eigen vector
     *
     *  @param k: first k eigen vector
     *
     */
    
    if (k > n_Dimensions || k > n_Samples)
    {
        std::cerr << "Too hign dimensions\n";
    }
    
    Eigen::JacobiSVD<MatrixXf> svd(OriginalData, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    EigenVector = svd.singularValues();
    std::cout << "The accuracy is " << EigenVector.head(k).sum() / EigenVector.sum() * 1.0 << std::endl;
    
    ONB =  svd.matrixV().block(0, 0, n_Dimensions, k);
}

void PCA::sole_k_ONB(const int k)
{
    /*
     *  Description:
     *  Find the k-th eigen vector
     *
     *  @param k: k-th eigen vector
     *
     */

    if (k > n_Dimensions || k > n_Samples)
    {
        std::cerr << "Too hign dimensions\n";
    }
    Eigen::JacobiSVD<MatrixXf> svd(OriginalData, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    EigenVector = svd.singularValues();
    std::cout << "The accuracy is " << EigenVector(k - 1) / EigenVector.sum() * 1.0 << std::endl;
    
    ONB =  svd.matrixV().block(0, k - 1, n_Dimensions, 1);
}

VectorXf PCA::Proj2LowDim(VectorXf OriginalVector)
{
    /*
     *  Description:
     *  Mapping the original vector to a lower space
     *
     *  @param OriginalVector: Could be one of the sample or test
     *                          set
     */
    
    proj = ONB.transpose() * OriginalVector;
    return proj;
}

VectorXf PCA::Reconstruction()
{
    /*
     *  Description:
     *  Mapping the projected vector back to the original space
     *
     */
    return ONB * proj + Average;
}

MatrixXf PCA::Whitening()
{
    /*
     *  Description:
     *  Eliminate the correlation between the data
     *
     */
    
    first_k_ONB(n_Dimensions);

    MatrixXf whitenData(OriginalData.rows(), OriginalData.cols());
    for (int i = 0; i < n_Samples; i++)
    {
        for (int j = 0; j < n_Dimensions; j++)
        {
            whitenData(i, j) = (Proj2LowDim(OriginalData.row(i)))(j) / sqrt(EigenVector(j));
        }
    }
    
    return whitenData;
}

VectorXf PCA::GetOriginalVector(const int k)
{
    /*
     *  Description:
     *  Get the k-th original vector
     *
     */
    assert(k < n_Samples);
    return OriginalData.row(k);
}

void PCA::writeProj(std::string add) const
{
    /*
     *  Description:
     *  Write the projected space on files
     */
    std::ofstream f(add);
    
    for (int i = 0; i < proj.size() - 1; i++)
    {
        f << proj(i) << ',';
    }
    f << proj(proj.size() - 1) << '\n';
    f.close();
}