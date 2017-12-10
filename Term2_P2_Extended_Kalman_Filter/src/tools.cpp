#include <iostream>
#include <cmath>
#include "tools.h"

Vector Tools::CalculateRMSE(const Vectors& estimations, const Vectors& groundTruth) 
{
    Eigen::VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if (estimations.empty() || groundTruth.empty())
    {
        std::cout << "Esitimation or groundTruth is empty." << std::endl;
        return rmse;
    }

    const size_t numEstimations = estimations.size();
    const size_t numGroundTruthData = groundTruth.size();
    if (numEstimations != numGroundTruthData)
    {
        std::cout << "Esitimation or groundTruth are of different size." << std::endl;
        return rmse;
    }

    for (size_t index = 0; index < numEstimations; index++)
    {
        Eigen::VectorXd diffVector = estimations[index] - groundTruth[index];
        Eigen::VectorXd diff = diffVector.array() * diffVector.array();
        rmse += diff;
    }

    rmse = rmse / numEstimations;
    rmse = rmse.array().sqrt();
    return rmse;
}

Matrix Tools::CalculateJacobian(const Vector& state) 
{
    const float px = state(0);
    const float py = state(1);
    const float vx = state(2);
    const float vy = state(3);
    const float pxpxpypy = px * px + py * py;
    const float pxpxpypySqrt = sqrt(pxpxpypy);
    
    // check divided by 0 case
    Eigen::MatrixXd jacobianMatrix(3, 4);
    if (Tools::FloatEqual(pxpxpypy, 0.0f))
    {
        std::cout << "Invalid px and py value" << std::endl;
        return jacobianMatrix;
    }
  
    // fill the jacobian matrix
    jacobianMatrix << px / pxpxpypySqrt, py / pxpxpypySqrt, 0, 0,
                      - py / pxpxpypy, px / pxpxpypy, 0, 0,
                      py * (vx * py - vy * px) / (pxpxpypy * pxpxpypySqrt), px * (vy * px - vx * py) / (pxpxpypy * pxpxpypySqrt), px / pxpxpypySqrt, py / pxpxpypySqrt;
  
    return jacobianMatrix;
}

Vector Tools::PolorToCartesian(float ro, float theta, float roDot)
{
    const float px = ro * cos(theta);      
    const float py = ro * sin(theta);
    const float vx = 0.0f;
    const float vy = 0.0f;

    Vector cartesian(4);
    cartesian << px, py, vx, vy;
    return cartesian;
}

Vector Tools::CartesianToPolor(float px, float py, float vx, float vy)
{    
    const float ro = sqrt(px * px + py * py);
    if (FloatEqual(ro, 0.0f))
    {
        Vector polor(3);
        polor << 0, 0, 0;
        return polor;
    }
    
    const float roDot = (px * vx + py * vy) / ro;

    // calculate theta value, handle the case when px is 0
    float theta = 0.0f;
    if (!FloatEqual(px, 0.0f))
    {
        theta = atan2(py, px);
    }
    else
    {
        if ( py > 0.0)
        {
            theta = M_PI / 2.0f;
        }
        else if (py < 0.0)
        {
            theta = -M_PI / 2.0f;
        }
        else 
        {
            assert(false);
        }
    }
    
    Vector polor(3);
    polor << ro, theta, roDot;
    return polor;
}

bool Tools::FloatEqual(float left, float right, float delta)
{
    return fabs(left - right) < delta;    
}

float Tools::ClampRadius(float radius, float low, float high)
{
    assert(low < high);

    while (radius < low)
    {
        radius += 2 * M_PI;
    }

    while (radius > high)
    {
        radius -= 2 * M_PI;
    }

    return radius;
}
