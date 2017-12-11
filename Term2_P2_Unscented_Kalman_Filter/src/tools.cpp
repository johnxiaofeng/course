#include <iostream>
#include "tools.h"

Vector Tools::CalculateRMSE(const Vectors &estimations, const Vectors &groundTruth)
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

bool Tools::DoubleEqual(double left, double right, double delta)
{
    return fabs(left - right) < delta;
}

double Tools::NormalizeAngleInRadius(double angleInRadius)
{
    while (angleInRadius > M_PI)
    {
        angleInRadius -= 2.*M_PI;
    }

    while (angleInRadius < -M_PI)
    {
        angleInRadius += 2.*M_PI;
    }

    return angleInRadius;
}