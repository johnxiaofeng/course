#include "Helpers.h"
#include "Eigen-3.3/Eigen/QR"
#include <iostream>

namespace Helpers
{
    double polyeval(const Eigen::VectorXd& coeffs, double x)
    {
        double result = 0.0;

        const int numCoeffs = coeffs.size();
        for (int index = 0; index < numCoeffs; index++) 
        {
            result += coeffs[index] * pow(x, index);
        }
        return result;
    }

    Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order)
    {
        assert(xvals.size() == yvals.size());
        assert(order >= 1 && order <= xvals.size() - 1);
        Eigen::MatrixXd A(xvals.size(), order + 1);

        for (int i = 0; i < xvals.size(); i++)
        {
            A(i, 0) = 1.0;
        }

        for (int j = 0; j < xvals.size(); j++)
        {
            for (int i = 0; i < order; i++)
            {
                A(j, i + 1) = A(j, i) * xvals(j);
            }
        }

        auto Q = A.householderQr();
        auto result = Q.solve(yvals);
        return result;
    }

    double getPsiTarget(const Eigen::VectorXd& coeffs, double x)
    {
        double tangent = 0.0;

        const int numCoeffs = coeffs.size();
        assert(numCoeffs >= 2);
        for (int index = 1; index < numCoeffs; index++)
        {
            tangent += static_cast<double>(index) * coeffs[index] * pow(x, index - 1);
        }

        return atan(tangent);
    }

    double clamp(double value, double min, double max)
    {
        assert(min < max);
        if (value > max)
        {
            value = max;
        }
        else if (value < min)
        {
            value = min;
        }

        return value;
    }

    void convertToCarCoordinate(
        double carX,
        double carY,
        double carPsi,
        double x,
        double y,
        double& convertedX,
        double& convertedY)
    {
        const double diffX = x - carX;
        const double diffY = y - carY;

        convertedX = diffX * cos(-carPsi) - diffY * sin(-carPsi);
        convertedY = diffX * sin(-carPsi) + diffY * cos(-carPsi);
    }
}