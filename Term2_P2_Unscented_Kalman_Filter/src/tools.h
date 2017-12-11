#ifndef TOOLS_H_
#define TOOLS_H_

#include "types.h"

class Tools
{
public:
    /**
    * A helper method to calculate RMSE.
    */
    static Vector CalculateRMSE(const Vectors &estimations, const Vectors &groundTruth);

    /**
    * A helper method to compare if two float values are equal
    */
    static bool DoubleEqual(double left, double right, double delta = 0.00001);

    /**
     * Normalize an angle in radius to be in [-M_PI, M_PI]
     */
    static double NormalizeAngleInRadius(double angleInRadius);
};

#endif /* TOOLS_H_ */