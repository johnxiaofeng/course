#ifndef TOOLS_H_
#define TOOLS_H_

#include <cmath>
#include "types.h"

class Tools
{
public:
    /**
    * A helper method to calculate RMSE.
    */
    static Vector CalculateRMSE(const Vectors &estimations, const Vectors &groundTruth);

    /**
    * A helper method to calculate Jacobians.
    */
    static Matrix CalculateJacobian(const Vector &x_state);

    /**
     * A helper method to convert polor coordinate to cartesian coordinate
     */
    static Vector PolorToCartesian(float ro, float theta, float roDot);

    /**
     * A helper method to convert cartesian coordinate to polor coordinate
     */
    static Vector CartesianToPolor(float px, float py, float vx, float vy);

    /**
     * A helper method to compare if two float values are equal
     */
    static bool FloatEqual(float left, float right, float delta = 0.00001f);

    /**
     * A helper function to clamp angle to [-PI, PI]
     */
    static float ClampRadius(float radius, float low = -M_PI, float high = M_PI);
};

#endif //TOOLS_H_
