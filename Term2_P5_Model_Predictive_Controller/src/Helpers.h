#ifndef HELPERS_H
#define HELPERS_H_

#include <cmath>
#include "Eigen-3.3/Eigen/Core"

namespace Helpers
{
    constexpr double pi()
    {
        return M_PI;
    }

    /**
     * @brief convert from degree to radius
     * @param x, input degree value 
     */
    inline double deg2rad(double x)
    {
        return x * pi() / 180;
    }

    /**
     * @brief convert from radius to degree
     * @param x, input radius value 
     */
    inline double rad2deg(double x)
    {
        return x * 180 / pi();
    }

    /**
     * @brief evaluate a polynomial value given coefficient and x value
     * @param coeffs, the coefficients of the polynomial curve
     * @param x, the given x value
     */
    double polyeval(const Eigen::VectorXd& coeffs, double x);

    /**
     * @brief Fit a polynomial given several points. Adapted from
     * https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
     * @param xvals, the x values of the points
     * @param yvals, the y values of the points
     */
    Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);

    /**
     * @brief get the radius value of the tangent at the given x of the curve specified by coeffs
     * @param coeffs, the coefficients specifying the curve
     * @param x, the x value of the point 
     */
    double getPsiTarget(const Eigen::VectorXd& coeffs, double x);

    /**
     * @brief clamp the value to be in [min, max]
     */
    double clamp(double value, double min, double max);

    /**
     * @brief convert a global coordinate to car coordinate
     * @param carX, the x coordinate of the car in global coordinate
     * @param carY, the y coordinate of the car in global coordinate
     * @param carPsi, the psi of the car in global coordinate
     * @param x, the x coordinate in global coordinate to be converted
     * @param y, the y coordinate in global coordinate to be converted
     * @param carCoordinateX, the converted x value in car coordinate
     * @param carCoordinateY, the converted y value in car coordinate
     */
    void convertToCarCoordinate(
        double carX,
        double carY,
        double carPsi,
        double x,
        double y,
        double& convertedX,
        double& convertedY);

} //namespace Helpers

#endif //HELPERS_H_
