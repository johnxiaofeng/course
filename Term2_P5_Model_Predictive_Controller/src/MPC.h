#ifndef MPC_H
#define MPC_H

#include <vector>
#include <cppad/cppad.hpp>
#include "Eigen-3.3/Eigen/Core"

class MPC
{
public:
    /**
     * @brief solve the model given an initial state and polynomial coefficients. It returns the first actuatotions as well as the
     * predicted way points
     * @param state, the initial state
     * @param coeffs, the polynomial coefficients of the reference way points
     * @param delta, the first steering value of the first actuatotion
     * @param a, the first throttle value of the first actuatotion
     */
    static std::vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs, double& delta, double& a);

private:
    using Dvector = CPPAD_TESTVECTOR(double);

    static void initVariables(const Eigen::VectorXd& state, Dvector& variables);
    static void setVariablesBounds(Dvector& lowerBound, Dvector& upperBound);
    static void setConstraintsBounds(const Eigen::VectorXd& state, Dvector& lowerBound, Dvector& upperBound);
    static std::string getIPOPTSolverOptions();
};

#endif /* MPC_H */
