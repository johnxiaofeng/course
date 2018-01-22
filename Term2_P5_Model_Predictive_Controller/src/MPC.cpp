#include "MPC.h"
#include "Constants.h"
#include "FGEvaluator.h"

#include <cppad/ipopt/solve.hpp>

std::vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs, double& delta, double& a)
{
    Dvector vars(Constants::NUM_VARIABLES);
    initVariables(state, vars);

    Dvector vars_lowerbound(Constants::NUM_VARIABLES);
    Dvector vars_upperbound(Constants::NUM_VARIABLES);
    setVariablesBounds(vars_lowerbound, vars_upperbound);

    Dvector constraints_lowerbound(Constants::NUM_CONSTRAINTS);
    Dvector constraints_upperbound(Constants::NUM_CONSTRAINTS);
    setConstraintsBounds(state, constraints_lowerbound, constraints_upperbound);

    // object that computes objective and constraints
    FGEvaluator fg_eval(coeffs);

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    std::string options = getIPOPTSolverOptions();
    CppAD::ipopt::solve<Dvector, FGEvaluator>(
        options,
        vars,
        vars_lowerbound,
        vars_upperbound,
        constraints_lowerbound,
        constraints_upperbound,
        fg_eval,
        solution);

    // Check some of the solution values
    if (solution.status != CppAD::ipopt::solve_result<Dvector>::success)
    {
        std::cout << "Fail to solve/n";
        assert(false);
    }

    std::cout << "Cost " << solution.obj_value << std::endl;

    // set the value to actuations
    delta = solution.x[Constants::DELTA_START];
    a = solution.x[Constants::A_START];

    std::vector<double> predictedTrajectory(2 * Constants::N);
    for (size_t index = 0; index < Constants::N; index++)
    {
        predictedTrajectory[index * 2] = solution.x[Constants::X_START + index];
        predictedTrajectory[index * 2 + 1] = solution.x[Constants::Y_START + index];
    }
    return predictedTrajectory;
}

void MPC::initVariables(const Eigen::VectorXd& state, Dvector& variables)
{
    for (size_t index = 0; index < Constants::NUM_VARIABLES; index++)
    {
        variables[index] = 0;
    }

    const double x = state[0];
    const double y = state[1];
    const double psi = state[2];
    const double v = state[3];
    const double cte = state[4];
    const double epsi = state[5];

    variables[Constants::X_START] = x;
    variables[Constants::Y_START] = y;
    variables[Constants::PSI_START] = psi;
    variables[Constants::V_START] = v;
    variables[Constants::CTE_START] = cte;
    variables[Constants::EPSI_START] = epsi;
}

void MPC::setVariablesBounds(Dvector& lowerBound, Dvector& upperBound)
{
    // for state variables, thers is no upper or lower bounds
    for (size_t index = 0; index < Constants::DELTA_START; index++)
    {
        lowerBound[index] = -1.0e19;
        upperBound[index] = 1.0e19;
    }

    // The upper and lower limits of delta are set to -25 and 25 degrees (values in radians).
    for (size_t index = Constants::DELTA_START; index < Constants::A_START; index++)
    {
        lowerBound[index] = -0.436332;
        upperBound[index] = 0.436332;
    }

    // Acceleration/decceleration upper and lower limits.
    for (size_t index = Constants::A_START; index < Constants::NUM_VARIABLES; index++)
    {
        lowerBound[index] = -1.0;
        upperBound[index] = 1.0;
    }
}

void MPC::setConstraintsBounds(const Eigen::VectorXd& state, Dvector& lowerBound, Dvector& upperBound)
{
    // set all bounds to 0 first
    for (size_t index = 0; index < Constants::NUM_CONSTRAINTS; index++)
    {
        lowerBound[index] = 0;
        upperBound[index] = 0;
    }

    const double x = state[0];
    const double y = state[1];
    const double psi = state[2];
    const double v = state[3];
    const double cte = state[4];
    const double epsi = state[5];

    lowerBound[Constants::X_START] = x;
    lowerBound[Constants::Y_START] = y;
    lowerBound[Constants::PSI_START] = psi;
    lowerBound[Constants::V_START] = v;
    lowerBound[Constants::CTE_START] = cte;
    lowerBound[Constants::EPSI_START] = epsi;

    upperBound[Constants::X_START] = x;
    upperBound[Constants::Y_START] = y;
    upperBound[Constants::PSI_START] = psi;
    upperBound[Constants::V_START] = v;
    upperBound[Constants::CTE_START] = cte;
    upperBound[Constants::EPSI_START] = epsi;
}

std::string MPC::getIPOPTSolverOptions()
{
    // NOTE: You don't have to worry about these options
    //
    // options for IPOPT solver
    std::string options;

    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";

    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";

    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          0.5\n";

    return options;
}