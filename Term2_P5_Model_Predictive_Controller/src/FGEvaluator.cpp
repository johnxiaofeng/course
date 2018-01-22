#include "FGEvaluator.h"
#include "Constants.h"
#include "Helpers.h"

FGEvaluator::FGEvaluator(const Eigen::VectorXd& coeffs)
: m_coeffs(coeffs)
{

}

void FGEvaluator::operator()(ADvector& fg, const ADvector& vars)
{
    setupCost(fg, vars);
    setupConstraints(fg, vars);
}

void FGEvaluator::setupCost(ADvector& fg, const ADvector& vars)
{
    fg[0] = 0;

    // The part of the cost based on the reference state.
    for (size_t t = 0; t < Constants::N; t++)
    {
        fg[0] += CppAD::pow(vars[Constants::CTE_START + t], 2);
        fg[0] += CppAD::pow(vars[Constants::EPSI_START + t], 2);
        fg[0] += CppAD::pow(vars[Constants::V_START + t] - Constants::REFERENCE_V, 2);
    }

    // Minimize the use of actuators.
    for (size_t t = 0; t < Constants::N - 1; t++)
    {
        fg[0] += CppAD::pow(vars[Constants::DELTA_START + t], 2);
        fg[0] += CppAD::pow(vars[Constants::A_START + t], 2);
        fg[0] += 20 * CppAD::pow(vars[Constants::DELTA_START + t] * vars[Constants::V_START + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (size_t t = 0; t < Constants::N - 2; t++)
    {
        fg[0] += 1000.0 * CppAD::pow(vars[Constants::DELTA_START + t + 1] - vars[Constants::DELTA_START + t], 2);
        fg[0] += 10.0 * CppAD::pow(vars[Constants::A_START + t + 1] - vars[Constants::A_START + t], 2);
    }
}

void FGEvaluator::setupConstraints(ADvector& fg, const ADvector& vars)
{
    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + Constants::X_START] = vars[Constants::X_START];
    fg[1 + Constants::Y_START] = vars[Constants::Y_START];
    fg[1 + Constants::PSI_START] = vars[Constants::PSI_START];
    fg[1 + Constants::V_START] = vars[Constants::V_START];
    fg[1 + Constants::CTE_START] = vars[Constants::CTE_START];
    fg[1 + Constants::EPSI_START] = vars[Constants::EPSI_START];

    for (size_t t = 1; t < Constants::N; t++)
    {
        CppAD::AD<double> x0 = vars[Constants::X_START + t - 1];
        CppAD::AD<double> y0 = vars[Constants::Y_START + t - 1];
        CppAD::AD<double> psi0 = vars[Constants::PSI_START + t - 1];
        CppAD::AD<double> v0 = vars[Constants::V_START + t - 1];
        CppAD::AD<double> cte0 = vars[Constants::CTE_START + t - 1];
        CppAD::AD<double> epsi0 = vars[Constants::EPSI_START + t - 1];
        CppAD::AD<double> delta0 = vars[Constants::DELTA_START + t - 1];
        CppAD::AD<double> a0 = vars[Constants::A_START + t - 1];

        CppAD::AD<double> f0 = polyEvaluate(m_coeffs, x0);
        CppAD::AD<double> psides0 = getPsiTarget(m_coeffs, x0);

        CppAD::AD<double> x1 = vars[Constants::X_START + t];
        CppAD::AD<double> y1 = vars[Constants::Y_START + t];
        CppAD::AD<double> psi1 = vars[Constants::PSI_START + t];
        CppAD::AD<double> v1 = vars[Constants::V_START + t];
        CppAD::AD<double> cte1 = vars[Constants::CTE_START + t];
        CppAD::AD<double> epsi1 = vars[Constants::EPSI_START + t];

        fg[1 + Constants::X_START + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * Constants::DT);
        fg[1 + Constants::Y_START + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * Constants::DT);
        fg[1 + Constants::PSI_START + t ] = psi1 - (psi0 + v0 * delta0/ Constants::Lf * Constants::DT);
        fg[1 + Constants::V_START + t] = v1 - (v0 + a0 * Constants::DT);
        fg[1 + Constants::CTE_START + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * Constants::DT));
        fg[1 + Constants::EPSI_START + t] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Constants::Lf * Constants::DT);
     }
}

CppAD::AD<double> FGEvaluator::polyEvaluate(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x)
{
    CppAD::AD<double> result = 0.0;

    const int numCoeffs = coeffs.size();
    for (int index = 0; index < numCoeffs; index++)
    {
        result += coeffs[index] * CppAD::pow(x, index);
    }
    return result;
}

CppAD::AD<double> FGEvaluator::getPsiTarget(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x)
{
    CppAD::AD<double> tangent = 0.0;

    const int numCoeffs = coeffs.size();
    assert(numCoeffs >= 2);
    for (int index = 1; index < numCoeffs; index++)
    {
        tangent += static_cast<double>(index) * coeffs[index] * CppAD::pow(x, index - 1);
    }

    return CppAD::atan(tangent);
}