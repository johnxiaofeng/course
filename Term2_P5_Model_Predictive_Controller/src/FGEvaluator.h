#ifndef FGEVALUATOR_H
#define FGEVALUATOR_H

#include "Eigen-3.3/Eigen/Core"
#include <cppad/cppad.hpp>

class FGEvaluator
{
public:
    using ADvector = CPPAD_TESTVECTOR(CppAD::AD<double>);

    FGEvaluator(const Eigen::VectorXd& coeffs);
    void operator()(ADvector& fg, const ADvector& vars);

private:
    void setupCost(ADvector& fg, const ADvector& vars);
    void setupConstraints(ADvector& fg, const ADvector& vars);

    static CppAD::AD<double> polyEvaluate(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x);
    static CppAD::AD<double> getPsiTarget(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x);

private:
    Eigen::VectorXd m_coeffs;
};

#endif //FGEVALUATOR_H