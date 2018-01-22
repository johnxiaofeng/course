#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdlib>

namespace Constants
{
    // This value assumes the model presented in the classroom is used.
    //
    // It was obtained by measuring the radius formed by running the vehicle in the
    // simulator around in a circle with a constant steering angle and velocity on a
    // flat terrain.
    //
    // Lf was tuned until the the radius formed by the simulating the model
    // presented in the classroom matched the previous radius.
    //
    // This is the length from front to CoG that has a similar radius.
    static constexpr double Lf = 2.67;

    // The reference velocity
    static constexpr double REFERENCE_V = 80.0;

    static constexpr size_t N = 10u;
    static constexpr double DT = 0.1;

    static constexpr size_t STATE_DIM = 6u;
    static constexpr size_t ACTUATION_DIM = 2u;

    // Set the number of model variables (includes both states and inputs).
    // For example: If the state is a 4 element vector, the actuators is a 2
    // element vector and there are 10 timesteps. The number of variables is:
    // 4 * 10 + 2 * 9
    static constexpr size_t NUM_VARIABLES = Constants::STATE_DIM * Constants::N + Constants::ACTUATION_DIM * (Constants::N - 1);
    static constexpr size_t NUM_CONSTRAINTS = Constants::STATE_DIM * Constants::N;

    static constexpr size_t X_START = 0u;
    static constexpr size_t Y_START = X_START + N;
    static constexpr size_t PSI_START = Y_START + N;
    static constexpr size_t V_START = PSI_START + N;
    static constexpr size_t CTE_START = V_START + N;
    static constexpr size_t EPSI_START = CTE_START + N;
    static constexpr size_t DELTA_START = EPSI_START + N;
    static constexpr size_t A_START = DELTA_START + N - 1;
}

#endif //CONSTANTS_H