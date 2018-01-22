#include "PID.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <fstream>
#include <numeric>

namespace
{
    const int WAITING_WINDOW_SIZE = 100;
    const int TUNING_WINDOW_SIZE = 2000;
    const double TUNING_TOLERANCE = 0.0002;
}

PID::PID()
{
    reset();
    createResultFile();
}

void PID::Init(double Kp, double Ki, double Kd)
{
    // set tau parameters
    m_parameters[0] = Kp;
    m_parameters[1] = Ki;
    m_parameters[2] = Kd;

    // set delta to be 0.1 of the initial tau parameters
    m_deltaParameters[0] = 0.1 * Kp;
    m_deltaParameters[1] = 0.1 * Ki;
    m_deltaParameters[2] = 0.1 * Kd;
}

void PID::UpdateError(double cte)
{
    // set previous error the current cte if it is the first update
    if (m_isFirstUpdate)
    {
        m_prevCte = cte;
        m_isFirstUpdate = false;
    }

    // update the p error
    m_pError = -m_parameters[0] * cte;

    // updat the total error and i error
    m_totalError += cte;
    m_iError = -m_parameters[1] * m_totalError;

    // update the d error
    m_dError = -m_parameters[2] * (cte - m_prevCte);

    // update the previous cte value
    m_prevCte = cte;
}

float PID::getDError() const
{
    return m_dError;
}

double PID::GetValue(double min, double max) const
{
    // get the total error
    double totalError = m_pError + m_iError + m_dError;

    // clamp the total to [min, max]
    if (totalError > max)
    {
        totalError = max;
    }
    else if (totalError < min)
    {
        totalError = min;
    }

    return totalError;
}

void PID::UpdateParameters(double cte)
{
    m_numCteCollected++;

    // skip those cte for the waiting phase, wait for the new parameters to take effect
    if (m_numCteCollected <= WAITING_WINDOW_SIZE)
    {
        return;
    }

    // start updating after the waiting phase
    m_totalTuningError += cte * cte;
    std::cout << "# " << m_numCteCollected - WAITING_WINDOW_SIZE << std::endl;

    // when the number of cte received reaches TUNING_WINDOW_SIZE, trigger tuning.
    if (m_numCteCollected >= WAITING_WINDOW_SIZE + TUNING_WINDOW_SIZE)
    {
        double error = m_totalTuningError / static_cast<double>(TUNING_WINDOW_SIZE);
        twiddleParameters(error);
        writeParameterInfo(error);

        // reset variables for next tuning
        m_numCteCollected = 0;
        m_totalTuningError = 0.0;
    }
}

void PID::twiddleParameters(double error)
{
    if (sumDeltaParameters() < TUNING_TOLERANCE)
    {
        return;
    }

    if (!m_isTuningStarted)
    {
        m_isTuningStarted = true;

        updateBestParameters(error);
        m_parameters[m_currentTuningParameterIndex] += m_deltaParameters[m_currentTuningParameterIndex];
        reset();
        return;
    }

    if (m_isTuningIncresingError)
    {
        m_isTuningIncresingError = false;
        if (error < m_tuningBestError)
        {
            updateBestParameters(error);
            m_deltaParameters[m_currentTuningParameterIndex] *= 1.1;
        }
        else
        {
            m_parameters[m_currentTuningParameterIndex] += m_deltaParameters[m_currentTuningParameterIndex];
            m_deltaParameters[m_currentTuningParameterIndex] *= 0.9;
        }

        incrementCurrentTuningParameterIndex();
        m_parameters[m_currentTuningParameterIndex] += m_deltaParameters[m_currentTuningParameterIndex];
        reset();
        return;
    }

    if (error < m_tuningBestError)
    {
        updateBestParameters(error);
        m_deltaParameters[m_currentTuningParameterIndex] *= 1.1;

        incrementCurrentTuningParameterIndex();
        m_parameters[m_currentTuningParameterIndex] += m_deltaParameters[m_currentTuningParameterIndex];
        reset();
        return;
    }
    else
    {
        m_parameters[m_currentTuningParameterIndex] -= 2.0 * m_deltaParameters[m_currentTuningParameterIndex];
        m_isTuningIncresingError = true;
        reset();
        return;
    }
}

void PID::incrementCurrentTuningParameterIndex()
{
    m_currentTuningParameterIndex++;
    if (m_currentTuningParameterIndex >= static_cast<int>(m_parameters.size()))
    {
        m_currentTuningParameterIndex = 0;
    }
}

void PID::reset()
{
    m_isFirstUpdate = true;
    m_totalError = 0.0;
    m_pError = 0.0;
    m_iError = 0.0;
    m_dError = 0.0;
    m_prevCte = 0.0;
}

double PID::sumDeltaParameters() const
{
    return std::accumulate(m_deltaParameters.begin(), m_deltaParameters.end(), 0.0);
}

void PID::createResultFile()
{
    std::ofstream parameterfile;
    parameterfile.open("parameter.txt", std::ios::out);
    parameterfile << "Parameter Tuning Results: " << std::endl;
}

void PID::writeParameterInfo(double error)
{
    std::ofstream parameterfile;
    parameterfile.open("parameter.txt", std::ios::out | std::ios::app);
    parameterfile << "Current error is: " << error << ", Best Error = " << m_tuningBestError << std::endl;
    parameterfile << "Parameter: p = " << m_parameters[0] << ", i = " << m_parameters[1] << ", d = " << m_parameters[2] << std::endl;
    parameterfile << "Delta Param: p = " << m_deltaParameters[0] << ", i = " << m_deltaParameters[1] << ", d = " << m_deltaParameters[2] << std::endl;
    parameterfile << "Best Parameter is : p = " << m_bestParameters[0] << ", i = " << m_bestParameters[1] << ", d = " << m_bestParameters[2] << std::endl << std::endl;
}

void PID::updateBestParameters(double newError)
{
    m_tuningBestError = newError;
    m_bestParameters = m_parameters;
}
