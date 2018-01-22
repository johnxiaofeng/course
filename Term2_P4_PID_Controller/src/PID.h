#ifndef PID_H
#define PID_H

#include <vector>

class PID
{
public:
    PID();
    ~PID() = default;

    /**
     * @brief initialize the pid controller with given tau parameters
     */
    void Init(double Kp, double Ki, double Kd);

    /**
     * @brief Update the PID error variables given cross track error.
     * @param cte, the cte error got.
     */
    void UpdateError(double cte);

    /**
     * @brief get the value from the PID controller, and clamp the value to [min, max]
     * @return the value got from the controller
     * @param min, the lower end of the value
     * @param max, the upper end of the value
     */
    double GetValue(double min = -1.0, double max = 1.0) const;

    /**
     * @brief update the parameters based on cte value, internally use twiddle to tune the
     * parameters based on feed cte values
     */
    void UpdateParameters(double cte);

    float getDError() const;

private:
    /**
     * @brief output the parameter infomation to a sepearate file, and this will create the file.
     */
    void createResultFile();

    /**
     * @brief write out the parameter inforamtion to the file
     */
    void writeParameterInfo(double error);

    /**
     * @brief helper function to get the sum of delta parameters
     */
    double sumDeltaParameters() const;

    /**
     * @brief implementation of the twiddle algorithm to update the parameters based on error
     */
    void twiddleParameters(double error);

    /**
     * @brief proceed to the next paramter in the twiddle process
     */
    void incrementCurrentTuningParameterIndex();

    /**
     * @brief update best parameters info, update the best error and also the best parameters
     */
    void updateBestParameters(double newError);

    /**
     * @brief reset the PID varaibles to use new parameters
     */
    void reset();

private:
    // variables for PID filtering
    double m_prevCte;
    double m_pError;
    double m_iError;
    double m_dError;
    double m_totalError;
    bool m_isFirstUpdate;

    // variables for PID parameter tuning
    // contains the tauP, tauI and tauD values in the parameters vector
    std::vector<double> m_parameters = {0.0, 0.0, 0.0};
    std::vector<double> m_deltaParameters = {0.0, 0.0, 0.0};
    std::vector<double> m_bestParameters = {0.0, 0.0, 0.0};

    bool m_isTuningStarted = {false};
    bool m_isTuningIncresingError = {false};
    double m_tuningBestError = {0.0};

    int m_currentTuningParameterIndex = {0};
    int m_numCteCollected = {0};
    double m_totalTuningError = {0.0f};
};

#endif /* PID_H */
