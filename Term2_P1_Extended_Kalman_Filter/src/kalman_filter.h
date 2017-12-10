#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "types.h"

class KalmanFilter 
{
public:
    // state vector
    Vector x_;

    // state covariance matrix
    Matrix P_;

    // state transition matrix
    Matrix F_;

    // process covariance matrix
    Matrix Q_;

    // measurement matrix
    Matrix H_;

    // measurement covariance matrix
    Matrix R_;

    KalmanFilter();
    virtual ~KalmanFilter();

    /**
     * Init Initializes Kalman filter
     * @param x_in Initial state
     * @param P_in Initial state covariance
     * @param F_in Transition matrix
     * @param H_in Measurement matrix
     * @param R_in Measurement covariance matrix
     * @param Q_in Process covariance matrix
     */
    void Init(const Vector& x_in, const Matrix& P_in, const Matrix& F_in, const Matrix& H_in, const Matrix& R_in, const Matrix& Q_in);

    /**
     * Prediction Predicts the state and the state covariance
     * using the process model
     */
    void Predict();

    /**
     * Updates the state by using standard Kalman Filter equations
     * @param z The measurement at k+1
     */
    void Update(const Vector &z);

    /**
     * Updates the state by using Extended Kalman Filter equations
     * @param z The measurement at k+1
     */
    void UpdateEKF(const Vector &z);
};

#endif //KALMAN_FILTER_H_
