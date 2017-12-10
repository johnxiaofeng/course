#ifndef FUSIONEKF_H_
#define FUSIONEKF_H_

#include "kalman_filter.h"
#include "types.h"

struct MeasurementPackage;

class FusionEKF 
{
public:
    FusionEKF();
    virtual ~FusionEKF();

    /**
    * Run the whole flow of the Kalman Filter from here.
    */
    void ProcessMeasurement(const MeasurementPackage &measurement_pack);

    /**
    * Kalman Filter update and prediction math lives in here.
    */
    KalmanFilter ekf_;

private:
    /**
     * Initialize the state ekf_ with the first measurement. 
     */
    void Initialize(const MeasurementPackage& measurement_pack);
    
    /**
     * perform the prediction step
     */
    void Predict(const MeasurementPackage& measurement_pack);
    
    /**
     * perform the update step
     */
    void Update(const MeasurementPackage& measurement_pack);

    /**
     * perform the update step for laser measurement
     */
    void UpdateLaser(const MeasurementPackage& measurement_pack);

    /**
     * perform the update step for radar measurement
     */
    void UpdateRadar(const MeasurementPackage& measurement_pack);

private:
    // check whether the tracking toolbox was initialized or not (first measurement)
    bool is_initialized_;

    // previous timestamp
    long long previous_timestamp_;

    // tool object used to compute Jacobian and RMSE
    Matrix R_laser_;
    Matrix H_laser_;
    Matrix R_radar_;
};

#endif //FUSIONEKF_H_