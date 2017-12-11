#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "types.h"

class UKF
{
public:
    UKF();
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(const MeasurementPackage& measurementPackage);

    /**
     * @brief get the state vector from the UKF
     */
    const Vector& GetState() const;

private:
    /**
    * Initialize the state and covariance matrix with the first measurement.
    * @param measurementPackage, the first measurement received
    */
    void Initialize(const MeasurementPackage& measurementPackage);

    /**
     * Predicts sigma points, the state, and the state covariance matrix.
     * @param deltaTimeInSecond, the change in time (in seconds) between the last
     * measurement and this one.
     */
    void Prediction(double deltaTimeInSecond);

    /**
     * perform the update step for the measurementPackage
     * @param measurementPackage, the measurement received
     */
    void Update(const MeasurementPackage& measurementPackage);

    /**
     * Updates the state and the state covariance matrix using a lidar measurement
     * @param measurementPackage, the lidar measurement received
     */
    void UpdateLidar(const MeasurementPackage& measurementPackage);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param measurementPackage, the radar measurement received
     */
    void UpdateRadar(const MeasurementPackage& measurementPackage);

    /**
     * Generate Augmented Sigma Points
     * @param augmentedSigmaPoints, the augmented sigma points generated
     */
    void GenerateAugmentedSigmaPoints(Matrix& augmentedSigmaPoints);

    /**
     * Predict Sigma Points
     * @param deltaTimeInSecond, the time step in seconds
     * @param sigmaPoints, the sigma points generated
     * @param predSigmaPoints, the predicted sigma points
     */
    void PredictSigmaPoints(double deltaTimeInSecond, const Matrix& sigmaPoints, Matrix& predSigmaPoints);

    /**
     * Predict mean and covariance of Sigma Points
     * @param predSigmaPoints, the predicted sigma points
     * @param mean, the state mean predicted
     * @param covariance, the state covariance predicted
     */
    void PredictStateMeanAndCovariance(const Matrix& predSigmaPoints, Vector& mean, Matrix& covariance);

    /**
     * Predict Radar measurement
     * @param predMeasurement, predicted radar measurement
     * @param covariance, the covariance matrix of the prediction
     * @param sigmaPointsInMeasurementSpace, The sigma points in radar measurement space
     */
    void PredictRadarMeasurement(Vector& predMeasurement, Matrix& covariance, Matrix& sigmaPointsInMeasurementSpace);

    /**
     * Update the state and state covariance from the radar measurement
     * @param predMeasurement, the predicted radar measurement
     * @param covariance, the covariance matrix of the prediction
     * @param sigmaPointsInMeasurementSpace, the sigma points transformed to the radar measurement space
     * @param measurement, the radar measurement received
     */
    void UpdateStateFromRadarMeasurement(const Vector& predMeasurement, const Matrix& covariance, const Matrix& sigmaPointsInMeasurementSpace, const Vector& measurement);

    /**
     * Predict Lidar measurement
     * @param predMeasurement, predicted lidar measurement
     * @param covariance, the covariance matrix of the prediction
     * @param sigmaPointsInMeasurementSpace, The sigma points in lidar measurement space
     */
    void PredictLidarMeasurement(Vector& predMeasurement, Matrix& covariance, Matrix& sigmaPointsInMeasurementSpace);

    /**
     * Update the state and state covariance from the lidar measurement
     * @param predMeasurement, the predicted lidar measurement
     * @param covariance, the covariance matrix of the prediction
     * @param sigmaPointsInMeasurementSpace, the sigma points transformed to the lidar measurement space
     * @param measurement, the lidar measurement received
     */
    void UpdateStateFromLidarMeasurement(const Vector& mean, const Matrix& covariance, const Matrix& sigmaPointsInMeasurementSpace, const Vector& measurement);

    /**
     * Update the NIS value for Radar
     * @param predMeasurement, the predicted radar measurement
     * @param covariance, the covariance matrix
     * @param measurement, the radar measurement received
     */
    void UpdateRadarNIS(const Vector& predMeasurement, const Matrix& covariance, const Vector& measurement);

    /**
     * Update the NIS value for Lidar
     * @param predMeasurement, the predicted lidar measurement
     * @param covariance, the covariance matrix
     * @param measurement, the lidar measurement received
     */
    void UpdateLidarNIS(const Vector& predMeasurement, const Matrix& covariance, const Vector& measurement);

    /**
     * Transform the sigma points to radar measurement space
     * @param sigmaPoints, the sigma points to be transformed
     * @param sigmaPointsInMeasurementSpace, the transformed sigma points in the radar measurement space
     */
    static void TransformSigmaPointsToRadarMeasurementSpace(const Matrix& sigmaPoints, Matrix& sigmaPointsInMeasurementSpace);

    /**
     * Transform the sigma points to lidar measurement space
     * @param sigmaPoints, the sigma points to be transformmed
     * @param sigmaPointsInMeasurementSpace, the transformed sigma points in the lidar measurement space
     */
    static void TransformSigmaPointsToLidarMeasurementSpace(const Matrix& sigmaPoints, Matrix& sigmaPointsInMeasurementSpace);

    /**
     * Get the mean of the given sigma points
     * @param sigmaPoints, the input sigma points
     * @param weights, the weights used to calculate the mean
     * @param mean, the mean of the sigma points
     */
    static void GetSigmaPointsMean(const Matrix& sigmaPoints, const Vector& weights, Vector& mean);

    /**
     * Calculate the NIS given the predicted measurement, covariance and the measurement received
     * @param predMeasurement, the predicted measurement
     * @param covariance, the covariance of the predicted measurement
     * @param measurement, the measurement received
     */
    static double CalculateNIS(const Vector& predMeasurement, const Matrix& covariance, const Vector& measurement);

private:
    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool m_initialized;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool m_laserUsed;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool m_radarUsed;

    ///* time when the state is true, in us
    long long m_prevTimestamp;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double m_stdAccelaration;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double m_stdYawdd;

    ///* Weights of sigma points
    Vector m_sigmaPointWeights;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Vector m_state;

    ///* state covariance matrix
    Matrix m_stateCovariance;

    ///* predicted sigma points matrix
    Matrix m_predSigmaPoints;

    ///* counter for the number of radar measurements
    int m_radarMeasurmentCount;

    ///* counter for the number of radar measurements out of the NIS limitation for 95%
    int m_radarMeasurmentNotSatisfyingCount;

    ///* counter for the number of lidar measurements
    int m_lidarMeasurementCount;

    ///* counter for the number of lidar measurements out of the NIS limitation for 95%
    int m_lidarMeasurmentNotSatisfyingCount;
};

#endif /* UKF_H */
