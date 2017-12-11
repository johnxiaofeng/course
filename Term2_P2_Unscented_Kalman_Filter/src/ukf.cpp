#include "ukf.h"
#include "tools.h"
#include <iostream>

namespace
{
    const int STATE_DIMENSION = 5;
    const int AUG_STATE_DIMENSION = 7;

    // radar measurement: r, phi, rDot
    const int RADAR_MEASUREMENT_DIMENSION = 3;

    // lidar measurement: px, py
    const int LIDAR_MEASUREMENT_DIMENSION = 2;

    // sigma point spreading parameter
    const double LAMBDA = 3 - AUG_STATE_DIMENSION;

    // Laser measurement noise standard deviation position1 in m
    const double STD_LASER_PX = 0.15;

    // Laser measurement noise standard deviation position2 in m
    const double STD_LASER_PY = 0.15;

    // Radar measurement noise standard deviation radius in m
    const double STD_RADAR_R = 0.3;

    // Radar measurement noise standard deviation angle in rad
    const double STD_RADAR_PHI = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    const double STD_RADAR_RD = 0.3;
}

UKF::UKF()
: m_initialized(false)
, m_laserUsed(true)
, m_radarUsed(true)
, m_prevTimestamp(0)
, m_sigmaPointWeights(2 * AUG_STATE_DIMENSION + 1)
, m_predSigmaPoints(STATE_DIMENSION, 2 * AUG_STATE_DIMENSION + 1)
, m_radarMeasurmentCount(0)
, m_radarMeasurmentNotSatisfyingCount(0)
, m_lidarMeasurementCount(0)
, m_lidarMeasurmentNotSatisfyingCount(0)
{
    m_stdAccelaration = 1.0; // Process noise standard deviation longitudinal acceleration in m/s^2
    m_stdYawdd = 0.5; // Process noise standard deviation yaw acceleration in rad/s^2

    m_state = Vector::Zero(STATE_DIMENSION);
    m_stateCovariance = Matrix::Zero(STATE_DIMENSION, STATE_DIMENSION);

    // initialize sigma points weights
    m_sigmaPointWeights(0) = LAMBDA / (LAMBDA + AUG_STATE_DIMENSION);
    for (int i = 1; i < 2 * AUG_STATE_DIMENSION + 1; i++)
    {
        m_sigmaPointWeights(i) = 0.5 / (AUG_STATE_DIMENSION + LAMBDA);
    }
}

UKF::~UKF()
{

}

void UKF::ProcessMeasurement(const MeasurementPackage& measurementPackage)
{
    if (!m_initialized)
    {
        Initialize(measurementPackage);
        return;
    }

    MeasurementPackage::SensorType type = measurementPackage.sensor_type_;
    if (type == MeasurementPackage::LASER && !m_laserUsed)
    {
        return;
    }
    else if (type == MeasurementPackage::RADAR && !m_radarUsed)
    {
        return;
    }

    const double deltaTimeInSecond = (measurementPackage.timestamp_ - m_prevTimestamp) / 1000000.0;
    Prediction(deltaTimeInSecond);
    Update(measurementPackage);

    m_prevTimestamp = measurementPackage.timestamp_;
}

const Vector& UKF::GetState() const
{
    return m_state;
}

void UKF::Initialize(const MeasurementPackage& measurement_pack)
{
    const float initialV = 1.0f;
    const float initialPhi = 0.0f;
    const float initialPhiDot = 0.0f;

    // create covariance matrix
    m_stateCovariance.setIdentity();

    // initialize the state ekf_x_ with first measurement
    switch (measurement_pack.sensor_type_)
    {
        case MeasurementPackage::RADAR:
        {
            // convert radar from polor to cartesian coordinates and initialize state
            const float ro = measurement_pack.raw_measurements_[0];
            const float theta = measurement_pack.raw_measurements_[1];
            const float roDot = measurement_pack.raw_measurements_[2];

            const float px = ro * cos(theta);
            const float py = ro * sin(theta);
            m_state << px, py, initialV, initialPhi, initialPhiDot;
            break;
        }
        case MeasurementPackage::LASER:
        {
            // initialie state from laser measurement
            const float px = measurement_pack.raw_measurements_[0];
            const float py = measurement_pack.raw_measurements_[1];
            m_state << px, py, initialV, initialPhi, initialPhiDot;

            m_stateCovariance(0, 0) = 0.7;
            m_stateCovariance(1, 1) = 0.7;
            m_stateCovariance(2, 2) = 3.0;
            m_stateCovariance(3, 3) = 3.0;
            m_stateCovariance(4, 4) = 3.0;
            break;
        }
        default:
        {
            // non supported sensor type
            std::cout << "Sensor Type is not supported" << std::endl;
            assert(false);
            return;
        }
    }

    // update the previous time stamp
    m_prevTimestamp = measurement_pack.timestamp_;

    // done initializing
    m_initialized = true;
}

void UKF::Prediction(double deltaTimeInSecond)
{
    // generate sigma points with augmentation
    Matrix augmentedSigmaPoints;
    GenerateAugmentedSigmaPoints(augmentedSigmaPoints);

    // predict sigma points
    PredictSigmaPoints(deltaTimeInSecond, augmentedSigmaPoints, m_predSigmaPoints);

    // use predicted sigma points to predict state mean and covariance
    PredictStateMeanAndCovariance(m_predSigmaPoints, m_state, m_stateCovariance);
}

void UKF::GenerateAugmentedSigmaPoints(Matrix& sigmaPoints)
{
    //create augmented mean vector
    Vector x_aug = Vector::Zero(AUG_STATE_DIMENSION);
    x_aug.head(STATE_DIMENSION) = m_state;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented state covariance
    Matrix P_aug = Matrix::Zero(AUG_STATE_DIMENSION, AUG_STATE_DIMENSION);
    P_aug.topLeftCorner(STATE_DIMENSION, STATE_DIMENSION) = m_stateCovariance;
    P_aug(STATE_DIMENSION, STATE_DIMENSION) = m_stdAccelaration * m_stdAccelaration;
    P_aug(STATE_DIMENSION + 1, STATE_DIMENSION + 1) = m_stdYawdd * m_stdYawdd;

    //create sigma point matrix
    sigmaPoints = Matrix::Zero(AUG_STATE_DIMENSION, 2 * AUG_STATE_DIMENSION + 1);

    //create square root matrix
    Matrix A = P_aug.llt().matrixL();
    Matrix spreadingFactor = sqrt(LAMBDA + AUG_STATE_DIMENSION) * A;

    //create augmented sigma points
    int sigmaPointIndex = 0;
    sigmaPoints.col(sigmaPointIndex++) = x_aug;
    for (int index = 0; index < AUG_STATE_DIMENSION; index++)
    {
        sigmaPoints.col(sigmaPointIndex++) = x_aug + spreadingFactor.col(index);
    }
    for (int index = 0; index < AUG_STATE_DIMENSION; index++)
    {
        sigmaPoints.col(sigmaPointIndex++) = x_aug - spreadingFactor.col(index);
    }
}

void UKF::PredictSigmaPoints(double deltaTimeInSecond, const Matrix& augmentedSigmaPoints, Matrix& predSigmaPoints)
{
    //create matrix with predicted sigma points as columns
    predSigmaPoints = Matrix::Zero(STATE_DIMENSION, 2 * AUG_STATE_DIMENSION + 1);

    const int numColumns = augmentedSigmaPoints.cols();
    for (int index = 0; index < numColumns; index++)
    {
        const double v = augmentedSigmaPoints(2, index);
        const double rotation = augmentedSigmaPoints(3, index);
        const double rotationSpeed = augmentedSigmaPoints(4, index);
        const double mu_a = augmentedSigmaPoints(5, index);
        const double mu_rotation = augmentedSigmaPoints(6, index);

        Vector A(5);
        if (Tools::DoubleEqual(rotationSpeed, 0))
        {
            A << v * cos(rotation) * deltaTimeInSecond,
                 v * sin(rotation) * deltaTimeInSecond,
                 0,
                 0,
                 0;
        }
        else
        {
            A << v * (sin(rotation + rotationSpeed * deltaTimeInSecond) - sin(rotation)) / rotationSpeed,
                 v * (-cos(rotation + rotationSpeed * deltaTimeInSecond) + cos(rotation)) / rotationSpeed,
                 0,
                 rotationSpeed * deltaTimeInSecond,
                 0;
        }

        Vector B(5);
        B << 0.5 * deltaTimeInSecond * deltaTimeInSecond * cos(rotation) * mu_a,
             0.5 * deltaTimeInSecond * deltaTimeInSecond * sin(rotation) * mu_a,
             deltaTimeInSecond * mu_a,
             0.5 * deltaTimeInSecond * deltaTimeInSecond * mu_rotation,
             deltaTimeInSecond * mu_rotation;

        Vector x = augmentedSigmaPoints.col(index).head(5);
        predSigmaPoints.col(index) = x + A + B;
    }
}

void UKF::PredictStateMeanAndCovariance(const Matrix& predSigmaPoints, Vector& mean, Matrix& covariance)
{
    // predict state mean
    GetSigmaPointsMean(predSigmaPoints, m_sigmaPointWeights, mean);

    // predict state covariance matrix
    covariance = Matrix::Zero(STATE_DIMENSION, STATE_DIMENSION);
    for (int index = 0; index < predSigmaPoints.cols(); index++)
    {
        // state difference
        Vector diff = predSigmaPoints.col(index) - mean;
        diff(3) = Tools::NormalizeAngleInRadius(diff(3));

        covariance += m_sigmaPointWeights(index) * diff * diff.transpose();
    }
}

void UKF::Update(const MeasurementPackage& measurementPackage)
{
    switch (measurementPackage.sensor_type_)
    {
        case MeasurementPackage::RADAR:
            UpdateRadar(measurementPackage);
            break;

        case MeasurementPackage::LASER:
            UpdateLidar(measurementPackage);
            break;

        default:
            std::cout << "Sensor Type is not supported" << std::endl;
            assert(false);
            return;
    }
}

void UKF::UpdateLidar(const MeasurementPackage& measurementPackage)
{
    Vector mean;
    Matrix covariance;
    Matrix sigmaPointsInMeasurementSpace;
    PredictLidarMeasurement(mean, covariance, sigmaPointsInMeasurementSpace);

    const Vector& measurement = measurementPackage.raw_measurements_;
    UpdateStateFromLidarMeasurement(mean, covariance, sigmaPointsInMeasurementSpace, measurement);

    UpdateLidarNIS(mean, covariance, measurement);
}

void UKF::PredictLidarMeasurement(Vector& predMeasurement, Matrix& covariance, Matrix& sigmaPointsInMeasurementSpace)
{
    TransformSigmaPointsToLidarMeasurementSpace(m_predSigmaPoints, sigmaPointsInMeasurementSpace);

    //calculate mean predicted measurement
    GetSigmaPointsMean(sigmaPointsInMeasurementSpace, m_sigmaPointWeights, predMeasurement);

    //calculate measurement covariance matrix S
    covariance = Matrix::Zero(LIDAR_MEASUREMENT_DIMENSION, LIDAR_MEASUREMENT_DIMENSION);
    const int numSigmaPoints = sigmaPointsInMeasurementSpace.cols();
    for (int index = 0; index < numSigmaPoints; index++)
    {
        Vector z_diff = sigmaPointsInMeasurementSpace.col(index) - predMeasurement;
        covariance += m_sigmaPointWeights(index) * z_diff * z_diff.transpose();
    }

    Matrix R = Matrix::Zero(LIDAR_MEASUREMENT_DIMENSION, LIDAR_MEASUREMENT_DIMENSION);
    R(0, 0) = STD_LASER_PX * STD_LASER_PX;
    R(1, 1) = STD_LASER_PY * STD_LASER_PY;
    covariance += R;
}

void UKF::TransformSigmaPointsToLidarMeasurementSpace(const Matrix& sigmaPoints, Matrix& sigmaPointsInMeasurementSpace)
{
    const int numSigmaPoints = sigmaPoints.cols();

    sigmaPointsInMeasurementSpace = Matrix::Zero(LIDAR_MEASUREMENT_DIMENSION, numSigmaPoints);
    for (int index = 0; index < numSigmaPoints; index++)
    {
        const double px = sigmaPoints(0, index);
        const double py = sigmaPoints(1, index);
        sigmaPointsInMeasurementSpace.col(index) << px, py;
    }
}

void UKF::UpdateStateFromLidarMeasurement(const Vector& mean, const Matrix& covariance, const Matrix& sigmaPointsInMeasurementSpace, const Vector& measurement)
{
    //calculate cross correlation matrix
    Matrix Tc = Matrix::Zero(STATE_DIMENSION, LIDAR_MEASUREMENT_DIMENSION);
    const int numSigmaPoints = sigmaPointsInMeasurementSpace.cols();
    for (int index = 0; index < numSigmaPoints; index++)
    {
        //residual
        Vector z_diff = sigmaPointsInMeasurementSpace.col(index) - mean;

        // state difference
        Vector x_diff = m_predSigmaPoints.col(index) - m_state;
        x_diff(3) = Tools::NormalizeAngleInRadius(x_diff(3));

        Tc += m_sigmaPointWeights(index) * x_diff * z_diff.transpose();
    }

    //calculate Kalman gain K;
    Matrix K = Tc * covariance.inverse();

    //update state mean and covariance matrix
    Vector z_diff = measurement - mean;
    m_state = m_state + K * z_diff;
    m_stateCovariance = m_stateCovariance - K * covariance * K.transpose();
}

void UKF::UpdateLidarNIS(const Vector& predMeasurement, const Matrix& covariance, const Vector& measurement)
{
    m_lidarMeasurementCount++;
    const double nis = CalculateNIS(predMeasurement, covariance, measurement);
    if (nis > 5.991)
    {
        m_lidarMeasurmentNotSatisfyingCount++;
    }

    std::cout << "Lidar 95% NIS error rate: " << static_cast<float>(m_lidarMeasurmentNotSatisfyingCount) / static_cast<float>(m_lidarMeasurementCount) << std::endl;
}

void UKF::UpdateRadar(const MeasurementPackage& measurementPackage)
{
    Vector mean;
    Matrix covariance;
    Matrix sigmaPointsInMeasurementSpace;
    PredictRadarMeasurement(mean, covariance, sigmaPointsInMeasurementSpace);

    const Vector& measurement = measurementPackage.raw_measurements_;
    UpdateStateFromRadarMeasurement(mean, covariance, sigmaPointsInMeasurementSpace, measurement);

    UpdateRadarNIS(mean, covariance, measurement);
}

void UKF::PredictRadarMeasurement(Vector& predMeasurement, Matrix& covariance, Matrix& sigmaPointsInMeasurementSpace)
{
    TransformSigmaPointsToRadarMeasurementSpace(m_predSigmaPoints, sigmaPointsInMeasurementSpace);

    //calculate mean predicted measurement
    GetSigmaPointsMean(sigmaPointsInMeasurementSpace, m_sigmaPointWeights, predMeasurement);

    //calculate measurement covariance matrix S
    covariance = Matrix::Zero(RADAR_MEASUREMENT_DIMENSION, RADAR_MEASUREMENT_DIMENSION);
    const int numSigmaPoints = sigmaPointsInMeasurementSpace.cols();
    for (int index = 0; index < numSigmaPoints; index++)
    {
        Vector z_diff = sigmaPointsInMeasurementSpace.col(index) - predMeasurement;
        z_diff(1) = Tools::NormalizeAngleInRadius(z_diff(1));

        covariance += m_sigmaPointWeights(index) * z_diff * z_diff.transpose();
    }

    Matrix R = Matrix::Zero(RADAR_MEASUREMENT_DIMENSION, RADAR_MEASUREMENT_DIMENSION);
    R(0, 0) = STD_RADAR_R * STD_RADAR_R;
    R(1, 1) = STD_RADAR_PHI * STD_RADAR_PHI;
    R(2, 2) = STD_RADAR_RD * STD_RADAR_RD;
    covariance += R;
}

void UKF::TransformSigmaPointsToRadarMeasurementSpace(const Matrix& sigmaPoints, Matrix& sigmaPointsInMeasurementSpace)
{
    const int numSigmaPoints = sigmaPoints.cols();
    sigmaPointsInMeasurementSpace = Matrix::Zero(RADAR_MEASUREMENT_DIMENSION, numSigmaPoints);

    for (int index = 0; index < numSigmaPoints; index++)
    {
        Vector sigmaPoint = sigmaPoints.col(index);
        const double px = sigmaPoints(0, index);
        const double py = sigmaPoints(1, index);
        const double v = sigmaPoints(2, index);
        const double yaw = sigmaPoints(3, index);

        const double vx = v * cos(yaw);
        const double vy = v * sin(yaw);

        const double pxpxpypy = (px * px + py * py);
        if (Tools::DoubleEqual(pxpxpypy, 0))
        {
            sigmaPointsInMeasurementSpace.col(index) << 0.0, 0.0, 0.0;
        }
        else if (Tools::DoubleEqual(px, 0))
        {
            sigmaPointsInMeasurementSpace.col(index) << py, 90.0, vy;
        }
        else
        {
            const double r = sqrt(px * px + py * py);
            const double phi = atan2(py, px);
            const double rDot = (px * vx + py * vy) / r;
            sigmaPointsInMeasurementSpace.col(index) << r, phi, rDot;
        }
    }
}

void UKF::UpdateStateFromRadarMeasurement(const Vector& predMeasurement, const Matrix& covariance, const Matrix& sigmaPointsInMeasurementSpace, const Vector& measurement)
{
    //calculate cross correlation matrix
    Matrix Tc = Matrix::Zero(STATE_DIMENSION, RADAR_MEASUREMENT_DIMENSION);
    const int numSigmaPoints = sigmaPointsInMeasurementSpace.cols();
    for (int index = 0; index < numSigmaPoints; index++)
    {
        //residual
        Vector z_diff = sigmaPointsInMeasurementSpace.col(index) - predMeasurement;
        z_diff(1) = Tools::NormalizeAngleInRadius(z_diff(1));

        // state difference
        Vector x_diff = m_predSigmaPoints.col(index) - m_state;
        x_diff(3) = Tools::NormalizeAngleInRadius(x_diff(3));

        Tc += m_sigmaPointWeights(index) * x_diff * z_diff.transpose();
    }

    //calculate Kalman gain K;
    Matrix K = Tc * covariance.inverse();

    //update state mean and covariance matrix
    Vector z_diff = measurement - predMeasurement;
    z_diff(1) = Tools::NormalizeAngleInRadius(z_diff(1));
    m_state = m_state + K * z_diff;
    m_stateCovariance = m_stateCovariance - K * covariance * K.transpose();
}

void UKF::UpdateRadarNIS(const Vector& mean, const Matrix& covariance, const Vector& measurement)
{
    m_radarMeasurmentCount++;

    const double nis = CalculateNIS(mean, covariance, measurement);
    if (nis > 7.815)
    {
        m_radarMeasurmentNotSatisfyingCount++;
    }

    std::cout << "Radar 95% NIS error rate: " << static_cast<float>(m_radarMeasurmentNotSatisfyingCount) / static_cast<float>(m_radarMeasurmentCount) << std::endl;
}

void UKF::GetSigmaPointsMean(const Matrix& sigmaPoints, const Vector& weights, Vector& mean)
{
    const int numSigmaPoints = sigmaPoints.cols();
    const int numWeights = weights.rows();
    assert(numSigmaPoints == numWeights);

    mean = Vector::Zero(sigmaPoints.rows());
    for (int index = 0; index < numSigmaPoints; index++)
    {
        mean += weights(index) * sigmaPoints.col(index);
    }
}

double UKF::CalculateNIS(const Vector& predMeasurement, const Matrix& covariance, const Vector& measurement)
{
    return (measurement - predMeasurement).transpose() * covariance.inverse() * (measurement - predMeasurement);
}
