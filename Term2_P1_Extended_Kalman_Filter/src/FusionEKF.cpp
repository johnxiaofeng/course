#include "FusionEKF.h"
#include <iostream>

#include "measurement_package.h"
#include "tools.h"

namespace
{
    const float NOISE_AX = 9.0f;
    const float NOISE_AY = 9.0f;
}

FusionEKF::FusionEKF() 
: is_initialized_(false)
, previous_timestamp_(0)
, R_laser_(2, 2)
, H_laser_(2, 4)
, R_radar_(3, 3)
{
    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    // measurement matrix - laser
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // the initial transition matrix F_
    ekf_.F_ = Matrix(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;
}

FusionEKF::~FusionEKF() 
{

}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
    if (!is_initialized_)
    {
        Initialize(measurement_pack);
        return;
    }

    Predict(measurement_pack);
    Update(measurement_pack);

    //update the timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // print the output
    std::cout << "x_ = " << ekf_.x_ << std::endl;
    std::cout << "P_ = " << ekf_.P_ << std::endl;
}

/**
* Initialize the state ekf_.x_ with the first measurement.
* Create the covariance matrix.
* Remember: you'll need to convert radar from polar to cartesian coordinates.
*/
void FusionEKF::Initialize(const MeasurementPackage& measurement_pack)
{
    // initialize the state ekf_x_ with first measurement
    switch (measurement_pack.sensor_type_)
    {
        case MeasurementPackage::RADAR:
        {
            // convert radar from polor to cartesian coordinates and initialize state
            const float ro = measurement_pack.raw_measurements_[0];
            const float theta = measurement_pack.raw_measurements_[1];
            const float roDot = measurement_pack.raw_measurements_[2];
            ekf_.x_ = Tools::PolorToCartesian(ro, theta, roDot);
            break;
        }
        case MeasurementPackage::LASER:
        {
            // initialie state from laser measurement
            const float px = measurement_pack.raw_measurements_[0];
            const float py = measurement_pack.raw_measurements_[1];
            ekf_.x_ = Vector(4);
            ekf_.x_ << px, py, 0, 0;
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

    // create covariance matrix, since only px and py is applied, set the covariance for vx, vy to big value
    ekf_.P_ = Matrix(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    // update the previous time stamp
    previous_timestamp_ = measurement_pack.timestamp_;
 
    // done initializing, no need to predict or update
    is_initialized_ = true;
}

/**
* Update the state transition matrix F according to the new elapsed time.
    - Time is measured in seconds.
* Update the process noise covariance matrix.
* Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
*/
void FusionEKF::Predict(const MeasurementPackage& measurement_pack)
{
    // get the time different between current and previous timestamp.
    const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    const double dtPow2 = dt * dt;
    const double dtPow3 = dtPow2 * dt;
    const double dtPow4 = dtPow3 * dt;
    
    // update state transition matrix F according to the new elapsed time.
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // update the process covariance matrix Q, using noise_ax = 9 and noise_ay = 9
    ekf_.Q_ = Eigen::MatrixXd(4, 4);
    ekf_.Q_ << (dtPow4 * NOISE_AX) / 4.0, 0, (dtPow3 * NOISE_AX) / 2.0, 0,
			   0, (dtPow4 * NOISE_AY) / 4.0, 0, (dtPow3 * NOISE_AY) / 2.0,
			   (dtPow3 * NOISE_AX) / 2.0, 0, dtPow2 * NOISE_AX, 0,
			   0, (dtPow3 * NOISE_AY) / 2.0, 0, dtPow2 * NOISE_AY;
    
    // call predict of the ekf_
    ekf_.Predict();
}

/**
* Use the sensor type to perform the update step.
* Update the state and covariance matrices.
*/
void FusionEKF::Update(const MeasurementPackage& measurement_pack)
{
    switch (measurement_pack.sensor_type_)
    {
        case MeasurementPackage::RADAR:
            UpdateRadar(measurement_pack);
            break;
            
        case MeasurementPackage::LASER:
            UpdateLaser(measurement_pack);
            break;

        default:
            std::cout << "Sensor Type is not supported" << std::endl;
            assert(false);
            break;
    }
}

void FusionEKF::UpdateLaser(const MeasurementPackage& measurement_pack)
{
    // set measurement matrix and measurement covariance matrix for laser
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    // call the update of ekf_
    ekf_.Update(measurement_pack.raw_measurements_);
}

void FusionEKF::UpdateRadar(const MeasurementPackage& measurement_pack)
{
    // set measurement matrix and measurement covariance matrix for laser
    ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    const float ro = measurement_pack.raw_measurements_[0];
    const float theta = measurement_pack.raw_measurements_[1];
    const float roDot = measurement_pack.raw_measurements_[2];

    // call the updateEKF of ekf_
    Vector z(3);
    z << ro, theta, roDot;
    ekf_.UpdateEKF(z);
}
