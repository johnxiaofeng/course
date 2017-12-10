#include "kalman_filter.h"
#include "tools.h"

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.
KalmanFilter::KalmanFilter() 
{

}

KalmanFilter::~KalmanFilter() 
{

}

void KalmanFilter::Init(const Vector& x_in, const Matrix& P_in, const Matrix& F_in, const Matrix& H_in, const Matrix& R_in, const Matrix& Q_in) 
{
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
    // predict the state
    x_ = F_ * x_;
    Matrix Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

/**
 * update the state by using Kalman Filter equations
 */
void KalmanFilter::Update(const Vector &z) 
{
    Vector z_pred = H_ * x_;
    Vector y = z - z_pred;
    Matrix Ht = H_.transpose();
    Matrix S = H_ * P_ * Ht + R_;
    Matrix Si = S.inverse();
    Matrix PHt = P_ * Ht;
    Matrix K = PHt * Si;
  
    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    Matrix I = Matrix::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

/**
 * update the state by using Extended Kalman Filter equations
 */
void KalmanFilter::UpdateEKF(const Vector &z) 
{
    const float px = x_[0]; 
    const float py = x_[1];
    const float vx = x_[2];
    const float vy = x_[3];
    const Vector z_pred = Tools::CartesianToPolor(px, py, vx, vy);
    Vector y = z - z_pred;
    y[1] = Tools::ClampRadius(y[1]);

    const Matrix Ht = H_.transpose();
    const Matrix S = H_ * P_ * Ht + R_;
    const Matrix Si = S.inverse();
    const Matrix PHt = P_ * Ht;
    const Matrix K = PHt * Si;

    x_ = x_ + (K * y);

    const long x_size = x_.size();
    const Matrix I = Matrix::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
