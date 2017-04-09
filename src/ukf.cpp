#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_[0] = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i < weights_.size(); i++) {
    weights_[i] = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  // INITIALIZE
  if (!is_initialized_) {
    Initialize(meas_package);
    return;
  }

  if (
    (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
    (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)
  )
    return;

  // PREDICT
  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  GenerateSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanCovariance();

  // UPDATE
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);

  previous_timestamp_ = meas_package.timestamp_;
}


void UKF::Initialize(MeasurementPackage meas_package) {
  // Initialize either by radar or lidar first measurement
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    double range = meas_package.raw_measurements_[0];
    double bearing = meas_package.raw_measurements_[1];

    x_ << range * std::cos(bearing), // px
          range * std::sin(bearing), // py
          0,                         // vx
          0,                         // ksi
          0;                         // ksi_rate
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_ << meas_package.raw_measurements_[0], // px
          meas_package.raw_measurements_[1], // py
          0,                                 // v
          0,                                 // ksi
          0;                                 // ksi_rate
  }

  // Initialize state covariance matrix (with big variance for v, ksi and ksi_rate)
  P_ << 1, 0, 0,   0,   0,
        0, 1, 0,   0,   0,
        0, 0, 999, 0,   0,
        0, 0, 0,   999, 0,
        0, 0, 0,   0,   999;

  // Initialize timestamp
  previous_timestamp_ = meas_package.timestamp_;

  // Sigma points
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  is_initialized_ = true;
}


void UKF::GenerateSigmaPoints() {
  // Augmented state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0);
  x_aug.head(n_x_) = x_;

  // Augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug.bottomRightCorner(2, 2) << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

  // Square root matrix
  MatrixXd Psq = P_aug.llt().matrixL();

  // Create augmented sigma points
  double lnx = std::sqrt(lambda_ + n_aug_);
  Xsig_aug_.col(0) << x_aug;
  Xsig_aug_.middleCols(1, n_aug_) << ((lnx * Psq).colwise() + x_aug);
  Xsig_aug_.middleCols(n_aug_ + 1, n_aug_) << ((-lnx * Psq).colwise() + x_aug);
}


void UKF::PredictSigmaPoints(double delta_t) {
  // Predict sigma points, avoiding division by zero
  for (int j=0; j < Xsig_aug_.cols(); j++) {
    VectorXd xinput = VectorXd(7);
    VectorXd xout = VectorXd(5);
    xinput = Xsig_aug_.col(j);

    double px    = xinput[0];
    double py    = xinput[1];
    double v     = xinput[2];
    double ksi   = xinput[3];
    double ksi_r = xinput[4];
    double nua   = xinput[5];
    double nuksi = xinput[6];

    xout[0] = px + 0.5 * delta_t * delta_t * std::cos(ksi) * nua;
    xout[1] = py + 0.5 * delta_t * delta_t * std::sin(ksi) * nua;
    xout[2] = v + delta_t * nua;
    xout[3] = ksi + ksi_r * delta_t + 0.5 * delta_t * delta_t * nuksi;
    xout[4] = ksi_r + delta_t * nuksi;

    if(fabs(ksi_r) < 0.001) {
      xout[0] += v * std::cos(ksi) * delta_t;
      xout[1] += v * std::sin(ksi) * delta_t;
    }
    else {
      xout[0] += (v / ksi_r) * (std::sin(ksi + ksi_r * delta_t) - std::sin(ksi));
      xout[1] += (v / ksi_r) * (-std::cos(ksi + ksi_r * delta_t) + std::cos(ksi));
    }

    Xsig_pred_.col(j) = xout;
  }
}


void UKF::PredictMeanCovariance() {
  // Predict state mean
  x_.fill(0);
  for(int i = 0; i < weights_.size(); i++) {
    x_ += weights_[i] * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0);
  for(int i = 0; i < weights_.size(); i++) {
    VectorXd Xdiff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (Xdiff(3)> M_PI) Xdiff(3)-=2.*M_PI;
    while (Xdiff(3)<-M_PI) Xdiff(3)+=2.*M_PI;

    P_ += weights_[i] * Xdiff * Xdiff.transpose();
  }
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */


  // Laser measurement space dimension
  int n_z = 2;

  // Matrix for sigma points in radar measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for(int j=0; j < Xsig_pred_.cols(); j++) {
    VectorXd input = Xsig_pred_.col(j);
    VectorXd out = VectorXd(n_z);
    out << input[0], input[1];
    Zsig.col(j) = out;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int j=0; j < Zsig.cols(); j++) {
    z_pred += weights_[j] * Zsig.col(j);
  }

  // Calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  S.fill(0.0);
  for(int j=0; j < Zsig.cols(); j++) {
    MatrixXd Zdiff = Zsig.col(j) - z_pred;
    S += weights_[j] * Zdiff * Zdiff.transpose();
  }
  S += R;

  // Matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int j=0; j < Xsig_pred_.cols(); j++) {
    VectorXd diffX = Xsig_pred_.col(j) - x_;
    //angle normalization
    while (diffX(3)> M_PI) diffX(3)-=2.*M_PI;
    while (diffX(3)<-M_PI) diffX(3)+=2.*M_PI;

    VectorXd diffZ = Zsig.col(j) - z_pred;
    Tc += weights_[j] * diffX * diffZ.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc * Sinv;

  // Update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  VectorXd diffZ = z - z_pred;

  x_ = x_ + K * diffZ;
  P_ = P_ - K * S * K.transpose();

  //angle normalization
  while (x_(3)> M_PI) x_(3)-=2.*M_PI;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

  NIS_laser_ = CalculateNIS(z, z_pred, Sinv);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Radar measurement space dimension
  int n_z = 3;

  // Matrix for sigma points in radar measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for(int j=0; j < Xsig_pred_.cols(); j++) {
    VectorXd input = Xsig_pred_.col(j);
    VectorXd out = VectorXd(n_z);

    double sq =  std::sqrt(std::pow(input[0], 2) + std::pow(input[1], 2));
    out << sq,
           std::atan2(input[1], input[0]),
           (input[0] * std::cos(input[3]) * input[2] + input[1] * std::sin(input[3]) * input[2]) / sq;

    Zsig.col(j) = out;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int j=0; j < Zsig.cols(); j++) {
    z_pred += weights_[j] * Zsig.col(j);
  }

  // Calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  S.fill(0.0);
  for(int j=0; j < Zsig.cols(); j++) {
    MatrixXd Zdiff = Zsig.col(j) - z_pred;

    //angle normalization
    while (Zdiff(1)> M_PI) Zdiff(1)-=2.*M_PI;
    while (Zdiff(1)<-M_PI) Zdiff(1)+=2.*M_PI;

    S += weights_[j] * Zdiff * Zdiff.transpose();
  }
  S += R;

  // Matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int j=0; j < Xsig_pred_.cols(); j++) {
    VectorXd diffX = Xsig_pred_.col(j) - x_;
    //angle normalization
    while (diffX(3)> M_PI) diffX(3)-=2.*M_PI;
    while (diffX(3)<-M_PI) diffX(3)+=2.*M_PI;

    VectorXd diffZ = Zsig.col(j) - z_pred;
    //angle normalization
    while (diffZ(1)> M_PI) diffZ(1)-=2.*M_PI;
    while (diffZ(1)<-M_PI) diffZ(1)+=2.*M_PI;

    Tc += weights_[j] * diffX * diffZ.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc * Sinv;

  // Update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  VectorXd diffZ = z - z_pred;

  //angle normalization
  while (diffZ(1)> M_PI) diffZ(1)-=2.*M_PI;
  while (diffZ(1)<-M_PI) diffZ(1)+=2.*M_PI;

  x_ = x_ + K * diffZ;
  P_ = P_ - K * S * K.transpose();

  //angle normalization
  while (x_(3)> M_PI) x_(3)-=2.*M_PI;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

//  std::cout << "Updated x" << std::endl << x_ << std::endl << std::endl;
//  std::cout << "Updated P" << std::endl << P_ << std::endl << std::endl;

  NIS_radar_ = CalculateNIS(z, z_pred, Sinv);
}


double UKF::CalculateNIS(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &Sinv) {
  VectorXd zdiff = z - z_pred;
  return zdiff.transpose() * Sinv * zdiff;
}
