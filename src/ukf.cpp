#include "ukf.h"
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

  // state dimensions
  n_x_ = 5;

  // augmented state dimensions
  n_aug_ = n_x_ + 2;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 16.0;

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

  // Not initialised yet
  is_initialized_ = false;
  time_us_ = 0;

  // Which measurement sensors are we going to use
  use_laser_ = true;
  use_radar_ = true;

  // Initial matrix for predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Initialise vector for sigma point weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  // Initialise (if required)
  if (!is_initialized_)
  {
    // First measurement recieved
    cout << "Initializing UKF: " << endl;

    double x, y, v, yaw, yaw_rate;
    double x_sig, y_sig, v_sig, yaw_sig, yaw_rate_sig;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      double r = meas_package.raw_measurements_(0);
      double b = meas_package.raw_measurements_(1);
      double r_rate = meas_package.raw_measurements_(2);

      // Initialise state
      x = r * cos(b);
      y = r * sin(b);

      // Initial state uncertainty
      double tangential_noise = r * sin(std_radphi_);
      x_sig = std_radr_ * cos(b) + tangential_noise * sin(b);
      y_sig = std_radr_ * sin(b) + tangential_noise * cos(b);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state
      x = meas_package.raw_measurements_(0);
      y = meas_package.raw_measurements_(1);

      // Initial state uncertainty
      x_sig = std_laspx_;
      y_sig = std_laspy_;
    }

    // Can't observe these from initial measurement, initialize as zeros
    v = 0.0;
    yaw = 0.0;
    yaw_rate = 0.0;

    // Can't calculate these from initial measurement, initialize based maximum expected error in initial estimate
    v_sig = 5.0;                  // Assume max velocity of 10 m/s
    yaw_sig = M_PI / 8.0;         // Assume max steer angle of 45 degrees
    yaw_rate_sig = M_PI / 16.0;   // Assume max steer angle rate of 22.5 degrees/s

    // Set initial state
    x_ << x, y, v, yaw, yaw_rate;

    cout << "Initial state: " << endl << x_ << endl;

    // Set initial state covariance
    P_ << x_sig * x_sig, 0, 0, 0, 0,
          0, y_sig * y_sig, 0, 0, 0,
          0, 0, v_sig * v_sig, 0, 0,
          0, 0, 0, yaw_sig * yaw_sig, 0,
          0, 0, 0, 0, yaw_rate_sig * yaw_rate_sig;

    cout << "Initial covariance: " << endl << P_ << endl;

    // Store timestamp ready for next measurement update
    time_us_ = meas_package.timestamp_;

    // We are now initialized
    is_initialized_ = true;

    // Done initializing, no need to predict or update state
    return;
  }
  
  // Compute the time elapsed between the current and previous measurements
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  
  // Predict state
  Prediction(delta_t);

  // Update state
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else // RADAR
  {
	  UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // Calculate current sigma points (augmented with noise state)
  MatrixXd Xsig_aug;
  AugmentedSigmaPoints(Xsig_aug);

  // Calculate predicted sigma points
  SigmaPointPrediction(Xsig_aug, delta_t);

  // Calculate predicted state and covariance
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // Measurement vector
  const VectorXd& z = meas_package.raw_measurements_;

  // Predict radar measurement based on current state
  MatrixXd Zsig;
  VectorXd z_pred;
  MatrixXd S;
  PredictLidarMeasurement(Zsig, z_pred, S);

  // Update state based on new measurement
  UpdateState(z, Zsig, z_pred, S);

  // Calculate normalized innovation squared
  VectorXd innov = z - z_pred;
  VectorXd NIS = innov.transpose() * S.inverse() * innov;

  cout << "Laser NIS: " << NIS << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // Measurement vector
  const VectorXd& z = meas_package.raw_measurements_;
  
  // Predict radar measurement based on current state
  MatrixXd Zsig;
  VectorXd z_pred;
  MatrixXd S;
  PredictRadarMeasurement(Zsig, z_pred, S);

  // Update state based on new measurement
  UpdateState(z, Zsig, z_pred, S);

  // Calculate normalized innovation squared
  VectorXd innov = z - z_pred;
  VectorXd NIS = innov.transpose() * S.inverse() * innov;

  cout << "Radar NIS: " << NIS << endl;
}

void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_aug) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
	  Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double delta_t) {

  //predict sigma points
  Xsig_pred_.fill(0.0);
  for (int i = 0; i< 2 * n_aug_ + 1; i++) {
    //extract values for better readability
	  double p_x = Xsig_aug(0, i);
	  double p_y = Xsig_aug(1, i);
	  double v = Xsig_aug(2, i);
	  double yaw = Xsig_aug(3, i);
	  double yawd = Xsig_aug(4, i);
	  double nu_a = Xsig_aug(5, i);
	  double nu_yawdd = Xsig_aug(6, i);

	  //predicted state values
	  double px_p, py_p;

	  //avoid division by zero
	  if (fabs(yawd) > 0.001) {
	    px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
	    py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
	  else {
	    px_p = p_x + v*delta_t*cos(yaw);
	    py_p = p_y + v*delta_t*sin(yaw);
    }

	  double v_p = v;
	  double yaw_p = yaw + yawd*delta_t;
	  double yawd_p = yawd;

	  //add noise
	  px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
	  py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
	  v_p = v_p + nu_a*delta_t;

	  yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
	  yawd_p = yawd_p + nu_yawdd*delta_t;

	  //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //transform sigma points into measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                       //r
    Zsig(1, i) = atan2(p_y, p_x);                               //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0, std_radrd_*std_radrd_;
  S += R;
}

void UKF::PredictLidarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S) {

  //set measurement dimension, lidar can measure x and y position
  int n_z = 2;

  //transform sigma points into measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x; //x
    Zsig(1, i) = p_y; //y
  }

  //mean predicted measurement
  z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = S + R;
}

void UKF::UpdateState(const VectorXd& z, const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S) {

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, z.size());

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //measurement residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K*S*K.transpose();
}
