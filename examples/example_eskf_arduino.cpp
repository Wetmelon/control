// Example Arduino sketch using eskf_update_imu for orientation estimation
// Assumes you have an IMU sensor (e.g., MPU6050 with magnetometer).

// Include the library headers (adjust paths as needed for your project structure)

#include "kalman.hpp"
#include "matrix.hpp"
#include "rotation.hpp"
#include "sensor_fusion.hpp"

using namespace wetmelon::control;

// Mock functions to simulate IMU readings (replace with actual sensor code)
Vec3<float> readAccelerometer() {
    return {0.0f, 0.0f, 9.81f};
}
Vec3<float> readGyroscope() {
    return {0.0f, 0.0f, 0.0f};
}
Vec3<float> readMagnetometer() {
    return {0.0f, 1.0f, 0.0f};
}

constexpr float dt = 0.01f; // 10ms time step

static ErrorStateKalmanFilter<6, 6, float> eksf = design::eskf_design(
    0.003f,  // gyro noise density [rad/s/sqrt(Hz)]
    0.03f,   // accel noise density [m/s^2/sqrt(Hz)]
    0.3f,    // mag noise density [unit/sqrt(Hz)]
    0.0001f, // gyro bias random walk [rad/s^1.5]
    dt       // dt = 10ms
);

void setup() {
}

static Euler<float> euler_angles;
static Quaternion   q_nom = Quaternion<float>::identity();
static Vec3         gyro_bias = {0.0f, 0.0f, 0.0f}; // Gyro bias

void loop() {
    // Read IMU sensor data (replace with actual sensor reading code)
    Vec3<float> accel_meas = readAccelerometer(); // [m/s^2]
    Vec3<float> gyro_meas = readGyroscope();      // [rad/s]
    Vec3<float> mag_meas = readMagnetometer();    // [normalized]

    // Perform ESKF update
    eskf_update_imu(
        eksf,
        q_nom,
        gyro_bias,
        accel_meas,
        gyro_meas,
        mag_meas,
        dt // dt = 10ms
    );

    // Convert quaternion to Euler angles for output (if needed)
    euler_angles = q_nom.to_euler<EulerOrder::ZYX>();

    // Here you can use roll, pitch, yaw for control or display
}

int main() {
    return 0;
}
