#pragma once

#include <cmath>
#include <optional>
#include <span>
#include <type_traits>

#include "matrix.hpp"

// Euler rotation order enumeration
enum class EulerOrder {
    XYZ, // Roll-Pitch-Yaw (aerospace)
    ZYX, // Yaw-Pitch-Roll
    ZXY,
    YXZ,
    YZX,
    XZY,
};

// Forward declarations
template<typename T>
struct Quaternion;

template<typename T>
struct DCM;

template<typename T, EulerOrder Order>
struct Euler;

// ============================================================================
// DCM: Direction Cosine Matrix (3x3 rotation matrix wrapper)
// ============================================================================
template<typename T>
struct DCM : public Mat3<T> {
    using value_type = T;

    static_assert(std::is_floating_point_v<T>, "DCM element type must be floating point");

    constexpr DCM() : Mat3<T>(Mat3<T>::identity()) {}
    constexpr DCM(const DCM&) = default;
    constexpr DCM& operator=(const DCM&) = default;
    constexpr DCM(DCM&&) = default;
    constexpr DCM& operator=(DCM&&) = default;
    constexpr ~DCM() = default;

    // Construct from raw Mat3
    constexpr explicit DCM(const Mat3<T>& m) : Mat3<T>(m) {}

    // Identity rotation
    [[nodiscard]] static constexpr DCM identity() { return DCM{}; }

    // Basic rotation matrices about principal axes
    [[nodiscard]] static constexpr DCM rotate_x(T angle) {
        T   c = std::cos(angle);
        T   s = std::sin(angle);
        DCM R;
        R(0, 0) = T{1};
        R(0, 1) = T{0};
        R(0, 2) = T{0};
        R(1, 0) = T{0};
        R(1, 1) = c;
        R(1, 2) = -s;
        R(2, 0) = T{0};
        R(2, 1) = s;
        R(2, 2) = c;
        return R;
    }

    [[nodiscard]] static constexpr DCM rotate_y(T angle) {
        T   c = std::cos(angle);
        T   s = std::sin(angle);
        DCM R;
        R(0, 0) = c;
        R(0, 1) = T{0};
        R(0, 2) = s;
        R(1, 0) = T{0};
        R(1, 1) = T{1};
        R(1, 2) = T{0};
        R(2, 0) = -s;
        R(2, 1) = T{0};
        R(2, 2) = c;
        return R;
    }

    [[nodiscard]] static constexpr DCM rotate_z(T angle) {
        T   c = std::cos(angle);
        T   s = std::sin(angle);
        DCM R;
        R(0, 0) = c;
        R(0, 1) = -s;
        R(0, 2) = T{0};
        R(1, 0) = s;
        R(1, 1) = c;
        R(1, 2) = T{0};
        R(2, 0) = T{0};
        R(2, 1) = T{0};
        R(2, 2) = T{1};
        return R;
    }

    // Compose rotations
    [[nodiscard]] constexpr DCM operator*(const DCM& rhs) const {
        return DCM(static_cast<const Mat3<T>&>(*this) * static_cast<const Mat3<T>&>(rhs));
    }

    constexpr DCM& operator*=(const DCM& rhs) {
        return *this = *this * rhs;
    }

    // Rotate a vector
    [[nodiscard]] constexpr Vec3<T> operator*(const Vec3<T>& v) const {
        return Vec3<T>(static_cast<const Mat3<T>&>(*this) * static_cast<const Matrix<3, 1, T>&>(v));
    }

    // Transpose (inverse for orthonormal)
    [[nodiscard]] constexpr DCM transpose() const {
        return DCM(Mat3<T>::transpose());
    }

    // Inverse (same as transpose for orthonormal)
    [[nodiscard]] constexpr DCM inverse() const {
        return transpose();
    }

    // Access underlying matrix
    [[nodiscard]] constexpr const Mat3<T>& matrix() const { return *this; }
    [[nodiscard]] constexpr Mat3<T>&       matrix() { return *this; }

    // Convert to quaternion
    [[nodiscard]] constexpr std::optional<Quaternion<T>> to_quaternion(T eps = T{1e-6}) const;

    // Convert to Euler angles
    template<EulerOrder Order>
    [[nodiscard]] constexpr Euler<T, Order> to_euler() const;

    // Construct from Euler angles
    template<EulerOrder Order>
    [[nodiscard]] static constexpr DCM from_euler(const Euler<T, Order>& e);

    // Construct from quaternion
    [[nodiscard]] static constexpr DCM from_quaternion(const Quaternion<T>& q);

    // Construct from axis-angle
    [[nodiscard]] static constexpr std::optional<DCM> from_axis_angle(const Vec3<T>& axis, T angle, T eps = T{1e-9}) {
        T axis_norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if (axis_norm2 <= eps) {
            return std::nullopt;
        }
        T inv_norm = T{1} / wet::sqrt(axis_norm2);
        T ux = axis[0] * inv_norm;
        T uy = axis[1] * inv_norm;
        T uz = axis[2] * inv_norm;

        T c = std::cos(angle);
        T s = std::sin(angle);
        T t = T{1} - c;

        DCM R;
        R(0, 0) = t * ux * ux + c;
        R(0, 1) = t * ux * uy - s * uz;
        R(0, 2) = t * ux * uz + s * uy;

        R(1, 0) = t * ux * uy + s * uz;
        R(1, 1) = t * uy * uy + c;
        R(1, 2) = t * uy * uz - s * ux;

        R(2, 0) = t * ux * uz - s * uy;
        R(2, 1) = t * uy * uz + s * ux;
        R(2, 2) = t * uz * uz + c;
        return R;
    }
};

// ============================================================================
// Euler: Euler angles templated on rotation order
// ============================================================================
template<typename T, EulerOrder Order = EulerOrder::ZYX>
struct Euler {
    using value_type = T;
    static constexpr EulerOrder order = Order;

    T angle1{}; // First rotation
    T angle2{}; // Second rotation
    T angle3{}; // Third rotation

    constexpr Euler() = default;
    constexpr Euler(T a1, T a2, T a3) : angle1(a1), angle2(a2), angle3(a3) {}
    constexpr Euler(const Euler&) = default;
    constexpr Euler& operator=(const Euler&) = default;
    constexpr Euler(Euler&&) = default;
    constexpr Euler& operator=(Euler&&) = default;
    constexpr ~Euler() = default;

    // Named accessors for common conventions
    // For ZYX (yaw-pitch-roll): angle1=yaw, angle2=pitch, angle3=roll
    // For XYZ (roll-pitch-yaw): angle1=roll, angle2=pitch, angle3=yaw

    // ZYX specific accessors
    [[nodiscard]] constexpr T yaw() const
        requires(Order == EulerOrder::ZYX)
    { return angle1; }
    [[nodiscard]] constexpr T& yaw()
        requires(Order == EulerOrder::ZYX)
    { return angle1; }
    [[nodiscard]] constexpr T pitch() const
        requires(Order == EulerOrder::ZYX)
    { return angle2; }
    [[nodiscard]] constexpr T& pitch()
        requires(Order == EulerOrder::ZYX)
    { return angle2; }
    [[nodiscard]] constexpr T roll() const
        requires(Order == EulerOrder::ZYX)
    { return angle3; }
    [[nodiscard]] constexpr T& roll()
        requires(Order == EulerOrder::ZYX)
    { return angle3; }

    // XYZ specific accessors
    [[nodiscard]] constexpr T roll_xyz() const
        requires(Order == EulerOrder::XYZ)
    { return angle1; }
    [[nodiscard]] constexpr T& roll_xyz()
        requires(Order == EulerOrder::XYZ)
    { return angle1; }
    [[nodiscard]] constexpr T pitch_xyz() const
        requires(Order == EulerOrder::XYZ)
    { return angle2; }
    [[nodiscard]] constexpr T& pitch_xyz()
        requires(Order == EulerOrder::XYZ)
    { return angle2; }
    [[nodiscard]] constexpr T yaw_xyz() const
        requires(Order == EulerOrder::XYZ)
    { return angle3; }
    [[nodiscard]] constexpr T& yaw_xyz()
        requires(Order == EulerOrder::XYZ)
    { return angle3; }

    // Convert to DCM
    [[nodiscard]] constexpr DCM<T> to_dcm() const {
        if constexpr (Order == EulerOrder::ZYX) {
            return DCM<T>::rotate_z(angle1) * DCM<T>::rotate_y(angle2) * DCM<T>::rotate_x(angle3);
        } else if constexpr (Order == EulerOrder::XYZ) {
            return DCM<T>::rotate_x(angle1) * DCM<T>::rotate_y(angle2) * DCM<T>::rotate_z(angle3);
        } else if constexpr (Order == EulerOrder::ZXY) {
            return DCM<T>::rotate_z(angle1) * DCM<T>::rotate_x(angle2) * DCM<T>::rotate_y(angle3);
        } else if constexpr (Order == EulerOrder::YXZ) {
            return DCM<T>::rotate_y(angle1) * DCM<T>::rotate_x(angle2) * DCM<T>::rotate_z(angle3);
        } else if constexpr (Order == EulerOrder::YZX) {
            return DCM<T>::rotate_y(angle1) * DCM<T>::rotate_z(angle2) * DCM<T>::rotate_x(angle3);
        } else if constexpr (Order == EulerOrder::XZY) {
            return DCM<T>::rotate_x(angle1) * DCM<T>::rotate_z(angle2) * DCM<T>::rotate_y(angle3);
        }
    }

    // Convert to quaternion
    [[nodiscard]] constexpr Quaternion<T> to_quaternion() const;

    // Construct from DCM
    [[nodiscard]] static constexpr Euler from_dcm(const DCM<T>& R) {
        Euler e;

        if constexpr (Order == EulerOrder::ZYX) {
            // Yaw-Pitch-Roll
            T sp = -R(2, 0);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp); // pitch

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(R(1, 0), R(0, 0)); // yaw
                e.angle3 = std::atan2(R(2, 1), R(2, 2)); // roll
            } else {
                // Gimbal lock
                e.angle1 = std::atan2(-R(0, 1), R(1, 1));
                e.angle3 = T{0};
            }
        } else if constexpr (Order == EulerOrder::XYZ) {
            // Roll-Pitch-Yaw
            T sp = R(0, 2);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp); // pitch

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(-R(1, 2), R(2, 2)); // roll
                e.angle3 = std::atan2(-R(0, 1), R(0, 0)); // yaw
            } else {
                e.angle1 = std::atan2(R(2, 1), R(1, 1));
                e.angle3 = T{0};
            }
        } else if constexpr (Order == EulerOrder::ZXY) {
            T sp = R(2, 1);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp);

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(-R(0, 1), R(1, 1));
                e.angle3 = std::atan2(-R(2, 0), R(2, 2));
            } else {
                e.angle1 = std::atan2(R(1, 0), R(0, 0));
                e.angle3 = T{0};
            }
        } else if constexpr (Order == EulerOrder::YXZ) {
            T sp = -R(1, 2);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp);

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(R(0, 2), R(2, 2));
                e.angle3 = std::atan2(R(1, 0), R(1, 1));
            } else {
                e.angle1 = std::atan2(-R(2, 0), R(0, 0));
                e.angle3 = T{0};
            }
        } else if constexpr (Order == EulerOrder::YZX) {
            T sp = R(1, 0);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp);

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(-R(2, 0), R(0, 0));
                e.angle3 = std::atan2(-R(1, 2), R(1, 1));
            } else {
                e.angle1 = std::atan2(R(0, 2), R(2, 2));
                e.angle3 = T{0};
            }
        } else if constexpr (Order == EulerOrder::XZY) {
            T sp = -R(0, 1);
            if (sp > T{1})
                sp = T{1};
            if (sp < T{-1})
                sp = T{-1};
            e.angle2 = std::asin(sp);

            if (std::abs(sp) < T{1} - T{1e-6}) {
                e.angle1 = std::atan2(R(2, 1), R(1, 1));
                e.angle3 = std::atan2(R(0, 2), R(0, 0));
            } else {
                e.angle1 = std::atan2(-R(1, 2), R(2, 2));
                e.angle3 = T{0};
            }
        }
        return e;
    }

    // Construct from quaternion
    [[nodiscard]] static constexpr Euler from_quaternion(const Quaternion<T>& q);
};

// Common Euler type aliases
template<typename T>
using EulerZYX = Euler<T, EulerOrder::ZYX>;

template<typename T>
using EulerXYZ = Euler<T, EulerOrder::XYZ>;

// 4x4 homogeneous transform matrix alias
template<typename T>
using Transform4 = Mat4<T>;

// ============================================================================
// Quaternion (moved/refactored from quaternion.hpp)
// ============================================================================
template<typename T>
struct Quaternion : public Matrix<4, 1, T> {
    using value_type = T;

    static_assert(std::is_floating_point_v<T>, "Quaternion element type must be floating point");

    // Component accessors (w + vector part x,y,z)
    constexpr T&       w() { return this->data_[0][0]; }
    constexpr const T& w() const { return this->data_[0][0]; }
    constexpr T&       x() { return this->data_[1][0]; }
    constexpr const T& x() const { return this->data_[1][0]; }
    constexpr T&       y() { return this->data_[2][0]; }
    constexpr const T& y() const { return this->data_[2][0]; }
    constexpr T&       z() { return this->data_[3][0]; }
    constexpr const T& z() const { return this->data_[3][0]; }

    // Default constructor (identity)
    constexpr Quaternion() : Matrix<4, 1, T>() { w() = T{1}; }
    constexpr Quaternion(const Quaternion&) = default;
    constexpr Quaternion& operator=(const Quaternion&) = default;
    constexpr Quaternion(Quaternion&&) = default;
    constexpr Quaternion& operator=(Quaternion&&) = default;
    constexpr ~Quaternion() = default;

    constexpr Quaternion(T w_, T x_, T y_, T z_) : Matrix<4, 1, T>() {
        w() = w_;
        x() = x_;
        y() = y_;
        z() = z_;
    }

    constexpr Quaternion(std::initializer_list<T> values) : Matrix<4, 1, T>() {
        size_t i = 0;
        for (const auto& val : values) {
            if (i < 4) {
                this->data_[i][0] = val;
            }
            ++i;
        }
    }

    template<typename SpanType>
        requires std::is_same_v<SpanType, std::span<const T>>
    constexpr explicit Quaternion(SpanType values) : Matrix<4, 1, T>() {
        size_t i = 0;
        for (const auto& val : values) {
            if (i < 4) {
                this->data_[i][0] = val;
            }
            ++i;
        }
    }

    template<typename U>
    constexpr Quaternion(const Quaternion<U>& other) : Matrix<4, 1, T>(other) {}

    template<typename U>
    constexpr explicit Quaternion(const Matrix<4, 1, U>& other) : Matrix<4, 1, T>(other) {}

    template<typename U>
    constexpr Quaternion& operator=(const Matrix<4, 1, U>& other) {
        Matrix<4, 1, T>::operator=(other);
        return *this;
    }

    // Identity (no rotation)
    [[nodiscard]] static constexpr Quaternion identity() { return Quaternion{T{1}, T{0}, T{0}, T{0}}; }

    // Norms
    [[nodiscard]] constexpr T norm_squared() const { return (w() * w()) + (x() * x()) + (y() * y()) + (z() * z()); }
    [[nodiscard]] constexpr T norm() const { return wet::sqrt(norm_squared()); }

    // Normalization utilities
    [[nodiscard]] constexpr std::optional<Quaternion> normalized_safe(T eps = T{1e-9}) const {
        T n2 = norm_squared();
        if (n2 <= eps) {
            return std::nullopt;
        }
        T inv_n = T{1} / wet::sqrt(n2);
        return Quaternion{w() * inv_n, x() * inv_n, y() * inv_n, z() * inv_n};
    }

    constexpr bool normalize_in_place(T eps = T{1e-9}) {
        T n2 = norm_squared();
        if (n2 <= eps) {
            return false;
        }
        T inv_n = T{1} / wet::sqrt(n2);
        w() *= inv_n;
        x() *= inv_n;
        y() *= inv_n;
        z() *= inv_n;
        return true;
    }

    [[nodiscard]] constexpr Quaternion normalized(T eps = T{1e-9}) const {
        auto n = normalized_safe(eps);
        return n.value_or(*this);
    }

    // Conjugate and inverse
    [[nodiscard]] constexpr Quaternion conjugate() const { return Quaternion{w(), -x(), -y(), -z()}; }

    [[nodiscard]] constexpr std::optional<Quaternion> inverse(T eps = T{1e-9}) const {
        T n2 = norm_squared();
        if (n2 <= eps) {
            return std::nullopt;
        }
        T inv_n2 = T{1} / n2;
        return Quaternion{w() * inv_n2, -x() * inv_n2, -y() * inv_n2, -z() * inv_n2};
    }

    // Hamilton product
    [[nodiscard]] constexpr Quaternion operator*(const Quaternion& rhs) const {
        return Quaternion{
            w() * rhs.w() - x() * rhs.x() - y() * rhs.y() - z() * rhs.z(),
            w() * rhs.x() + x() * rhs.w() + y() * rhs.z() - z() * rhs.y(),
            w() * rhs.y() - x() * rhs.z() + y() * rhs.w() + z() * rhs.x(),
            w() * rhs.z() + x() * rhs.y() - y() * rhs.x() + z() * rhs.w()
        };
    }

    constexpr Quaternion& operator*=(const Quaternion& rhs) {
        return *this = (*this) * rhs;
    }

    // Scalar multiply/divide
    [[nodiscard]] constexpr Quaternion operator*(T scalar) const {
        return Quaternion{w() * scalar, x() * scalar, y() * scalar, z() * scalar};
    }

    [[nodiscard]] constexpr Quaternion operator/(T scalar) const {
        return Quaternion{w() / scalar, x() / scalar, y() / scalar, z() / scalar};
    }

    // Rotate 3D vector
    [[nodiscard]] constexpr Vec3<T> rotate(const Vec3<T>& v) const {
        Vec3<T> qv{x(), y(), z()};
        Vec3<T> t;
        t[0] = T{2} * (qv[1] * v[2] - qv[2] * v[1]);
        t[1] = T{2} * (qv[2] * v[0] - qv[0] * v[2]);
        t[2] = T{2} * (qv[0] * v[1] - qv[1] * v[0]);

        Vec3<T> result{
            v[0] + w() * t[0] + (qv[1] * t[2] - qv[2] * t[1]),
            v[1] + w() * t[1] + (qv[2] * t[0] - qv[0] * t[2]),
            v[2] + w() * t[2] + (qv[0] * t[1] - qv[1] * t[0])
        };
        return result;
        return result;
    }

    // Convert to DCM
    [[nodiscard]] constexpr DCM<T> to_dcm() const {
        Quaternion qn = normalized();
        const T    ww = qn.w();
        const T    xx = qn.x();
        const T    yy = qn.y();
        const T    zz = qn.z();

        DCM<T> R;
        R(0, 0) = T{1} - T{2} * (yy * yy + zz * zz);
        R(0, 1) = T{2} * (xx * yy - zz * ww);
        R(0, 2) = T{2} * (xx * zz + yy * ww);

        R(1, 0) = T{2} * (xx * yy + zz * ww);
        R(1, 1) = T{1} - T{2} * (xx * xx + zz * zz);
        R(1, 2) = T{2} * (yy * zz - xx * ww);

        R(2, 0) = T{2} * (xx * zz - yy * ww);
        R(2, 1) = T{2} * (yy * zz + xx * ww);
        R(2, 2) = T{1} - T{2} * (xx * xx + yy * yy);
        return R;
    }

    // Convert to Euler
    template<EulerOrder Order = EulerOrder::ZYX>
    [[nodiscard]] constexpr Euler<T, Order> to_euler() const {
        return Euler<T, Order>::from_dcm(to_dcm());
    }

    // Construct from DCM
    [[nodiscard]] static constexpr std::optional<Quaternion> from_dcm(const DCM<T>& R, T eps = T{1e-6}) {
        T          trace = R(0, 0) + R(1, 1) + R(2, 2);
        Quaternion q;
        if (trace > T{0}) {
            T s = wet::sqrt(trace + T{1});
            q.w() = T{0.5} * s;
            T inv_s = T{0.5} / s;
            q.x() = (R(2, 1) - R(1, 2)) * inv_s;
            q.y() = (R(0, 2) - R(2, 0)) * inv_s;
            q.z() = (R(1, 0) - R(0, 1)) * inv_s;
        } else {
            if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
                T s = wet::sqrt(T{1} + R(0, 0) - R(1, 1) - R(2, 2));
                if (s <= eps)
                    return std::nullopt;
                T inv_s = T{0.5} / s;
                q.x() = T{0.5} * s;
                q.y() = (R(0, 1) + R(1, 0)) * inv_s;
                q.z() = (R(0, 2) + R(2, 0)) * inv_s;
                q.w() = (R(2, 1) - R(1, 2)) * inv_s;
            } else if (R(1, 1) > R(2, 2)) {
                T s = wet::sqrt(T{1} + R(1, 1) - R(0, 0) - R(2, 2));
                if (s <= eps)
                    return std::nullopt;
                T inv_s = T{0.5} / s;
                q.x() = (R(0, 1) + R(1, 0)) * inv_s;
                q.y() = T{0.5} * s;
                q.z() = (R(1, 2) + R(2, 1)) * inv_s;
                q.w() = (R(0, 2) - R(2, 0)) * inv_s;
            } else {
                T s = wet::sqrt(T{1} + R(2, 2) - R(0, 0) - R(1, 1));
                if (s <= eps)
                    return std::nullopt;
                T inv_s = T{0.5} / s;
                q.x() = (R(0, 2) + R(2, 0)) * inv_s;
                q.y() = (R(1, 2) + R(2, 1)) * inv_s;
                q.z() = T{0.5} * s;
                q.w() = (R(1, 0) - R(0, 1)) * inv_s;
            }
        }
        if (!q.normalize_in_place(eps))
            return std::nullopt;
        return q;
    }

    // Construct from Euler
    template<EulerOrder Order>
    [[nodiscard]] static constexpr Quaternion from_euler(const Euler<T, Order>& e) {
        auto q_opt = from_dcm(e.to_dcm());
        return q_opt.value_or(identity());
    }

    // Construct from axis-angle
    [[nodiscard]] static constexpr std::optional<Quaternion> from_axis_angle(const Vec3<T>& axis, T angle, T eps = T{1e-9}) {
        T axis_norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if (axis_norm2 <= eps) {
            return std::nullopt;
        }
        T inv_axis_norm = T{1} / wet::sqrt(axis_norm2);
        T half = angle * T{0.5};
        T s = std::sin(half) * inv_axis_norm;
        return Quaternion{std::cos(half), axis[0] * s, axis[1] * s, axis[2] * s};
    }

    // Integrate body rates (first-order approximation)
    [[nodiscard]] constexpr Quaternion integrate_body_rates(const Vec3<T>& omega, T dt) const {
        T          half_dt = dt * T{0.5};
        Quaternion dq{T{1}, omega[0] * half_dt, omega[1] * half_dt, omega[2] * half_dt};
        return (dq * (*this)).normalized();
    }

    // Spherical linear interpolation
    [[nodiscard]] static constexpr Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) {
        if (t < T{0})
            t = T{0};
        if (t > T{1})
            t = T{1};

        T          cos_theta = a.w() * b.w() + a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
        Quaternion b_adj = b;
        if (cos_theta < T{0}) {
            cos_theta = -cos_theta;
            b_adj = Quaternion{-b.w(), -b.x(), -b.y(), -b.z()};
        }

        const T k_small = T{1e-6};
        if (cos_theta > T{1} - k_small) {
            Quaternion result{
                a.w() + t * (b_adj.w() - a.w()),
                a.x() + t * (b_adj.x() - a.x()),
                a.y() + t * (b_adj.y() - a.y()),
                a.z() + t * (b_adj.z() - a.z())
            };
            return result.normalized();
        }

        T          theta = std::acos(cos_theta);
        T          sin_theta = std::sin(theta);
        T          w1 = std::sin((T{1} - t) * theta) / sin_theta;
        T          w2 = std::sin(t * theta) / sin_theta;
        Quaternion result{
            a.w() * w1 + b_adj.w() * w2,
            a.x() * w1 + b_adj.x() * w2,
            a.y() * w1 + b_adj.y() * w2,
            a.z() * w1 + b_adj.z() * w2
        };
        return result.normalized();
    }
};

// Scalar multiply (scalar on left)
template<typename T>
[[nodiscard]] constexpr Quaternion<T> operator*(T scalar, const Quaternion<T>& q) {
    return q * scalar;
}

// ============================================================================
// Deferred implementations (need full type definitions)
// ============================================================================

template<typename T>
constexpr std::optional<Quaternion<T>> DCM<T>::to_quaternion(T eps) const {
    return Quaternion<T>::from_dcm(*this, eps);
}

template<typename T>
template<EulerOrder Order>
constexpr Euler<T, Order> DCM<T>::to_euler() const {
    return Euler<T, Order>::from_dcm(*this);
}

template<typename T>
template<EulerOrder Order>
constexpr DCM<T> DCM<T>::from_euler(const Euler<T, Order>& e) {
    return e.to_dcm();
}

template<typename T>
constexpr DCM<T> DCM<T>::from_quaternion(const Quaternion<T>& q) {
    return q.to_dcm();
}

template<typename T, EulerOrder Order>
constexpr Quaternion<T> Euler<T, Order>::to_quaternion() const {
    return Quaternion<T>::from_euler(*this);
}

template<typename T, EulerOrder Order>
constexpr Euler<T, Order> Euler<T, Order>::from_quaternion(const Quaternion<T>& q) {
    return q.template to_euler<Order>();
}

// Convenience type aliases
using Quatf = Quaternion<float>;
using Quatd = Quaternion<double>;

using DCMf = DCM<float>;
using DCMd = DCM<double>;

using EulerZYXf = EulerZYX<float>;
using EulerZYXd = EulerZYX<double>;

using EulerXYZf = EulerXYZ<float>;
using EulerXYZd = EulerXYZ<double>;
