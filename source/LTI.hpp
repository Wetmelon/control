#pragma once

#include <cmath>
#include <optional>
#include <ostream>

#include "Eigen/Dense"

namespace control {
using Matrix = Eigen::MatrixXd;

enum class Method {
  ZOH,
  FOH,
  Bilinear,
  Tustin,
};

struct TransferFunction {
  Eigen::MatrixXd num, den;
};

struct StateSpace {
  StateSpace(const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D,
             const std::optional<double> &Ts = std::nullopt,
             const std::optional<Method> &method = std::nullopt,
             const std::optional<double> &prewarp = std::nullopt)
      : A(A), B(B), C(C), D(D), Ts(Ts), method(method), prewarp(prewarp) {};

  Matrix step(const Matrix &x, const Matrix &u) const { return A * x + B * u; }

  Matrix output(const Matrix &x, const Matrix &u) const { return C * x + D * u; }

  StateSpace c2d(const double Ts, const Method method = Method::ZOH,
                 std::optional<double> prewarp = std::nullopt) const;

  const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};

  const std::optional<double> Ts = std::nullopt;
  const std::optional<Method> method = std::nullopt;
  const std::optional<double> prewarp = std::nullopt;

  friend std::ostream &operator<<(std::ostream &os, const StateSpace &sys) {
    os << "A = \n" << sys.A << '\n' << '\n';
    os << "B = \n" << sys.B << '\n' << '\n';
    os << "C = \n" << sys.C << '\n' << '\n';
    os << "D = \n" << sys.D << '\n' << '\n';

    return os;
  }
};

}; // namespace control
