#pragma once
#include <Eigen/Dense>

namespace control {
class LTI {
};

class TransferFunction {
   public:
    TransferFunction(Eigen::MatrixXd num, Eigen::MatrixXd den, double dt = 0.0) : num(std::move(num)), den(std::move(den)), dt(dt) {}

   private:
    Eigen::MatrixXd num, den;
    double dt = 0.0;
};

class StateSpace {
   public:
    StateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C, Eigen::MatrixXd D, double dt = 0.0)
        : A{std::move(A)}, B{std::move(B)}, C{std::move(C)}, D{std::move(D)}, dt_(dt) {}

   private:
    Eigen::MatrixXd A, B, C, D;
    double dt_ = 0.0;
};

};  // namespace control

#include <doctest/doctest.h>

TEST_SUITE("LTI") {
#include <iostream>
    TEST_CASE("Create tf") {
        control::TransferFunction{{{1}}, {{1, 1}}};
    }

    TEST_CASE("Create ss") {
        Eigen::MatrixXd A = {{1, 0.0}, {0.0, 0.0}};
        Eigen::MatrixXd B = {{1.0}};
        Eigen::MatrixXd C = {{1.0}};
        Eigen::MatrixXd D = {{0.0}};

        control::StateSpace{A, B, C, D};
        control::StateSpace{{{1.0, 0.0}, {0.0, 1.0}}, {{1.0}, {0.0}}, {{1.0}}, {{0.0}}};
        control::StateSpace{{{1.0}}, B, C, D};
        control::StateSpace{{{-1}}, {{1.0}}, C, {{0}}};
        control::StateSpace{{{1.0, 0.0}, {0.0, 1.0}}, B, {{1.0}}, D};
    }
}