#pragma once
#include <Eigen/Dense>
#include <initializer_list>
#include <utility>

namespace control {
class LTI {
};

class tf : public LTI {
};

class ss : public LTI {
   public:
    ss(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C, Eigen::MatrixXd D)
        : A_{std::move(A)}, B_{std::move(B)}, C_{std::move(C)}, D_{std::move(D)} {}

   private:
    Eigen::MatrixXd A_, B_, C_, D_;
};

};  // namespace control

#include <doctest/doctest.h>

TEST_SUITE("LTI") {
#include <iostream>
    TEST_CASE("Create ss") {
        Eigen::MatrixXd A{{1.0, 0.0}, {0.0, 0.0}};
        Eigen::MatrixXd B{{1.0}};
        Eigen::MatrixXd C{{1.0}};
        Eigen::MatrixXd D{{0.0}};

        control::ss{A, B, C, D};
        control::ss{{{1.0, 0.0}, {0.0, 1.0}}, {{1.0}, {0.0}}, {{1.0}}, {{0.0}}};
        control::ss{{{1.0}}, B, C, D};
        control::ss{{{-1.0}}, {{1.0}}, C, {{0.0}}};
        control::ss{{{1.0, 0.0}, {0.0, 1.0}}, B, {{1.0}}, D};
    }
}