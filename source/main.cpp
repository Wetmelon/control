
#include <iostream>
#include <print>

#include "LTI.hpp"

int main() {
    using namespace control;

    constexpr double m = 250.0;  // system mass
    constexpr double k = 40.0;   // spring constant
    constexpr double b = 60.0;   // damping constant

    Matrix A{{0, 1.0}, {-k / m, -b / m}};
    Matrix B{{0}, {1.0 / m}};
    Matrix C{{1.0, 0}};
    Matrix D{{0.0}};

    const StateSpace sys = {A, B, C, D};

    std::cout << "Continuous time matrix sys: \n";
    std::cout << sys;

    const auto sysd = sys.c2d(0.01, Method::Tustin);

    std::cout << "Discrete time matrix sysd: \n";
    std::cout << sysd;

    return 0;
}