
#include <iostream>
#include <print>

#include "LTI.hpp"

int main() {
    constexpr double m = 250.0;  // system mass
    constexpr double k = 40.0;   // spring constant
    constexpr double b = 60.0;   // damping constant

    const control::StateSpace sys = {
        .A = {{0, 1.0}, {-k / m, -b / m}},
        .B = {{0}, {1.0 / m}},
        .C = {{1.0, 0}},
        .D = {{0.0}},
    };

    std::cout << "Continuous time matrix sys: \n";
    std::cout << "A = \n" << sys.A << '\n';
    std::cout << "B = \n" << sys.B << '\n';
    std::cout << "C = \n" << sys.C << '\n';
    std::cout << "D = \n" << sys.D << '\n';

    const auto sysd = control::c2d(sys, 0.01, control::DiscretizationMethod::Tustin, 100.0);

    std::cout << "Discrete time matrix sysd: \n";
    std::cout << "A = \n" << sysd.A << '\n';
    std::cout << "B = \n" << sysd.B << '\n';
    std::cout << "C = \n" << sysd.C << '\n';
    std::cout << "D = \n" << sysd.D << '\n';

    return 0;
}