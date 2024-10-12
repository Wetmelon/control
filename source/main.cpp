
#include <iostream>
#include <print>

#include "LTI.hpp"

int main() {
    constexpr double m = 250.0;  // system mass
    constexpr double k = 40.0;   // spring constant
    constexpr double b = 60.0;   // damping constant

    control::mat A = {{0, 1.0}, {-k / m, -b / m}};
    control::mat B = {{0}, {1.0 / m}};
    control::mat C = {{1.0, 0}};
    control::mat D = {{0.0}};

    const control::StateSpace sys = {A, B, C, D};

    std::cout << "Continuous time matrix sys: \n";
    std::cout << "A = \n" << sys.A << '\n';
    std::cout << "B = \n" << sys.B << '\n';
    std::cout << "C = \n" << sys.C << '\n';
    std::cout << "D = \n" << sys.D << '\n';

    const auto sysd = sys.c2d(0.01, control::DiscretizationMethod::Tustin);

    std::cout << "Discrete time matrix sysd: \n";
    std::cout << "A = \n" << sysd.A << '\n';
    std::cout << "B = \n" << sysd.B << '\n';
    std::cout << "C = \n" << sysd.C << '\n';
    std::cout << "D = \n" << sysd.D << '\n';

    return 0;
}