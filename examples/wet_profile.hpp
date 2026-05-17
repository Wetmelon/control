#pragma once

// Explicit std:: backend selection for host/example builds.
// Suppresses the "wet_profile.hpp not found" warning from math_backend.hpp.
// For embedded targets, replace this include with your platform backend,
// e.g.: #include "ti_arm_backend.hpp"
#include "std_backend.hpp"
