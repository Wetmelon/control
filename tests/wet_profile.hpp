#pragma once

// Explicit std:: backend selection for host/test builds.
// Suppresses the "wet_profile.hpp not found" warning from math_backend.hpp.
#include "std_backend.hpp"
