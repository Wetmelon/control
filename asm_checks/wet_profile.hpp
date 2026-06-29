#pragma once

// Host profile for the examples: stdlib containers + std:: math backend (the
// library defaults). wet_profile.hpp is MACRO-ONLY configuration — see
// wet/config.hpp for the recognized macros. The host defaults need none.
//
// For an embedded target you would instead set, e.g.:
#define WET_BACKEND_ETL
#define WET_MATH_BACKEND_WET
