#pragma once

// Host profile for tests: stdlib containers + std:: math backend (the library
// defaults). wet_profile.hpp is MACRO-ONLY configuration — see wet/config.hpp
// for the recognized macros. The host defaults need none, so this file is empty
// apart from existing: its presence selects an explicit, warning-free config.

// #define WET_BACKEND_ETL
// #define WET_MATH_BACKEND_WET
