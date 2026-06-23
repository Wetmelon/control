#pragma once

// ETL profile for this standalone suite: select the Embedded Template Library
// container backend instead of the stdlib default. wet_profile.hpp is MACRO-ONLY
// configuration — see wet/config.hpp. Found ahead of the host tests/wet_profile.hpp
// because this directory is on the include path first (-I.).

#define WET_BACKEND_ETL
