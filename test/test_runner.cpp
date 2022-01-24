
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

TEST_CASE("Basic") {
    int a = 1;
    CHECK(a == a);
}
