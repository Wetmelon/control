#pragma once

namespace wetmelon::control {

// Result type goes here

namespace online {
// Functions for runtime evaluation can be added here

struct ResultType {
    // Define the structure of the result type
};

// Example factory function (placeholder, give it a better name))
constexpr ResultType makeResultType(int parameter) {
    // Runtime computations go here
    return ResultType{};
};

} // namespace online

namespace design {

// Consteval functions for design-time evaluation can be added here
struct ResultType {
    // Define the structure of the result type
};

// Example consteval factory function (placeholder, give it a better name))
consteval ResultType makeResultType(int parameter) {
    // Heavy compile-time computations go here.  May delegate to online::makeResultType
    return ResultType{};
};

} // namespace design

class TemplateItem {
    // Internal state variables
    // Don't include all result data here; only what is necessary for computations

public:
    using OnlineType = online::ResultType;
    using DesignType = design::ResultType;

    consteval TemplateItem(DesignType /* possibly more parameters */)
        : /* Initialize internal state variables here */ {
    }

    constexpr TemplateItem(OnlineType /* possibly more parameters */)
        : /* Initialize internal state variables here */ {
    }

    constexpr void step() {
        // Update internal state variables here
    }
};

} // namespace wetmelon::control