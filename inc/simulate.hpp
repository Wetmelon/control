#pragma once

/**
 * @defgroup simulate Closed-Loop Simulation
 * @brief Runtime simulation of nonlinear plants with linear/nonlinear controllers
 *
 * Provides functions to simulate closed-loop systems using the ODE solvers
 * from solver.hpp and integrators from integrator.hpp.
 *
 * Usage:
 * @code
 *   // Nonlinear plant: dx/dt = f(t, x, u)
 *   auto plant = [](double t, auto x, auto u) { return ...; };
 *   auto output = [](auto x) { return ...; };
 *   auto controller = [&](auto y) { return -K * y; };
 *
 *   auto result = simulate(plant, output, controller, solver, x0, {0.0, 10.0});
 * @endcode
 */

#include <utility>
#include <vector>

#include "solver.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

/**
 * @brief Result of a closed-loop simulation
 *
 * @tparam NX Number of plant states
 * @tparam NU Number of control inputs
 * @tparam NY Number of plant outputs
 * @tparam T  Scalar type
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct SimulationResult {
    std::vector<T>             t; ///< Time points
    std::vector<ColVec<NX, T>> x; ///< State history
    std::vector<ColVec<NY, T>> y; ///< Output history
    std::vector<ColVec<NU, T>> u; ///< Control input history
};

/**
 * @brief Simulate a nonlinear plant with a controller in closed loop
 *
 * The plant dynamics are dx/dt = plant(t, x, u), the output is y = output(x),
 * and the controller computes u = controller(y). At each solver step, the
 * controller output is computed from the current output, then held constant
 * for the duration of the integration step (zero-order hold).
 *
 * @tparam Plant      Callable: (T t, ColVec<NX,T> x, ColVec<NU,T> u) -> ColVec<NX,T>
 * @tparam Output     Callable: (ColVec<NX,T> x) -> ColVec<NY,T>
 * @tparam Controller Callable: (ColVec<NY,T> y) -> ColVec<NU,T>
 * @tparam Solver     FixedStepSolver or AdaptiveStepSolver
 *
 * @param plant      Plant dynamics: dx/dt = plant(t, x, u)
 * @param output     Output function: y = output(x)
 * @param controller Controller: u = controller(y)
 * @param solver     ODE solver instance
 * @param x0         Initial state
 * @param t_span     Simulation time span {t0, tf}
 * @return SimulationResult with time, state, output, and input history
 */
template<size_t NX, size_t NU, size_t NY, typename T, typename Plant, typename Output, typename Controller, typename Solver>
SimulationResult<NX, NU, NY, T> simulate(
    Plant&&                plant,
    Output&&               output,
    Controller&&           controller,
    const Solver&          solver,
    const ColVec<NX, T>&   x0,
    const std::pair<T, T>& t_span
) {
    // Capture controller output so the plant closure can use it
    ColVec<NU, T> u_current{};

    // Wrap plant + controller into f(t, x) for the solver
    auto f = [&](T t, const ColVec<NX, T>& x) -> ColVec<NX, T> {
        return plant(t, x, u_current);
    };

    // Solve, but intercept each step to record outputs and update controller
    SimulationResult<NX, NU, NY, T> sim;

    // Compute initial output and control
    ColVec<NY, T> y0 = output(x0);
    u_current = controller(y0);

    // Record initial conditions
    sim.t.push_back(t_span.first);
    sim.x.push_back(x0);
    sim.y.push_back(y0);
    sim.u.push_back(u_current);

    // Create a copy of the solver so we can add a step callback
    auto sim_solver = solver;
    sim_solver.set_on_step([&](T t, const ColVec<NX, T>& x) {
        ColVec<NY, T> y = output(x);
        u_current = controller(y);

        sim.t.push_back(t);
        sim.x.push_back(x);
        sim.y.push_back(y);
        sim.u.push_back(u_current);
    });

    sim_solver.solve(f, x0, t_span);

    return sim;
}

/**
 * @brief Simulate a nonlinear plant with state-feedback controller
 *
 * Convenience overload where the controller receives the full state x
 * rather than the output y. Output is computed for recording only.
 *
 * @param plant      dx/dt = plant(t, x, u)
 * @param output     y = output(x) — for recording only
 * @param controller u = controller(x) — state feedback
 * @param solver     ODE solver instance
 * @param x0         Initial state
 * @param t_span     {t0, tf}
 */
template<size_t NX, size_t NU, size_t NY, typename T, typename Plant, typename Output, typename Controller, typename Solver>
SimulationResult<NX, NU, NY, T> simulate_state_feedback(
    Plant&&                plant,
    Output&&               output,
    Controller&&           controller,
    const Solver&          solver,
    const ColVec<NX, T>&   x0,
    const std::pair<T, T>& t_span
) {
    ColVec<NU, T> u_current{};

    auto f = [&](T t, const ColVec<NX, T>& x) -> ColVec<NX, T> {
        return plant(t, x, u_current);
    };

    SimulationResult<NX, NU, NY, T> sim;

    ColVec<NY, T> y0 = output(x0);
    u_current = controller(x0);

    sim.t.push_back(t_span.first);
    sim.x.push_back(x0);
    sim.y.push_back(y0);
    sim.u.push_back(u_current);

    auto sim_solver = solver;
    sim_solver.set_on_step([&](T t, const ColVec<NX, T>& x) {
        ColVec<NY, T> y = output(x);
        u_current = controller(x);

        sim.t.push_back(t);
        sim.x.push_back(x);
        sim.y.push_back(y);
        sim.u.push_back(u_current);
    });

    sim_solver.solve(f, x0, t_span);

    return sim;
}

/**
 * @brief Simulate a continuous LTI system with a controller
 *
 * Uses StateSpace matrices (A, B, C, D) as the plant.
 * Controller receives output y = Cx + Du and produces u = controller(y).
 *
 * @param sys        Continuous-time state-space system (Ts == 0)
 * @param controller u = controller(y)
 * @param solver     ODE solver
 * @param x0         Initial state
 * @param t_span     {t0, tf}
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T, typename Controller, typename Solver>
SimulationResult<NX, NU, NY, T> simulate_lti(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    Controller&&                             controller,
    const Solver&                            solver,
    const ColVec<NX, T>&                     x0,
    const std::pair<T, T>&                   t_span
) {
    auto plant = [&](T /*t*/, const ColVec<NX, T>& x, const ColVec<NU, T>& u) -> ColVec<NX, T> {
        return sys.A * x + sys.B * u;
    };

    auto output = [&](const ColVec<NX, T>& x) -> ColVec<NY, T> {
        return sys.C * x;
    };

    return simulate<NX, NU, NY, T>(plant, output, controller, solver, x0, t_span);
}

/**
 * @brief Simulate a discrete-time system with a controller
 *
 * Steps the discrete system x_{k+1} = Ax_k + Bu_k with u_k = controller(y_k).
 *
 * @param sys        Discrete-time state-space system (Ts > 0)
 * @param controller u = controller(y)
 * @param x0         Initial state
 * @param n_steps    Number of time steps to simulate
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T, typename Controller>
SimulationResult<NX, NU, NY, T> simulate_discrete(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    Controller&&                             controller,
    const ColVec<NX, T>&                     x0,
    size_t                                   n_steps
) {
    SimulationResult<NX, NU, NY, T> sim;
    sim.t.reserve(n_steps + 1);
    sim.x.reserve(n_steps + 1);
    sim.y.reserve(n_steps + 1);
    sim.u.reserve(n_steps + 1);

    ColVec<NX, T> x = x0;
    T             t = T(0);

    for (size_t k = 0; k <= n_steps; ++k) {
        ColVec<NY, T> y = sys.C * x + sys.D * ColVec<NU, T>{};
        ColVec<NU, T> u = controller(y);

        sim.t.push_back(t);
        sim.x.push_back(x);
        sim.y.push_back(y);
        sim.u.push_back(u);

        if (k < n_steps) {
            x = sys.A * x + sys.B * u;
            t += sys.Ts;
        }
    }

    return sim;
}

} // namespace wetmelon::control
