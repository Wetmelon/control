#pragma once

/**
 * @defgroup plot_export Plot Data Export
 * @brief CSV and gnuplot export utilities for control system visualization
 *
 * Enables users to export frequency response, step response, and time-domain
 * simulation data from unit tests for plotting with external tools.
 *
 * Supported formats:
 * - CSV (comma-separated values) — importable by any spreadsheet or plotting tool
 * - Gnuplot script — auto-generates a .gp file that produces plots
 *
 * Usage in user tests:
 * @code
 *   auto bode_data = analysis::bode(sys, omega);
 *   plot::to_csv("my_bode.csv", bode_data);
 *   plot::bode_to_gnuplot("my_bode", bode_data);  // creates .csv + .gp
 * @endcode
 */

#include <cstdio>
#include <string>
#include <vector>

#include "analysis.hpp"
#include "state_space.hpp"
#include "utility.hpp"

namespace wetmelon::control {
namespace plot {

/**
 * @brief Export Bode plot data to CSV
 *
 * Columns: frequency_rad_s, magnitude, magnitude_dB, phase_deg
 *
 * @param filename Output CSV file path
 * @param data     BodeResult from analysis::bode()
 * @return true on success, false on file open failure
 */
template<typename T>
bool to_csv(const std::string& filename, const analysis::BodeResult<T>& data) {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) {
        return false;
    }
    std::fprintf(f, "frequency_rad_s,magnitude,magnitude_dB,phase_deg\n");
    for (const auto& pt : data.points) {
        std::fprintf(f, "%.10g,%.10g,%.10g,%.10g\n", static_cast<double>(pt.omega), static_cast<double>(pt.magnitude), static_cast<double>(pt.magnitude_db), static_cast<double>(pt.phase_deg));
    }
    std::fclose(f);
    return true;
}

/**
 * @brief Export time-domain data to CSV
 *
 * Generic time/value export for step responses, simulations, etc.
 *
 * @param filename Output CSV file path
 * @param time     Time vector
 * @param values   Value vector (same length as time)
 * @param header   Column names (default: "time,value")
 * @return true on success
 */
template<typename T>
bool to_csv(const std::string& filename, const std::vector<T>& time, const std::vector<T>& values, const std::string& header = "time,value") {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) {
        return false;
    }
    std::fprintf(f, "%s\n", header.c_str());
    size_t n = (time.size() < values.size()) ? time.size() : values.size();
    for (size_t i = 0; i < n; ++i) {
        std::fprintf(f, "%.10g,%.10g\n", static_cast<double>(time[i]), static_cast<double>(values[i]));
    }
    std::fclose(f);
    return true;
}

/**
 * @brief Export multi-channel time-domain data to CSV
 *
 * @param filename Output CSV file path
 * @param time     Time vector
 * @param channels Vector of value vectors (one per channel)
 * @param names    Channel names for CSV header
 * @return true on success
 */
template<typename T>
bool to_csv(const std::string& filename, const std::vector<T>& time, const std::vector<std::vector<T>>& channels, const std::vector<std::string>& names) {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) {
        return false;
    }
    // Header
    std::fprintf(f, "time");
    for (const auto& name : names) {
        std::fprintf(f, ",%s", name.c_str());
    }
    std::fprintf(f, "\n");
    // Data
    for (size_t i = 0; i < time.size(); ++i) {
        std::fprintf(f, "%.10g", static_cast<double>(time[i]));
        for (const auto& ch : channels) {
            if (i < ch.size()) {
                std::fprintf(f, ",%.10g", static_cast<double>(ch[i]));
            } else {
                std::fprintf(f, ",");
            }
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return true;
}

/**
 * @brief Generate a gnuplot script for Bode plot visualization
 *
 * Creates both a CSV data file and a .gp gnuplot script.
 * Run with: gnuplot my_bode.gp
 * Or in test: system("gnuplot my_bode.gp") to auto-generate PNG.
 *
 * @param basename Base filename (without extension). Creates basename.csv and basename.gp
 * @param data     BodeResult from analysis::bode()
 * @param title    Plot title (optional)
 * @return true on success
 */
template<typename T>
bool bode_to_gnuplot(const std::string& basename, const analysis::BodeResult<T>& data, const std::string& title = "Bode Plot") {
    // Write CSV data
    std::string csv_file = basename + ".csv";
    if (!to_csv(csv_file, data)) {
        return false;
    }

    // Write gnuplot script
    std::string gp_file = basename + ".gp";
    FILE*       f = std::fopen(gp_file.c_str(), "w");
    if (!f) {
        return false;
    }

    std::fprintf(f, "set terminal pngcairo size 800,600 enhanced\n");
    std::fprintf(f, "set output '%s.png'\n", basename.c_str());
    std::fprintf(f, "set multiplot layout 2,1 title '%s'\n", title.c_str());
    std::fprintf(f, "set datafile separator ','\n");
    std::fprintf(f, "\n");
    std::fprintf(f, "# Magnitude plot\n");
    std::fprintf(f, "set xlabel 'Frequency (rad/s)'\n");
    std::fprintf(f, "set ylabel 'Magnitude (dB)'\n");
    std::fprintf(f, "set logscale x\n");
    std::fprintf(f, "set grid\n");
    std::fprintf(f, "set key off\n");
    std::fprintf(f, "plot '%s' skip 1 using 1:3 with lines lw 2 lc rgb '#0060ad'\n", csv_file.c_str());
    std::fprintf(f, "\n");
    std::fprintf(f, "# Phase plot\n");
    std::fprintf(f, "set xlabel 'Frequency (rad/s)'\n");
    std::fprintf(f, "set ylabel 'Phase (deg)'\n");
    std::fprintf(f, "set logscale x\n");
    std::fprintf(f, "set grid\n");
    std::fprintf(f, "set key off\n");
    std::fprintf(f, "plot '%s' skip 1 using 1:4 with lines lw 2 lc rgb '#dd181f'\n", csv_file.c_str());
    std::fprintf(f, "\n");
    std::fprintf(f, "unset multiplot\n");

    std::fclose(f);
    return true;
}

/**
 * @brief Generate a gnuplot script for step response visualization
 *
 * @param basename Base filename (without extension)
 * @param time     Time vector
 * @param values   Step response values
 * @param title    Plot title
 * @return true on success
 */
template<typename T>
bool step_to_gnuplot(const std::string& basename, const std::vector<T>& time, const std::vector<T>& values, const std::string& title = "Step Response") {
    std::string csv_file = basename + ".csv";
    if (!to_csv(csv_file, time, values)) {
        return false;
    }

    std::string gp_file = basename + ".gp";
    FILE*       f = std::fopen(gp_file.c_str(), "w");
    if (!f) {
        return false;
    }

    std::fprintf(f, "set terminal pngcairo size 800,400 enhanced\n");
    std::fprintf(f, "set output '%s.png'\n", basename.c_str());
    std::fprintf(f, "set title '%s'\n", title.c_str());
    std::fprintf(f, "set datafile separator ','\n");
    std::fprintf(f, "set xlabel 'Time (s)'\n");
    std::fprintf(f, "set ylabel 'Output'\n");
    std::fprintf(f, "set grid\n");
    std::fprintf(f, "set key off\n");
    std::fprintf(f, "plot '%s' skip 1 using 1:2 with lines lw 2 lc rgb '#0060ad'\n", csv_file.c_str());

    std::fclose(f);
    return true;
}

/**
 * @brief Simulate step response of a discrete-time SISO system
 *
 * Runs the state-space equations forward in time with a unit step input.
 *
 * @param sys     Discrete-time SISO state-space system
 * @param n_steps Number of time steps to simulate
 * @return pair of {time_vector, output_vector}
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr std::pair<std::vector<T>, std::vector<T>>
step_response(const StateSpace<NX, 1, 1, NW, NV, T>& sys, size_t n_steps) {
    std::vector<T> time;
    std::vector<T> output;
    time.reserve(n_steps);
    output.reserve(n_steps);

    ColVec<NX, T>   x = ColVec<NX, T>::zeros();
    Matrix<1, 1, T> u{T{1}};

    for (size_t k = 0; k < n_steps; ++k) {
        time.push_back(static_cast<T>(k) * sys.Ts);
        auto y = sys.C * x + sys.D * u;
        output.push_back(y(0, 0));
        x = sys.A * x + sys.B * u;
    }
    return {time, output};
}

/**
 * @brief Simulate impulse response of a discrete-time SISO system
 *
 * @param sys     Discrete-time SISO state-space system
 * @param n_steps Number of time steps
 * @return pair of {time_vector, output_vector}
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr std::pair<std::vector<T>, std::vector<T>>
impulse_response(const StateSpace<NX, 1, 1, NW, NV, T>& sys, size_t n_steps) {
    std::vector<T> time;
    std::vector<T> output;
    time.reserve(n_steps);
    output.reserve(n_steps);

    ColVec<NX, T> x = ColVec<NX, T>::zeros();

    for (size_t k = 0; k < n_steps; ++k) {
        time.push_back(static_cast<T>(k) * sys.Ts);
        Matrix<1, 1, T> u{(k == 0) ? T{1} / sys.Ts : T{0}}; // Scaled impulse
        auto            y = sys.C * x + sys.D * u;
        output.push_back(y(0, 0));
        x = sys.A * x + sys.B * u;
    }
    return {time, output};
}

} // namespace plot
} // namespace wetmelon::control
