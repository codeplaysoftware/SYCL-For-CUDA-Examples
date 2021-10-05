#pragma once

#include <string>
#include <iostream>
#include <chrono>

/**
 * Small Chrono class that prints the time spent in a scope.
 */
class Chrono {
public:
    inline Chrono();

    inline explicit Chrono(std::string &&caller_name);

    inline Chrono(const Chrono &) = delete;

    Chrono &operator=(const Chrono &) = delete;

    inline double stop();

    inline ~Chrono();

private:
    std::string caller;

    const std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::duration<long int, std::ratio<1, 1000000000>>> start;
};

inline Chrono::Chrono()
        : start(std::chrono::high_resolution_clock::now()) {
}

inline Chrono::~Chrono() {
    double elapsed_seconds = Chrono::stop();
    if (!caller.empty()) {
        std::cerr << "time in " << caller << " : " << elapsed_seconds << "s" << std::endl;
    } else {
        std::cerr << "time " << elapsed_seconds << "s" << std::endl;
    }
}

inline Chrono::Chrono(std::string &&caller_name)
        : Chrono() {
    caller = caller_name;
}

inline double Chrono::stop() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / 1000000.0;
}
