#pragma once

// Inclusive wall-clock timings. Nested buckets must not be added together.
// All instrumentation disappears when profiling is disabled at build time.
#ifdef MLIR_PSP_ENABLE_PROFILING
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace MLIRPSP::Profiling {
enum Bucket { Lift, Lower, Compile, Dispatch, Execute, Entry, Read, Write,
    Passes, EngineCreate, Initialize, Lookup, Count };
struct Counters {
    std::array<uint64_t, Count> calls{};
    std::array<uint64_t, Count> ns{};
    ~Counters() {
        bool used = false;
        for (auto count : calls)
            used = used || count != 0;
        if (!used)
            return;
        static constexpr const char *names[] = {
            "lift", "lower", "compile", "dispatch", "execute", "entry", "read", "write",
            "passes", "engine_create", "initialize", "lookup"};
        for (int i = 0; i < Count; ++i)
            std::fprintf(stderr, "MLIR_PROFILE bucket=%s calls=%llu inclusive_ms=%.3f\n",
                names[i], static_cast<unsigned long long>(calls[i]), ns[i] / 1.0e6);
    }
};
inline thread_local Counters counters;
class Scope {
    Bucket bucket;
    std::chrono::steady_clock::time_point start;
public:
    explicit Scope(Bucket b) : bucket(b), start(std::chrono::steady_clock::now()) {}
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() {
        counters.ns[bucket] += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        ++counters.calls[bucket];
    }
};
}
#define MLIR_PSP_PROFILE(bucket) MLIRPSP::Profiling::Scope profileScope(MLIRPSP::Profiling::bucket)
#else
#define MLIR_PSP_PROFILE(bucket) ((void)0)
#endif

#if defined(MLIR_PSP_ENABLE_PROFILING) && defined(MLIR_PSP_PROFILE_EXECUTION)
#define MLIR_PSP_PROFILE_HOT(bucket) MLIR_PSP_PROFILE(bucket)
#else
#define MLIR_PSP_PROFILE_HOT(bucket) ((void)0)
#endif
