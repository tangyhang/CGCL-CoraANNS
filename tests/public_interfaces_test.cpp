#include "coraanns/search/hybrid.cuh"
#include "coraanns/baselines/gpu.cuh"
#include "coraanns/experimental/cpu_gpu.cuh"
#include "coraanns/experimental/gpu_global.cuh"
#include "coraanns/experimental/asymmetric.cuh"
#include "coraanns/experimental/backlog_trace.cuh"
#include <type_traits>

template <class Engine>
constexpr bool unique_owner =
    !std::is_copy_constructible_v<Engine> && !std::is_copy_assignable_v<Engine>;
static_assert(unique_owner<coraanns::HybridIndex>);
static_assert(unique_owner<coraanns::baselines::GpuIndex>);
static_assert(unique_owner<coraanns::experimental::cpu_gpu::Index>);
static_assert(unique_owner<coraanns::experimental::gpu_global::Index>);
static_assert(unique_owner<coraanns::experimental::asymmetric::Index>);
static_assert(unique_owner<coraanns::experimental::backlog_trace::Index>);
int main() {}
