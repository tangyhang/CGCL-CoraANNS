// Standalone host-only test; no CUDA runtime or large dataset is needed.
// g++ -std=c++17 -fopenmp -Isample tests/experiment_options_test.cpp -o /tmp/experiment_options_test
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "coraanns/benchmark/mechanism_options.hpp"

namespace {
rabitq_benchmark::ExperimentOptions parse(
    std::initializer_list<const char*> extra, std::size_t default_vanguard = 5) {
    std::vector<std::string> arguments{"experiment", "index", "query", "gt"};
    for (const auto* argument : extra) arguments.emplace_back(argument);
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return {static_cast<int>(argv.size()), argv.data(), default_vanguard};
}

void rejects(std::initializer_list<const char*> extra) {
    bool threw = false;
    try {
        (void)parse(extra);
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

struct FakeIndex {
    std::size_t vanguard = 999;
    bool secondary = true;
    bool profiling = true;
    void set_vanguard_size(std::size_t value) { vanguard = value; }
    void set_cpu_secondary_filter(bool value) { secondary = value; }
    void set_profiling(bool value) { profiling = value; }
};
}  // namespace

int main() {
    const auto defaults = parse({});
    assert(defaults.topk == 100 && defaults.rounds == 1 && defaults.warmup == 1);
    assert(defaults.vanguard_size == 5 && defaults.cpu_secondary_filter);
    assert(defaults.batch_q == 4 && defaults.threads == 72 && !defaults.profile);
    assert(parse({}, 0).vanguard_size == 0);
    assert(parse({}, 0).cpu_secondary_filter);
    assert(parse({"10"}).topk == 10);  // Legacy positional topk remains supported.

    const auto simple = parse({"--topk", "50", "--nprobes", "123,1966",
        "--rounds", "2", "--warmup", "0", "--batch-q", "6", "--threads", "1",
        "--vanguard-size", "0", "--cpu-secondary-filter", "false", "--profile", "false"});
    assert(simple.topk == 50 && simple.rounds == 2 && simple.warmup == 0);
    assert(simple.vanguard_size == 0 && !simple.cpu_secondary_filter);
    assert(simple.batch_q == 6 && simple.threads == 1);
    const auto nprobes = rabitq_benchmark::parse_nprobes(simple.cli.get("nprobes", "123"), 65536);
    assert((nprobes == std::vector<std::size_t>{123, 1966}));
    FakeIndex index;
    simple.configure(index);
    assert(index.vanguard == 0 && !index.secondary && !index.profiling);
    assert(omp_get_max_threads() == 1);

    assert(!parse({"--cpu-secondary-filter=0"}).cpu_secondary_filter);
    assert(parse({"--cpu-secondary-filter=true"}).cpu_secondary_filter);
    assert(parse({"--profile", "true"}).profile);
    assert(parse({"--output-dir", "results/a directory"}).output_dir == "results/a directory");
    rejects({"--cpu-secondary-filter", "invalid"});
    rejects({"--cpu-secondary-filter"});
    rejects({"--cpu-secondary-filter", "true", "--cpu-secondary-filter", "false"});
    rejects({"--topk", "0"});
    rejects({"--rounds", "0"});
    rejects({"--batch-q", "0"});
    rejects({"--batch-q", "7"});
    rejects({"--threads", "0"});
    rejects({"--threads", "2147483648"});
    rejects({"--vanguard-size", "-1"});
    rejects({"--output-dir="});
    rejects({"--unknown", "true"});
    rejects({"10", "--rounds", "1"});
    rejects({"-1"});
    rejects({"10bad"});
    std::cout << "experiment options: all tests passed\n";
}
