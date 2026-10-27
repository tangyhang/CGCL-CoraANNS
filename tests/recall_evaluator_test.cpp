// Standalone, host-only regression tests (including the -Ofast build used by
// the benchmarks). Optional arguments: BASE QUERY GT, to time a real-data
// evaluation pass using GT ranks 2..11 as predictions for recall@10.
// g++ -std=c++17 -Ofast -march=native -fopenmp -Iinclude -Isample tests/recall_evaluator_test.cpp -o /tmp/recall_evaluator_test
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/utils/io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

using Floats = rabitqlib::RowMajorArray<float>;
using Ids = rabitqlib::RowMajorArray<std::uint32_t>;

namespace {

template <typename Function>
void rejects(Function&& function) {
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

struct Fixtures {
    std::filesystem::path directory;
    Fixtures() {
        char name[] = "/tmp/rabitq-recall-test.XXXXXX";
        const auto* path = ::mkdtemp(name);
        assert(path != nullptr);
        directory = path;
    }
    ~Fixtures() { std::filesystem::remove_all(directory); }

    std::string write(const Floats& base, bool fvecs) const {
        const auto path = (directory / (fvecs ? "base.fvecs" : "base.fbin")).string();
        std::ofstream output(path, std::ios::binary);
        const std::uint32_t rows = base.rows(), dim = base.cols();
        if (!fvecs) {
            output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        for (std::uint32_t i = 0; i < rows; ++i) {
            if (fvecs) output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            output.write(reinterpret_cast<const char*>(&base(i, 0)), dim * sizeof(float));
        }
        assert(output.good());
        return path;
    }
};

void test_options() {
    const auto parse = [](std::initializer_list<const char*> extra) {
        std::vector<std::string> arguments{"search", "index", "query", "gt"};
        for (const auto* text : extra) arguments.emplace_back(text);
        std::vector<char*> argv;
        for (auto& argument : arguments) argv.push_back(argument.data());
        int argc = argv.size();
        argv.push_back(nullptr);
        const auto path = recall_eval::take_base_option(argc, argv.data());
        std::vector<std::string> remaining;
        for (int i = 0; i < argc; ++i) remaining.emplace_back(argv[i]);
        assert(argv[argc] == nullptr);
        return std::make_pair(path, remaining);
    };
    assert(parse({}).first.empty());
    const auto legacy = parse({"10", "--recall-base", "/base with spaces.fbin"});
    assert(legacy.first == "/base with spaces.fbin");
    assert(legacy.second.size() == 5 && legacy.second.back() == "10");
    const auto named = parse({"--topk", "10", "--recall-base=x.fvecs", "--rounds", "2"});
    assert(named.first == "x.fvecs" && named.second.size() == 8);
    assert(named.second[4] == "--topk" && named.second.back() == "2");
    rejects([&] { parse({"--recall-base"}); });
    rejects([&] { parse({"--recall-base", "--topk", "10"}); });
    rejects([&] { parse({"--recall-base="}); });
    rejects([&] { parse({"--recall-base=a", "--recall-base", "b"}); });
}

void test_distance_and_ids(const std::string& base_file) {
    Floats queries(2, 2);
    queries << 0, 0, 2, 0;
    Ids gt(2, 3);
    gt << 0, 1, 2, 5, 1, 0;
    recall_eval::Evaluator evaluator(queries, gt, 3, base_file);
    const auto check = [&](std::size_t q, std::initializer_list<std::int64_t> ids) {
        const std::vector<std::int64_t> results(ids);
        return evaluator.count_query(q, results.data(), results.size());
    };
    auto counts = check(0, {0, 1, 2});
    assert(counts.id_hits == 3 && counts.exact_checks == 0);
    counts = check(0, {7, 3, 4});  // Duplicate vector and two different equal-distance vectors.
    assert(counts.id_hits == 0 && counts.distance_hits == 3 && counts.exact_checks == 3);
    counts = check(0, {7, 3, 4});
    assert(counts.correct() == 3 && counts.exact_checks == 0 && counts.cache_hits == 3);
    counts = check(0, {0, 3, 5});
    assert(counts.id_hits == 1 && counts.distance_hits == 1);
    assert(check(0, {6}).correct() == 0);  // Just outside radius; no epsilon.
    counts = check(0, {0, 0, 3});
    assert(counts.id_hits == 1 && counts.distance_hits == 1 && counts.total == 3);
    assert(check(0, {0, 1}).correct() == 2);  // Missing results remain in denominator.
    assert(check(0, {0, 1, 5, 2}).correct() == 2);  // Never examine beyond top-k.
    assert(check(0, {-1, 4294967295LL, 4294967297LL}).correct() == 0);
    assert(check(0, {11}).exact_checks == 0);  // Out-of-range base ID.
    assert(check(0, {8, 9}).correct() == 0);  // NaN / infinity are not hits, even at -Ofast.
    assert(check(1, {3}).correct() == 0);  // Per-query caches cannot leak q0's decision.
    assert(check(1, {7}).distance_hits == 1);  // Kth GT ID is 0, not largest ID (5).
    rejects([&] { evaluator.evaluate(std::vector<std::uint32_t>{0}); });

    recall_eval::Evaluator id_only(queries, gt, 3, "");
    const std::uint32_t predictions[] = {0, 3, 4};
    counts = id_only.count_query(0, predictions, 3);
    assert(counts.correct() == 1 && counts.exact_checks == 0);

    Floats zero_query(1, 2);
    zero_query << 0, 0;
    Ids zero_gt(1, 1);
    zero_gt << 0;
    recall_eval::Evaluator zero(zero_query, zero_gt, 1, base_file);
    const std::uint32_t duplicate = 7, near_zero = 10;
    assert(zero.count_query(0, &duplicate, 1).correct() == 1);
    assert(zero.count_query(0, &near_zero, 1).correct() == 0);

    Floats ip_query(1, 2);
    ip_query << 1, 0;
    Ids ip_gt(1, 2);
    ip_gt << 5, 1;
    recall_eval::Evaluator ip(ip_query, ip_gt, 2, base_file, rabitqlib::METRIC_IP);
    const std::uint32_t ip_ids[] = {6, 3};
    counts = ip.count_query(0, ip_ids, 2);
    assert(counts.id_hits == 0 && counts.distance_hits == 1);

    rejects([&] { recall_eval::Evaluator bad(queries, gt, 0, base_file); });
    rejects([&] { recall_eval::Evaluator bad(queries, gt, 4, base_file); });
    Floats wrong_dim(2, 3);
    rejects([&] { recall_eval::Evaluator bad(wrong_dim, gt, 3, base_file); });
    Ids bad_gt = gt;
    bad_gt(0, 0) = 11;
    rejects([&] { recall_eval::Evaluator bad(queries, bad_gt, 3, base_file); });
    bad_gt = gt;
    bad_gt(0, 1) = 0;
    rejects([&] { recall_eval::Evaluator bad(queries, bad_gt, 3, base_file); });
    bad_gt = gt;
    bad_gt(0, 2) = 8;
    rejects([&] { recall_eval::Evaluator bad(queries, bad_gt, 3, base_file); });
    rejects([&] { recall_eval::Evaluator bad(queries, gt, 3, base_file + ".missing"); });
}

void dataset_probe(const char* base, const char* query_file, const char* gt_file) {
    Floats queries;
    Ids gt;
    rabitqlib::load_auto<float, Floats>(query_file, queries);
    rabitqlib::load_auto<std::uint32_t, Ids>(gt_file, gt);
    assert(gt.cols() >= 20);
    constexpr std::size_t topk = 10;
    const auto started = std::chrono::steady_clock::now();
    recall_eval::Evaluator evaluator(queries, gt, topk, base);
    std::cout << "[Probe] prepare_ms=" << std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count() << '\n';
    std::vector<std::uint32_t> results(queries.rows() * topk);
    for (std::size_t q = 0; q < static_cast<std::size_t>(queries.rows()); ++q) {
        for (std::size_t k = 0; k < topk; ++k) results[q * topk + k] = gt(q, k + 1);
    }
    const double first = evaluator.evaluate(results);
    const double second = evaluator.evaluate(results);
    assert(first == second);
    if (queries.rows() > 5) {
        std::vector<std::uint32_t> alternatives(topk);
        for (std::size_t k = 0; k < topk; ++k) alternatives[k] = gt(5, k + topk);
        const auto counts = evaluator.count_query(5, alternatives.data(), topk);
        std::cout << "[Probe] query6_GT_ranks11_to20 ";
        evaluator.report(counts);
    }
}

void test_parallel_cache(const std::string& base_file) {
    constexpr std::size_t nq = 256, topk = 3;
    Floats queries(nq, 2);
    Ids gt(nq, topk);
    std::vector<std::uint32_t> results(nq * topk);
    for (std::size_t q = 0; q < nq; ++q) {
        queries(q, 0) = q % 2 ? 2.0f : 0.0f;
        queries(q, 1) = 0;
        gt(q, 0) = q % 2 ? 5 : 0;
        gt(q, 1) = 1;
        gt(q, 2) = q % 2 ? 0 : 2;
        results[q * topk] = 3;
        results[q * topk + 1] = 7;
        results[q * topk + 2] = 6;
    }
    recall_eval::Evaluator evaluator(queries, gt, topk, base_file);
    assert(evaluator.evaluate(results) == 2.0 / 3.0);
    assert(evaluator.evaluate(results) == 2.0 / 3.0);
}

void test_paper_topks(const Fixtures& fixtures) {
    Floats base = Floats::Zero(400, 768);
    Floats query = Floats::Zero(1, 768);
    Ids gt(1, 200);
    for (std::size_t k = 0; k < 200; ++k) gt(0, k) = k;
    const auto path = fixtures.write(base, false);
    for (const std::size_t topk : {10, 50, 100, 200}) {
        recall_eval::Evaluator evaluator(query, gt, topk, path);
        std::vector<std::uint32_t> results(topk);
        for (std::size_t k = 0; k < topk; ++k) results[k] = 200 + k;
        auto counts = evaluator.count_query(0, results.data(), topk);
        assert(counts.id_hits == 0 && counts.distance_hits == topk);
        results.back() = std::numeric_limits<std::uint32_t>::max();
        counts = evaluator.count_query(0, results.data(), topk);
        assert(counts.correct() == topk - 1 && counts.total == topk);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4) {
        dataset_probe(argv[1], argv[2], argv[3]);
        return 0;
    }
    test_options();
    Fixtures fixtures;
    Floats base(11, 2);
    base << 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 2, 0,
        std::nextafter(1.0f, 2.0f), 0, 0, 0,
        std::numeric_limits<float>::quiet_NaN(), 0,
        std::numeric_limits<float>::infinity(), 0, 1e-9f, 0;
    test_distance_and_ids(fixtures.write(base, false));
    test_distance_and_ids(fixtures.write(base, true));
    test_parallel_cache(fixtures.write(base, false));
    test_paper_topks(fixtures);
    std::cout << "recall evaluator: all tests passed\n";
}
