#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rabitq_benchmark {

inline bool starts_with_option(const char* text) {
    return text != nullptr && text[0] == '-' && text[1] == '-';
}

class CliOptions {
public:
    CliOptions(int argc, char** argv, int begin) {
        for (int i = begin; i < argc; ++i) {
            std::string argument(argv[i]);
            if (argument.rfind("--", 0) != 0) {
                throw std::invalid_argument(
                    "Unexpected positional argument: " + argument);
            }

            argument.erase(0, 2);
            const auto equals = argument.find('=');
            std::string key;
            std::string value;
            if (equals != std::string::npos) {
                key = argument.substr(0, equals);
                value = argument.substr(equals + 1);
            } else {
                key = argument;
                if (i + 1 >= argc || starts_with_option(argv[i + 1])) {
                    throw std::invalid_argument("Missing value for --" + key);
                }
                value = argv[++i];
            }
            if (key.empty() || values_.count(key) != 0) {
                throw std::invalid_argument("Invalid or duplicate option --" + key);
            }
            values_.emplace(std::move(key), std::move(value));
        }
    }

    void require_known(const std::vector<std::string>& known) const {
        for (const auto& entry : values_) {
            if (std::find(known.begin(), known.end(), entry.first) == known.end()) {
                throw std::invalid_argument("Unknown option --" + entry.first);
            }
        }
    }

    bool contains(const std::string& key) const {
        return values_.count(key) != 0;
    }

    std::string get(const std::string& key, const std::string& fallback) const {
        const auto iterator = values_.find(key);
        return iterator == values_.end() ? fallback : iterator->second;
    }

    std::size_t get_size(const std::string& key, std::size_t fallback) const {
        const auto text = get(key, "");
        if (text.empty()) return fallback;
        if (text.front() == '-' || text.front() == '+') {
            throw std::invalid_argument("Invalid integer for --" + key + ": " + text);
        }
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed);
        if (consumed != text.size() ||
            value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("Invalid integer for --" + key + ": " + text);
        }
        return static_cast<std::size_t>(value);
    }

    bool get_bool(const std::string& key, bool fallback) const {
        const auto text = get(key, fallback ? "true" : "false");
        if (text == "true" || text == "1" || text == "yes" || text == "on") {
            return true;
        }
        if (text == "false" || text == "0" || text == "no" || text == "off") {
            return false;
        }
        throw std::invalid_argument(
            "Invalid boolean for --" + key + ": " + text);
    }

private:
    std::unordered_map<std::string, std::string> values_;
};

inline std::vector<std::size_t> parse_nprobes(
    const std::string& text,
    std::size_t nlist,
    const std::vector<std::size_t>& defaults = {
        40, 50, 60, 75, 95, 128, 180, 255, 400, 535, 775, 1400}) {
    std::vector<std::size_t> result;
    if (text.empty()) {
        for (const auto value : defaults) {
            if (value <= nlist) result.push_back(value);
        }
        if (result.empty()) result.push_back(nlist);
        return result;
    }

    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto comma = text.find(',', begin);
        const auto token = text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (token.empty() || token.front() == '-' || token.front() == '+') {
            throw std::invalid_argument("Invalid value in --nprobes: " + token);
        }
        std::size_t consumed = 0;
        const auto value = std::stoull(token, &consumed);
        if (consumed != token.size() || value == 0 || value > nlist) {
            throw std::invalid_argument(
                "Each nprobe must be in [1, nlist]; got: " + token);
        }
        const auto parsed = static_cast<std::size_t>(value);
        if (std::find(result.begin(), result.end(), parsed) != result.end()) {
            throw std::invalid_argument("Duplicate nprobe value: " + token);
        }
        result.push_back(parsed);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return result;
}

inline std::string join_values(const std::vector<std::size_t>& values) {
    std::string result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) result.push_back(',');
        result += std::to_string(values[i]);
    }
    return result;
}

inline void validate_benchmark_shape(
    std::size_t query_rows,
    std::size_t query_dim,
    std::size_t gt_rows,
    std::size_t gt_dim,
    std::size_t index_dim,
    std::size_t topk,
    std::size_t rounds) {
    if (topk == 0) throw std::invalid_argument("topk must be positive");
    if (rounds == 0) throw std::invalid_argument("rounds must be positive");
    if (query_rows == 0 || query_dim == 0) {
        throw std::invalid_argument("The query set must not be empty");
    }
    if (query_rows != gt_rows) {
        throw std::invalid_argument(
            "Query/ground-truth row mismatch: " + std::to_string(query_rows) +
            " vs " + std::to_string(gt_rows));
    }
    if (query_dim != index_dim) {
        throw std::invalid_argument(
            "Query/index dimension mismatch: " + std::to_string(query_dim) +
            " vs " + std::to_string(index_dim));
    }
    if (gt_dim < topk) {
        throw std::invalid_argument(
            "Ground truth has only " + std::to_string(gt_dim) +
            " neighbors per query, fewer than topk=" + std::to_string(topk));
    }
}

}  // namespace rabitq_benchmark
