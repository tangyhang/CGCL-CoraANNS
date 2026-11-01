#include "coraanns/search/parameters.hpp"
#include <cassert>
#include <stdexcept>

template <class F> void rejects(F &&f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}
int main() {
    using coraanns::validate_search_shape;
    validate_search_shape(10000, 100, 640, 65536, 128);
    validate_search_shape(65535, 10, 8, 8, 128);
    validate_search_shape(1, 100, 65536, 65536, 128);
    rejects([] { validate_search_shape(65536, 10, 8, 8, 128); });
    rejects([] { validate_search_shape(1, 10, 65537, 65537, 128); });
    rejects([] { validate_search_shape(0, 10, 8, 8, 128); });
    rejects([] { validate_search_shape(1, 0, 8, 8, 128); });
    rejects([] { validate_search_shape(1, 10, 0, 8, 128); });
    rejects([] { validate_search_shape(1, 10, 9, 8, 128); });
    rejects([] { validate_search_shape(65535, 10, 65536, 65536, 128); });
}
