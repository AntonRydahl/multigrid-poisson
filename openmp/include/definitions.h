#ifndef POISSON_DEFINITIONS
#define POISSON_DEFINITIONS
#include <cstdint>
namespace Poisson{
    #ifndef CHUNK_SIZE
    #define CHUNK_SIZE 1
    #endif

    using double_t = double;
    using float_t = float;
    using uint_t = std::size_t;
    using int_t = long int;
}
#endif
