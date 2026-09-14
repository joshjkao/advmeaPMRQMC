#include "parameters.hpp"
#include <bitset>
#include <fstream>
#include <mpi.h>
#include <random>
#include <ranges>

// template <size_t NBITS>
// double calculate_Oi(const std::bitset<NBITS> &l) {
//   return static_cast<double>(l.to_ullong());
// }

#ifdef TEST_OPERATOR_RANDOM_ZSUM
template <size_t NBITS> double calculate_Oi(const std::bitset<NBITS> &l) {
  static std::array<double, NBITS> coeffs = [] {
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::array<double, NBITS> ret{};
    for (size_t ii = 0; ii < NBITS; ++ii) {
      ret[ii] = dist(gen);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::ofstream ofile("O.txt");
      for (const auto &[i, c] : ret | std::views::enumerate) {
        ofile << c << " " << (i + 1) << " Z" << "\n";
      }
    }
    return ret;
  }();
  double O_val = 0.0;

  for (size_t i = 0; i < NBITS; ++i) {
    // Map bit 0 -> spin -1.0, bit 1 -> spin +1.0
    // l.test(i) is safer and faster than l[i] for std::bitset
    double spin = l.test(i) ? 1.0 : -1.0;

    O_val += coeffs[i] * spin;
  }

  return O_val;
}
#else
static_assert(false, "A test operator is not specified");
#endif
