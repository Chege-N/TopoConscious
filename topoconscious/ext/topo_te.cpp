/*
 * topo_te.cpp  -  C++17 / OpenMP extension for fast Transfer Entropy
 * between topological feature time series.
 *
 * Build:
 *   g++ -O3 -std=c++17 -fopenmp -shared -fPIC -o _topo_te.so topo_te.cpp
 *
 * Python bindings via pybind11 (see setup.py).
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <omp.h>

namespace py = pybind11;

/* ---------- helpers ---------- */
static int digitize(double v, double lo, double hi, int bins) {
    if (hi <= lo) return 0;
    int b = static_cast<int>((v - lo) / (hi - lo) * bins);
    return std::clamp(b, 0, bins - 1);
}

static double cond_entropy(
    const std::vector<int>& y,
    const std::vector<int>& cond,
    int n_cond, int n_y_bins)
{
    using Map = std::unordered_map<int, std::unordered_map<int,int>>;
    Map joint;
    std::unordered_map<int,int> cond_counts;
    int n = (int)y.size();

    for (int i = 0; i < n; ++i) {
        joint[cond[i]][y[i]]++;
        cond_counts[cond[i]]++;
    }

    double h = 0.0;
    for (auto& [c, ymap] : joint) {
        double pc = static_cast<double>(cond_counts[c]) / n;
        double h_y = 0.0;
        for (auto& [yv, cnt] : ymap) {
            double p = static_cast<double>(cnt) / cond_counts[c];
            h_y -= p * std::log2(p + 1e-12);
        }
        h += pc * h_y;
    }
    return h;
}

/* ---------- main function ---------- */
py::array_t<double> transfer_entropy_matrix(
    py::array_t<double> region_ts_np,  /* (n_windows, n_regions) */
    int lag,
    int n_bins)
{
    auto buf = region_ts_np.request();
    int n_w = buf.shape[0];
    int n_r = buf.shape[1];
    double* data = static_cast<double*>(buf.ptr);

    std::vector<double> rmin(n_r, 1e300), rmax(n_r, -1e300);
    for (int w = 0; w < n_w; ++w)
        for (int r = 0; r < n_r; ++r) {
            double v = data[w * n_r + r];
            rmin[r] = std::min(rmin[r], v);
            rmax[r] = std::max(rmax[r], v);
        }

    std::vector<std::vector<int>> disc(n_r, std::vector<int>(n_w));
    for (int r = 0; r < n_r; ++r)
        for (int w = 0; w < n_w; ++w)
            disc[r][w] = digitize(data[w*n_r+r], rmin[r], rmax[r], n_bins);

    int n = n_w - lag;
    std::vector<double> te_flat(n_r * n_r, 0.0);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n_r; ++i) {
        for (int j = 0; j < n_r; ++j) {
            if (i == j) continue;
            const auto& xi = disc[i];
            const auto& xj = disc[j];

            std::vector<int> yt (n), yt1(n), xt1(n), joint_cond(n);
            for (int t = 0; t < n; ++t) {
                yt [t] = xj[t + lag];
                yt1[t] = xj[t];
                xt1[t] = xi[t];
                joint_cond[t] = yt1[t] * n_bins + xt1[t];
            }

            double h1 = cond_entropy(yt, yt1, n_bins, n_bins);
            double h2 = cond_entropy(yt, joint_cond, n_bins*n_bins, n_bins);
            te_flat[i * n_r + j] = std::max(0.0, h1 - h2);
        }
    }

    auto result = py::array_t<double>({n_r, n_r});
    auto rbuf = result.request();
    double* rptr = static_cast<double*>(rbuf.ptr);
    std::copy(te_flat.begin(), te_flat.end(), rptr);
    return result;
}

PYBIND11_MODULE(_topo_te, m) {
    m.doc() = "Fast C++/OpenMP Transfer Entropy for TopoConscious";
    m.def("transfer_entropy_matrix", &transfer_entropy_matrix,
          "Compute n_regions x n_regions TE matrix from region time series",
          py::arg("region_ts"), py::arg("lag")=1, py::arg("n_bins")=10);
}
