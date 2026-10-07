#pragma once

#include "core/Binary.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <random>
#include <vector>

namespace howspeak::nn {

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using Index = Eigen::Index;

struct Parameter {
    Matrix value;
    Matrix grad;
    Matrix m;
    Matrix v;

    void init(Index rows, Index cols, float scale, std::mt19937_64& rng);
    void zeroGrad() { grad.setZero(); }
};

class Adam {
public:
    explicit Adam(float learningRate, float weightDecay = 0.0F, float beta1 = 0.9F, float beta2 = 0.999F,
                  float epsilon = 1e-8F);

    void step(const std::vector<Parameter*>& parameters);
    void setLearningRate(float lr) noexcept { learningRate_ = lr; }

private:
    float learningRate_;
    float weightDecay_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_ = 0;
};

// Column-major batches: one example per column.
class Linear {
public:
    Linear() = default;
    Linear(Index in, Index out, std::mt19937_64& rng);

    [[nodiscard]] Matrix forward(const Matrix& x) const;
    // Accumulates parameter gradients; returns the gradient with respect to x.
    [[nodiscard]] Matrix backward(const Matrix& x, const Matrix& dy);
    void accumulate(const Matrix& x, const Matrix& dy);

    [[nodiscard]] Index inputs() const noexcept { return weight.value.cols(); }
    [[nodiscard]] Index outputs() const noexcept { return weight.value.rows(); }
    void collect(std::vector<Parameter*>& out) { out.push_back(&weight); out.push_back(&bias); }
    void save(core::BinaryWriter& w) const;
    void load(core::BinaryReader& r);

    Parameter weight;
    Parameter bias;
};

[[nodiscard]] Matrix relu(const Matrix& x);
[[nodiscard]] Matrix reluBackward(const Matrix& dy, const Matrix& y);

void saveMatrix(core::BinaryWriter& w, const Matrix& m);
[[nodiscard]] Matrix loadMatrix(core::BinaryReader& r);

}
