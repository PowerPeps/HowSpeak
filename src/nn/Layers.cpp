#include "nn/Layers.hpp"

#include <cmath>
#include <span>

namespace howspeak::nn {

void Parameter::init(Index rows, Index cols, float scale, std::mt19937_64& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    value.resize(rows, cols);
    for (Index i = 0; i < value.size(); ++i) {
        value.data()[i] = dist(rng);
    }
    grad = Matrix::Zero(rows, cols);
    m = Matrix::Zero(rows, cols);
    v = Matrix::Zero(rows, cols);
}

Adam::Adam(float learningRate, float weightDecay, float beta1, float beta2, float epsilon)
    : learningRate_(learningRate), weightDecay_(weightDecay), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

void Adam::step(const std::vector<Parameter*>& parameters) {
    ++t_;
    const float c1 = 1.0F - std::pow(beta1_, static_cast<float>(t_));
    const float c2 = 1.0F - std::pow(beta2_, static_cast<float>(t_));
    for (Parameter* p : parameters) {
        p->m = beta1_ * p->m + (1.0F - beta1_) * p->grad;
        p->v = beta2_ * p->v + (1.0F - beta2_) * p->grad.cwiseProduct(p->grad);
        const float lr = learningRate_;
        const float eps = epsilon_;
        p->value.array() -= lr * ((p->m.array() / c1) / ((p->v.array() / c2).sqrt() + eps));
        if (weightDecay_ > 0.0F && p->value.cols() > 1) {
            p->value *= 1.0F - lr * weightDecay_;
        }
        p->zeroGrad();
    }
}

Linear::Linear(Index in, Index out, std::mt19937_64& rng) {
    weight.init(out, in, std::sqrt(6.0F / static_cast<float>(in)), rng);
    bias.init(out, 1, 0.0F, rng);
}

Matrix Linear::forward(const Matrix& x) const {
    Matrix y = weight.value * x;
    y.colwise() += bias.value.col(0);
    return y;
}

void Linear::accumulate(const Matrix& x, const Matrix& dy) {
    weight.grad.noalias() += dy * x.transpose();
    bias.grad.col(0) += dy.rowwise().sum();
}

Matrix Linear::backward(const Matrix& x, const Matrix& dy) {
    accumulate(x, dy);
    return weight.value.transpose() * dy;
}

void Linear::save(core::BinaryWriter& w) const {
    saveMatrix(w, weight.value);
    saveMatrix(w, bias.value);
}

void Linear::load(core::BinaryReader& r) {
    weight.value = loadMatrix(r);
    bias.value = loadMatrix(r);
    if (bias.value.rows() != weight.value.rows() || bias.value.cols() != 1) {
        r.fail("inconsistent linear layer");
    }
    weight.grad = Matrix::Zero(weight.value.rows(), weight.value.cols());
    weight.m = weight.grad;
    weight.v = weight.grad;
    bias.grad = Matrix::Zero(bias.value.rows(), 1);
    bias.m = bias.grad;
    bias.v = bias.grad;
}

Matrix relu(const Matrix& x) {
    return x.cwiseMax(0.0F);
}

Matrix reluBackward(const Matrix& dy, const Matrix& y) {
    return (y.array() > 0.0F).select(dy, 0.0F);
}

void saveMatrix(core::BinaryWriter& w, const Matrix& m) {
    w.write<std::int64_t>(m.rows());
    w.write<std::int64_t>(m.cols());
    w.writeArray(std::span<const float>(m.data(), static_cast<std::size_t>(m.size())));
}

Matrix loadMatrix(core::BinaryReader& r) {
    const auto rows = r.read<std::int64_t>();
    const auto cols = r.read<std::int64_t>();
    if (rows < 0 || cols < 0 || rows * cols > (1LL << 28)) {
        r.fail("invalid matrix shape");
    }
    Matrix m(rows, cols);
    r.readArray(std::span<float>(m.data(), static_cast<std::size_t>(m.size())));
    return m;
}

}
