#include "nn/Attention.hpp"

#include <cmath>

namespace howspeak::nn {

namespace {

// Residual branches start small so that the stack begins close to the identity.
void shrink(Linear& layer, float factor) {
    layer.weight.value *= factor;
}

}

AttentionBlock::AttentionBlock(Index width, Index heads, Index feedForward, std::mt19937_64& rng)
    : heads_(heads),
      q_(width, width, rng),
      k_(width, width, rng),
      v_(width, width, rng),
      o_(width, width, rng),
      f1_(width, feedForward, rng),
      f2_(feedForward, width, rng) {
    shrink(q_, 0.5F);
    shrink(k_, 0.5F);
    shrink(o_, 0.2F);
    shrink(f2_, 0.2F);
}

Matrix AttentionBlock::forward(const Matrix& x, const Ranges& sets, Cache& c) const {
    const Index width = x.rows();
    const Index dk = width / heads_;
    const float scale = 1.0F / std::sqrt(static_cast<float>(dk));
    c.x = x;
    c.q = q_.forward(x);
    c.k = k_.forward(x);
    c.v = v_.forward(x);
    c.mixed = Matrix::Zero(width, x.cols());
    c.weights.assign(sets.size() * static_cast<std::size_t>(heads_), Matrix());

#pragma omp parallel for schedule(dynamic, 8)
    for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(sets.size()); ++s) {
        const auto [begin, end] = sets[static_cast<std::size_t>(s)];
        const Index n = end - begin;
        for (Index h = 0; h < heads_; ++h) {
            Matrix p = (c.q.block(h * dk, begin, dk, n).transpose() * c.k.block(h * dk, begin, dk, n)) * scale;
            for (Index i = 0; i < n; ++i) {
                auto row = p.row(i);
                row.array() -= row.maxCoeff();
                row = row.array().exp();
                row /= row.sum();
            }
            c.mixed.block(h * dk, begin, dk, n).noalias() = c.v.block(h * dk, begin, dk, n) * p.transpose();
            c.weights[static_cast<std::size_t>(s) * static_cast<std::size_t>(heads_) + static_cast<std::size_t>(h)] =
                std::move(p);
        }
    }

    c.u = x + o_.forward(c.mixed);
    c.hidden = relu(f1_.forward(c.u));
    return c.u + f2_.forward(c.hidden);
}

Matrix AttentionBlock::backward(const Matrix& dy, const Ranges& sets, const Cache& c) {
    const Index width = c.x.rows();
    const Index dk = width / heads_;
    const float scale = 1.0F / std::sqrt(static_cast<float>(dk));

    Matrix du = dy + f1_.backward(c.u, reluBackward(f2_.backward(c.hidden, dy), c.hidden));
    const Matrix dMixed = o_.backward(c.mixed, du);
    Matrix dq = Matrix::Zero(width, c.x.cols());
    Matrix dk_ = Matrix::Zero(width, c.x.cols());
    Matrix dv = Matrix::Zero(width, c.x.cols());

#pragma omp parallel for schedule(dynamic, 8)
    for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(sets.size()); ++s) {
        const auto [begin, end] = sets[static_cast<std::size_t>(s)];
        const Index n = end - begin;
        for (Index h = 0; h < heads_; ++h) {
            const Matrix& p =
                c.weights[static_cast<std::size_t>(s) * static_cast<std::size_t>(heads_) + static_cast<std::size_t>(h)];
            const auto dOut = dMixed.block(h * dk, begin, dk, n);
            const auto values = c.v.block(h * dk, begin, dk, n);
            dv.block(h * dk, begin, dk, n).noalias() = dOut * p;
            const Matrix dp = dOut.transpose() * values;
            const Eigen::VectorXf dot = (dp.cwiseProduct(p)).rowwise().sum();
            const Matrix ds = (p.array() * (dp.colwise() - dot).array()).matrix() * scale;
            dq.block(h * dk, begin, dk, n).noalias() = c.k.block(h * dk, begin, dk, n) * ds.transpose();
            dk_.block(h * dk, begin, dk, n).noalias() = c.q.block(h * dk, begin, dk, n) * ds;
        }
    }

    Matrix dx = du;
    dx += q_.backward(c.x, dq);
    dx += k_.backward(c.x, dk_);
    dx += v_.backward(c.x, dv);
    return dx;
}

void AttentionBlock::collect(std::vector<Parameter*>& out) {
    for (Linear* l : {&q_, &k_, &v_, &o_, &f1_, &f2_}) {
        l->collect(out);
    }
}

void AttentionBlock::save(core::BinaryWriter& w) const {
    w.write<std::int64_t>(heads_);
    for (const Linear* l : {&q_, &k_, &v_, &o_, &f1_, &f2_}) {
        l->save(w);
    }
}

void AttentionBlock::load(core::BinaryReader& r) {
    heads_ = r.read<std::int64_t>();
    if (heads_ <= 0) {
        r.fail("invalid head count");
    }
    for (Linear* l : {&q_, &k_, &v_, &o_, &f1_, &f2_}) {
        l->load(r);
    }
}

}
