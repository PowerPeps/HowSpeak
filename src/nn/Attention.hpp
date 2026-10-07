#pragma once

#include "nn/Layers.hpp"

#include <utility>
#include <vector>

namespace howspeak::nn {

using Ranges = std::vector<std::pair<Index, Index>>;  // [begin, end) columns of each set

// Pre-activation-free transformer block over independent sets of columns:
// u = x + O(MultiHeadAttention(x)), y = u + F2(relu(F1(u))). Attention never crosses sets.
class AttentionBlock {
public:
    struct Cache {
        Matrix x;
        Matrix q;
        Matrix k;
        Matrix v;
        Matrix mixed;
        Matrix u;
        Matrix hidden;
        std::vector<Matrix> weights;  // per set and head, row-stochastic
    };

    AttentionBlock() = default;
    AttentionBlock(Index width, Index heads, Index feedForward, std::mt19937_64& rng);

    [[nodiscard]] Matrix forward(const Matrix& x, const Ranges& sets, Cache& cache) const;
    // Accumulates parameter gradients; returns the gradient with respect to x.
    [[nodiscard]] Matrix backward(const Matrix& dy, const Ranges& sets, const Cache& cache);

    void collect(std::vector<Parameter*>& out);
    void save(core::BinaryWriter& w) const;
    void load(core::BinaryReader& r);

private:
    Index heads_ = 1;
    Linear q_;
    Linear k_;
    Linear v_;
    Linear o_;
    Linear f1_;
    Linear f2_;
};

}
