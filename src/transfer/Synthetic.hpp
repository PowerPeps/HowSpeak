#pragma once

#include "syntax/Sentence.hpp"
#include "transfer/Grammar.hpp"

#include <cstdint>
#include <random>

namespace howspeak::transfer {

// Random dependency trees covering the constructions the WALS features talk about.
// Words carry no lexical content: the network only ever sees structure.
class SyntheticTreebank {
public:
    explicit SyntheticTreebank(std::uint64_t seed) : rng_(seed) {}

    // A tree linearised by the given source grammar (orders are set accordingly).
    [[nodiscard]] syntax::Sentence next(const Grammar& source);

private:
    enum class Clause { Main, Relative, Complement, Adverbial, Coordinate };

    int add(std::string upos, std::string deprel, int head, syntax::Features feats = {});
    int clause(int head, std::string deprel, int depth, Clause kind);
    int nounPhrase(int head, std::string deprel, int depth, bool interrogative);
    bool chance(double p) { return std::bernoulli_distribution(p)(rng_); }

    std::mt19937_64 rng_;
    syntax::Sentence s_;
};

}
