#pragma once

#include "nn/Layers.hpp"
#include "transfer/Grammar.hpp"
#include "transfer/Groups.hpp"
#include "transfer/Rules.hpp"

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace howspeak::transfer {

// Neural transfer model: for every head-dependents group it scores each member (sorting the
// scores gives the target order) and predicts its morphological operation, conditioned on the
// target typology distribution. Group context is pooled DeepSets-style.
class Reorderer {
public:
    struct Options {
        int steps = 40000;
        int batch = 32;
        float learningRate = 2e-3F;
        double unseenTypologies = 0.3;  // share of training targets drawn outside the atlas
        int evaluation = 1500;
        unsigned seed = 11;
    };

    struct Metrics {
        double pairs = 0.0;      // pairwise order agreement with the rules
        double groups = 0.0;     // groups ordered exactly as the rules
        double operations = 0.0; // operation agreement on dependents
        double sentences = 0.0;  // identical realisation
    };

    struct Report {
        Metrics atlas;   // targets from the WALS 200-language sample
        Metrics unseen;  // random typologies never seen in training
    };

    Reorderer();

    Report train(const TypologyBank& bank, const Options& options, std::ostream& log);
    [[nodiscard]] std::vector<std::vector<Decision>> decide(const std::vector<Group>& groups,
                                                            const Profile& target) const;

    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

private:
    struct Batch;
    struct Activations;

    void initialise(unsigned seed);
    void forward(const Batch& batch, Activations& a) const;
    [[nodiscard]] Metrics evaluate(const TypologyBank& bank, bool unseen, int count, unsigned seed) const;

    nn::Linear l1_;
    nn::Linear l2_;
    nn::Linear l3_;
    nn::Linear score_;
    nn::Linear operation_;
};

}
