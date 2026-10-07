#pragma once

#include "nn/Attention.hpp"
#include "nn/Layers.hpp"
#include "transfer/Grammar.hpp"
#include "transfer/Groups.hpp"
#include "transfer/Rules.hpp"

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace howspeak::transfer {

// Neural transfer model. Every member of a head-dependents group is embedded together with the
// target typology, contextualised by attention restricted to its group, then scored (sorting the
// scores gives the target order) and assigned a morphological operation.
class Reorderer {
public:
    struct Options {
        int steps = 150000;
        int batch = 128;
        float learningRate = 1e-3F;
        int warmup = 2000;
        double unseenTypologies = 0.5;    // random typologies outside the atlas
        double blendedTypologies = 0.15;  // soft mixtures of two atlas profiles
        int checkpointEvery = 10000;
        int checkpointEvaluation = 3000;  // sentences per typology kind at each checkpoint
        int evaluation = 20000;           // sentences per typology kind for the final report
        unsigned seed = 11;
    };

    struct Metrics {
        double pairs = 0.0;       // pairwise order agreement with the rules
        double groups = 0.0;      // groups ordered exactly as the rules
        double operations = 0.0;  // operation agreement on dependents
        double sentences = 0.0;   // identical realisation
    };

    struct Report {
        Metrics atlas;   // targets from the WALS 200-language sample
        Metrics unseen;  // random typologies never seen in training
        std::vector<std::pair<std::string, std::size_t>> errors;  // most frequent disagreements
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
    enum class Kind { Atlas, Unseen, Blended };

    void initialise(unsigned seed);
    void forward(const Batch& batch, Activations& a) const;
    void backward(const Batch& batch, const Activations& a, const nn::Matrix& dScore, const nn::Matrix& dOp);
    void gradients(const Batch& batch, double pairTotal, double itemTotal, double& lossPairs, double& lossOps);
    std::vector<nn::Parameter*> parameters();
    [[nodiscard]] Metrics evaluate(const TypologyBank& bank, Kind kind, int count, std::uint64_t seed,
                                   std::vector<std::pair<std::string, std::size_t>>* errors) const;

    nn::Linear input_;
    std::vector<nn::AttentionBlock> blocks_;
    nn::Linear head_;
    nn::Linear score_;
    nn::Linear operation_;
};

}
