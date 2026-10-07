#pragma once

#include "nn/Layers.hpp"
#include "wals/Database.hpp"

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <unordered_map>
#include <vector>

namespace howspeak::nn {

// Denoising autoencoder over the WALS language x feature matrix, conditioned on genealogy and geography.
class Imputer {
public:
    struct Options {
        int epochs = 60;
        int batch = 64;
        float learningRate = 1e-3F;
        float hideRate = 0.3F;
        float holdout = 0.1F;
        unsigned seed = 7;
    };

    struct Report {
        std::size_t heldout = 0;
        double accuracy = 0.0;
        double globalMajority = 0.0;
        double genusMajority = 0.0;
    };

    explicit Imputer(const wals::Database& db);

    Report train(const Options& options, std::ostream& log);

    // Probability of each code of each feature (observed features are returned as one-hot).
    [[nodiscard]] std::vector<std::vector<float>> distributions(std::size_t language) const;

    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

private:
    using Mask = std::vector<std::vector<bool>>;  // [language][feature] visible to the network

    void initialise(unsigned seed);
    void fit(const Options& options, const Mask& usable, std::ostream& log);
    [[nodiscard]] Matrix encode(const std::vector<std::size_t>& languages, const Mask& visible) const;
    [[nodiscard]] Matrix logits(const Matrix& input) const;
    void softmaxGroups(Matrix& logits) const;
    [[nodiscard]] Index inputSize() const;

    const wals::Database* db_;
    std::unordered_map<std::string, std::size_t> families_;
    std::unordered_map<std::string, std::size_t> genera_;
    std::unordered_map<std::string, std::size_t> areas_;
    Linear l1_;
    Linear l2_;
    Linear l3_;
};

}
