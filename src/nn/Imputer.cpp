#include "nn/Imputer.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace howspeak::nn {

namespace {

constexpr std::string_view kMagic = "howspeak-imputer";
constexpr std::uint32_t kVersion = 1;
constexpr Index kHidden1 = 512;
constexpr Index kHidden2 = 256;

Index idx(std::size_t i) {
    return static_cast<Index>(i);
}

}

Imputer::Imputer(const wals::Database& db) : db_(&db) {
    for (const auto& l : db.languages()) {
        families_.try_emplace(l.family, families_.size());
        genera_.try_emplace(l.family + "/" + l.genus, genera_.size());
        areas_.try_emplace(l.macroarea, areas_.size());
    }
    initialise(7);
}

Index Imputer::inputSize() const {
    return idx(db_->oneHotSize() + families_.size() + genera_.size() + areas_.size() + 3);
}

void Imputer::initialise(unsigned seed) {
    std::mt19937_64 rng(seed);
    l1_ = Linear(inputSize(), kHidden1, rng);
    l2_ = Linear(kHidden1, kHidden2, rng);
    l3_ = Linear(kHidden2, idx(db_->oneHotSize()), rng);
}

Matrix Imputer::encode(const std::vector<std::size_t>& languages, const Mask& visible) const {
    const auto& features = db_->features();
    Matrix x = Matrix::Zero(inputSize(), idx(languages.size()));
    const auto metaBase = db_->oneHotSize();
    for (std::size_t c = 0; c < languages.size(); ++c) {
        const auto li = languages[c];
        const auto col = idx(c);
        for (std::size_t f = 0; f < features.size(); ++f) {
            const int v = db_->value(li, f);
            if (v != wals::kMissing && visible[li][f]) {
                x(idx(features[f].offset + static_cast<std::size_t>(v)), col) = 1.0F;
            }
        }
        const auto& l = db_->languages()[li];
        x(idx(metaBase + families_.at(l.family)), col) = 1.0F;
        x(idx(metaBase + families_.size() + genera_.at(l.family + "/" + l.genus)), col) = 1.0F;
        x(idx(metaBase + families_.size() + genera_.size() + areas_.at(l.macroarea)), col) = 1.0F;
        const auto geo = metaBase + families_.size() + genera_.size() + areas_.size();
        const double lon = l.longitude * std::numbers::pi / 180.0;
        x(idx(geo), col) = static_cast<float>(l.latitude / 90.0);
        x(idx(geo + 1), col) = static_cast<float>(std::sin(lon));
        x(idx(geo + 2), col) = static_cast<float>(std::cos(lon));
    }
    return x;
}

Matrix Imputer::logits(const Matrix& input) const {
    return l3_.forward(relu(l2_.forward(relu(l1_.forward(input)))));
}

void Imputer::softmaxGroups(Matrix& z) const {
    for (const auto& f : db_->features()) {
        auto block = z.middleRows(idx(f.offset), idx(f.codes.size()));
        for (Index c = 0; c < block.cols(); ++c) {
            auto col = block.col(c);
            col.array() -= col.maxCoeff();
            col = col.array().exp();
            col /= col.sum();
        }
    }
}

void Imputer::fit(const Options& options, const Mask& usable, std::ostream& log) {
    initialise(options.seed);
    std::mt19937_64 rng(options.seed + 1);
    std::bernoulli_distribution hide(options.hideRate);
    Adam adam(options.learningRate, 1e-4F);
    std::vector<Parameter*> params;
    l1_.collect(params);
    l2_.collect(params);
    l3_.collect(params);

    const auto& features = db_->features();
    const auto nLang = db_->languages().size();
    std::vector<std::size_t> order(nLang);
    std::iota(order.begin(), order.end(), 0);
    Mask visible = usable;

    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        std::ranges::shuffle(order, rng);
        double lossSum = 0.0;
        std::size_t targetSum = 0;
        for (std::size_t start = 0; start < nLang; start += static_cast<std::size_t>(options.batch)) {
            const auto end = std::min(nLang, start + static_cast<std::size_t>(options.batch));
            std::vector<std::size_t> batch(order.begin() + static_cast<std::ptrdiff_t>(start),
                                           order.begin() + static_cast<std::ptrdiff_t>(end));
            std::vector<std::vector<std::size_t>> targets(batch.size());
            for (std::size_t b = 0; b < batch.size(); ++b) {
                const auto li = batch[b];
                for (std::size_t f = 0; f < features.size(); ++f) {
                    visible[li][f] = usable[li][f];
                    if (usable[li][f] && hide(rng)) {
                        visible[li][f] = false;
                        targets[b].push_back(f);
                    }
                }
            }

            const Matrix x = encode(batch, visible);
            const Matrix h1 = relu(l1_.forward(x));
            const Matrix h2 = relu(l2_.forward(h1));
            Matrix probs = l3_.forward(h2);
            softmaxGroups(probs);

            Matrix grad = Matrix::Zero(probs.rows(), probs.cols());
            std::size_t count = 0;
            for (const auto& t : targets) {
                count += t.size();
            }
            const float scale = 1.0F / static_cast<float>(std::max<std::size_t>(count, 1));
            for (std::size_t b = 0; b < batch.size(); ++b) {
                for (const auto f : targets[b]) {
                    const auto& feat = features[f];
                    const auto rows = idx(feat.offset);
                    const auto size = idx(feat.codes.size());
                    const auto truth = idx(feat.offset + static_cast<std::size_t>(db_->value(batch[b], f)));
                    grad.block(rows, idx(b), size, 1) = probs.block(rows, idx(b), size, 1) * scale;
                    grad(truth, idx(b)) -= scale;
                    lossSum -= std::log(std::max(probs(truth, idx(b)), 1e-12F));
                }
                visible[batch[b]] = usable[batch[b]];
            }
            targetSum += count;

            const Matrix d2 = reluBackward(l3_.backward(h2, grad), h2);
            const Matrix d1 = reluBackward(l2_.backward(h1, d2), h1);
            l1_.accumulate(x, d1);
            adam.step(params);
        }
        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            log << std::format("  epoch {:4d}  masked cross-entropy {:.4f}\n", epoch + 1,
                               lossSum / static_cast<double>(std::max<std::size_t>(targetSum, 1)));
        }
    }
}

Imputer::Report Imputer::train(const Options& options, std::ostream& log) {
    const auto& features = db_->features();
    const auto& languages = db_->languages();
    Mask all(languages.size(), std::vector<bool>(features.size(), false));
    for (std::size_t l = 0; l < languages.size(); ++l) {
        for (std::size_t f = 0; f < features.size(); ++f) {
            all[l][f] = db_->value(l, f) != wals::kMissing;
        }
    }

    Report report;
    if (options.holdout > 0.0F) {
        std::mt19937_64 rng(options.seed + 99);
        std::bernoulli_distribution pick(options.holdout);
        Mask train = all;
        std::vector<std::pair<std::size_t, std::size_t>> heldout;
        for (std::size_t l = 0; l < languages.size(); ++l) {
            for (std::size_t f = 0; f < features.size(); ++f) {
                if (all[l][f] && pick(rng)) {
                    train[l][f] = false;
                    heldout.emplace_back(l, f);
                }
            }
        }

        std::vector<std::vector<std::size_t>> global(features.size());
        std::unordered_map<std::string, std::vector<std::vector<std::size_t>>> genus;
        for (std::size_t f = 0; f < features.size(); ++f) {
            global[f].assign(features[f].codes.size(), 0);
        }
        for (std::size_t l = 0; l < languages.size(); ++l) {
            auto& g = genus[languages[l].family + "/" + languages[l].genus];
            if (g.empty()) {
                g.resize(features.size());
                for (std::size_t f = 0; f < features.size(); ++f) {
                    g[f].assign(features[f].codes.size(), 0);
                }
            }
            for (std::size_t f = 0; f < features.size(); ++f) {
                if (train[l][f]) {
                    const auto v = static_cast<std::size_t>(db_->value(l, f));
                    ++global[f][v];
                    ++g[f][v];
                }
            }
        }
        const auto argmax = [](const std::vector<std::size_t>& counts) {
            return static_cast<std::size_t>(std::ranges::max_element(counts) - counts.begin());
        };

        log << std::format("held-out evaluation on {} observed values\n", heldout.size());
        fit(options, train, log);

        std::size_t hit = 0;
        std::size_t hitGlobal = 0;
        std::size_t hitGenus = 0;
        std::vector<std::size_t> langs(languages.size());
        std::iota(langs.begin(), langs.end(), 0);
        Matrix probs = logits(encode(langs, train));
        softmaxGroups(probs);
        for (const auto& [l, f] : heldout) {
            const auto truth = static_cast<std::size_t>(db_->value(l, f));
            const auto& feat = features[f];
            Index best = 0;
            probs.col(idx(l)).segment(idx(feat.offset), idx(feat.codes.size())).maxCoeff(&best);
            hit += static_cast<std::size_t>(best) == truth ? 1U : 0U;
            hitGlobal += argmax(global[f]) == truth ? 1U : 0U;
            const auto& g = genus[languages[l].family + "/" + languages[l].genus][f];
            const bool genusKnown = std::ranges::any_of(g, [](std::size_t c) { return c > 0; });
            hitGenus += (genusKnown ? argmax(g) : argmax(global[f])) == truth ? 1U : 0U;
        }
        const auto n = static_cast<double>(std::max<std::size_t>(heldout.size(), 1));
        report.heldout = heldout.size();
        report.accuracy = static_cast<double>(hit) / n;
        report.globalMajority = static_cast<double>(hitGlobal) / n;
        report.genusMajority = static_cast<double>(hitGenus) / n;
        log << "final fit on all observed values\n";
    }
    fit(options, all, log);
    return report;
}

std::vector<std::vector<float>> Imputer::distributions(std::size_t language) const {
    const auto& features = db_->features();
    Mask visible(db_->languages().size());
    visible[language].resize(features.size());
    for (std::size_t f = 0; f < features.size(); ++f) {
        visible[language][f] = db_->value(language, f) != wals::kMissing;
    }
    Matrix probs = logits(encode({language}, visible));
    softmaxGroups(probs);

    std::vector<std::vector<float>> out(features.size());
    for (std::size_t f = 0; f < features.size(); ++f) {
        const auto& feat = features[f];
        out[f].resize(feat.codes.size());
        const int v = db_->value(language, f);
        for (std::size_t c = 0; c < feat.codes.size(); ++c) {
            out[f][c] = v == wals::kMissing ? probs(idx(feat.offset + c), 0)
                                            : (static_cast<std::size_t>(v) == c ? 1.0F : 0.0F);
        }
    }
    return out;
}

void Imputer::save(const std::filesystem::path& path) const {
    core::BinaryWriter w(path, kMagic, kVersion);
    w.write<std::uint64_t>(db_->oneHotSize());
    w.write<std::int64_t>(inputSize());
    l1_.save(w);
    l2_.save(w);
    l3_.save(w);
    w.close();
}

void Imputer::load(const std::filesystem::path& path) {
    core::BinaryReader r(path, kMagic, kVersion);
    if (r.read<std::uint64_t>() != db_->oneHotSize() || r.read<std::int64_t>() != inputSize()) {
        r.fail("model does not match the WALS data");
    }
    l1_.load(r);
    l2_.load(r);
    l3_.load(r);
}

}
