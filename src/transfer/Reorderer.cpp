#include "transfer/Reorderer.hpp"

#include "transfer/Encoder.hpp"
#include "transfer/Realizer.hpp"
#include "transfer/Restructure.hpp"
#include "transfer/Synthetic.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <numbers>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ostream>
#include <span>

namespace howspeak::transfer {

namespace {

using nn::Index;
using nn::Matrix;

constexpr std::string_view kMagic = "howspeak-reorderer";
constexpr std::uint32_t kVersion = 3;
constexpr Index kWidth = 128;
constexpr Index kHeads = 4;
constexpr Index kFeedForward = 256;
constexpr std::size_t kBlocks = 3;
constexpr Index kHead = 128;
constexpr float kClip = 1.0F;
constexpr int kMaxShards = 16;  // more replicas land on efficiency cores and slow the step down

constexpr std::array<std::string_view, kOperationCount> kOperationNames = {
    "keep", "drop", "prefix-affix", "suffix-affix", "prefix-clitic", "suffix-clitic"};

Index idx(std::size_t i) {
    return static_cast<Index>(i);
}

int workerThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

struct Example {
    syntax::Sentence sentence;
    Profile target;
    Plan rules;
};

Profile blend(const Profile& a, const Profile& b, float alpha) {
    Profile p;
    for (std::size_t f = 0; f < kFeatureCount; ++f) {
        p.distribution[f].resize(a.distribution[f].size());
        for (std::size_t c = 0; c < p.distribution[f].size(); ++c) {
            p.distribution[f][c] = alpha * a.distribution[f][c] + (1.0F - alpha) * b.distribution[f][c];
        }
    }
    return p;
}

// Labels always come from the most likely grammar, exactly as the rule engine is used at inference.
Example makeExample(const TypologyBank& typology, std::uint64_t seed, int kind) {
    std::mt19937_64 rng(mix(seed));
    SyntheticTreebank treebank(mix(seed ^ 0xA5A5A5A5ULL));
    const auto& sample = typology.sample200();
    std::uniform_int_distribution<std::size_t> pick(0, sample.size() - 1);
    std::bernoulli_distribution coin(0.5);

    const Profile source = coin(rng) ? typology.randomProfile(rng) : typology.profile(sample[pick(rng)]);
    Example e;
    e.sentence = treebank.next(source.sample(rng));
    switch (kind) {
    case 0: e.target = typology.profile(sample[pick(rng)]); break;
    case 1: e.target = typology.randomProfile(rng); break;
    default:
        e.target = blend(typology.profile(sample[pick(rng)]), typology.profile(sample[pick(rng)]),
                         std::uniform_real_distribution<float>(0.0F, 1.0F)(rng));
        break;
    }
    const Grammar g = e.target.mostLikely();
    restructure(e.sentence, g);
    e.rules.groups = buildGroups(e.sentence);
    const Rules rules(g);
    for (const auto& group : e.rules.groups) {
        e.rules.decisions.push_back(rules.decide(group));
    }
    return e;
}

}

struct Reorderer::Batch {
    Matrix items;     // Encoder::width() x N
    Matrix typology;  // Profile::width() x G
    nn::Ranges ranges;
    std::vector<int> rank;  // target position inside the group (training only)
    std::vector<int> op;    // target operation (training only)

    void add(const Group& group, const Profile& profile, const std::vector<Decision>* labels) {
        const Index start = columns_;
        const auto width = Encoder::width();
        const auto offset = itemData_.size();
        itemData_.resize(offset + width * group.items.size());
        for (std::size_t i = 0; i < group.items.size(); ++i) {
            Encoder::encode(group, group.items[i], std::span(itemData_).subspan(offset + i * width, width));
        }
        columns_ += idx(group.items.size());
        profile.appendTo(typologyData_);
        ranges.emplace_back(start, columns_);
        if (labels != nullptr) {
            const auto order = orderGroup(group, *labels);
            std::vector<int> r(group.items.size());
            for (std::size_t k = 0; k < order.size(); ++k) {
                r[order[k]] = static_cast<int>(k);
            }
            rank.insert(rank.end(), r.begin(), r.end());
            for (const auto& d : *labels) {
                op.push_back(static_cast<int>(d.op));
            }
        }
    }

    void append(const Batch& other) {
        for (const auto& [b, e] : other.ranges) {
            ranges.emplace_back(b + columns_, e + columns_);
        }
        columns_ += other.columns_;
        itemData_.insert(itemData_.end(), other.itemData_.begin(), other.itemData_.end());
        typologyData_.insert(typologyData_.end(), other.typologyData_.begin(), other.typologyData_.end());
        rank.insert(rank.end(), other.rank.begin(), other.rank.end());
        op.insert(op.end(), other.op.begin(), other.op.end());
    }

    void finish() {
        items = Eigen::Map<const Matrix>(itemData_.data(), idx(Encoder::width()), columns_);
        typology = Eigen::Map<const Matrix>(typologyData_.data(), idx(Profile::width()), idx(ranges.size()));
    }

private:
    std::vector<float> itemData_;
    std::vector<float> typologyData_;
    Index columns_ = 0;
};

struct Reorderer::Activations {
    Matrix input;
    Matrix h0;
    std::vector<nn::AttentionBlock::Cache> caches;
    Matrix z;
    Matrix g;
    Matrix score;
    Matrix op;
};

Reorderer::Reorderer() {
    initialise(11);
}

void Reorderer::initialise(unsigned seed) {
    std::mt19937_64 rng(seed);
    input_ = nn::Linear(idx(Encoder::width() + Profile::width()), kWidth, rng);
    blocks_.clear();
    for (std::size_t b = 0; b < kBlocks; ++b) {
        blocks_.emplace_back(kWidth, kHeads, kFeedForward, rng);
    }
    head_ = nn::Linear(2 * kWidth, kHead, rng);
    score_ = nn::Linear(kHead, 1, rng);
    operation_ = nn::Linear(kHead, idx(kOperationCount), rng);
}

std::vector<nn::Parameter*> Reorderer::parameters() {
    std::vector<nn::Parameter*> params;
    input_.collect(params);
    for (auto& b : blocks_) {
        b.collect(params);
    }
    head_.collect(params);
    score_.collect(params);
    operation_.collect(params);
    return params;
}

void Reorderer::forward(const Batch& b, Activations& a) const {
    const Index n = b.items.cols();
    const Index itemRows = b.items.rows();
    a.input.resize(itemRows + b.typology.rows(), n);
    a.input.topRows(itemRows) = b.items;
    for (std::size_t g = 0; g < b.ranges.size(); ++g) {
        const auto [start, end] = b.ranges[g];
        a.input.block(itemRows, start, b.typology.rows(), end - start).colwise() = b.typology.col(idx(g));
    }
    a.h0 = nn::relu(input_.forward(a.input));
    a.caches.resize(blocks_.size());
    Matrix x = a.h0;
    for (std::size_t k = 0; k < blocks_.size(); ++k) {
        x = blocks_[k].forward(x, b.ranges, a.caches[k]);
    }
    a.z.resize(2 * kWidth, n);
    a.z.topRows(kWidth) = x;
    for (const auto& [start, end] : b.ranges) {
        const Eigen::VectorXf mean = x.middleCols(start, end - start).rowwise().mean();
        a.z.block(kWidth, start, kWidth, end - start).colwise() = mean;
    }
    a.g = nn::relu(head_.forward(a.z));
    a.score = score_.forward(a.g);
    a.op = operation_.forward(a.g);
}

void Reorderer::backward(const Batch& b, const Activations& a, const Matrix& dScore, const Matrix& dOp) {
    Matrix dg = score_.backward(a.g, dScore) + operation_.backward(a.g, dOp);
    const Matrix dz = head_.backward(a.z, nn::reluBackward(dg, a.g));
    Matrix dx = dz.topRows(kWidth);
    for (const auto& [start, end] : b.ranges) {
        const Eigen::VectorXf pooled =
            dz.block(kWidth, start, kWidth, end - start).rowwise().sum() / static_cast<float>(end - start);
        dx.middleCols(start, end - start).colwise() += pooled;
    }
    for (std::size_t k = blocks_.size(); k-- > 0;) {
        dx = blocks_[k].backward(dx, b.ranges, a.caches[k]);
    }
    input_.accumulate(a.input, nn::reluBackward(dx, a.h0));
}

std::vector<std::vector<Decision>> Reorderer::decide(const std::vector<Group>& groups, const Profile& target) const {
    std::vector<std::vector<Decision>> out;
    if (groups.empty()) {
        return out;
    }
    Batch b;
    for (const auto& g : groups) {
        b.add(g, target, nullptr);
    }
    b.finish();
    Activations a;
    forward(b, a);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const auto [start, end] = b.ranges[g];
        std::vector<Decision> ds;
        for (Index c = start; c < end; ++c) {
            Index best = 0;
            a.op.col(c).maxCoeff(&best);
            const auto& item = groups[g].items[static_cast<std::size_t>(c - start)];
            ds.push_back({static_cast<double>(a.score(0, c)),
                          item.isHead ? Operation::Keep : static_cast<Operation>(best)});
        }
        out.push_back(std::move(ds));
    }
    return out;
}

void Reorderer::gradients(const Batch& b, double pairTotal, double itemTotal, double& lossPairs, double& lossOps) {
    Activations a;
    forward(b, a);
    const Index n = b.items.cols();

    Matrix dScore = Matrix::Zero(1, n);
    for (const auto& [start, end] : b.ranges) {
        for (Index i = start; i < end; ++i) {
            for (Index j = start; j < end; ++j) {
                if (b.rank[static_cast<std::size_t>(i)] >= b.rank[static_cast<std::size_t>(j)]) {
                    continue;
                }
                const float d = a.score(0, i) - a.score(0, j);
                const float sigma = 1.0F / (1.0F + std::exp(-d));
                dScore(0, i) += sigma;
                dScore(0, j) -= sigma;
                lossPairs += d > 20.0F ? d : std::log1p(std::exp(d));
            }
        }
    }
    dScore /= static_cast<float>(pairTotal);

    Matrix dOp = a.op;
    for (Index c = 0; c < n; ++c) {
        auto col = dOp.col(c);
        col.array() -= col.maxCoeff();
        col = col.array().exp();
        col /= col.sum();
        const auto truth = idx(static_cast<std::size_t>(b.op[static_cast<std::size_t>(c)]));
        lossOps -= std::log(std::max(col(truth), 1e-12F));
        col(truth) -= 1.0F;
    }
    dOp /= static_cast<float>(itemTotal);
    backward(b, a, dScore, dOp);
}

Reorderer::Report Reorderer::train(const TypologyBank& typology, const Options& options, std::ostream& log) {
    initialise(options.seed);
    nn::Adam adam(options.learningRate, 1e-5F);
    const auto params = parameters();
    std::mt19937_64 rng(options.seed + 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Data parallelism: each replica runs forward and backward on its shard, gradients are summed.
    const int shards = std::max(1, std::min({workerThreads(), kMaxShards, options.batch}));
    std::vector<Reorderer> replicas(static_cast<std::size_t>(shards), *this);
    std::vector<std::vector<nn::Parameter*>> replicaParams;
    for (auto& r : replicas) {
        replicaParams.push_back(r.parameters());
    }

    Reorderer best = *this;
    double bestScore = -1.0;
    double lossPairs = 0.0;
    double lossOps = 0.0;
    auto clock = std::chrono::steady_clock::now();
    const std::uint64_t trainSeed = mix(options.seed);
    log << std::format("  {} data-parallel shards\n", shards) << std::flush;

    for (int step = 1; step <= options.steps; ++step) {
        const double warm = std::min(1.0, static_cast<double>(step) / std::max(1, options.warmup));
        const double progress = static_cast<double>(step) / static_cast<double>(options.steps);
        const double cosine = 0.02 + 0.98 * 0.5 * (1.0 + std::cos(std::numbers::pi * progress));
        adam.setLearningRate(static_cast<float>(options.learningRate * warm * cosine));

        std::vector<int> kinds(static_cast<std::size_t>(options.batch));
        for (auto& k : kinds) {
            const double u = unit(rng);
            k = u < options.unseenTypologies ? 1 : (u < options.unseenTypologies + options.blendedTypologies ? 2 : 0);
        }
        std::vector<Batch> batches(static_cast<std::size_t>(shards));
        std::vector<double> pairCounts(static_cast<std::size_t>(shards), 0.0);
#pragma omp parallel for num_threads(shards) schedule(static, 1)
        for (int r = 0; r < shards; ++r) {
            const auto ur = static_cast<std::size_t>(r);
            for (int k = r; k < options.batch; k += shards) {
                const auto e = makeExample(
                    typology, trainSeed + static_cast<std::uint64_t>(step) * 4096U + static_cast<std::uint64_t>(k),
                    kinds[static_cast<std::size_t>(k)]);
                for (std::size_t g = 0; g < e.rules.groups.size(); ++g) {
                    batches[ur].add(e.rules.groups[g], e.target, &e.rules.decisions[g]);
                    const auto m = static_cast<double>(e.rules.groups[g].items.size());
                    pairCounts[ur] += m * (m - 1.0) / 2.0;
                }
            }
            batches[ur].finish();
            for (std::size_t i = 0; i < params.size(); ++i) {
                replicaParams[ur][i]->value = params[i]->value;
            }
        }
        double pairTotal = 0.0;
        double itemTotal = 0.0;
        for (std::size_t r = 0; r < batches.size(); ++r) {
            pairTotal += pairCounts[r];
            itemTotal += static_cast<double>(batches[r].items.cols());
        }

        std::vector<double> shardPairs(static_cast<std::size_t>(shards), 0.0);
        std::vector<double> shardOps(static_cast<std::size_t>(shards), 0.0);
#pragma omp parallel for num_threads(shards) schedule(static, 1)
        for (int r = 0; r < shards; ++r) {
            const auto ur = static_cast<std::size_t>(r);
            if (batches[ur].items.cols() > 0) {
                replicas[ur].gradients(batches[ur], std::max(pairTotal, 1.0), std::max(itemTotal, 1.0),
                                       shardPairs[ur], shardOps[ur]);
            }
        }
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(params.size()); ++i) {
            const auto ui = static_cast<std::size_t>(i);
            for (auto& replica : replicaParams) {
                params[ui]->grad += replica[ui]->grad;
                replica[ui]->grad.setZero();
            }
        }
        adam.step(params, kClip);
        for (std::size_t r = 0; r < shardPairs.size(); ++r) {
            lossPairs += shardPairs[r];
            lossOps += shardOps[r];
        }

        if (step % 1000 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - clock).count();
            clock = now;
            log << std::format("  step {:7d}  pairwise {:.5f}  operation {:.5f}  ({:.1f} steps/s)\n", step,
                               lossPairs / 1000.0 / static_cast<double>(options.batch),
                               lossOps / 1000.0 / static_cast<double>(options.batch), 1000.0 / seconds)
                << std::flush;
            lossPairs = 0.0;
            lossOps = 0.0;
        }
        if (step % options.checkpointEvery == 0 || step == options.steps) {
            const auto atlas = evaluate(typology, Kind::Atlas, options.checkpointEvaluation, 77, nullptr);
            const auto unseen = evaluate(typology, Kind::Unseen, options.checkpointEvaluation, 78, nullptr);
            const double score = std::min(atlas.sentences, unseen.sentences);
            log << std::format("  checkpoint {:7d}  sentences atlas {:.4f}  unseen {:.4f}{}\n", step, atlas.sentences,
                               unseen.sentences, score > bestScore ? "  *" : "")
                << std::flush;
            if (score > bestScore) {
                bestScore = score;
                best = *this;
            }
        }
    }
    *this = best;

    Report report;
    report.atlas = evaluate(typology, Kind::Atlas, options.evaluation, 1001, &report.errors);
    report.unseen = evaluate(typology, Kind::Unseen, options.evaluation, 2002, &report.errors);
    return report;
}

Reorderer::Metrics Reorderer::evaluate(const TypologyBank& typology, Kind kind, int count, std::uint64_t seed,
                                       std::vector<std::pair<std::string, std::size_t>>* errors) const {
    std::size_t pairs = 0;
    std::size_t pairHits = 0;
    std::size_t groups = 0;
    std::size_t groupHits = 0;
    std::size_t ops = 0;
    std::size_t opHits = 0;
    std::size_t sentenceHits = 0;
    std::map<std::string, std::size_t> mistakes;
    const std::uint64_t base = mix(seed) ^ 0x5555555555555555ULL;
    const int kindIndex = kind == Kind::Atlas ? 0 : (kind == Kind::Unseen ? 1 : 2);

#pragma omp parallel
    {
        std::size_t lPairs = 0;
        std::size_t lPairHits = 0;
        std::size_t lGroups = 0;
        std::size_t lGroupHits = 0;
        std::size_t lOps = 0;
        std::size_t lOpHits = 0;
        std::size_t lSentences = 0;
        std::map<std::string, std::size_t> lMistakes;
#pragma omp for schedule(dynamic, 16)
        for (int k = 0; k < count; ++k) {
            const auto e = makeExample(typology, base + static_cast<std::uint64_t>(k), kindIndex);
            Plan net;
            net.groups = e.rules.groups;
            net.decisions = decide(net.groups, e.target);
            for (std::size_t g = 0; g < net.groups.size(); ++g) {
                const auto& group = net.groups[g];
                const auto want = orderGroup(group, e.rules.decisions[g]);
                const auto got = orderGroup(group, net.decisions[g]);
                std::vector<std::size_t> pos(want.size());
                for (std::size_t i = 0; i < got.size(); ++i) {
                    pos[got[i]] = i;
                }
                bool reported = false;
                for (std::size_t i = 0; i < want.size(); ++i) {
                    for (std::size_t j = i + 1; j < want.size(); ++j) {
                        ++lPairs;
                        const bool hit = pos[want[i]] < pos[want[j]];
                        lPairHits += hit ? 1U : 0U;
                        if (!hit && !reported) {
                            reported = true;
                            ++lMistakes[std::format("order  {} < {}  (head {})", group.items[want[i]].relation,
                                                    group.items[want[j]].relation, group.headUpos)];
                        }
                    }
                }
                ++lGroups;
                lGroupHits += want == got ? 1U : 0U;
                for (std::size_t i = 0; i < group.items.size(); ++i) {
                    if (group.items[i].isHead) {
                        continue;
                    }
                    ++lOps;
                    const auto w = e.rules.decisions[g][i].op;
                    const auto h = net.decisions[g][i].op;
                    lOpHits += w == h ? 1U : 0U;
                    if (w != h) {
                        ++lMistakes[std::format("op     {} {} -> {}", group.items[i].relation,
                                                kOperationNames[static_cast<std::size_t>(w)],
                                                kOperationNames[static_cast<std::size_t>(h)])];
                    }
                }
            }
            lSentences += realize(e.sentence, net) == realize(e.sentence, e.rules) ? 1U : 0U;
        }
#pragma omp critical
        {
            pairs += lPairs;
            pairHits += lPairHits;
            groups += lGroups;
            groupHits += lGroupHits;
            ops += lOps;
            opHits += lOpHits;
            sentenceHits += lSentences;
            for (const auto& [key, n] : lMistakes) {
                mistakes[key] += n;
            }
        }
    }

    if (errors != nullptr) {
        for (const auto& [key, n] : *errors) {
            mistakes[key] += n;
        }
        std::vector<std::pair<std::string, std::size_t>> sorted(mistakes.begin(), mistakes.end());
        std::ranges::sort(sorted, std::greater{}, &std::pair<std::string, std::size_t>::second);
        *errors = std::move(sorted);
    }
    const auto ratio = [](std::size_t a, std::size_t b) {
        return b == 0 ? 0.0 : static_cast<double>(a) / static_cast<double>(b);
    };
    return {ratio(pairHits, pairs), ratio(groupHits, groups), ratio(opHits, ops),
            ratio(sentenceHits, static_cast<std::size_t>(count))};
}

void Reorderer::save(const std::filesystem::path& path) const {
    core::BinaryWriter w(path, kMagic, kVersion);
    w.write<std::uint64_t>(Encoder::width());
    w.write<std::uint64_t>(Profile::width());
    w.write<std::uint64_t>(blocks_.size());
    input_.save(w);
    for (const auto& b : blocks_) {
        b.save(w);
    }
    head_.save(w);
    score_.save(w);
    operation_.save(w);
    w.close();
}

void Reorderer::load(const std::filesystem::path& path) {
    core::BinaryReader r(path, kMagic, kVersion);
    if (r.read<std::uint64_t>() != Encoder::width() || r.read<std::uint64_t>() != Profile::width()) {
        r.fail("model does not match the encoder layout");
    }
    const auto count = r.read<std::uint64_t>();
    if (count > 16) {
        r.fail("invalid block count");
    }
    input_.load(r);
    blocks_.assign(count, nn::AttentionBlock());
    for (auto& b : blocks_) {
        b.load(r);
    }
    head_.load(r);
    score_.load(r);
    operation_.load(r);
}

}
