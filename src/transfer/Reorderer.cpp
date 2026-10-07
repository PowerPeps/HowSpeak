#include "transfer/Reorderer.hpp"

#include "transfer/Encoder.hpp"
#include "transfer/Realizer.hpp"
#include "transfer/Restructure.hpp"
#include "transfer/Synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <span>
#include <ostream>

namespace howspeak::transfer {

namespace {

using nn::Index;
using nn::Matrix;

constexpr std::string_view kMagic = "howspeak-reorderer";
constexpr std::uint32_t kVersion = 1;
constexpr Index kHidden1 = 128;
constexpr Index kHidden2 = 256;
constexpr Index kHidden3 = 128;

Index idx(std::size_t i) {
    return static_cast<Index>(i);
}

struct Example {
    syntax::Sentence sentence;
    Profile target;
    Plan rules;
};

Example makeExample(SyntheticTreebank& bank, const TypologyBank& typology, std::mt19937_64& rng, bool unseen) {
    const auto& sample = typology.sample200();
    std::uniform_int_distribution<std::size_t> pick(0, sample.size() - 1);
    std::bernoulli_distribution coin(0.5);

    const Profile source = coin(rng) ? typology.randomProfile(rng) : typology.profile(sample[pick(rng)]);
    Example e;
    e.sentence = bank.next(source.sample(rng));
    e.target = unseen ? typology.randomProfile(rng) : typology.profile(sample[pick(rng)]);
    const Grammar g = e.target.sample(rng);
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
    std::vector<std::pair<Index, Index>> ranges;  // columns of each group
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
    Matrix h1;
    Matrix z;
    Matrix h2;
    Matrix h3;
    Matrix score;
    Matrix op;
};

Reorderer::Reorderer() {
    initialise(11);
}

void Reorderer::initialise(unsigned seed) {
    std::mt19937_64 rng(seed);
    l1_ = nn::Linear(idx(Encoder::width()), kHidden1, rng);
    l2_ = nn::Linear(2 * kHidden1 + idx(Profile::width()), kHidden2, rng);
    l3_ = nn::Linear(kHidden2, kHidden3, rng);
    score_ = nn::Linear(kHidden3, 1, rng);
    operation_ = nn::Linear(kHidden3, idx(kOperationCount), rng);
}

void Reorderer::forward(const Batch& b, Activations& a) const {
    const Index n = b.items.cols();
    a.h1 = nn::relu(l1_.forward(b.items));
    a.z.resize(2 * kHidden1 + b.typology.rows(), n);
    a.z.topRows(kHidden1) = a.h1;
    for (std::size_t g = 0; g < b.ranges.size(); ++g) {
        const auto [start, end] = b.ranges[g];
        const Eigen::VectorXf mean = a.h1.middleCols(start, end - start).rowwise().mean();
        for (Index c = start; c < end; ++c) {
            a.z.block(kHidden1, c, kHidden1, 1) = mean;
            a.z.block(2 * kHidden1, c, b.typology.rows(), 1) = b.typology.col(idx(g));
        }
    }
    a.h2 = nn::relu(l2_.forward(a.z));
    a.h3 = nn::relu(l3_.forward(a.h2));
    a.score = score_.forward(a.h3);
    a.op = operation_.forward(a.h3);
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

Reorderer::Report Reorderer::train(const TypologyBank& typology, const Options& options, std::ostream& log) {
    initialise(options.seed);
    std::mt19937_64 rng(options.seed + 1);
    SyntheticTreebank treebank(options.seed + 2);
    std::bernoulli_distribution unseen(options.unseenTypologies);
    nn::Adam adam(options.learningRate, 1e-5F);
    std::vector<nn::Parameter*> params;
    l1_.collect(params);
    l2_.collect(params);
    l3_.collect(params);
    score_.collect(params);
    operation_.collect(params);

    double lossPairs = 0.0;
    double lossOps = 0.0;
    for (int step = 1; step <= options.steps; ++step) {
        const float progress = static_cast<float>(step) / static_cast<float>(options.steps);
        adam.setLearningRate(options.learningRate * (1.0F - 0.9F * progress));

        Batch b;
        for (int k = 0; k < options.batch; ++k) {
            const auto e = makeExample(treebank, typology, rng, unseen(rng));
            for (std::size_t g = 0; g < e.rules.groups.size(); ++g) {
                b.add(e.rules.groups[g], e.target, &e.rules.decisions[g]);
            }
        }
        b.finish();
        Activations a;
        forward(b, a);
        const Index n = b.items.cols();

        Matrix dScore = Matrix::Zero(1, n);
        std::size_t pairCount = 0;
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
                    ++pairCount;
                }
            }
        }
        dScore /= static_cast<float>(std::max<std::size_t>(pairCount, 1));

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
        dOp /= static_cast<float>(n);

        Matrix dh3 = score_.backward(a.h3, dScore) + operation_.backward(a.h3, dOp);
        dh3 = nn::reluBackward(dh3, a.h3);
        const Matrix dh2 = nn::reluBackward(l3_.backward(a.h2, dh3), a.h2);
        const Matrix dz = l2_.backward(a.z, dh2);
        Matrix dh1 = dz.topRows(kHidden1);
        for (const auto& [start, end] : b.ranges) {
            const Eigen::VectorXf pooled =
                dz.block(kHidden1, start, kHidden1, end - start).rowwise().sum() / static_cast<float>(end - start);
            dh1.middleCols(start, end - start).colwise() += pooled;
        }
        l1_.accumulate(b.items, nn::reluBackward(dh1, a.h1));
        adam.step(params);

        if (step % 1000 == 0) {
            log << std::format("  step {:6d}  pairwise loss {:.4f}  operation loss {:.4f}\n", step,
                               lossPairs / 1000.0 / static_cast<double>(options.batch),
                               lossOps / 1000.0 / static_cast<double>(options.batch))
                << std::flush;
            lossPairs = 0.0;
            lossOps = 0.0;
        }
    }

    Report report;
    report.atlas = evaluate(typology, false, options.evaluation, options.seed + 1000);
    report.unseen = evaluate(typology, true, options.evaluation, options.seed + 2000);
    return report;
}

Reorderer::Metrics Reorderer::evaluate(const TypologyBank& typology, bool unseen, int count, unsigned seed) const {
    std::mt19937_64 rng(seed);
    SyntheticTreebank treebank(seed + 1);
    std::size_t pairs = 0;
    std::size_t pairHits = 0;
    std::size_t groups = 0;
    std::size_t groupHits = 0;
    std::size_t ops = 0;
    std::size_t opHits = 0;
    std::size_t sentenceHits = 0;
    for (int k = 0; k < count; ++k) {
        const auto e = makeExample(treebank, typology, rng, unseen);
        Plan net;
        net.groups = e.rules.groups;
        net.decisions = decide(net.groups, e.target);
        for (std::size_t g = 0; g < net.groups.size(); ++g) {
            const auto want = orderGroup(net.groups[g], e.rules.decisions[g]);
            const auto got = orderGroup(net.groups[g], net.decisions[g]);
            std::vector<std::size_t> pos(want.size());
            for (std::size_t i = 0; i < got.size(); ++i) {
                pos[got[i]] = i;
            }
            for (std::size_t i = 0; i < want.size(); ++i) {
                for (std::size_t j = i + 1; j < want.size(); ++j) {
                    ++pairs;
                    pairHits += pos[want[i]] < pos[want[j]] ? 1U : 0U;
                }
            }
            ++groups;
            groupHits += want == got ? 1U : 0U;
            for (std::size_t i = 0; i < net.groups[g].items.size(); ++i) {
                if (!net.groups[g].items[i].isHead) {
                    ++ops;
                    opHits += net.decisions[g][i].op == e.rules.decisions[g][i].op ? 1U : 0U;
                }
            }
        }
        sentenceHits += realize(e.sentence, net) == realize(e.sentence, e.rules) ? 1U : 0U;
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
    l1_.save(w);
    l2_.save(w);
    l3_.save(w);
    score_.save(w);
    operation_.save(w);
    w.close();
}

void Reorderer::load(const std::filesystem::path& path) {
    core::BinaryReader r(path, kMagic, kVersion);
    if (r.read<std::uint64_t>() != Encoder::width() || r.read<std::uint64_t>() != Profile::width()) {
        r.fail("model does not match the encoder layout");
    }
    l1_.load(r);
    l2_.load(r);
    l3_.load(r);
    score_.load(r);
    operation_.load(r);
}

}
