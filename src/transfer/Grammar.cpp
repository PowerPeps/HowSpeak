#include "transfer/Grammar.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace howspeak::transfer {

namespace {

template <class E>
E as(const CodeVector& codes, std::size_t i) {
    return static_cast<E>(codes[i]);
}

}

Grammar Grammar::fromCodes(const CodeVector& c) {
    Grammar g;
    g.affixation = as<Affixation>(c, 0);
    g.plural = as<Plural>(c, 1);
    g.definite = as<Definite>(c, 2);
    g.indefinite = as<Indefinite>(c, 3);
    g.caseCount = as<CaseCount>(c, 4);
    g.caseAffix = as<CaseAffix>(c, 5);
    g.tenseAffix = as<TenseAffix>(c, 6);
    g.clause = as<ClauseOrder>(c, 7);
    g.subjectVerb = as<SubjectVerb>(c, 8);
    g.objectVerb = as<ObjectVerb>(c, 9);
    g.oblique = as<ObliqueOrder>(c, 10);
    g.adposition = as<Adposition>(c, 11);
    g.genitive = as<Genitive>(c, 12);
    g.adjective = as<Adjective>(c, 13);
    g.demonstrative = as<Demonstrative>(c, 14);
    g.numeral = as<Numeral>(c, 15);
    g.relative = as<Relative>(c, 16);
    g.degree = as<Degree>(c, 17);
    g.questionParticle = as<QuestionParticle>(c, 18);
    g.wh = as<WhPosition>(c, 19);
    g.subordinator = as<Subordinator>(c, 20);
    g.subjectPronoun = as<SubjectPronoun>(c, 21);
    g.polar = as<PolarQuestion>(c, 22);
    g.predicativeAdjective = as<PredicativeAdjective>(c, 23);
    g.zeroCopula = as<ZeroCopula>(c, 24);
    g.relativization = as<Relativization>(c, 25);
    g.negation = as<NegationVerb>(c, 26);
    g.negationClause = as<NegationClause>(c, 27);
    return g;
}

Grammar Profile::mostLikely() const {
    CodeVector codes{};
    for (std::size_t f = 0; f < kFeatureCount; ++f) {
        const auto& d = distribution[f];
        codes[f] = static_cast<int>(std::ranges::max_element(d) - d.begin()) + 1;
    }
    return Grammar::fromCodes(codes);
}

Grammar Profile::sample(std::mt19937_64& rng) const {
    CodeVector codes{};
    for (std::size_t f = 0; f < kFeatureCount; ++f) {
        const auto& d = distribution[f];
        std::discrete_distribution<int> pick(d.begin(), d.end());
        codes[f] = pick(rng) + 1;
    }
    return Grammar::fromCodes(codes);
}

void Profile::appendTo(std::vector<float>& out) const {
    for (const auto& d : distribution) {
        out.insert(out.end(), d.begin(), d.end());
    }
}

std::size_t Profile::width() {
    std::size_t w = 0;
    for (const auto& f : kFeatures) {
        w += f.codes;
    }
    return w;
}

TypologyBank::TypologyBank(const wals::Database& db, const nn::Imputer& imputer) : db_(&db) {
    for (std::size_t f = 0; f < kFeatureCount; ++f) {
        walsIndex_[f] = db.feature(kFeatures[f].id);
        const auto& feature = db.features()[walsIndex_[f]];
        if (feature.codes.size() != kFeatures[f].codes) {
            throw std::runtime_error(std::format("WALS feature {} has {} codes, expected {}", kFeatures[f].id,
                                                 feature.codes.size(), kFeatures[f].codes));
        }
        for (std::size_t c = 0; c < feature.codes.size(); ++c) {
            if (feature.codes[c].number != static_cast<int>(c) + 1) {
                throw std::runtime_error(std::format("WALS feature {} has non-contiguous codes", kFeatures[f].id));
            }
        }
        marginal_[f].assign(kFeatures[f].codes, 0.0F);
    }

    const auto& languages = db.languages();
    profiles_.resize(languages.size());
    for (std::size_t l = 0; l < languages.size(); ++l) {
        const auto all = imputer.distributions(l);
        auto& p = profiles_[l];
        for (std::size_t f = 0; f < kFeatureCount; ++f) {
            p.distribution[f] = all[walsIndex_[f]];
            const int v = db.value(l, walsIndex_[f]);
            p.observed[f] = v != wals::kMissing;
            if (p.observed[f]) {
                marginal_[f][static_cast<std::size_t>(v)] += 1.0F;
            }
        }
        if (languages[l].sample200) {
            sample200_.push_back(l);
        }
    }
}

Profile TypologyBank::profile(std::size_t language) const {
    return profiles_.at(language);
}

Profile TypologyBank::randomProfile(std::mt19937_64& rng) const {
    Profile p;
    for (std::size_t f = 0; f < kFeatureCount; ++f) {
        std::discrete_distribution<std::size_t> pick(marginal_[f].begin(), marginal_[f].end());
        p.distribution[f].assign(kFeatures[f].codes, 0.0F);
        p.distribution[f][pick(rng)] = 1.0F;
        p.observed[f] = true;
    }
    return p;
}

}
