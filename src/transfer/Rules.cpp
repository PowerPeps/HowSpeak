#include "transfer/Rules.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace howspeak::transfer {

namespace {

constexpr double kPunct = 1000.0;
constexpr double kParataxis = 400.0;
constexpr double kConj = 300.0;
constexpr double kQuestion = 300.0;
constexpr double kMark = 200.0;
constexpr double kCoordinator = 150.0;
constexpr double kNegClause = 150.0;
constexpr double kOther = 120.0;
constexpr double kWh = 100.0;
constexpr double kAdverbialClause = 60.0;
constexpr double kRelative = 60.0;
constexpr double kCase = 50.0;
constexpr double kSubject = 40.0;
constexpr double kArticle = 40.0;
constexpr double kAdverb = 35.0;
constexpr double kOblique = 30.0;
constexpr double kDemonstrative = 30.0;
constexpr double kNumeral = 25.0;
constexpr double kComplement = 25.0;
constexpr double kObject = 20.0;
constexpr double kGenitive = 20.0;
constexpr double kAdjective = 10.0;
constexpr double kAuxiliary = 5.0;
constexpr double kNegation = 3.0;
constexpr double kDegree = 3.0;
constexpr double kFronted = -45.0;  // fronted auxiliary in interrogative word order

double sideKey(const Item& it, double distance) {
    return it.side < 0 ? -distance : distance;
}

bool clauseLevel(std::string_view rel) {
    return rel == "nsubj" || rel == "obj" || rel == "iobj" || rel == "obl" || rel == "advmod" || rel == "ccomp" ||
           rel == "xcomp" || rel == "csubj";
}

}

bool Rules::objectBeforeVerb() const {
    switch (g_.clause) {
    case ClauseOrder::SOV:
    case ClauseOrder::OVS:
    case ClauseOrder::OSV:
        return true;
    case ClauseOrder::SVO:
    case ClauseOrder::VSO:
    case ClauseOrder::VOS:
        return false;
    default:
        return g_.objectVerb == ObjectVerb::OV;
    }
}

bool Rules::objectAfterVerb() const {
    switch (g_.clause) {
    case ClauseOrder::SVO:
    case ClauseOrder::VSO:
    case ClauseOrder::VOS:
        return true;
    case ClauseOrder::SOV:
    case ClauseOrder::OVS:
    case ClauseOrder::OSV:
        return false;
    default:
        return g_.objectVerb == ObjectVerb::VO;
    }
}

Operation Rules::affix(bool prefix) const {
    return prefix ? Operation::PrefixAffix : Operation::SuffixAffix;
}

Rules::Slots Rules::clauseSlots() const {
    Slots s;
    const auto set = [&](double subject, double object) {
        s.subject = subject;
        s.object = object;
        s.subjectKnown = true;
        s.objectKnown = true;
    };
    switch (g_.clause) {
    case ClauseOrder::SOV: set(-kSubject, -kObject); break;
    case ClauseOrder::SVO: set(-kSubject, kObject); break;
    case ClauseOrder::VSO: set(kObject, kOblique); break;
    case ClauseOrder::VOS: set(kOblique, kObject); break;
    case ClauseOrder::OVS: set(kObject, -kSubject); break;
    case ClauseOrder::OSV: set(-kOblique, -kSubject); break;
    default:
        if (g_.subjectVerb == SubjectVerb::SV || g_.subjectVerb == SubjectVerb::VS) {
            s.subject = g_.subjectVerb == SubjectVerb::SV ? -kSubject : kObject;
            s.subjectKnown = true;
        }
        if (g_.objectVerb == ObjectVerb::OV || g_.objectVerb == ObjectVerb::VO) {
            s.object = g_.objectVerb == ObjectVerb::OV ? -kObject : kObject + 5.0;
            s.objectKnown = true;
        }
        break;
    }

    s.obliqueKnown = true;
    switch (g_.oblique) {
    case ObliqueOrder::VOX: s.oblique = s.objectKnown && s.object > 0 ? s.object + 10.0 : kOblique; break;
    case ObliqueOrder::XVO: s.oblique = -10.0; break;
    case ObliqueOrder::XOV: s.oblique = s.objectKnown && s.object < 0 ? s.object - 5.0 : -kNumeral; break;
    case ObliqueOrder::OXV: s.oblique = -10.0; break;
    case ObliqueOrder::OVX: s.oblique = kOblique; break;
    default:
        if (s.objectKnown) {
            s.oblique = s.object < 0 ? s.object - 5.0 : s.object + 10.0;
        } else {
            s.obliqueKnown = false;
        }
        break;
    }
    return s;
}

Decision Rules::negation(const Group& group, const Item& item, const Slots& slots) const {
    Decision d{sideKey(item, kNegation), Operation::Keep};
    const auto between = [&](double fallback) {
        return slots.subjectKnown && slots.objectKnown ? (slots.subject + slots.object) / 2.0 : fallback;
    };
    bool single = true;
    if (g_.negation == NegationVerb::NegPrefix) {
        d = {-1.0, Operation::PrefixAffix};
    } else if (g_.negation == NegationVerb::NegSuffix) {
        d = {1.0, Operation::SuffixAffix};
    } else {
        switch (g_.negationClause) {
        case NegationClause::NegSVO:
        case NegationClause::NegSOV:
        case NegationClause::NegVSO:
        case NegationClause::NegVOS: d.key = -kNegClause; break;
        case NegationClause::SNegVO:
        case NegationClause::SONegV:
        case NegationClause::ONegVS: d.key = -kNegation; break;
        case NegationClause::SNegOV: d.key = between(-kNegation); break;
        case NegationClause::VSNegO: d.key = between(kNegation); break;
        case NegationClause::SVNegO:
        case NegationClause::SOVNeg:
        case NegationClause::OVNegS:
        case NegationClause::OSVNeg: d.key = kNegation; break;
        case NegationClause::SVONeg:
        case NegationClause::VSONeg: d.key = kNegClause; break;
        default:
            switch (g_.negation) {
            case NegationVerb::NegV: d.key = -kNegation; break;
            case NegationVerb::VNeg: d.key = kNegation; break;
            default: single = false; break;
            }
            break;
        }
    }
    if (group.negatives > 1 && single && !item.lastNegative) {
        d.op = Operation::Drop;
    }
    return d;
}

Decision Rules::dependent(const Group& group, const Item& item, const Slots& slots) const {
    const auto rel = item.baseRelation;
    const bool ov = objectBeforeVerb();
    const bool vo = objectAfterVerb();

    if (item.relation == "discourse:q") {
        switch (g_.questionParticle) {
        case QuestionParticle::Initial: return {-kQuestion, Operation::Keep};
        case QuestionParticle::Second: return {0.0, Operation::Keep};
        default: return {kQuestion, Operation::Keep};
        }
    }
    if (rel == "punct") {
        return {sideKey(item, kPunct), Operation::Keep};
    }
    if (item.negative) {
        return negation(group, item, slots);
    }

    Decision d{sideKey(item, kOther), Operation::Keep};
    if (rel == "nsubj" || rel == "csubj" || rel == "expl") {
        d.key = slots.subjectKnown ? slots.subject : sideKey(item, kSubject);
        const bool pronoun = (item.personalPronoun || rel == "expl") && rel != "csubj" && !item.marker;
        if (pronoun && (g_.subjectPronoun == SubjectPronoun::VerbAffix || g_.subjectPronoun == SubjectPronoun::Optional)) {
            d.op = Operation::Drop;
        }
        if (group.polarQuestion && g_.polar == PolarQuestion::WordOrder && d.key < 0) {
            const bool hasAux = std::ranges::any_of(group.items, [](const Item& i) {
                return i.baseRelation == "aux" || i.baseRelation == "cop";
            });
            if (!hasAux) {
                d.key = 0.5;
            }
        }
    } else if (rel == "obj") {
        d.key = slots.objectKnown ? slots.object : sideKey(item, kObject);
    } else if (rel == "iobj") {
        d.key = (slots.objectKnown ? slots.object : sideKey(item, kObject)) - 1.0;
    } else if (rel == "ccomp" || rel == "xcomp") {
        d.key = ov ? -15.0 : (vo ? kComplement + 5.0 : sideKey(item, kComplement));
    } else if (rel == "obl") {
        d.key = slots.obliqueKnown ? slots.oblique : sideKey(item, kOblique);
    } else if (rel == "advcl") {
        d.key = sideKey(item, kAdverbialClause);
    } else if (rel == "advmod") {
        if (group.headAdjectival) {
            d.key = g_.degree == Degree::DegAdj ? -kDegree
                  : g_.degree == Degree::AdjDeg ? kDegree
                                                : sideKey(item, kDegree);
        } else {
            d.key = sideKey(item, kAdverb);
        }
    } else if (rel == "aux" || rel == "cop") {
        d.key = ov ? kAuxiliary : (vo ? -kAuxiliary : sideKey(item, kAuxiliary));
        if (group.polarQuestion && g_.polar == PolarQuestion::WordOrder) {
            d.key = kFronted;
        } else if (rel == "aux" && ov && g_.tenseAffix == TenseAffix::Suffix) {
            d.op = Operation::SuffixAffix;
        } else if (rel == "aux" && vo && g_.tenseAffix == TenseAffix::Prefix) {
            d.op = Operation::PrefixAffix;
        }
        if (rel == "cop" && ((group.headNominal && g_.zeroCopula == ZeroCopula::Possible) ||
                             (group.headAdjectival && g_.predicativeAdjective == PredicativeAdjective::Verbal))) {
            d.op = Operation::Drop;
        }
    } else if (rel == "mark") {
        switch (g_.subordinator) {
        case Subordinator::Initial: d.key = -kMark; break;
        case Subordinator::Final: d.key = kMark; break;
        case Subordinator::Suffix: d = {kMark, Operation::SuffixAffix}; break;
        case Subordinator::Unknown: d.key = ov ? kMark : (vo ? -kMark : sideKey(item, kMark)); break;
        default: d.key = sideKey(item, kMark); break;
        }
    } else if (rel == "case") {
        const bool prefixing = g_.caseAffix == CaseAffix::Prefix || g_.caseAffix == CaseAffix::PreClitic;
        switch (g_.adposition) {
        case Adposition::Pre: d.key = -kCase; break;
        case Adposition::Post:
        case Adposition::In: d.key = kCase; break;
        case Adposition::Absent: d.key = prefixing ? -kCase : kCase; break;
        default: d.key = sideKey(item, kCase); break;
        }
        const bool postposed = g_.adposition == Adposition::Post || g_.adposition == Adposition::Absent;
        const bool preposed = g_.adposition == Adposition::Pre || g_.adposition == Adposition::Absent;
        if (d.key > 0 && postposed && g_.caseAffix == CaseAffix::Suffix) {
            d.op = Operation::SuffixAffix;
        } else if (d.key > 0 && postposed && g_.caseAffix == CaseAffix::PostClitic) {
            d.op = Operation::SuffixClitic;
        } else if (d.key < 0 && preposed && g_.caseAffix == CaseAffix::Prefix) {
            d.op = Operation::PrefixAffix;
        } else if (d.key < 0 && preposed && g_.caseAffix == CaseAffix::PreClitic) {
            d.op = Operation::PrefixClitic;
        }
    } else if (rel == "det") {
        const bool prefixing = g_.affixation == Affixation::WeakPrefix || g_.affixation == Affixation::StrongPrefix;
        const auto demonstrative = [&] {
            switch (g_.demonstrative) {
            case Demonstrative::DemN: return Decision{-kDemonstrative, Operation::Keep};
            case Demonstrative::NDem: return Decision{kDemonstrative, Operation::Keep};
            case Demonstrative::Prefix: return Decision{-kDemonstrative, Operation::PrefixAffix};
            case Demonstrative::Suffix: return Decision{kDemonstrative, Operation::SuffixAffix};
            default: return Decision{sideKey(item, kDemonstrative), Operation::Keep};
            }
        };
        if (item.article && !item.indefinite) {
            switch (g_.definite) {
            case Definite::Demonstrative: d = demonstrative(); break;
            case Definite::Affix: d = {prefixing ? -kArticle : kArticle, affix(prefixing)}; break;
            case Definite::OnlyIndefinite:
            case Definite::None: d = {sideKey(item, kArticle), Operation::Drop}; break;
            default: d.key = sideKey(item, kArticle); break;
            }
        } else if (item.article) {
            switch (g_.indefinite) {
            case Indefinite::Affix: d = {prefixing ? -kArticle : kArticle, affix(prefixing)}; break;
            case Indefinite::OnlyDefinite:
            case Indefinite::None: d = {sideKey(item, kArticle), Operation::Drop}; break;
            default: d.key = sideKey(item, kArticle); break;
            }
        } else if (item.demonstrative) {
            d = demonstrative();
        } else if (item.possessive) {
            d.key = g_.genitive == Genitive::GenN ? -kGenitive
                  : g_.genitive == Genitive::NGen ? kGenitive
                                                  : sideKey(item, kGenitive);
        } else {
            d.key = sideKey(item, kDemonstrative);
        }
    } else if (rel == "nummod") {
        d.key = g_.numeral == Numeral::NumN ? -kNumeral
              : g_.numeral == Numeral::NNum ? kNumeral
                                            : sideKey(item, kNumeral);
    } else if (rel == "amod") {
        d.key = g_.adjective == Adjective::AdjN ? -kAdjective
              : g_.adjective == Adjective::NAdj ? kAdjective
                                                : sideKey(item, kAdjective);
    } else if (rel == "nmod" || (rel == "compound" && item.relation != "compound:prt")) {
        d.key = g_.genitive == Genitive::GenN ? -kGenitive
              : g_.genitive == Genitive::NGen ? kGenitive
                                              : sideKey(item, kGenitive);
    } else if (rel == "acl") {
        d.key = g_.relative == Relative::NRel ? kRelative
              : g_.relative == Relative::RelN ? -kRelative
                                              : sideKey(item, kRelative);
    } else if (rel == "fixed" || rel == "flat" || rel == "goeswith" || rel == "compound") {
        d.key = sideKey(item, 1.0);
    } else if (rel == "cc" || rel == "appos") {
        d.key = sideKey(item, kCoordinator);
    } else if (rel == "conj") {
        d.key = sideKey(item, kConj);
    } else if (rel == "parataxis") {
        d.key = sideKey(item, kParataxis);
    } else if (rel == "clf") {
        d.key = sideKey(item, kNumeral);
    }

    if (item.interrogative && clauseLevel(rel) && g_.wh == WhPosition::Initial) {
        d.key = -kWh;
    }
    if (item.relativePronoun && g_.relativization == Relativization::Gap) {
        d.op = Operation::Drop;
    }
    return d;
}

std::vector<Decision> Rules::decide(const Group& group) const {
    const Slots slots = clauseSlots();
    std::vector<Decision> out(group.items.size());
    for (std::size_t i = 0; i < group.items.size(); ++i) {
        const auto& item = group.items[i];
        if (item.isHead) {
            continue;
        }
        out[i] = dependent(group, item, slots);
        if (out[i].key != 0.0) {
            out[i].key += (out[i].key > 0 ? 0.01 : -0.01) * static_cast<double>(item.proximity);
        }
    }

    // Inner punctuation sticks to the constituent that follows it in the source.
    const auto n = group.items.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (group.items[i].baseRelation != "punct" || i == 0) {
            continue;
        }
        std::size_t next = i + 1;
        while (next < n && group.items[next].baseRelation == "punct") {
            ++next;
        }
        if (next < n) {
            out[i].key = out[next].key - 1e-4;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (group.items[i].relation != "discourse:q" || g_.questionParticle != QuestionParticle::Second) {
            continue;
        }
        double first = std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i && group.items[j].baseRelation != "punct") {
                first = std::min(first, out[j].key);
            }
        }
        out[i].key = first + 1e-3;
    }
    return out;
}

std::vector<std::size_t> orderGroup(const Group& group, const std::vector<Decision>& decisions) {
    std::vector<std::size_t> order(group.items.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, {}, [&](std::size_t i) { return decisions[i].key; });
    return order;
}

}
