#include "transfer/Restructure.hpp"

#include "transfer/Groups.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace howspeak::transfer {

namespace {

constexpr std::array<std::string_view, 22> kObliqueCases = {
    "Loc", "Abl", "All", "Ill", "Ine", "Ela", "Ade", "Ins", "Com", "Dat", "Ben",
    "Abe", "Ess", "Tra", "Ter", "Lat", "Per", "Sbl", "Sup", "Del", "Cau", "Equ",
};
constexpr std::array<std::string_view, 4> kCoreCases = {"Nom", "Acc", "Erg", "Abs"};

std::string gloss(std::string_view text) {
    std::string out = "[";
    for (const char c : text) {
        out.push_back(c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c);
    }
    out.push_back(']');
    return out;
}

void append(syntax::Sentence& s, std::string form, std::string upos, std::string deprel, int head, double order,
            syntax::Features feats = {}) {
    syntax::Word w;
    w.form = std::move(form);
    w.lemma = w.form;
    w.upos = std::move(upos);
    w.deprel = std::move(deprel);
    w.head = head;
    w.order = order;
    w.marker = true;
    w.feats = std::move(feats);
    static_cast<void>(s.insert(s.words.size(), std::move(w)));
}

bool hasChild(const syntax::Sentence& s, const std::vector<int>& kids, auto&& predicate) {
    return std::ranges::any_of(kids, [&](int k) { return predicate(s.words[static_cast<std::size_t>(k)]); });
}

bool finite(const syntax::Word& w) {
    return w.has("VerbForm", "Fin") && !w.has("Mood", "Imp");
}

void questionParticle(syntax::Sentence& s, const Grammar& g) {
    const bool particle = g.polar == PolarQuestion::Particle || g.polar == PolarQuestion::Mixture;
    if (!particle || g.questionParticle == QuestionParticle::None || g.questionParticle == QuestionParticle::Unknown ||
        !isPolarQuestion(s)) {
        return;
    }
    double last = 0.0;
    for (const auto& w : s.words) {
        if (w.upos != "PUNCT") {
            last = std::max(last, w.order);
        }
    }
    append(s, "[Q]", "PART", "discourse:q", s.root(), last + 0.5);
}

void subjectMarkers(syntax::Sentence& s, const Grammar& g) {
    if (g.subjectPronoun != SubjectPronoun::Obligatory) {
        return;
    }
    const auto kids = s.children();
    const auto original = s.words.size();
    for (std::size_t i = 0; i < original; ++i) {
        const auto& w = s.words[i];
        const auto rel = w.head < 0 ? std::string_view("root") : w.baseRelation();
        if (rel != "root" && rel != "ccomp" && rel != "advcl" && rel != "parataxis") {
            continue;
        }
        const syntax::Word* inflected = finite(w) ? &w : nullptr;
        for (const int k : kids[i]) {
            const auto& child = s.words[static_cast<std::size_t>(k)];
            const auto childRel = child.baseRelation();
            if (!inflected && (childRel == "aux" || childRel == "cop") && finite(child)) {
                inflected = &child;
            }
        }
        if (!inflected || inflected->feat("Person").empty()) {
            continue;
        }
        if (hasChild(s, kids[i], [](const syntax::Word& c) {
                const auto r = c.baseRelation();
                return r == "nsubj" || r == "csubj" || r == "expl";
            })) {
            continue;
        }
        const auto number = inflected->feat("Number");
        const std::string form = "[" + std::string(inflected->feat("Person")) +
                                 (number == "Plur" ? "PL" : (number == "Sing" ? "SG" : "")) + "]";
        syntax::Features feats;
        feats.emplace("PronType", "Prs");
        append(s, form, "PRON", "nsubj", static_cast<int>(i), w.order - 0.45, std::move(feats));
    }
}

void caseMarkers(syntax::Sentence& s, const Grammar& g) {
    const bool adpositional =
        g.adposition == Adposition::Pre || g.adposition == Adposition::Post || g.adposition == Adposition::In;
    const bool caseless = g.caseCount == CaseCount::None;
    if (!caseless) {
        return;
    }
    const auto kids = s.children();
    const auto original = s.words.size();
    for (std::size_t i = 0; i < original; ++i) {
        auto& w = s.words[i];
        const auto c = w.feat("Case");
        if (c.empty() || (w.upos != "NOUN" && w.upos != "PROPN" && w.upos != "PRON")) {
            continue;
        }
        const bool oblique = std::ranges::find(kObliqueCases, c) != kObliqueCases.end() ||
                             (c == "Gen" && w.baseRelation() == "nmod");
        const bool core = std::ranges::find(kCoreCases, c) != kCoreCases.end();
        if (core && w.upos != "PRON") {
            w.useLemma = true;
        }
        if (!oblique || !adpositional ||
            hasChild(s, kids[i], [](const syntax::Word& k) { return k.baseRelation() == "case"; })) {
            continue;
        }
        w.useLemma = w.upos != "PRON";
        const auto order = w.order;
        append(s, gloss(c), "ADP", "case", static_cast<int>(i), order - 0.4);
    }
}

void plural(syntax::Sentence& s, const Grammar& g) {
    if (g.plural != Plural::None && g.plural != Plural::Word) {
        return;
    }
    const auto original = s.words.size();
    for (std::size_t i = 0; i < original; ++i) {
        auto& w = s.words[i];
        if (w.upos != "NOUN" || !w.has("Number", "Plur") || w.marker) {
            continue;
        }
        w.useLemma = true;
        if (g.plural == Plural::Word) {
            const auto order = w.order;
            append(s, "[PL]", "DET", "det", static_cast<int>(i), order - 0.35);
        }
    }
}

}

void restructure(syntax::Sentence& sentence, const Grammar& target) {
    questionParticle(sentence, target);
    subjectMarkers(sentence, target);
    caseMarkers(sentence, target);
    plural(sentence, target);
}

}
