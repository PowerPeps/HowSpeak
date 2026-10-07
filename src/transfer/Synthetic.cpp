#include "transfer/Synthetic.hpp"

#include "transfer/Groups.hpp"
#include "transfer/Realizer.hpp"
#include "transfer/Rules.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <string_view>

namespace howspeak::transfer {

namespace {

constexpr std::array<std::string_view, 8> kMorphologicalCases = {"Loc", "Abl", "All", "Ins", "Com", "Dat", "Ine", "Ela"};

syntax::Features finiteVerb(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> person(1, 3);
    std::bernoulli_distribution plural(0.3);
    return {{"VerbForm", "Fin"}, {"Mood", "Ind"}, {"Person", std::to_string(person(rng))},
            {"Number", plural(rng) ? "Plur" : "Sing"}};
}

}

int SyntheticTreebank::add(std::string upos, std::string deprel, int head, syntax::Features feats) {
    syntax::Word w;
    w.form = std::format("w{}", s_.words.size());
    w.lemma = w.form;
    w.upos = std::move(upos);
    w.deprel = std::move(deprel);
    w.head = head;
    w.feats = std::move(feats);
    w.order = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    s_.words.push_back(std::move(w));
    return static_cast<int>(s_.words.size()) - 1;
}

int SyntheticTreebank::nounPhrase(int head, std::string deprel, int depth, bool interrogative) {
    const bool needsCase = deprel == "obl" || deprel.starts_with("nmod");
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double kind = u(rng_);

    if (interrogative) {
        const int w = add("PRON", std::move(deprel), head, {{"PronType", "Int"}});
        if (needsCase) {
            add("ADP", "case", w);
        }
        return w;
    }
    if (kind < 0.12 && deprel != "nmod") {
        const int w = add("PRON", std::move(deprel), head, {{"PronType", "Prs"}, {"Person", "3"}});
        if (needsCase) {
            add("ADP", "case", w);
        }
        return w;
    }

    const bool proper = kind < 0.22;
    syntax::Features feats;
    if (!proper && chance(0.3)) {
        feats.emplace("Number", "Plur");
    }
    const bool morphologicalCase = needsCase && chance(0.08);
    if (morphologicalCase) {
        std::uniform_int_distribution<std::size_t> pick(0, kMorphologicalCases.size() - 1);
        feats.emplace("Case", std::string(kMorphologicalCases[pick(rng_)]));
    }
    const int n = add(proper ? "PROPN" : "NOUN", std::move(deprel), head, std::move(feats));
    if (needsCase && !morphologicalCase) {
        add("ADP", "case", n);
    }
    if (proper) {
        if (chance(0.2)) {
            add("PROPN", "flat", n);
        }
        return n;
    }

    if (chance(0.5)) {
        add("DET", "det", n, {{"PronType", "Art"}, {"Definite", chance(0.6) ? "Def" : "Ind"}});
    } else if (chance(0.15)) {
        add("DET", "det", n, {{"PronType", "Dem"}});
    } else if (chance(0.1)) {
        add("DET", "det", n, {{"Poss", "Yes"}, {"PronType", "Prs"}});
    } else if (chance(0.05)) {
        add("DET", "det", n, {{"PronType", "Tot"}});
    }
    for (int a = 0; a < 2 && chance(a == 0 ? 0.3 : 0.15); ++a) {
        const int adj = add("ADJ", "amod", n);
        if (chance(0.15)) {
            add("ADV", "advmod", adj);
        }
    }
    if (chance(0.1)) {
        add("NUM", "nummod", n);
    }
    if (chance(0.05)) {
        add("NOUN", "compound", n);
    }
    if (depth < 2 && chance(0.12)) {
        nounPhrase(n, chance(0.3) ? "nmod:poss" : "nmod", depth + 1, false);
    }
    if (depth < 2 && chance(0.07)) {
        clause(n, "acl:relcl", depth + 1, Clause::Relative);
    }
    if (depth < 2 && chance(0.03)) {
        add("VERB", "acl", n, {{"VerbForm", "Part"}});
    }
    if (depth < 2 && chance(0.04)) {
        const int conj = nounPhrase(n, "conj", depth + 1, false);
        add("CCONJ", "cc", conj);
    }
    if (chance(0.02)) {
        nounPhrase(n, "appos", depth + 1, false);
    }
    return n;
}

int SyntheticTreebank::clause(int head, std::string deprel, int depth, Clause kind) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double type = u(rng_);
    const bool copularNoun = type < 0.08;
    const bool copularAdj = !copularNoun && type < 0.16;
    const bool hasAux = !copularNoun && !copularAdj && chance(0.3);

    syntax::Features predFeats;
    if (!copularNoun && !copularAdj) {
        predFeats = hasAux ? syntax::Features{{"VerbForm", "Part"}} : finiteVerb(rng_);
    }
    const int pred = add(copularNoun ? "NOUN" : (copularAdj ? "ADJ" : "VERB"), std::move(deprel), head,
                         std::move(predFeats));
    if (copularNoun || copularAdj) {
        add("AUX", "cop", pred, finiteVerb(rng_));
    }
    if (hasAux) {
        add("AUX", "aux", pred, finiteVerb(rng_));
        if (chance(0.15)) {
            add("AUX", "aux", pred, {{"VerbForm", "Part"}});
        }
    }

    const bool question = kind == Clause::Main && chance(0.25);
    const bool whQuestion = question && chance(0.5);
    std::uniform_int_distribution<int> whSlot(0, 3);
    const int wh = whQuestion ? whSlot(rng_) : -1;

    if (kind == Clause::Relative) {
        const bool pronoun = chance(0.6);
        if (pronoun) {
            add("PRON", chance(0.5) ? "nsubj" : "obj", pred, {{"PronType", "Rel"}});
        }
    }
    const bool needsSubject = std::ranges::none_of(s_.words, [&](const syntax::Word& w) {
        return w.head == pred && w.baseRelation() == "nsubj";
    });
    if (needsSubject && kind != Clause::Coordinate && chance(0.85)) {
        const double r = u(rng_);
        if (wh == 0) {
            nounPhrase(pred, "nsubj", depth, true);
        } else if (r < 0.03) {
            add("PRON", "expl", pred, {{"PronType", "Prs"}});
        } else {
            nounPhrase(pred, "nsubj", depth, false);
        }
    }
    if (!copularNoun && !copularAdj) {
        if (wh == 1 || chance(0.55)) {
            nounPhrase(pred, "obj", depth, wh == 1);
        }
        if (chance(0.08)) {
            nounPhrase(pred, "iobj", depth, false);
        }
    }
    for (int k = 0; k < 2 && (k == 0 ? (wh == 2 || chance(0.4)) : chance(0.15)); ++k) {
        nounPhrase(pred, chance(0.15) ? "obl:tmod" : "obl", depth, k == 0 && wh == 2);
    }
    if (wh == 3) {
        add("ADV", "advmod", pred, {{"PronType", "Int"}});
    }
    if (chance(0.2)) {
        add("ADV", "advmod", pred);
    }
    if (chance(0.15)) {
        add("PART", "advmod", pred, {{"Polarity", "Neg"}});
        if (chance(0.3)) {
            add("ADV", "advmod", pred, {{"Polarity", "Neg"}});
        }
    }
    if (kind == Clause::Complement || kind == Clause::Adverbial) {
        add("SCONJ", "mark", pred);
    }
    if (kind == Clause::Coordinate) {
        add("CCONJ", "cc", pred);
    }
    if (depth < 2) {
        if (chance(0.08)) {
            clause(pred, "ccomp", depth + 1, Clause::Complement);
        }
        if (chance(0.08)) {
            clause(pred, "advcl", depth + 1, Clause::Adverbial);
        }
        if (chance(0.05)) {
            const int x = add("VERB", "xcomp", pred, {{"VerbForm", "Inf"}});
            if (chance(0.5)) {
                nounPhrase(x, "obj", depth + 1, false);
            }
        }
        if (kind == Clause::Main && chance(0.06)) {
            clause(pred, "conj", depth + 1, Clause::Coordinate);
        }
    }
    if (kind == Clause::Main) {
        const int p = add("PUNCT", "punct", pred);
        s_.words[static_cast<std::size_t>(p)].form = question ? "?" : ".";
        s_.words[static_cast<std::size_t>(p)].order = 2.0;
    }
    return pred;
}

syntax::Sentence SyntheticTreebank::next(const Grammar& source) {
    s_ = syntax::Sentence{};
    clause(-1, "root", 0, Clause::Main);

    Plan plan;
    plan.groups = buildGroups(s_);
    const Rules rules(source);
    for (const auto& g : plan.groups) {
        plan.decisions.push_back(rules.decide(g));
    }
    const auto order = linearize(s_, plan);
    for (std::size_t i = 0; i < order.size(); ++i) {
        s_.words[static_cast<std::size_t>(order[i])].order = static_cast<double>(i);
    }
    s_.validate();
    return std::move(s_);
}

}
