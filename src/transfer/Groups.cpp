#include "transfer/Groups.hpp"

#include <algorithm>
#include <cstdlib>

namespace howspeak::transfer {

namespace {

bool nominal(std::string_view upos) {
    return upos == "NOUN" || upos == "PROPN" || upos == "PRON" || upos == "NUM";
}

}

bool isNegative(const syntax::Word& w) {
    return w.has("Polarity", "Neg") &&
           (w.upos == "PART" || w.upos == "ADV" || w.upos == "AUX" || w.upos == "VERB" || w.marker);
}

bool isPolarQuestion(const syntax::Sentence& s) {
    if (!s.isQuestion()) {
        return false;
    }
    return std::ranges::none_of(s.words, [](const syntax::Word& w) { return w.has("PronType", "Int"); });
}

std::vector<Group> buildGroups(const syntax::Sentence& s) {
    const auto kids = s.children();
    const bool polar = isPolarQuestion(s);

    std::vector<bool> interrogative(s.words.size(), false);
    for (std::size_t i = 0; i < s.words.size(); ++i) {
        if (!s.words[i].has("PronType", "Int")) {
            continue;
        }
        // Mark the path up to the nearest clause-level dependent.
        for (int cur = static_cast<int>(i); cur >= 0; cur = s.words[static_cast<std::size_t>(cur)].head) {
            interrogative[static_cast<std::size_t>(cur)] = true;
            const auto rel = s.words[static_cast<std::size_t>(cur)].baseRelation();
            if (rel == "nsubj" || rel == "obj" || rel == "iobj" || rel == "obl" || rel == "advmod" ||
                rel == "csubj" || rel == "ccomp" || rel == "xcomp" || rel == "root") {
                break;
            }
        }
    }

    std::vector<Group> groups;
    for (std::size_t h = 0; h < s.words.size(); ++h) {
        if (kids[h].empty()) {
            continue;
        }
        const auto& head = s.words[h];
        Group g;
        g.head = static_cast<int>(h);
        g.headUpos = head.upos;
        g.headRelation = head.head < 0 ? std::string_view("root") : head.baseRelation();
        g.headNominal = nominal(head.upos);
        g.headAdjectival = head.upos == "ADJ" || head.upos == "ADV";
        g.polarQuestion = polar;

        std::vector<int> members = kids[h];
        members.push_back(static_cast<int>(h));
        std::ranges::sort(members, [&](int a, int b) {
            return s.words[static_cast<std::size_t>(a)].order < s.words[static_cast<std::size_t>(b)].order;
        });
        const auto headPos = static_cast<int>(std::ranges::find(members, static_cast<int>(h)) - members.begin());
        const auto n = static_cast<int>(members.size());

        for (int k = 0; k < n; ++k) {
            const int wi = members[static_cast<std::size_t>(k)];
            const auto& w = s.words[static_cast<std::size_t>(wi)];
            Item it;
            it.word = wi;
            it.isHead = wi == static_cast<int>(h);
            it.relation = it.isHead ? std::string_view("HEAD") : std::string_view(w.deprel);
            it.baseRelation = it.isHead ? std::string_view("HEAD") : w.baseRelation();
            it.upos = w.upos;
            it.side = k < headPos ? -1 : (k > headPos ? 1 : 0);
            it.proximity = std::max(0, std::abs(k - headPos) - 1);
            it.rank = n > 1 ? static_cast<double>(k) / static_cast<double>(n - 1) : 0.0;
            it.article = w.has("PronType", "Art");
            it.definite = w.has("Definite", "Def");
            it.indefinite = w.has("Definite", "Ind");
            it.demonstrative = w.has("PronType", "Dem") && w.upos == "DET";
            it.possessive = w.has("Poss", "Yes") && w.upos == "DET";
            it.interrogative = !it.isHead && interrogative[static_cast<std::size_t>(wi)];
            it.relativePronoun = w.has("PronType", "Rel");
            it.personalPronoun = w.upos == "PRON" && (w.has("PronType", "Prs") || w.feat("PronType").empty());
            it.negative = !it.isHead && isNegative(w);
            it.marker = w.marker;
            g.negatives += it.negative ? 1 : 0;
            g.items.push_back(it);
        }
        for (auto it = g.items.rbegin(); it != g.items.rend(); ++it) {
            if (it->negative) {
                it->lastNegative = true;
                break;
            }
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

}
