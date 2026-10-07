#include "transfer/Realizer.hpp"

#include "core/Utf8.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>

namespace howspeak::transfer {

namespace {

struct Piece {
    std::string text;
    Operation op = Operation::Keep;
    int first = -1;
    int last = -1;
    bool closing = false;
    bool opening = false;
};

bool closingPunctuation(std::string_view s) {
    return s == "." || s == "," || s == ";" || s == ":" || s == "!" || s == "?" || s == ")" || s == "]" ||
           s == "»" || s == "”" || s == "…" || s == "。" || s == "、" || s == "？" || s == "！";
}

bool openingPunctuation(std::string_view s) {
    return s == "(" || s == "[" || s == "«" || s == "“" || s == "¿" || s == "¡";
}

}

std::vector<int> linearize(const syntax::Sentence& s, const Plan& plan) {
    std::unordered_map<int, std::size_t> groupOf;
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        groupOf.emplace(plan.groups[g].head, g);
    }
    std::vector<int> out;
    out.reserve(s.words.size());
    const std::function<void(int)> visit = [&](int node) {
        const auto it = groupOf.find(node);
        if (it == groupOf.end()) {
            out.push_back(node);
            return;
        }
        const auto& group = plan.groups[it->second];
        for (const auto i : orderGroup(group, plan.decisions[it->second])) {
            const auto& item = group.items[i];
            if (item.isHead) {
                out.push_back(node);
            } else {
                visit(item.word);
            }
        }
    };
    if (!s.words.empty()) {
        visit(s.root());
    }
    return out;
}

std::vector<Operation> wordOperations(const syntax::Sentence& s, const Plan& plan) {
    std::vector<Operation> ops(s.words.size(), Operation::Keep);
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        const auto& group = plan.groups[g];
        for (std::size_t i = 0; i < group.items.size(); ++i) {
            if (!group.items[i].isHead) {
                ops[static_cast<std::size_t>(group.items[i].word)] = plan.decisions[g][i].op;
            }
        }
    }
    return ops;
}

std::string realize(const syntax::Sentence& s, const Plan& plan) {
    const auto order = linearize(s, plan);
    const auto ops = wordOperations(s, plan);

    int sourceFirst = -1;
    double firstOrder = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < s.words.size(); ++i) {
        if (!s.words[i].marker && s.words[i].order < firstOrder) {
            firstOrder = s.words[i].order;
            sourceFirst = static_cast<int>(i);
        }
    }

    std::vector<Piece> pieces;
    for (const int wi : order) {
        const auto& w = s.words[static_cast<std::size_t>(wi)];
        const auto op = ops[static_cast<std::size_t>(wi)];
        if (op == Operation::Drop) {
            continue;
        }
        Piece p;
        p.text = w.useLemma && !w.lemma.empty() ? w.lemma : w.form;
        p.op = op;
        p.first = wi;
        p.last = wi;
        p.closing = w.upos == "PUNCT" && closingPunctuation(w.form);
        p.opening = w.upos == "PUNCT" && openingPunctuation(w.form);
        pieces.push_back(std::move(p));
    }

    for (const auto& m : s.multiwords) {
        for (std::size_t i = 0; i + static_cast<std::size_t>(m.last - m.first) < pieces.size(); ++i) {
            bool match = true;
            for (int k = 0; k <= m.last - m.first && match; ++k) {
                const auto& p = pieces[i + static_cast<std::size_t>(k)];
                match = p.first == m.first + k && p.op == Operation::Keep &&
                        !s.words[static_cast<std::size_t>(p.first)].useLemma;
            }
            if (match) {
                pieces[i].text = m.form;
                pieces[i].last = m.last;
                pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                             pieces.begin() + static_cast<std::ptrdiff_t>(i) + 1 + (m.last - m.first));
                break;
            }
        }
    }

    if (sourceFirst >= 0) {
        const auto& w = s.words[static_cast<std::size_t>(sourceFirst)];
        const bool lowerLemma = !w.lemma.empty() && !core::utf8::startsUpper(w.lemma);
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            if (pieces[i].first == sourceFirst && i > 0 && w.upos != "PROPN" && lowerLemma) {
                pieces[i].text = core::utf8::lower(pieces[i].text);
            }
        }
    }

    std::string out;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        const auto& p = pieces[i];
        if (i > 0) {
            const auto& prev = pieces[i - 1];
            if (p.op == Operation::SuffixAffix || prev.op == Operation::PrefixAffix) {
                out += '-';
            } else if (p.op == Operation::SuffixClitic || prev.op == Operation::PrefixClitic) {
                out += '=';
            } else if (!p.closing && !prev.opening) {
                out += ' ';
            }
        }
        out += p.text;
    }
    if (!out.empty() && out.front() != '[') {
        const auto space = out.find(' ');
        const auto head = out.substr(0, space);
        out = core::utf8::capitalize(head) + (space == std::string::npos ? "" : out.substr(space));
    }
    return out;
}

}
