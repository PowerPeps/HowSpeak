#include "syntax/Sentence.hpp"

#include <format>
#include <stdexcept>

namespace howspeak::syntax {

std::string_view Word::feat(std::string_view key) const {
    const auto it = feats.find(key);
    return it == feats.end() ? std::string_view{} : std::string_view(it->second);
}

std::string_view Word::baseRelation() const {
    const std::string_view rel = deprel;
    return rel.substr(0, rel.find(':'));
}

std::vector<std::vector<int>> Sentence::children() const {
    std::vector<std::vector<int>> out(words.size());
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (words[i].head >= 0) {
            out[static_cast<std::size_t>(words[i].head)].push_back(static_cast<int>(i));
        }
    }
    return out;
}

int Sentence::root() const {
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (words[i].head < 0) {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("sentence without root");
}

bool Sentence::isQuestion() const {
    const Word* last = nullptr;
    for (const auto& w : words) {
        if (!w.marker && (last == nullptr || w.order > last->order)) {
            last = &w;
        }
    }
    return last != nullptr && last->upos == "PUNCT" && (last->form == "?" || last->form == "？" || last->form == "؟");
}

bool Sentence::subtreeHas(int node, const std::function<bool(const Word&)>& predicate) const {
    const auto kids = children();
    std::vector<int> stack{node};
    while (!stack.empty()) {
        const int n = stack.back();
        stack.pop_back();
        if (predicate(words[static_cast<std::size_t>(n)])) {
            return true;
        }
        for (const int k : kids[static_cast<std::size_t>(n)]) {
            stack.push_back(k);
        }
    }
    return false;
}

int Sentence::insert(std::size_t position, Word word) {
    if (position > words.size()) {
        throw std::out_of_range("Sentence::insert");
    }
    const int pos = static_cast<int>(position);
    for (auto& w : words) {
        if (w.head >= pos) {
            ++w.head;
        }
    }
    if (word.head >= pos) {
        ++word.head;
    }
    for (auto& m : multiwords) {
        if (m.first >= pos) {
            ++m.first;
        }
        if (m.last >= pos) {
            ++m.last;
        }
    }
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(position), std::move(word));
    return pos;
}

void Sentence::validate() const {
    int roots = 0;
    const int n = static_cast<int>(words.size());
    for (int i = 0; i < n; ++i) {
        const int h = words[static_cast<std::size_t>(i)].head;
        if (h < -1 || h >= n || h == i) {
            throw std::runtime_error(std::format("word {} has invalid head {}", i, h));
        }
        roots += h < 0 ? 1 : 0;
        int steps = 0;
        for (int cur = h; cur >= 0; cur = words[static_cast<std::size_t>(cur)].head) {
            if (++steps > n) {
                throw std::runtime_error("dependency cycle");
            }
        }
    }
    if (n > 0 && roots != 1) {
        throw std::runtime_error(std::format("{} roots in sentence", roots));
    }
}

Features parseFeatures(std::string_view text) {
    Features feats;
    if (text.empty() || text == "_") {
        return feats;
    }
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = std::min(text.find('|', start), text.size());
        const auto item = text.substr(start, end - start);
        const auto eq = item.find('=');
        if (eq != std::string_view::npos) {
            feats.emplace(std::string(item.substr(0, eq)), std::string(item.substr(eq + 1)));
        }
        start = end + 1;
    }
    return feats;
}

std::string formatFeatures(const Features& feats) {
    if (feats.empty()) {
        return "_";
    }
    std::string out;
    for (const auto& [k, v] : feats) {
        if (!out.empty()) {
            out += '|';
        }
        out += k;
        out += '=';
        out += v;
    }
    return out;
}

}
