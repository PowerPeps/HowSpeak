#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace howspeak::syntax {

using Features = std::map<std::string, std::string, std::less<>>;

struct Word {
    std::string form;
    std::string lemma;
    std::string upos;
    Features feats;
    int head = -1;  // index in Sentence::words, -1 for the root
    std::string deprel;
    double order = 0.0;  // source linear position, fractional for inserted words
    bool spaceAfter = true;
    bool marker = false;    // abstract morpheme inserted by restructuring, e.g. [Q]
    bool useLemma = false;  // realise the lemma instead of the inflected form

    [[nodiscard]] std::string_view feat(std::string_view key) const;
    [[nodiscard]] bool has(std::string_view key, std::string_view value) const { return feat(key) == value; }
    [[nodiscard]] std::string_view baseRelation() const;
};

struct MultiwordToken {
    int first = 0;
    int last = 0;
    std::string form;
};

struct Sentence {
    std::vector<Word> words;
    std::vector<MultiwordToken> multiwords;
    std::string text;

    [[nodiscard]] std::vector<std::vector<int>> children() const;
    [[nodiscard]] int root() const;
    [[nodiscard]] bool isQuestion() const;
    [[nodiscard]] bool subtreeHas(int node, const std::function<bool(const Word&)>& predicate) const;
    // Inserts a word, remapping heads and multiword ranges; returns its index.
    int insert(std::size_t position, Word word);
    // Validates heads (single root, no cycle); throws on malformed trees.
    void validate() const;
};

Features parseFeatures(std::string_view text);
std::string formatFeatures(const Features& feats);

}
