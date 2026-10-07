#pragma once

#include "syntax/Sentence.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace howspeak::transfer {

enum class Operation : std::uint8_t { Keep, Drop, PrefixAffix, SuffixAffix, PrefixClitic, SuffixClitic };
inline constexpr std::size_t kOperationCount = 6;

// Structural facts about one member of a head-dependents group, shared by the rules and the network.
struct Item {
    int word = -1;
    bool isHead = false;
    std::string_view relation;      // full deprel ("HEAD" for the group head)
    std::string_view baseRelation;  // deprel before ':'
    std::string_view upos;
    int side = 0;                   // -1 left of the head in the source, +1 right, 0 head
    int proximity = 0;              // group members between the item and the head in the source
    double rank = 0.0;              // normalised source position in the group
    bool article = false;
    bool definite = false;
    bool indefinite = false;
    bool demonstrative = false;
    bool possessive = false;
    bool interrogative = false;  // the item's subtree contains an interrogative word
    bool relativePronoun = false;
    bool personalPronoun = false;
    bool negative = false;
    bool lastNegative = false;
    bool marker = false;
};

struct Group {
    int head = -1;
    std::string_view headUpos;
    std::string_view headRelation;
    bool headNominal = false;
    bool headAdjectival = false;
    bool polarQuestion = false;
    int negatives = 0;
    std::vector<Item> items;  // source order, head included
};

[[nodiscard]] std::vector<Group> buildGroups(const syntax::Sentence& sentence);
[[nodiscard]] bool isPolarQuestion(const syntax::Sentence& sentence);
[[nodiscard]] bool isNegative(const syntax::Word& word);

}
