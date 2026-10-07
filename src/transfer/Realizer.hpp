#pragma once

#include "syntax/Sentence.hpp"
#include "transfer/Groups.hpp"
#include "transfer/Rules.hpp"

#include <string>
#include <vector>

namespace howspeak::transfer {

struct Plan {
    std::vector<Group> groups;
    std::vector<std::vector<Decision>> decisions;  // parallel to groups[i].items
};

// Projective linearisation of the tree; dropped words are kept so that callers can inspect them.
[[nodiscard]] std::vector<int> linearize(const syntax::Sentence& sentence, const Plan& plan);
[[nodiscard]] std::vector<Operation> wordOperations(const syntax::Sentence& sentence, const Plan& plan);
// Calque: source words in target structure; affixes joined with '-' and clitics with '='
// following the Leipzig Glossing Rules.
[[nodiscard]] std::string realize(const syntax::Sentence& sentence, const Plan& plan);

}
