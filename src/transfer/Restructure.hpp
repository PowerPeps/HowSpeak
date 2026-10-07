#pragma once

#include "syntax/Sentence.hpp"
#include "transfer/Grammar.hpp"

namespace howspeak::transfer {

// Inserts the abstract morphemes the target grammar needs but the source left implicit
// ([Q] particle, [3SG] subject, [LOC] adposition, [PL] plural word) and flags words whose
// inflection the target cannot express.
void restructure(syntax::Sentence& sentence, const Grammar& target);

}
