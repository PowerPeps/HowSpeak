#pragma once

#include "syntax/Sentence.hpp"

#include <iosfwd>
#include <string_view>
#include <vector>

namespace howspeak::syntax {

std::vector<Sentence> readConllu(std::string_view text);
void writeConllu(std::ostream& out, const Sentence& sentence);

}
