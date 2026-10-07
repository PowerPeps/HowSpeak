#pragma once

#include "syntax/Sentence.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ufal::udpipe {
class model;
}

namespace howspeak::parse {

// Local UDPipe 1 pipeline: tokenisation, tagging, lemmatisation and dependency parsing.
class Parser {
public:
    explicit Parser(const std::filesystem::path& modelPath);
    ~Parser();
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    [[nodiscard]] std::vector<syntax::Sentence> parse(std::string_view text) const;
    // Universal POS of a word tagged in isolation.
    [[nodiscard]] std::string tagWord(std::string_view word) const;

private:
    std::unique_ptr<ufal::udpipe::model> model_;
};

}
