#include "parse/Parser.hpp"

#include <udpipe.h>

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

namespace howspeak::parse {

Parser::Parser(const std::filesystem::path& modelPath) {
    std::ifstream in(modelPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::format("cannot open UDPipe model '{}'", modelPath.string()));
    }
    model_.reset(ufal::udpipe::model::load(in));
    if (!model_) {
        throw std::runtime_error(std::format("invalid UDPipe model '{}'", modelPath.string()));
    }
}

Parser::~Parser() = default;

std::vector<syntax::Sentence> Parser::parse(std::string_view text) const {
    std::unique_ptr<ufal::udpipe::input_format> tokenizer(model_->new_tokenizer(ufal::udpipe::model::DEFAULT));
    if (!tokenizer) {
        throw std::runtime_error("the UDPipe model has no tokenizer");
    }
    const std::string owned(text);
    tokenizer->set_text(owned);

    std::vector<syntax::Sentence> out;
    ufal::udpipe::sentence s;
    std::string error;
    while (tokenizer->next_sentence(s, error)) {
        if (!model_->tag(s, ufal::udpipe::model::DEFAULT, error) ||
            !model_->parse(s, ufal::udpipe::model::DEFAULT, error)) {
            throw std::runtime_error(std::format("UDPipe: {}", error));
        }
        syntax::Sentence sentence;
        for (std::size_t i = 1; i < s.words.size(); ++i) {
            const auto& w = s.words[i];
            syntax::Word word;
            word.form = w.form;
            word.lemma = w.lemma;
            word.upos = w.upostag;
            word.feats = syntax::parseFeatures(w.feats);
            word.head = w.head - 1;
            word.deprel = w.deprel;
            word.order = static_cast<double>(i - 1);
            word.spaceAfter = w.get_space_after();
            sentence.words.push_back(std::move(word));
        }
        if (!s.get_text(sentence.text)) {
            for (const auto& w : sentence.words) {
                sentence.text += w.form + (w.spaceAfter ? " " : "");
            }
        }
        for (const auto& m : s.multiword_tokens) {
            sentence.multiwords.push_back({m.id_first - 1, m.id_last - 1, m.form});
            if (!m.get_space_after()) {
                sentence.words[static_cast<std::size_t>(m.id_last - 1)].spaceAfter = false;
            }
        }
        sentence.validate();
        out.push_back(std::move(sentence));
    }
    if (!error.empty()) {
        throw std::runtime_error(std::format("UDPipe: {}", error));
    }
    return out;
}

std::string Parser::tagWord(std::string_view word) const {
    ufal::udpipe::sentence s;
    s.add_word(std::string(word));
    std::string error;
    if (!model_->tag(s, ufal::udpipe::model::DEFAULT, error)) {
        throw std::runtime_error(std::format("UDPipe: {}", error));
    }
    return s.words.at(1).upostag;
}

}
