#pragma once

#include "syntax/Sentence.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class Hunspell;

namespace howspeak::parse {

// Splits closed compounds (e.g. German "Haustür" -> "Haus" + "Tür") with a Hunspell dictionary.
class CompoundSplitter {
public:
    CompoundSplitter(const std::filesystem::path& aff, const std::filesystem::path& dic);
    ~CompoundSplitter();
    CompoundSplitter(const CompoundSplitter&) = delete;
    CompoundSplitter& operator=(const CompoundSplitter&) = delete;

    // Parts of the compound in surface order, or an empty vector if the word is not split.
    [[nodiscard]] std::vector<std::string> split(std::string_view word) const;

    // Rewrites every NOUN of the sentence into a head noun with `compound` dependents.
    void apply(syntax::Sentence& sentence) const;

private:
    [[nodiscard]] bool spell(const std::u32string& word) const;
    [[nodiscard]] bool isNoun(const std::u32string& part) const;

    std::unique_ptr<Hunspell> hunspell_;
    bool latin1_ = false;
};

}
