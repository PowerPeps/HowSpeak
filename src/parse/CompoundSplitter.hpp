#pragma once

#include "syntax/Sentence.hpp"

#include <filesystem>
#include <functional>
#include <unordered_map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class Hunspell;

namespace howspeak::parse {

// Splits closed compounds (e.g. German "Haustür" -> "Haus" + "Tür") with a Hunspell dictionary.
class CompoundSplitter {
public:
    using NounOracle = std::function<bool(std::string_view)>;

    // The oracle, when given, confirms that a candidate part is a noun (e.g. a POS tagger).
    CompoundSplitter(const std::filesystem::path& aff, const std::filesystem::path& dic, NounOracle oracle = {});
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
    [[nodiscard]] bool capitalisedEntry(const std::u32string& word) const;

    std::unique_ptr<Hunspell> hunspell_;
    NounOracle oracle_;
    mutable std::unordered_map<std::u32string, bool> nouns_;
    bool latin1_ = false;
};

}
