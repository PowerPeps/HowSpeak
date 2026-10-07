#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace howspeak::wals {

struct Code {
    std::string id;
    std::string name;
    int number = 0;
};

struct Feature {
    std::string id;
    std::string name;
    std::vector<Code> codes;
    std::size_t offset = 0;  // first column of this feature in the one-hot layout
};

struct Language {
    std::string id;
    std::string name;
    std::string iso;
    std::string family;
    std::string genus;
    std::string macroarea;
    double latitude = 0.0;
    double longitude = 0.0;
    bool sample100 = false;
    bool sample200 = false;
};

inline constexpr int kMissing = -1;

class Database {
public:
    static Database load(const std::filesystem::path& directory);

    [[nodiscard]] const std::vector<Feature>& features() const noexcept { return features_; }
    [[nodiscard]] const std::vector<Language>& languages() const noexcept { return languages_; }
    [[nodiscard]] std::size_t oneHotSize() const noexcept { return oneHotSize_; }

    [[nodiscard]] std::optional<std::size_t> findFeature(std::string_view id) const;
    [[nodiscard]] std::optional<std::size_t> findLanguage(std::string_view walsCode) const;
    [[nodiscard]] std::size_t feature(std::string_view id) const;
    [[nodiscard]] std::size_t language(std::string_view walsCode) const;

    // Index of the code inside Feature::codes, or kMissing.
    [[nodiscard]] int value(std::size_t language, std::size_t feature) const noexcept {
        return values_[language * features_.size() + feature];
    }
    [[nodiscard]] std::size_t observedCount() const noexcept { return observed_; }

private:
    std::vector<Feature> features_;
    std::vector<Language> languages_;
    std::vector<int> values_;
    std::unordered_map<std::string, std::size_t> featureIndex_;
    std::unordered_map<std::string, std::size_t> languageIndex_;
    std::size_t oneHotSize_ = 0;
    std::size_t observed_ = 0;
};

}
