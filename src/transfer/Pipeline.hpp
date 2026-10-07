#pragma once

#include "nn/Imputer.hpp"
#include "syntax/Sentence.hpp"
#include "transfer/Grammar.hpp"
#include "transfer/Realizer.hpp"
#include "transfer/Reorderer.hpp"
#include "wals/Database.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace howspeak::transfer {

enum class Engine { Neural, Rules };

struct Paths {
    std::filesystem::path root;

    [[nodiscard]] std::filesystem::path wals() const { return root / "data" / "wals"; }
    [[nodiscard]] std::filesystem::path udpipe(std::string_view iso) const;
    [[nodiscard]] std::filesystem::path hunspell(std::string_view iso, std::string_view extension) const;
    [[nodiscard]] std::filesystem::path imputer() const { return root / "models" / "imputer.bin"; }
    [[nodiscard]] std::filesystem::path reorderer() const { return root / "models" / "reorderer.bin"; }
};

struct Translation {
    syntax::Sentence sentence;  // after compound splitting and restructuring
    Plan plan;
    std::string calque;
};

class Pipeline {
public:
    explicit Pipeline(Paths paths);
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    [[nodiscard]] const wals::Database& database() const noexcept { return db_; }
    [[nodiscard]] const TypologyBank& typology() const;
    [[nodiscard]] std::size_t targetLanguage(std::string_view walsCode) const;

    [[nodiscard]] std::vector<syntax::Sentence> parse(std::string_view text, std::string_view sourceWals) const;
    [[nodiscard]] std::vector<Translation> transfer(std::vector<syntax::Sentence> sentences,
                                                    std::string_view sourceWals, std::string_view targetWals,
                                                    Engine engine) const;

private:
    void ensureTypology() const;
    [[nodiscard]] std::string iso(std::string_view walsCode) const;

    Paths paths_;
    wals::Database db_;
    mutable std::unique_ptr<nn::Imputer> imputer_;
    mutable std::unique_ptr<TypologyBank> typology_;
    mutable std::unique_ptr<Reorderer> reorderer_;
};

}
