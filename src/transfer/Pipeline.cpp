#include "transfer/Pipeline.hpp"

#include "parse/CompoundSplitter.hpp"
#include "parse/Parser.hpp"
#include "transfer/Groups.hpp"
#include "transfer/Restructure.hpp"
#include "transfer/Rules.hpp"

#include <format>
#include <stdexcept>

namespace howspeak::transfer {

std::filesystem::path Paths::udpipe(std::string_view iso) const {
    return root / "data" / "udpipe" / std::format("{}.udpipe", iso);
}

std::filesystem::path Paths::hunspell(std::string_view iso, std::string_view extension) const {
    return root / "data" / "hunspell" / std::format("{}.{}", iso, extension);
}

Pipeline::Pipeline(Paths paths) : paths_(std::move(paths)), db_(wals::Database::load(paths_.wals())) {}

Pipeline::~Pipeline() = default;

void Pipeline::ensureTypology() const {
    if (typology_) {
        return;
    }
    if (!std::filesystem::exists(paths_.imputer())) {
        throw std::runtime_error(std::format("missing '{}': run `HowSpeak train-imputer` first", paths_.imputer().string()));
    }
    imputer_ = std::make_unique<nn::Imputer>(db_);
    imputer_->load(paths_.imputer());
    typology_ = std::make_unique<TypologyBank>(db_, *imputer_);
}

const TypologyBank& Pipeline::typology() const {
    ensureTypology();
    return *typology_;
}

std::string Pipeline::iso(std::string_view walsCode) const {
    const auto& l = db_.languages()[db_.language(walsCode)];
    if (l.iso.empty()) {
        throw std::runtime_error(std::format("WALS language '{}' has no ISO 639-3 code", walsCode));
    }
    return l.iso;
}

std::size_t Pipeline::targetLanguage(std::string_view walsCode) const {
    const auto index = db_.language(walsCode);
    if (!db_.languages()[index].sample200) {
        throw std::runtime_error(
            std::format("'{}' is outside the WALS 200-language sample (see `HowSpeak languages`)", walsCode));
    }
    return index;
}

std::vector<syntax::Sentence> Pipeline::parse(std::string_view text, std::string_view sourceWals) const {
    const auto model = paths_.udpipe(iso(sourceWals));
    if (!std::filesystem::exists(model)) {
        throw std::runtime_error(std::format(
            "no parser for '{}': expected '{}' (cmake -DLANGS={} -P scripts/fetch_resources.cmake)", sourceWals,
            model.string(), iso(sourceWals)));
    }
    const parse::Parser parser(model);
    return parser.parse(text);
}

std::vector<Translation> Pipeline::transfer(std::vector<syntax::Sentence> sentences, std::string_view sourceWals,
                                            std::string_view targetWals, Engine engine) const {
    ensureTypology();
    const Profile target = typology_->profile(targetLanguage(targetWals));
    const Grammar grammar = target.mostLikely();

    std::unique_ptr<parse::CompoundSplitter> splitter;
    const auto sourceIso = iso(sourceWals);
    if (std::filesystem::exists(paths_.hunspell(sourceIso, "dic"))) {
        splitter = std::make_unique<parse::CompoundSplitter>(paths_.hunspell(sourceIso, "aff"),
                                                             paths_.hunspell(sourceIso, "dic"));
    }
    if (engine == Engine::Neural && !reorderer_) {
        if (!std::filesystem::exists(paths_.reorderer())) {
            throw std::runtime_error(
                std::format("missing '{}': run `HowSpeak train-reorderer` first", paths_.reorderer().string()));
        }
        reorderer_ = std::make_unique<Reorderer>();
        reorderer_->load(paths_.reorderer());
    }

    std::vector<Translation> out;
    out.reserve(sentences.size());
    for (auto& sentence : sentences) {
        Translation t;
        t.sentence = std::move(sentence);
        if (splitter) {
            splitter->apply(t.sentence);
        }
        restructure(t.sentence, grammar);
        t.sentence.validate();
        t.plan.groups = buildGroups(t.sentence);
        if (engine == Engine::Neural) {
            t.plan.decisions = reorderer_->decide(t.plan.groups, target);
        } else {
            const Rules rules(grammar);
            for (const auto& g : t.plan.groups) {
                t.plan.decisions.push_back(rules.decide(g));
            }
        }
        t.calque = realize(t.sentence, t.plan);
        out.push_back(std::move(t));
    }
    return out;
}

}
