#include "core/Csv.hpp"
#include "core/Resources.hpp"
#include "nn/Imputer.hpp"
#include "syntax/Conllu.hpp"
#include "transfer/Pipeline.hpp"
#include "transfer/Reorderer.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <format>
#include <iostream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using namespace howspeak;

std::string readStdin() {
    return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}

void printProfile(const transfer::Pipeline& pipeline, std::string_view code) {
    const auto& db = pipeline.database();
    const auto lang = db.language(code);
    const auto profile = pipeline.typology().profile(lang);
    const auto& l = db.languages()[lang];
    std::println("{} ({}) — {}, {}{}", l.name, l.id, l.family, l.genus, l.sample200 ? ", WALS 200 sample" : "");
    for (std::size_t f = 0; f < transfer::kFeatureCount; ++f) {
        const auto& feature = db.features()[db.feature(transfer::kFeatures[f].id)];
        const auto& d = profile.distribution[f];
        const auto best = static_cast<std::size_t>(std::ranges::max_element(d) - d.begin());
        std::println("  {:>5} {:<52} {:<40} {}", feature.id, feature.name.substr(0, 52),
                     feature.codes[best].name.substr(0, 40),
                     profile.observed[f] ? std::string("attested") : std::format("imputed p={:.2f}", d[best]));
    }
}

}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    CLI::App app{"HowSpeak — structural calque between languages from WALS typology"};
    argv = app.ensure_utf8(argv);
    app.require_subcommand(1);

    std::string root = core::projectRoot().string();
    app.add_option("--root", root, "Project directory holding data/ and models/")->capture_default_str();

    auto* imputerCmd = app.add_subcommand("train-imputer", "Train the WALS feature imputer (denoising autoencoder)");
    nn::Imputer::Options imputerOptions;
    imputerCmd->add_option("--epochs", imputerOptions.epochs)->capture_default_str()->check(CLI::PositiveNumber);
    imputerCmd->add_option("--holdout", imputerOptions.holdout, "Share of attested values held out for evaluation")
        ->capture_default_str()
        ->check(CLI::Range(0.0F, 0.5F));

    auto* reordererCmd = app.add_subcommand("train-reorderer", "Train the neural transfer model on synthetic trees");
    transfer::Reorderer::Options reordererOptions;
    reordererCmd->add_option("--steps", reordererOptions.steps)->capture_default_str()->check(CLI::PositiveNumber);
    reordererCmd->add_option("--batch", reordererOptions.batch, "Sentences per step")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    reordererCmd->add_option("--unseen", reordererOptions.unseenTypologies,
                             "Share of targets drawn as random typologies outside the atlas")
        ->capture_default_str()
        ->check(CLI::Range(0.0, 1.0));
    reordererCmd->add_option("--evaluation", reordererOptions.evaluation, "Sentences per typology kind in the final report")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    reordererCmd->add_option("--checkpoint", reordererOptions.checkpointEvery, "Steps between checkpoints")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);

    auto* translateCmd = app.add_subcommand("translate", "Rewrite sentences in the structure of another language");
    std::string from;
    std::string to;
    std::string engineName = "neural";
    std::string conllu;
    bool trace = false;
    std::vector<std::string> words;
    translateCmd->add_option("--from,-f", from, "Source language (WALS code, e.g. fre)")->required();
    translateCmd->add_option("--to,-t", to, "Target language (WALS code, e.g. jpn)")->required();
    translateCmd->add_option("--engine,-e", engineName)
        ->capture_default_str()
        ->check(CLI::IsMember({"neural", "rules"}));
    translateCmd->add_option("--conllu", conllu, "Read pre-parsed CoNLL-U instead of running UDPipe")
        ->check(CLI::ExistingFile);
    translateCmd->add_flag("--trace", trace, "Print the restructured dependency tree");
    translateCmd->add_option("text", words, "Text to rewrite (stdin when omitted)");

    auto* profileCmd = app.add_subcommand("profile", "Show the WALS profile used for a language");
    std::string profileLang;
    profileCmd->add_option("language", profileLang, "WALS code")->required();

    auto* languagesCmd = app.add_subcommand("languages", "List target languages (WALS 200-language sample)");

    CLI11_PARSE(app, argc, argv);

    try {
        const transfer::Paths paths{root};
        if (*imputerCmd) {
            const auto db = wals::Database::load(paths.wals());
            std::println("WALS: {} languages, {} features, {} attested values", db.languages().size(),
                         db.features().size(), db.observedCount());
            nn::Imputer imputer(db);
            const auto report = imputer.train(imputerOptions, std::cout);
            if (report.heldout > 0) {
                std::println("held-out accuracy {:.3f} | genus majority {:.3f} | global majority {:.3f} (n={})",
                             report.accuracy, report.genusMajority, report.globalMajority, report.heldout);
            }
            imputer.save(paths.imputer());
            std::println("saved {}", paths.imputer().string());
        } else if (*reordererCmd) {
            const transfer::Pipeline pipeline(paths);
            transfer::Reorderer reorderer;
            const auto report = reorderer.train(pipeline.typology(), reordererOptions, std::cout);
            const auto show = [](std::string_view label, const transfer::Reorderer::Metrics& m) {
                std::println("{:<28} pairs {:.3f} | groups {:.3f} | operations {:.3f} | sentences {:.3f}", label,
                             m.pairs, m.groups, m.operations, m.sentences);
            };
            show("agreement, WALS sample:", report.atlas);
            show("agreement, unseen typology:", report.unseen);
            for (std::size_t i = 0; i < std::min<std::size_t>(report.errors.size(), 15); ++i) {
                std::println("  {:6d}  {}", report.errors[i].second, report.errors[i].first);
            }
            reorderer.save(paths.reorderer());
            std::println("saved {}", paths.reorderer().string());
        } else if (*translateCmd) {
            const transfer::Pipeline pipeline(paths);
            std::vector<syntax::Sentence> sentences;
            if (!conllu.empty()) {
                sentences = syntax::readConllu(core::readFile(conllu));
            } else {
                std::string text;
                for (const auto& w : words) {
                    text += (text.empty() ? "" : " ") + w;
                }
                sentences = pipeline.parse(text.empty() ? readStdin() : text, from);
            }
            const auto engine = engineName == "rules" ? transfer::Engine::Rules : transfer::Engine::Neural;
            for (const auto& t : pipeline.transfer(std::move(sentences), from, to, engine)) {
                if (trace) {
                    syntax::writeConllu(std::cout, t.sentence);
                }
                std::println("{}", t.calque);
            }
        } else if (*profileCmd) {
            const transfer::Pipeline pipeline(paths);
            printProfile(pipeline, profileLang);
        } else if (*languagesCmd) {
            const transfer::Pipeline pipeline(paths);
            const transfer::Paths p{root};
            for (const auto& l : pipeline.database().languages()) {
                if (l.sample200) {
                    const bool parser = !l.iso.empty() && std::filesystem::exists(p.udpipe(l.iso));
                    std::println("{:<5} {:<4} {:<32} {}", l.id, l.iso, l.name, parser ? "parser" : "");
                }
            }
        }
    } catch (const std::exception& e) {
        std::println(stderr, "error: {}", e.what());
        return 1;
    }
    return 0;
}
