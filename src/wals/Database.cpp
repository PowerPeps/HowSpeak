#include "wals/Database.hpp"

#include "core/Csv.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <stdexcept>

namespace howspeak::wals {

namespace {

template <class T>
T parseNumber(const std::string& text, std::string_view what) {
    T value{};
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error(std::format("invalid {} '{}'", what, text));
    }
    return value;
}

double parseCoordinate(const std::string& text) {
    return text.empty() ? 0.0 : parseNumber<double>(text, "coordinate");
}

}

Database Database::load(const std::filesystem::path& directory) {
    Database db;

    const auto parameters = core::CsvTable::load(directory / "parameters.csv");
    const auto pId = parameters.column("ID");
    const auto pName = parameters.column("Name");
    for (std::size_t r = 0; r < parameters.rows(); ++r) {
        Feature f;
        f.id = parameters.at(r, pId);
        f.name = parameters.at(r, pName);
        db.featureIndex_.emplace(f.id, db.features_.size());
        db.features_.push_back(std::move(f));
    }

    const auto codes = core::CsvTable::load(directory / "codes.csv");
    const auto cId = codes.column("ID");
    const auto cParam = codes.column("Parameter_ID");
    const auto cName = codes.column("Name");
    const auto cNumber = codes.column("Number");
    for (std::size_t r = 0; r < codes.rows(); ++r) {
        auto& f = db.features_.at(db.feature(codes.at(r, cParam)));
        f.codes.push_back({codes.at(r, cId), codes.at(r, cName), parseNumber<int>(codes.at(r, cNumber), "code number")});
    }
    std::unordered_map<std::string, std::pair<std::size_t, int>> codeIndex;
    for (std::size_t fi = 0; fi < db.features_.size(); ++fi) {
        auto& f = db.features_[fi];
        std::ranges::sort(f.codes, {}, &Code::number);
        f.offset = db.oneHotSize_;
        db.oneHotSize_ += f.codes.size();
        for (std::size_t ci = 0; ci < f.codes.size(); ++ci) {
            codeIndex.emplace(f.codes[ci].id, std::pair{fi, static_cast<int>(ci)});
        }
    }

    const auto languages = core::CsvTable::load(directory / "languages.csv");
    const auto lId = languages.column("ID");
    const auto lName = languages.column("Name");
    const auto lIso = languages.column("ISO639P3code");
    const auto lFamily = languages.column("Family");
    const auto lGenus = languages.column("Genus");
    const auto lArea = languages.column("Macroarea");
    const auto lLat = languages.column("Latitude");
    const auto lLon = languages.column("Longitude");
    const auto l100 = languages.column("Samples_100");
    const auto l200 = languages.column("Samples_200");
    for (std::size_t r = 0; r < languages.rows(); ++r) {
        if (languages.at(r, lArea).empty()) {
            continue;  // family and genus rows of the genealogy
        }
        Language l;
        l.id = languages.at(r, lId);
        l.name = languages.at(r, lName);
        l.iso = languages.at(r, lIso);
        l.family = languages.at(r, lFamily);
        l.genus = languages.at(r, lGenus);
        l.macroarea = languages.at(r, lArea);
        l.latitude = parseCoordinate(languages.at(r, lLat));
        l.longitude = parseCoordinate(languages.at(r, lLon));
        l.sample100 = languages.at(r, l100) == "true";
        l.sample200 = languages.at(r, l200) == "true";
        db.languageIndex_.emplace(l.id, db.languages_.size());
        db.languages_.push_back(std::move(l));
    }

    db.values_.assign(db.languages_.size() * db.features_.size(), kMissing);
    const auto values = core::CsvTable::load(directory / "values.csv");
    const auto vLang = values.column("Language_ID");
    const auto vCode = values.column("Code_ID");
    for (std::size_t r = 0; r < values.rows(); ++r) {
        const auto lang = db.findLanguage(values.at(r, vLang));
        const auto code = codeIndex.find(values.at(r, vCode));
        if (!lang || code == codeIndex.end()) {
            throw std::runtime_error(std::format("values.csv: unresolved row {}", r + 2));
        }
        auto& slot = db.values_[*lang * db.features_.size() + code->second.first];
        if (slot == kMissing) {
            ++db.observed_;
        }
        slot = code->second.second;
    }

    std::vector<Language> kept;
    std::vector<int> keptValues;
    db.languageIndex_.clear();
    for (std::size_t li = 0; li < db.languages_.size(); ++li) {
        const auto row = std::span(db.values_).subspan(li * db.features_.size(), db.features_.size());
        if (std::ranges::all_of(row, [](int v) { return v == kMissing; })) {
            continue;
        }
        db.languageIndex_.emplace(db.languages_[li].id, kept.size());
        kept.push_back(std::move(db.languages_[li]));
        keptValues.insert(keptValues.end(), row.begin(), row.end());
    }
    db.languages_ = std::move(kept);
    db.values_ = std::move(keptValues);
    return db;
}

std::optional<std::size_t> Database::findFeature(std::string_view id) const {
    const auto it = featureIndex_.find(std::string(id));
    return it == featureIndex_.end() ? std::nullopt : std::optional(it->second);
}

std::optional<std::size_t> Database::findLanguage(std::string_view walsCode) const {
    const auto it = languageIndex_.find(std::string(walsCode));
    return it == languageIndex_.end() ? std::nullopt : std::optional(it->second);
}

std::size_t Database::feature(std::string_view id) const {
    if (const auto f = findFeature(id)) {
        return *f;
    }
    throw std::runtime_error(std::format("unknown WALS feature '{}'", id));
}

std::size_t Database::language(std::string_view walsCode) const {
    if (const auto l = findLanguage(walsCode)) {
        return *l;
    }
    throw std::runtime_error(std::format("unknown WALS language '{}'", walsCode));
}

}
