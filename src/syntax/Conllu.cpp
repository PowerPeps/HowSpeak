#include "syntax/Conllu.hpp"

#include <charconv>
#include <format>
#include <ostream>
#include <stdexcept>

namespace howspeak::syntax {

namespace {

std::vector<std::string_view> splitTabs(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (true) {
        const auto tab = line.find('\t', start);
        out.push_back(line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start));
        if (tab == std::string_view::npos) {
            return out;
        }
        start = tab + 1;
    }
}

int toInt(std::string_view s, std::size_t line) {
    int v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        throw std::runtime_error(std::format("CoNLL-U line {}: invalid number '{}'", line, s));
    }
    return v;
}

std::string field(std::string_view s) {
    return s == "_" ? std::string{} : std::string(s);
}

}

std::vector<Sentence> readConllu(std::string_view text) {
    std::vector<Sentence> out;
    Sentence current;
    std::size_t lineNo = 0;
    const auto flush = [&] {
        if (!current.words.empty()) {
            current.validate();
            out.push_back(std::move(current));
        }
        current = Sentence{};
    };

    std::size_t start = 0;
    while (start <= text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        auto line = text.substr(start, end - start);
        start = end + 1;
        ++lineNo;
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            flush();
            continue;
        }
        if (line.front() == '#') {
            if (line.starts_with("# text = ")) {
                current.text = std::string(line.substr(9));
            }
            continue;
        }
        const auto cols = splitTabs(line);
        if (cols.size() != 10) {
            throw std::runtime_error(std::format("CoNLL-U line {}: expected 10 columns", lineNo));
        }
        if (cols[0].find('.') != std::string_view::npos) {
            continue;  // empty nodes are not used
        }
        if (const auto dash = cols[0].find('-'); dash != std::string_view::npos) {
            current.multiwords.push_back({toInt(cols[0].substr(0, dash), lineNo) - 1,
                                          toInt(cols[0].substr(dash + 1), lineNo) - 1, std::string(cols[1])});
            continue;
        }
        Word w;
        const int id = toInt(cols[0], lineNo);
        if (id != static_cast<int>(current.words.size()) + 1) {
            throw std::runtime_error(std::format("CoNLL-U line {}: non-sequential word id", lineNo));
        }
        w.form = std::string(cols[1]);
        w.lemma = field(cols[2]);
        w.upos = field(cols[3]);
        w.feats = parseFeatures(cols[5]);
        w.head = toInt(cols[6], lineNo) - 1;
        w.deprel = field(cols[7]);
        w.order = static_cast<double>(id - 1);
        w.spaceAfter = cols[9].find("SpaceAfter=No") == std::string_view::npos;
        current.words.push_back(std::move(w));
    }
    flush();
    return out;
}

void writeConllu(std::ostream& out, const Sentence& sentence) {
    if (!sentence.text.empty()) {
        out << "# text = " << sentence.text << '\n';
    }
    for (std::size_t i = 0; i < sentence.words.size(); ++i) {
        const auto& w = sentence.words[i];
        for (const auto& m : sentence.multiwords) {
            if (m.first == static_cast<int>(i)) {
                out << std::format("{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n", m.first + 1, m.last + 1, m.form);
            }
        }
        out << std::format("{}\t{}\t{}\t{}\t_\t{}\t{}\t{}\t_\t{}\n", i + 1, w.form, w.lemma.empty() ? "_" : w.lemma,
                           w.upos.empty() ? "_" : w.upos, formatFeatures(w.feats), w.head + 1,
                           w.deprel.empty() ? "_" : w.deprel, w.spaceAfter ? "_" : "SpaceAfter=No");
    }
    out << '\n';
}

}
