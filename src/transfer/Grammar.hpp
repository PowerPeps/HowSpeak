#pragma once

#include "nn/Imputer.hpp"
#include "wals/Database.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string_view>
#include <vector>

namespace howspeak::transfer {

// Enumerators follow the WALS code numbers; Unknown (0) means no value.
enum class Affixation { Unknown, Little, StrongSuffix, WeakSuffix, Equal, WeakPrefix, StrongPrefix };
enum class Plural { Unknown, Prefix, Suffix, StemChange, Tone, Reduplication, Mixed, Word, Clitic, None };
enum class Definite { Unknown, Word, Demonstrative, Affix, OnlyIndefinite, None };
enum class Indefinite { Unknown, Word, One, Affix, OnlyDefinite, None };
enum class CaseCount { Unknown, None, Two, Three, Four, Five, SixSeven, EightNine, TenPlus, Borderline };
enum class CaseAffix { Unknown, Suffix, Prefix, Tone, StemChange, Mixed, PostClitic, PreClitic, InClitic, None };
enum class TenseAffix { Unknown, Prefix, Suffix, Tone, Mixed, None };
enum class ClauseOrder { Unknown, SOV, SVO, VSO, VOS, OVS, OSV, None };
enum class SubjectVerb { Unknown, SV, VS, None };
enum class ObjectVerb { Unknown, OV, VO, None };
enum class ObliqueOrder { Unknown, VOX, XVO, XOV, OXV, OVX, None };
enum class Adposition { Unknown, Post, Pre, In, None, Absent };
enum class Genitive { Unknown, GenN, NGen, None };
enum class Adjective { Unknown, AdjN, NAdj, None, InternallyHeaded };
enum class Demonstrative { Unknown, DemN, NDem, Prefix, Suffix, BeforeAndAfter, Mixed };
enum class Numeral { Unknown, NumN, NNum, None, VerbOnly };
enum class Relative { Unknown, NRel, RelN, InternallyHeaded, Correlative, Adjoined, DoublyHeaded, Mixed };
enum class Degree { Unknown, DegAdj, AdjDeg, None };
enum class QuestionParticle { Unknown, Initial, Final, Second, Other, Either, None };
enum class WhPosition { Unknown, Initial, NotInitial, Mixed };
enum class Subordinator { Unknown, Initial, Final, Internal, Suffix, Mixed };
enum class SubjectPronoun { Unknown, Obligatory, VerbAffix, Clitic, OtherPosition, Optional, Mixed };
enum class PolarQuestion { Unknown, Particle, VerbMorphology, Mixture, WordOrder, NoDeclarative, Intonation, NoDistinction };
enum class PredicativeAdjective { Unknown, Verbal, Nonverbal, Mixed };
enum class ZeroCopula { Unknown, Impossible, Possible };
enum class Relativization { Unknown, RelativePronoun, NonReduction, PronounRetention, Gap };
enum class NegationVerb {
    Unknown, NegV, VNeg, NegPrefix, NegSuffix, Tone, T12, T13, T14, T23, T24, T34, T3Infix,
    OptSingle, ObligDouble, OptDouble, OptTripleObligDouble, OptTripleOptDouble
};
enum class NegationClause {
    Unknown, NegSVO, SNegVO, SVNegO, SVONeg, NegSOV, SNegOV, SONegV, SOVNeg, NegVSO, VSNegO, VSONeg, NegVOS,
    ONegVS, OVNegS, OSVNeg, MoreThanOne, OptSingle, ObligDouble, OptDouble, Morphological, Other
};

struct FeatureSpec {
    std::string_view id;
    std::size_t codes;
};

inline constexpr std::array<FeatureSpec, 28> kFeatures{{
    {"26A", 6}, {"33A", 9}, {"37A", 5}, {"38A", 5}, {"49A", 9}, {"51A", 9}, {"69A", 5},
    {"81A", 7}, {"82A", 3}, {"83A", 3}, {"84A", 6}, {"85A", 5}, {"86A", 3}, {"87A", 4},
    {"88A", 6}, {"89A", 4}, {"90A", 7}, {"91A", 3}, {"92A", 6}, {"93A", 3}, {"94A", 5},
    {"101A", 6}, {"116A", 7}, {"118A", 3}, {"120A", 2}, {"122A", 4}, {"143A", 17}, {"144A", 21},
}};
inline constexpr std::size_t kFeatureCount = kFeatures.size();

using CodeVector = std::array<int, kFeatureCount>;  // WALS code numbers, 0 = unknown

struct Grammar {
    Affixation affixation{};
    Plural plural{};
    Definite definite{};
    Indefinite indefinite{};
    CaseCount caseCount{};
    CaseAffix caseAffix{};
    TenseAffix tenseAffix{};
    ClauseOrder clause{};
    SubjectVerb subjectVerb{};
    ObjectVerb objectVerb{};
    ObliqueOrder oblique{};
    Adposition adposition{};
    Genitive genitive{};
    Adjective adjective{};
    Demonstrative demonstrative{};
    Numeral numeral{};
    Relative relative{};
    Degree degree{};
    QuestionParticle questionParticle{};
    WhPosition wh{};
    Subordinator subordinator{};
    SubjectPronoun subjectPronoun{};
    PolarQuestion polar{};
    PredicativeAdjective predicativeAdjective{};
    ZeroCopula zeroCopula{};
    Relativization relativization{};
    NegationVerb negation{};
    NegationClause negationClause{};

    static Grammar fromCodes(const CodeVector& codes);
};

// Probability over the codes of each feature: one-hot when attested, imputed otherwise.
struct Profile {
    std::array<std::vector<float>, kFeatureCount> distribution;
    std::array<bool, kFeatureCount> observed{};

    [[nodiscard]] Grammar mostLikely() const;
    [[nodiscard]] Grammar sample(std::mt19937_64& rng) const;
    void appendTo(std::vector<float>& out) const;
    [[nodiscard]] static std::size_t width();
};

class TypologyBank {
public:
    TypologyBank(const wals::Database& db, const nn::Imputer& imputer);

    [[nodiscard]] Profile profile(std::size_t language) const;
    // A typology outside the atlas: each feature drawn independently from its world-wide distribution.
    [[nodiscard]] Profile randomProfile(std::mt19937_64& rng) const;
    [[nodiscard]] const std::vector<std::size_t>& sample200() const noexcept { return sample200_; }
    [[nodiscard]] const wals::Database& database() const noexcept { return *db_; }

private:
    const wals::Database* db_;
    std::array<std::size_t, kFeatureCount> walsIndex_{};
    std::array<std::vector<float>, kFeatureCount> marginal_;
    std::vector<Profile> profiles_;
    std::vector<std::size_t> sample200_;
};

}
