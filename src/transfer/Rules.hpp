#pragma once

#include "syntax/Sentence.hpp"
#include "transfer/Grammar.hpp"
#include "transfer/Groups.hpp"

#include <vector>

namespace howspeak::transfer {

struct Decision {
    double key = 0.0;  // target position relative to the head (head = 0)
    Operation op = Operation::Keep;
};

// Deterministic transfer rules: WALS values, plus Dryer's (1992) word-order correlations where WALS is silent.
class Rules {
public:
    explicit Rules(const Grammar& grammar) : g_(grammar) {}

    [[nodiscard]] std::vector<Decision> decide(const Group& group) const;

private:
    struct Slots {
        double subject = 0.0;
        double object = 0.0;
        double oblique = 0.0;
        bool subjectKnown = false;
        bool objectKnown = false;
        bool obliqueKnown = false;
    };

    [[nodiscard]] Slots clauseSlots() const;
    [[nodiscard]] Decision dependent(const Group& group, const Item& item, const Slots& slots) const;
    [[nodiscard]] Decision negation(const Group& group, const Item& item, const Slots& slots) const;
    [[nodiscard]] bool objectBeforeVerb() const;
    [[nodiscard]] bool objectAfterVerb() const;
    [[nodiscard]] Operation affix(bool prefix) const;

    Grammar g_;
};

// Orders the group members: returns indices into group.items sorted by target position.
[[nodiscard]] std::vector<std::size_t> orderGroup(const Group& group, const std::vector<Decision>& decisions);

}
