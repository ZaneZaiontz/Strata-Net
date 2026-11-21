// include/strata/net/stage_split_facts.hpp - the facts `strata --plan-split FACTS` plans from, and its JSON answer.
#pragma once

#include "strata/net/stage_split.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::pool {

/// What the split planner needs about a model and its PCs, as read from a FACTS file (docs/REMOTE_STAGE.md).
struct SplitFacts {
    SplitModel model;
    std::vector<NodeCap> nodes;   ///< the head PC first
    std::string profile;          ///< the expert profile file; the caller reads it into model.profile (empty: none)
    int64_t n_expert = 0;         ///< experts per layer, which the profile file is checked against
};

/// Parses FACTS text: `key=value` lines about the model and one `node=NAME key=value ...` line per PC, head first.
/// False with `err` on an unknown key, a missing required fact, a repeated one or a value out of range.
bool parse_split_facts(const std::string& text, SplitFacts& f, std::string& err);

/// parse_split_facts on the content of the file at `path`.
bool read_split_facts(const std::string& path, SplitFacts& f, std::string& err);

/// The plan as one line of JSON: the PCs it uses in chain order with their layers, the estimate, and the inputs.
std::string split_plan_json(const SplitFacts& f, const SplitPlan& p);

}  // namespace strata::pool
