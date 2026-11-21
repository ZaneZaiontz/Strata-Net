// src/net/stage_split_facts.cpp - reads the planner's FACTS and writes its answer (see stage_split_facts.hpp).
#include "strata/net/stage_split_facts.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace strata::pool {

static constexpr int64_t kMaxLayers = 4096;
static constexpr size_t kMaxNodes = 8;

/// A whole word as a finite number of at least 0.
static bool to_real(const std::string& s, double& v) {
    char* end = nullptr;
    v = std::strtod(s.c_str(), &end);
    return !s.empty() && end == s.c_str() + s.size() && std::isfinite(v) && v >= 0;
}

/// A whole word as a whole number of at least 0.
static bool to_int(const std::string& s, int64_t& v) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 18) return false;
    v = std::atoll(s.c_str());
    return true;
}

/// A per-layer fact: one value for every layer, or exactly `n` comma-separated values.
static bool to_layers(const std::string& s, int64_t n, std::vector<int64_t>& out) {
    out.clear();
    std::istringstream in(s);
    std::string item;
    int64_t v = 0;
    while (std::getline(in, item, ',')) {
        if (!to_int(item, v)) return false;
        out.push_back(v);
    }
    if (out.size() == 1) out.assign((size_t) n, v);
    return (int64_t) out.size() == n;
}

/// A node name: letters, digits, '.', '_' and '-' (no spaces, '=', ':' or ',', which the format uses).
static bool good_name(const std::string& s) {
    if (s.empty()) return false;
    for (const char c : s)
        if (!std::isalnum((unsigned char) c) && c != '.' && c != '_' && c != '-') return false;
    return true;
}

/// Every word of a calib line is one of SplitCalib::encode's keys with a value.
static bool known_calib_words(const std::string& s) {
    const std::set<std::string> keys = {"windows", "fixed_ms", "net_ms", "scale", "seen", "failed"};
    std::istringstream words(s);
    std::string word;
    while (words >> word)
        if (word.find('=') == std::string::npos || !keys.count(word.substr(0, word.find('=')))) return false;
    return true;
}

bool parse_split_facts(const std::string& text, SplitFacts& f, std::string& err) {
    f = SplitFacts();
    SplitModel& m = f.model;
    m.n_layers = 0;
    int64_t ring = 0, session_base = 0;
    const std::map<std::string, int64_t*> model_ints = {
        {"layers", &m.n_layers},           {"first_min", &m.first_min},         {"kv_copies", &m.kv_copies},
        {"window_rows", &m.window_rows},   {"row_bytes", &m.row_bytes},         {"ring", &ring},
        {"prompt_row_bytes", &m.prompt_row_bytes}, {"prompt_chunk", &m.prompt_chunk},
        {"drafter_rows", &m.drafter_rows}, {"prompt_tokens", &m.prompt_tokens}, {"answer_tokens", &m.answer_tokens},
        {"experts", &f.n_expert},          {"session_base_bytes", &session_base}};
    const std::map<std::string, double*> model_reals = {{"miss_ms", &m.miss_ms},
                                                        {"hop_ms", &m.hop_ms},
                                                        {"hop_overhead_ms", &m.hop_overhead_ms},
                                                        {"window_tokens", &m.window_tokens}};
    const std::set<std::string> layer_keys = {"slot_bytes", "layer_arena_bytes", "session_layer_bytes", "layer_kv"};
    std::map<std::string, std::string> layer_facts, delays;
    std::set<std::string> seen;
    std::istringstream in(text);
    std::string line;
    for (int no = 1; std::getline(in, line); ++no) {
        const auto fail = [&](const std::string& why) {
            err = "line " + std::to_string(no) + ": " + why;
            return false;
        };
        if (!line.empty() && line[0] == '#') continue;
        if (line.rfind("calib ", 0) == 0) {
            if (!f.calib.scale.empty()) return fail("calib is given twice");
            if (!known_calib_words(line.substr(6)) || !f.calib.decode(line.substr(6)))
                return fail("calib is SplitCalib's text: windows=N fixed_ms=MS net_ms=MS scale=S,S,... seen=K;...");
            continue;
        }
        std::istringstream words(line);
        std::string word;
        NodeCap* node = nullptr;
        std::set<std::string> node_seen;
        for (bool first = true; words >> word; first = false) {
            const size_t eq = word.find('=');
            if (eq == std::string::npos) return fail("\"" + word + "\" is not key=value");
            const std::string k = word.substr(0, eq), v = word.substr(eq + 1);
            if (first && k == "node") {
                if (!good_name(v)) return fail("a node name is letters, digits, '.', '_' and '-'");
                for (const NodeCap& n : f.nodes)
                    if (n.name == v) return fail("node " + v + " is given twice");
                if (f.nodes.size() == kMaxNodes) return fail("at most " + std::to_string(kMaxNodes) + " nodes");
                f.nodes.emplace_back();
                node = &f.nodes.back();
                node->name = v;
                continue;
            }
            if (node != nullptr) {
                const std::map<std::string, int64_t*> ints = {{"vram_room", &node->vram_room},
                                                              {"ram_room", &node->ram_room},
                                                              {"kv_host_layer", &node->kv_host_layer},
                                                              {"pin_room", &node->pin_room}};
                const std::map<std::string, double*> reals = {{"layer_ms", &node->layer_ms},
                                                              {"up_mbit", &node->up_mbit},
                                                              {"down_mbit", &node->down_mbit},
                                                              {"pf_layer_ms", &node->pf_layer_ms}};
                if (!node_seen.insert(k).second) return fail(k + " is given twice for node " + node->name);
                const auto i = ints.find(k);
                const auto r = reals.find(k);
                if (i != ints.end()) {
                    if (!to_int(v, *i->second)) return fail(k + " must be a whole number of at least 0");
                } else if (r != reals.end()) {
                    if (!to_real(v, *r->second)) return fail(k + " must be a number of at least 0");
                } else if (k == "delay") {
                    delays[node->name] = v;
                } else {
                    return fail("unknown node fact \"" + k + "\"");
                }
                continue;
            }
            if (!seen.insert(k).second) return fail(k + " is given twice");
            const auto i = model_ints.find(k);
            const auto r = model_reals.find(k);
            if (i != model_ints.end()) {
                if (!to_int(v, *i->second)) return fail(k + " must be a whole number of at least 0");
            } else if (r != model_reals.end()) {
                if (!to_real(v, *r->second)) return fail(k + " must be a number of at least 0");
            } else if (layer_keys.count(k)) {
                layer_facts[k] = v;
            } else if (k == "profile") {
                f.profile = v;
            } else if (k == "at") {
                if (!parse_split(v, f.at)) return fail("at is the first layer of each later node, rising: 30 or 16,32");
            } else {
                return fail("unknown fact \"" + k + "\"");
            }
        }
        if (node != nullptr)
            for (const char* need : {"vram_room", "ram_room", "layer_ms"})
                if (!node_seen.count(need)) return fail("node " + node->name + " needs " + need);
        if (node != nullptr && !(node->layer_ms > 0)) return fail("layer_ms must be more than 0");
    }
    if (m.n_layers < 1 || m.n_layers > kMaxLayers) {
        err = "layers must be 1 to " + std::to_string(kMaxLayers);
        return false;
    }
    if (m.first_min > m.n_layers || ring > 1) {
        err = m.first_min > m.n_layers ? "first_min is more than layers" : "ring must be 0 or 1";
        return false;
    }
    if (!f.profile.empty() && f.n_expert < 1) {
        err = "a profile needs experts";
        return false;
    }
    m.ring = ring == 1;
    std::map<std::string, std::vector<int64_t>> per_layer;
    for (const char* need : {"slot_bytes", "layer_arena_bytes", "session_layer_bytes"})
        if (!layer_facts.count(need)) {
            err = std::string("missing ") + need;
            return false;
        }
    for (const auto& [k, v] : layer_facts)
        if (!to_layers(v, m.n_layers, per_layer[k])) {
            err = k + " must be one whole number or " + std::to_string(m.n_layers) + " of them, comma-separated";
            return false;
        }
    const std::vector<int64_t> slots = per_layer["slot_bytes"], session = per_layer["session_layer_bytes"];
    m.slot_bytes = [slots](int64_t l) { return slots[(size_t) l]; };
    m.session_bytes = [session, session_base](int64_t lb, int64_t le) {
        int64_t b = session_base;
        for (int64_t l = lb; l < le; ++l) b += session[(size_t) l];
        return b;
    };
    m.layer_arena_bytes = per_layer["layer_arena_bytes"];
    for (const int64_t kv : per_layer["layer_kv"]) m.layer_kv.push_back(kv ? 1 : 0);
    if (f.nodes.empty()) {
        err = "no node: give one node=NAME line per PC, the head PC first";
        return false;
    }
    if (!f.at.empty() && f.at.size() + 1 != f.nodes.size()) {
        err = "at needs one split point per node after the first";
        return false;
    }
    if (!f.calib.scale.empty() && !f.calib.usable(f.nodes.size())) {
        err = "calib needs windows of at least " + std::to_string(SplitCalib::kMinWindows) +
              ", one scale above 0 per node, and the splits it was measured with (seen)";
        return false;
    }
    for (NodeCap& n : f.nodes) {
        if (!delays.count(n.name)) continue;
        std::istringstream list(delays[n.name]);
        std::string item;
        while (std::getline(list, item, ',')) {
            const size_t colon = item.find(':');
            const std::string to = item.substr(0, colon);
            double ms = 0;
            bool known = false;
            for (const NodeCap& o : f.nodes) known = known || (o.name == to && o.name != n.name);
            if (colon == std::string::npos || !known || !to_real(item.substr(colon + 1), ms)) {
                err = "node " + n.name + ": delay is OTHER:MS,... with OTHER another node's name";
                return false;
            }
            n.delay_ms[to] = ms;
        }
    }
    return true;
}

bool read_split_facts(const std::string& path, SplitFacts& f, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot read " + path;
        return false;
    }
    std::ostringstream text;
    text << in.rdbuf();
    return parse_split_facts(text.str(), f, err);
}

SplitPlan plan_split_facts(const SplitFacts& f) {
    if (f.at.empty() && f.calib.scale.empty()) return auto_pool(f.model, f.nodes);
    SplitModel m = f.model;
    if (!f.calib.scale.empty()) m.calib = m.failures = &f.calib;
    SplitPlan p = f.at.empty() ? auto_split(m, f.nodes) : evaluate_split(m, f.nodes, f.at);
    p.order.resize(f.nodes.size());
    std::iota(p.order.begin(), p.order.end(), 0);
    return p;
}

std::string split_plan_json(const SplitFacts& f, const SplitPlan& p) {
    const SplitModel& m = f.model;
    std::string o;
    char b[64];
    const auto real = [&](const std::string& k, double v) {
        std::snprintf(b, sizeof b, "%.6g", v);
        o += "\"" + k + "\":" + b + ",";
    };
    const auto whole = [&](const std::string& k, int64_t v) { o += "\"" + k + "\":" + std::to_string(v) + ","; };
    const auto end_with = [&](char c) {
        if (o.back() == ',') o.pop_back();
        o += c;
    };
    o = p.ok ? "{\"ok\":true," : "{\"ok\":false,\"why\":\"" + p.why + "\",";
    if (p.ok) {
        o += "\"at\":[";
        for (const int64_t a : p.at) o += std::to_string(a) + ",";
        end_with(']');
        o += ",\"nodes\":[";
        for (size_t i = 0; i < p.order.size(); ++i) {
            o += "{\"name\":\"" + f.nodes[(size_t) p.order[i]].name + "\",";
            whole("first", i == 0 ? 0 : p.at[i - 1]);
            whole("last", (i < p.at.size() ? p.at[i] : m.n_layers) - 1);
            real("ms", p.node_ms[i]);
            whole("slots", p.node_slots[i]);
            whole("arena_bytes", p.node_arena[i]);
            whole("kv_bytes", p.node_kv[i]);
            end_with('}');
            o += ",";
        }
        end_with(']');
        o += ",";
        real("window_ms", p.ms);
        real("decode_tok_s", p.decode_tok_s);
        real("prompt_ms", p.prompt_ms);
        real("request_ms", p.request_ms);
        real("mass", p.mass);
        whole("held", p.held);
        o += std::string("\"measured\":") + (p.measured ? "true," : "false,");
    }
    o += "\"inputs\":{";
    whole("layers", m.n_layers);
    whole("experts", f.n_expert);
    whole("profile_pairs", (int64_t) m.profile.size());
    real("window_tokens", m.window_tokens);
    whole("window_rows", m.window_rows);
    whole("row_bytes", m.row_bytes);
    whole("ring", m.ring ? 1 : 0);
    real("hop_overhead_ms", m.hop_overhead_ms);
    real("hop_ms", m.hop_ms);
    real("miss_ms", m.miss_ms);
    whole("prompt_tokens", m.prompt_tokens);
    whole("prompt_chunk", m.prompt_chunk);
    whole("prompt_row_bytes", m.prompt_row_bytes);
    whole("drafter_rows", m.drafter_rows);
    whole("answer_tokens", m.answer_tokens);
    whole("kv_copies", m.kv_copies);
    whole("first_min", m.first_min);
    if (!f.calib.scale.empty()) o += "\"calib\":\"" + f.calib.encode() + "\",";
    o += "\"nodes\":[";
    for (const NodeCap& n : f.nodes) {
        o += "{\"name\":\"" + n.name + "\",";
        whole("vram_room", n.vram_room);
        whole("ram_room", n.ram_room);
        real("layer_ms", n.layer_ms);
        real("pf_layer_ms", n.pf_layer_ms);
        whole("kv_host_layer", n.kv_host_layer);
        whole("pin_room", n.pin_room);
        real("up_mbit", n.up_mbit);
        real("down_mbit", n.down_mbit);
        o += "\"delay_ms\":{";
        for (const auto& [to, ms] : n.delay_ms) real(to, ms);
        end_with('}');
        end_with('}');
        o += ",";
    }
    end_with(']');
    o += "}}";
    return o;
}

}  // namespace strata::pool
