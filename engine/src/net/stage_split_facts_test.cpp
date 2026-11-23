// src/net/stage_split_facts_test.cpp - `strata --plan-split FACTS` without a GPU: the facts it reads, the plan it
// picks from them over two and three PCs with and without link facts, the JSON line, and the facts it refuses.
#include "strata/net/stage_split_facts.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::pool;

static int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static const std::string kModel = "# an 8-layer model, 1 GiB of experts a layer\n"
                                  "layers=8\n"
                                  "experts=4\n"
                                  "slot_bytes=1048576\n"
                                  "layer_arena_bytes=1073741824\n"
                                  "session_base_bytes=4096 session_layer_bytes=8388608\n"
                                  "miss_ms=190\n";

/// Parses `text`, gives the model a profile of every (layer, expert) pair and plans as the mode does.
static SplitPlan plan_of(const std::string& text, SplitFacts& f) {
    std::string err;
    CHECK(parse_split_facts(text, f, err) && err.empty());
    for (int32_t e = 0; e < 4; ++e)
        for (int32_t l = 0; l < 8; ++l) f.model.profile.push_back({l, e});
    return plan_split_facts(f);
}

/// The window's network time: what the plan's window costs beyond its nodes' own time.
static double net_ms(const SplitPlan& p) {
    double ms = p.ms;
    for (const double n : p.node_ms) ms -= n;
    return ms;
}

/// Two PCs whose RAM cannot hold the model alone: the facts reach the planner, the split puts at least 3 layers on
/// the second, and the JSON line names both with their layers.  Without link facts each crossing costs hop_ms.
static void test_two_pcs() {
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "node=desktop vram_room=4294967296 ram_room=4294967296 layer_ms=0.5\n"
                                         "node=laptop vram_room=3221225472 ram_room=5368709120 layer_ms=0.6\n", f);
    CHECK(f.model.n_layers == 8 && f.nodes.size() == 2 && f.nodes[1].name == "laptop");
    CHECK(f.model.slot_bytes(3) == 1048576 && f.model.session_bytes(2, 5) == 4096 + 3 * 8388608);
    CHECK(p.ok && p.order.size() == 2 && p.at.size() == 1 && p.at[0] >= 3 && p.at[0] <= 4);
    CHECK(std::fabs(net_ms(p) - 2 * f.model.hop_ms) < 1e-9);
    const std::string json = split_plan_json(f, p);
    CHECK(json.rfind("{\"ok\":true,\"at\":[", 0) == 0 && json.back() == '}' && json.find('\n') == std::string::npos);
    CHECK(json.find("{\"name\":\"desktop\",\"first\":0,\"last\":" + std::to_string(p.at[0] - 1)) != std::string::npos);
    CHECK(json.find("{\"name\":\"laptop\",\"first\":" + std::to_string(p.at[0]) + ",\"last\":7") != std::string::npos);
    CHECK(json.find("\"decode_tok_s\":") != std::string::npos && json.find("\"inputs\":{\"layers\":8,") != std::string::npos);
}

/// The same two PCs with a 20 Mbit/s, 10 ms link: each crossing costs the delay, the overhead and a row over 20 Mbit/s.
static void test_link_facts() {
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "node=desktop vram_room=4294967296 ram_room=4294967296 layer_ms=0.5\n"
                                         "node=laptop vram_room=3221225472 ram_room=5368709120 layer_ms=0.6 "
                                         "up_mbit=20 down_mbit=20 delay=desktop:10\n", f);
    CHECK(p.ok && p.order.size() == 2 && f.nodes[1].delay_ms.at("desktop") == 10);
    CHECK(std::fabs(net_ms(p) - 2 * (10 + 1.2 + 51216 / (20 * 125.0))) < 1e-9);
    CHECK(split_plan_json(f, p).find("\"delay_ms\":{\"desktop\":10}") != std::string::npos);
}

/// Three PCs of 3 GiB of RAM each: all three are needed, each with at most 3 of the 8 layers.
static void test_three_pcs() {
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "node=a vram_room=1073741824 ram_room=3221225472 layer_ms=0.5\n"
                                         "node=b vram_room=1073741824 ram_room=3221225472 layer_ms=0.5 delay=c:2\n"
                                         "node=c vram_room=1073741824 ram_room=3221225472 layer_ms=0.5\n", f);
    CHECK(p.ok && p.order.size() == 3 && p.at.size() == 2);
    CHECK(p.at[0] >= 2 && p.at[0] <= 3 && p.at[1] - p.at[0] <= 3 && 8 - p.at[1] <= 3);
    CHECK(split_plan_json(f, p).find("\"last\":7,") != std::string::npos);
}

/// A split the facts fix is scored as given, the same as evaluate_split, even where the search would cut elsewhere;
/// one the RAM cannot hold gives ok false.
static void test_fixed_split() {
    const std::string pcs = "node=desktop vram_room=4294967296 ram_room=4294967296 layer_ms=0.5\n"
                            "node=laptop vram_room=3221225472 ram_room=5368709120 layer_ms=0.6\n";
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "at=3\n" + pcs, f);
    const SplitPlan same = evaluate_split(f.model, f.nodes, {3});
    CHECK(p.ok && p.at == std::vector<int64_t>{3} && p.order.size() == 2 && p.ms == same.ms && p.held == same.held);
    CHECK(split_plan_json(f, p).find("{\"name\":\"laptop\",\"first\":3,\"last\":7") != std::string::npos);
    const SplitPlan full = plan_of(kModel + "at=2\n" + pcs, f);
    CHECK(!full.ok && full.why.find("laptop's RAM") != std::string::npos);
}

/// A node's prompt facts: each chunk costs every layer of it pf_layer_chunk_ms plus pf_layer_ms per 1,000 tokens.
/// The rows coming back cross at drafter_row_bytes, or at prompt_row_bytes without it.
static void test_prompt_facts() {
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "prompt_tokens=1000 prompt_chunk=500\n"
                                         "node=a vram_room=1 ram_room=0 layer_ms=0.5 pf_layer_ms=10 "
                                         "pf_layer_chunk_ms=20\n", f);
    CHECK(p.ok && std::fabs(p.prompt_ms - 2 * 8 * (20 + 5)) < 1e-9);
    CHECK(split_plan_json(f, p).find("\"pf_layer_chunk_ms\":20") != std::string::npos);
    CHECK(split_plan_json(f, p).find("\"drafter_row_bytes\":40960") != std::string::npos);
    SplitFacts g;
    const SplitPlan q = plan_of(kModel + "prompt_tokens=1000 prompt_chunk=1000 drafter_row_bytes=20480\nat=4\n"
                                         "node=a vram_room=1 ram_room=0 layer_ms=0.5\n"
                                         "node=b vram_room=1 ram_room=0 layer_ms=0.5\n", g);
    CHECK(q.ok && std::fabs(q.prompt_ms - (2 * 1.2 + 1000 * (40960 + 20480) / 117500.0)) < 1e-9);
    CHECK(split_plan_json(g, q).find("\"drafter_row_bytes\":20480") != std::string::npos);
}

/// Measured timings in SplitCalib's text: the window is the coordinator's fixed time, the network's and each node's
/// predicted time times its scale, and the plan says it is measured.  The search keeps the nodes' order.
static void test_calib() {
    const std::string pcs = "node=desktop vram_room=4294967296 ram_room=4294967296 layer_ms=0.5\n"
                            "node=laptop vram_room=3221225472 ram_room=5368709120 layer_ms=0.6\n"
                            "calib windows=500 fixed_ms=12 net_ms=3 scale=2,1 seen=4\n";
    SplitFacts f;
    const SplitPlan p = plan_of(kModel + "at=4\n" + pcs, f);
    CHECK(p.ok && p.measured && std::fabs(p.ms - (12 + 3 + 2 * p.node_pred[0] + p.node_pred[1])) < 1e-9);
    const std::string json = split_plan_json(f, p);
    CHECK(json.find("\"measured\":true") != std::string::npos && json.find("\"calib\":\"windows=500 ") != std::string::npos);
    const SplitPlan searched = plan_of(kModel + pcs, f);
    CHECK(searched.ok && searched.measured && searched.order == std::vector<int>({0, 1}));
}

/// Facts the mode refuses, each with an error that says what is wrong; and facts that fit nowhere give ok false.
static void test_bad_facts() {
    const std::string node = "node=a vram_room=1 ram_room=0 layer_ms=0.5\n";
    const struct { std::string text, says; } bad[] = {
        {"", "layers"},
        {kModel, "no node"},
        {kModel + node + "colour=red\n", "unknown fact"},
        {kModel + "garbage\n" + node, "not key=value"},
        {kModel + "layers=9\n" + node, "twice"},
        {"layers=8 slot_bytes=1,2 layer_arena_bytes=1 session_layer_bytes=1\n" + node, "slot_bytes"},
        {"layers=8 slot_bytes=1 layer_arena_bytes=1\n" + node, "session_layer_bytes"},
        {"layers=8 slot_bytes=1 layer_arena_bytes=1 session_layer_bytes=1\n" + node, "miss_ms"},
        {kModel + "node=a vram_room=1 ram_room=0\n", "layer_ms"},
        {kModel + "node=a vram_room=-5 ram_room=0 layer_ms=0.5\n", "vram_room"},
        {kModel + "node=a vram_room=1 ram_room=0 layer_ms=0\n", "layer_ms"},
        {kModel + node + node, "twice"},
        {kModel + "node=a,b vram_room=1 ram_room=0 layer_ms=0.5\n", "name"},
        {kModel + "node=a vram_room=1 ram_room=0 layer_ms=0.5 colour=red\n", "unknown node fact"},
        {kModel + "node=a vram_room=1 ram_room=0 layer_ms=0.5 delay=b:3\n", "delay"},
        {"layers=8 slot_bytes=1 layer_arena_bytes=1 session_layer_bytes=1 profile=p.bin\n" + node, "experts"},
        {kModel + "ring=2\n" + node, "ring"},
        {kModel + "at=4\n" + node, "one split point per node"},
        {kModel + "at=x\n" + node, "at is"},
        {kModel + node + "calib windows=100 fixed_ms=1 net_ms=1 scale=1 seen=4\n", "calib needs"},
        {kModel + node + "calib windows=500 fixed_ms=1 net_ms=1 scale=1,1 seen=4\n", "calib needs"},
        {kModel + node + "calib windows=500 colour=red scale=1 seen=4\n", "SplitCalib's text"},
    };
    for (const auto& b : bad) {
        SplitFacts f;
        std::string err;
        const bool refused = !parse_split_facts(b.text, f, err) && err.find(b.says) != std::string::npos;
        CHECK(refused);
        if (!refused) std::fprintf(stderr, "  wanted \"%s\", got \"%s\"\n", b.says.c_str(), err.c_str());
    }
    SplitFacts f;
    std::string err;
    CHECK(!read_split_facts("no-such-facts-file.txt", f, err) && err.find("cannot read") != std::string::npos);
    const SplitPlan none = plan_of(kModel + "node=a vram_room=1 ram_room=1073741824 layer_ms=0.5\n", f);
    const std::string json = split_plan_json(f, none);
    CHECK(!none.ok && json.rfind("{\"ok\":false,\"why\":\"", 0) == 0 && json.find("RAM") != std::string::npos);
}

int main() {
    test_two_pcs();
    test_link_facts();
    test_three_pcs();
    test_fixed_split();
    test_prompt_facts();
    test_calib();
    test_bad_facts();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("stage_split_facts_test: all checks passed\n");
    return 0;
}
