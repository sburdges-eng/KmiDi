// Unit test for daiw::rt::DSPGraph.
//
// Standalone executable; depends only on the header-only DSPGraph.
//
// Coverage:
//   1. Topological sort produces a valid execution order
//   2. Cycle detection refuses to go live
//   3. End-to-end audio flow: source → gain → sum → sink
//   4. Multi-branch DAG: source → (gain_a, gain_b) → mixer → sink
//   5. Capacity guards: max nodes / edges, prepare-after-live rejected
//   6. Process() with num_samples > block_size is rejected (no UB)

#include "daiw/dsp_graph.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;
void check(bool condition, const char *label) {
  if (!condition) {
    std::printf("FAIL: %s\n", label);
    ++g_failures;
  }
}

// --- Test nodes ----------------------------------------------------------

// Source node: zero inputs, one output. Fills the output buffer with a
// caller-provided value each call.
class Source : public daiw::rt::DSPNode {
public:
  explicit Source(float value) : value_(value) {}
  void prepare(double, std::size_t) override {}
  void process(const float *const *, float *const *outputs,
               std::size_t num_samples) noexcept override {
    if (outputs[0] == nullptr)
      return;
    for (std::size_t i = 0; i < num_samples; ++i)
      outputs[0][i] = value_;
  }
  std::size_t num_inputs() const override { return 0; }
  std::size_t num_outputs() const override { return 1; }

private:
  float value_;
};

// Gain node: one input, one output. output = input * gain.
class Gain : public daiw::rt::DSPNode {
public:
  explicit Gain(float gain) : gain_(gain) {}
  void prepare(double, std::size_t) override {}
  void process(const float *const *inputs, float *const *outputs,
               std::size_t num_samples) noexcept override {
    if (!inputs[0] || !outputs[0])
      return;
    for (std::size_t i = 0; i < num_samples; ++i) {
      outputs[0][i] = inputs[0][i] * gain_;
    }
  }
  std::size_t num_inputs() const override { return 1; }
  std::size_t num_outputs() const override { return 1; }

private:
  float gain_;
};

// Mixer node: two inputs, one output. output = in0 + in1.
class Mixer : public daiw::rt::DSPNode {
public:
  void prepare(double, std::size_t) override {}
  void process(const float *const *inputs, float *const *outputs,
               std::size_t num_samples) noexcept override {
    if (!outputs[0])
      return;
    const float *a = inputs[0];
    const float *b = inputs[1];
    for (std::size_t i = 0; i < num_samples; ++i) {
      const float va = a ? a[i] : 0.0f;
      const float vb = b ? b[i] : 0.0f;
      outputs[0][i] = va + vb;
    }
  }
  std::size_t num_inputs() const override { return 2; }
  std::size_t num_outputs() const override { return 1; }
};

// Sink node: one input, zero outputs. Records the last sample seen.
class Sink : public daiw::rt::DSPNode {
public:
  float last_value = 0.0f;
  std::size_t process_calls = 0;
  void prepare(double, std::size_t) override {}
  void process(const float *const *inputs, float *const *,
               std::size_t num_samples) noexcept override {
    ++process_calls;
    if (!inputs[0] || num_samples == 0)
      return;
    last_value = inputs[0][num_samples - 1];
  }
  std::size_t num_inputs() const override { return 1; }
  std::size_t num_outputs() const override { return 0; }
};

// --- Tests ---------------------------------------------------------------

void test_linear_chain() {
  daiw::rt::DSPGraph<8, 16> graph;
  Source src(0.5f);
  Gain gain(2.0f);
  Sink sink;

  const auto s = graph.add_node(&src);
  const auto g = graph.add_node(&gain);
  const auto k = graph.add_node(&sink);

  check(s != daiw::rt::kInvalidNodeId, "linear: source registered");
  check(g != daiw::rt::kInvalidNodeId, "linear: gain registered");
  check(k != daiw::rt::kInvalidNodeId, "linear: sink registered");

  check(graph.connect(s, 0, g, 0), "linear: connect source→gain");
  check(graph.connect(g, 0, k, 0), "linear: connect gain→sink");
  check(graph.prepare(48000.0, 64), "linear: prepare succeeds");
  check(graph.is_live(), "linear: graph is live after prepare");

  // Execution order must place source before gain before sink.
  int pos_s = -1, pos_g = -1, pos_k = -1;
  for (std::size_t i = 0; i < 3; ++i) {
    const auto id = graph.exec_order(i);
    if (id == s)
      pos_s = static_cast<int>(i);
    if (id == g)
      pos_g = static_cast<int>(i);
    if (id == k)
      pos_k = static_cast<int>(i);
  }
  check(pos_s < pos_g && pos_g < pos_k,
        "linear: topo order is source < gain < sink");

  graph.process(64);
  check(std::fabs(sink.last_value - 1.0f) < 1e-6f,
        "linear: 0.5 × 2.0 = 1.0 reaches sink");
}

void test_diamond_dag() {
  daiw::rt::DSPGraph<8, 16> graph;
  Source src(0.25f);
  Gain a(4.0f);
  Gain b(2.0f);
  Mixer mix;
  Sink sink;

  const auto s = graph.add_node(&src);
  const auto na = graph.add_node(&a);
  const auto nb = graph.add_node(&b);
  const auto nm = graph.add_node(&mix);
  const auto nk = graph.add_node(&sink);

  check(graph.connect(s, 0, na, 0), "diamond: src→a");
  check(graph.connect(s, 0, nb, 0), "diamond: src→b (fan-out)");
  check(graph.connect(na, 0, nm, 0), "diamond: a→mix.in0");
  check(graph.connect(nb, 0, nm, 1), "diamond: b→mix.in1");
  check(graph.connect(nm, 0, nk, 0), "diamond: mix→sink");
  check(graph.prepare(48000.0, 32), "diamond: prepare succeeds");

  // src fan-out: source's output buffer is shared by edges to a AND b.
  // The graph stores edges in submission order; the LATER edge wins the
  // pointer slot. That's a known limitation (resolved by inserting an
  // explicit splitter node in higher layers). For now, just verify the
  // computation is still correct: with the src→a edge buffer driving 'a',
  // and the (potentially aliased) src→b edge buffer driving 'b', both
  // gains receive 0.25 and mixer sees 4*0.25 + 2*0.25 = 1.5.
  graph.process(32);
  check(std::fabs(sink.last_value - 1.5f) < 1e-6f,
        "diamond: 4×0.25 + 2×0.25 = 1.5 reaches sink");
}

void test_cycle_rejected() {
  daiw::rt::DSPGraph<4, 8> graph;
  Gain a(1.0f), b(1.0f), c(1.0f);
  const auto na = graph.add_node(&a);
  const auto nb = graph.add_node(&b);
  const auto nc = graph.add_node(&c);
  check(graph.connect(na, 0, nb, 0), "cycle: a→b");
  check(graph.connect(nb, 0, nc, 0), "cycle: b→c");
  check(graph.connect(nc, 0, na, 0), "cycle: c→a (cycle introduced)");

  check(!graph.prepare(48000.0, 64), "cycle: prepare rejects cyclic graph");
  check(!graph.is_live(), "cycle: graph not live after cycle rejection");
}

void test_capacity_guards() {
  daiw::rt::DSPGraph<2, 1> tiny;
  Gain g1(1.0f), g2(1.0f), g3(1.0f);

  const auto a = tiny.add_node(&g1);
  const auto b = tiny.add_node(&g2);
  const auto c = tiny.add_node(&g3);
  check(a != daiw::rt::kInvalidNodeId, "capacity: 1st node registered");
  check(b != daiw::rt::kInvalidNodeId, "capacity: 2nd node registered");
  check(c == daiw::rt::kInvalidNodeId,
        "capacity: 3rd node rejected (MaxNodes=2)");

  check(tiny.connect(a, 0, b, 0), "capacity: 1st edge registered");
  check(!tiny.connect(b, 0, a, 0), "capacity: 2nd edge rejected (MaxEdges=1)");
}

void test_reject_changes_after_live() {
  daiw::rt::DSPGraph<4, 8> graph;
  Source s(1.0f);
  Sink k;
  const auto ns = graph.add_node(&s);
  const auto nk = graph.add_node(&k);
  check(graph.connect(ns, 0, nk, 0), "live: connect");
  check(graph.prepare(48000.0, 64), "live: prepare succeeds");

  Gain extra(2.0f);
  check(graph.add_node(&extra) == daiw::rt::kInvalidNodeId,
        "live: add_node after prepare rejected");
  check(!graph.connect(ns, 0, nk, 0), "live: connect after prepare rejected");
  check(!graph.prepare(48000.0, 64), "live: re-prepare rejected");
}

void test_oversize_block_safe() {
  daiw::rt::DSPGraph<4, 8> graph;
  Source s(0.1f);
  Sink k;
  const auto ns = graph.add_node(&s);
  const auto nk = graph.add_node(&k);
  graph.connect(ns, 0, nk, 0);
  graph.prepare(48000.0, 32);

  // Process with num_samples > block_size — must be a no-op rather than
  // writing past the end of the per-edge buffer.
  k.process_calls = 0;
  graph.process(64);
  check(k.process_calls == 0,
        "guards: process(num_samples > block_size) is a no-op");
}

} // namespace

int main() {
  test_linear_chain();
  test_diamond_dag();
  test_cycle_rejected();
  test_capacity_guards();
  test_reject_changes_after_live();
  test_oversize_block_safe();

  if (g_failures == 0) {
    std::printf("OK: all DSP-graph checks passed\n");
    return 0;
  }
  std::printf("FAIL: %d check(s) failed\n", g_failures);
  return 1;
}
