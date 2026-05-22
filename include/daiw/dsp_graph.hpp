#pragma once

// daiw/dsp_graph.hpp — deterministic, allocation-free DSP node graph.
//
// Design
//   - Nodes are inserted on the non-RT thread before audio starts.
//     `add_node()` returns an opaque NodeId; `connect()` records a directed
//     dependency from producer to consumer.
//   - `prepare()` performs Kahn's topological sort once. It writes the linear
//     execution order into a fixed-size array, allocates per-edge audio
//     buffers, and switches the graph into "live" mode. From that point on,
//     no further structural changes are permitted.
//   - `process(num_samples)` walks the execution order and calls each node's
//     `process()` method with input/output buffer pointers. The walk is a
//     tight loop over preallocated indices — no heap, no locks, no syscalls.
//   - On cycle detection (toposort fails), `prepare()` returns false and the
//     graph stays in "build" mode. The audio thread must therefore check
//     `is_live()` before calling `process()`.
//
// Scope
//   This is the scheduling primitive — the routing skeleton. Concrete nodes
//   (gain, EQ, ML, mix bus, etc.) implement the `DSPNode` interface and are
//   composed by higher layers. The graph itself stays domain-agnostic so the
//   ML→EQ chain and any future routing topology (multi-bus, FX sends,
//   sidechain routing) can share the same machinery.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace daiw::rt {

using NodeId = uint16_t;
static constexpr NodeId kInvalidNodeId = 0xFFFFu;

// A node owns up to kMaxPorts input and output ports. The graph passes the
// node a fixed-size pointer array per direction; nullptr entries indicate
// unconnected ports. The node must tolerate partial connectivity.
constexpr std::size_t kMaxPorts = 8;

class DSPNode {
public:
  virtual ~DSPNode() = default;

  // Called once when the graph goes live. block_size is the maximum number
  // of samples that will be passed to process() in any single call.
  virtual void prepare(double sample_rate, std::size_t block_size) = 0;

  // RT contract: must be noexcept, allocation-free, lock-free.
  // inputs / outputs are arrays of length num_inputs / num_outputs;
  // unconnected ports are nullptr.
  virtual void process(const float *const *inputs, float *const *outputs,
                       std::size_t num_samples) noexcept = 0;

  virtual std::size_t num_inputs() const = 0;
  virtual std::size_t num_outputs() const = 0;
};

template <std::size_t MaxNodes = 64, std::size_t MaxEdges = 256>
class DSPGraph {
public:
  DSPGraph() noexcept
      : node_count_(0), edge_count_(0), exec_count_(0), live_(false),
        sample_rate_(0.0), block_size_(0) {}

  // ---- Build phase (non-RT) -------------------------------------------

  // Register a node. The graph stores the pointer but does not own it.
  // Returns kInvalidNodeId if the graph is full or already live.
  NodeId add_node(DSPNode *node) {
    if (live_ || node_count_ >= MaxNodes || node == nullptr) {
      return kInvalidNodeId;
    }
    const NodeId id = static_cast<NodeId>(node_count_++);
    nodes_[id] = node;
    return id;
  }

  // Wire src_node's output port to dst_node's input port. Both nodes must
  // already be registered. Fan-out (multiple edges from one output port) is
  // supported — every consumer reads from the same producer buffer. Fan-in
  // is rejected (an input port can only be driven by a single producer);
  // use an explicit Mixer node for summing.
  bool connect(NodeId src_node, std::size_t src_port, NodeId dst_node,
               std::size_t dst_port) {
    if (live_)
      return false;
    if (src_node >= node_count_ || dst_node >= node_count_)
      return false;
    if (src_port >= nodes_[src_node]->num_outputs())
      return false;
    if (dst_port >= nodes_[dst_node]->num_inputs())
      return false;
    if (edge_count_ >= MaxEdges)
      return false;
    // Reject fan-in (multiple edges into the same input port).
    for (std::size_t i = 0; i < edge_count_; ++i) {
      if (edges_[i].dst_node == dst_node && edges_[i].dst_port == dst_port) {
        return false;
      }
    }
    edges_[edge_count_++] =
        Edge{src_node, dst_node, static_cast<uint8_t>(src_port),
             static_cast<uint8_t>(dst_port), 0};
    return true;
  }

  // Topologically sort and pin the execution plan. Returns false on cycle.
  // After a successful prepare, no further add_node/connect calls succeed.
  bool prepare(double sample_rate, std::size_t block_size) {
    if (live_)
      return false;
    if (block_size == 0 || block_size > kMaxBlockSize)
      return false;

    // Kahn's algorithm. in_degree counted across edges.
    std::array<uint16_t, MaxNodes> in_degree{};
    for (std::size_t i = 0; i < edge_count_; ++i) {
      ++in_degree[edges_[i].dst_node];
    }

    // Seed queue with zero-in-degree nodes.
    std::array<NodeId, MaxNodes> queue{};
    std::size_t qhead = 0;
    std::size_t qtail = 0;
    for (std::size_t i = 0; i < node_count_; ++i) {
      if (in_degree[i] == 0) {
        queue[qtail++] = static_cast<NodeId>(i);
      }
    }

    // Walk in topo order.
    exec_count_ = 0;
    while (qhead < qtail) {
      const NodeId u = queue[qhead++];
      exec_order_[exec_count_++] = u;
      for (std::size_t i = 0; i < edge_count_; ++i) {
        if (edges_[i].src_node != u)
          continue;
        const NodeId v = edges_[i].dst_node;
        if (--in_degree[v] == 0) {
          queue[qtail++] = v;
        }
      }
    }

    if (exec_count_ != node_count_) {
      // Cycle detected — at least one node never reached zero in-degree.
      exec_count_ = 0;
      return false;
    }

    // Per-node input/output pointer scratch arrays — preallocated.
    node_input_ptrs_.assign(node_count_, std::array<float *, kMaxPorts>{});
    node_output_ptrs_.assign(node_count_, std::array<float *, kMaxPorts>{});

    // Allocate one audio buffer per unique (src_node, src_port) used by any
    // edge. Fan-out edges from the same output port share the same buffer
    // so the producer writes once and all consumers read the same samples.
    // Two-pass to keep buffer pointers stable across vector growth.
    std::array<std::array<bool, kMaxPorts>, MaxNodes> port_seen{};
    std::size_t unique_outputs = 0;
    for (std::size_t i = 0; i < edge_count_; ++i) {
      const auto &e = edges_[i];
      if (!port_seen[e.src_node][e.src_port]) {
        port_seen[e.src_node][e.src_port] = true;
        ++unique_outputs;
      }
    }
    edge_buffers_.assign(unique_outputs, std::vector<float>(block_size, 0.0f));

    std::array<std::array<float *, kMaxPorts>, MaxNodes> port_to_buf{};
    std::size_t buf_idx = 0;
    for (std::size_t i = 0; i < edge_count_; ++i) {
      const auto &e = edges_[i];
      float *&slot = port_to_buf[e.src_node][e.src_port];
      if (slot == nullptr) {
        slot = edge_buffers_[buf_idx++].data();
        node_output_ptrs_[e.src_node][e.src_port] = slot;
      }
      node_input_ptrs_[e.dst_node][e.dst_port] = slot;
    }

    // Prepare each node.
    for (std::size_t i = 0; i < node_count_; ++i) {
      nodes_[i]->prepare(sample_rate, block_size);
    }

    sample_rate_ = sample_rate;
    block_size_ = block_size;
    live_ = true;
    return true;
  }

  // ---- Audio thread ----------------------------------------------------

  void process(std::size_t num_samples) noexcept {
    if (!live_ || num_samples > block_size_)
      return;
    for (std::size_t k = 0; k < exec_count_; ++k) {
      const NodeId id = exec_order_[k];
      DSPNode *node = nodes_[id];
      node->process(
          reinterpret_cast<const float *const *>(node_input_ptrs_[id].data()),
          node_output_ptrs_[id].data(), num_samples);
    }
  }

  // ---- Diagnostics ----------------------------------------------------

  bool is_live() const noexcept { return live_; }
  std::size_t node_count() const noexcept { return node_count_; }
  std::size_t edge_count() const noexcept { return edge_count_; }
  NodeId exec_order(std::size_t k) const noexcept {
    return k < exec_count_ ? exec_order_[k] : kInvalidNodeId;
  }

  // Test-only: expose a buffer pointer for a specific node port (output).
  // Returns nullptr if the port is unconnected or out of range.
  float *output_buffer(NodeId node, std::size_t port) noexcept {
    if (!live_ || node >= node_count_ || port >= kMaxPorts)
      return nullptr;
    return node_output_ptrs_[node][port];
  }

  // Test-only: inject samples into a node's input buffer (e.g. for source
  // nodes that don't have an upstream feeder).
  float *input_buffer(NodeId node, std::size_t port) noexcept {
    if (!live_ || node >= node_count_ || port >= kMaxPorts)
      return nullptr;
    return node_input_ptrs_[node][port];
  }

private:
  struct Edge {
    NodeId src_node;
    NodeId dst_node;
    uint8_t src_port;
    uint8_t dst_port;
    uint8_t _pad;
  };

  static constexpr std::size_t kMaxBlockSize = 8192;

  std::array<DSPNode *, MaxNodes> nodes_{};
  std::array<Edge, MaxEdges> edges_{};
  std::array<NodeId, MaxNodes> exec_order_{};

  // Per-edge audio buffer + per-node port pointer arrays. Allocated once
  // in prepare(), never touched by the audio thread.
  std::vector<std::vector<float>> edge_buffers_;
  std::vector<std::array<float *, kMaxPorts>> node_input_ptrs_;
  std::vector<std::array<float *, kMaxPorts>> node_output_ptrs_;

  std::size_t node_count_;
  std::size_t edge_count_;
  std::size_t exec_count_;
  bool live_;
  double sample_rate_;
  std::size_t block_size_;
};

} // namespace daiw::rt
