#ifndef SM_2_SM_NETWORK_H
#define SM_2_SM_NETWORK_H

#include "../abstract_hardware_model.h"
#include "../intersim2/interconnect_interface.hpp"
#include "array"
#include "local_interconnect.h"
#include "shader.h"

void sm2sm_network_options(class OptionParser* opp);

class cluster_shmem_request {
 public:
  cluster_shmem_request(warp_inst_t* warp, addr_t address, bool is_write,
                        bool is_atomic, unsigned origin_shader_id,
                        unsigned target_shader_id, unsigned tid,
                        unsigned latency, unsigned size);
  void send_response() { m_is_response = true; }
  // Called when data came from the target SM, then the request can be treated
  // as a normal request
  void atomic_sendback() {
    m_is_atomic = false;
    m_is_response = false;
  }
  warp_inst_t* get_warp() { return m_warp; }
  void send_request() { m_is_send = true; }

 private:
  // Warp the request belongs to
  warp_inst_t* m_warp;
  bool m_is_write;
  bool m_is_atomic;
  bool m_is_response;
  bool m_is_complete;
  bool m_is_send;
  addr_t m_address;
  unsigned m_tid;
  unsigned m_origin_shader_id;
  unsigned m_target_shader_id;
  unsigned m_size;
  // Time it takes to process the request for the target SM.
  unsigned m_latency;

 public:
  const bool& is_write = m_is_write;
  const bool& is_atomic = m_is_atomic;
  const bool& is_response = m_is_response;
  const bool& is_send = m_is_send;
  const unsigned& origin_shader_id = m_origin_shader_id;
  const unsigned& target_shader_id = m_target_shader_id;
  const unsigned& size = m_size;
  const unsigned& tid = m_tid;
  const addr_t address = m_address;
  const unsigned& latency = m_latency;
};

class sm_2_sm_network {
 public:
  // Functions for local interconnect

  sm_2_sm_network(unsigned n_shader, const class shader_core_config* config,
                  const class gpgpu_sim* gpu);

  // void Init();
  virtual void Push(unsigned input_deviceID, unsigned output_deviceID,
                    void* data, unsigned int size,
                    Interconnect_type network) = 0;
  virtual void* Pop(unsigned ouput_deviceID, Interconnect_type network) = 0;
  virtual void Advance() = 0;
  virtual bool Busy() const = 0;
  virtual bool HasBuffer(unsigned deviceID, unsigned int size,
                         Interconnect_type network) const = 0;
  // virtual void DisplayStats() const;
  // virtual void DisplayOverallStats() const;
  // virtual unsigned GetFlitSize() const;

  // virtual void DisplayState(FILE* fp) const;

 protected:
  unsigned m_n_shader, m_n_mem;
  const class shader_core_config* m_config;
  const class gpgpu_sim* m_gpu;
  unsigned sid_to_gid(unsigned sid) const { return sid % m_n_shader; }
};

class local_crossbar : public sm_2_sm_network {
 public:
  local_crossbar(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu);
  ~local_crossbar();
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

 private:
  LocalInterconnect* m_localicnt_interface;

  std::ofstream m_request_net_in_log;
  std::ofstream m_request_net_out_log;
  std::ofstream m_reply_net_in_log;
  std::ofstream m_reply_net_out_log;
};

class Crossbar : public sm_2_sm_network {
 public:
  Crossbar(unsigned n_shader, const class shader_core_config* config,
           const class gpgpu_sim* gpu);
  ~Crossbar();
  void Init() {};
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);
  void Advance();
  bool Busy() const { return false; }
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const {
    return true;
  };

  struct Message {
    int src;
    int dst;
    int size;  // in bits
    int sent_bits = 0;
    uint64_t time_injected;
    void* data;
    Message(int s, int d, int sz, uint64_t t, void* data)
        : src(s), dst(d), size(sz), time_injected(t), data(data) {}
  };

 private:
  std::vector<std::queue<Message>> input_queues;
  std::vector<std::queue<void*>> output_queues;
  std::vector<std::pair<Message, uint64_t>> in_flight;
  std::vector<int> rr_pointers;
  uint64_t m_time;
  const int m_latency = 180;
  const int m_bandwidth = 11 * 8;
};

class ideal_network : public sm_2_sm_network {
 public:
  ideal_network(unsigned n_shader, const class shader_core_config* config,
                const class gpgpu_sim* gpu)
      : sm_2_sm_network(n_shader, config, gpu) {
    out_request.resize(n_shader);
    out_response.resize(n_shader);
    in_request.resize(n_shader);
    in_response.resize(n_shader);
  }
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);

  void Advance();
  bool Busy() const { return false; };
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const {
    return true;
  }
  void DisplayStats() const { ; };
  void DisplayOverallStats() const { ; };
  unsigned GetFlitSize() const { return 1; };

  void DisplayState(FILE* fp) const { ; };

 private:
  std::vector<std::queue<void*>> in_request;
  std::vector<std::queue<void*>> out_request;
  std::vector<std::queue<void*>> in_response;
  std::vector<std::queue<void*>> out_response;
};

class ringbus : public sm_2_sm_network {
 public:
  ringbus(unsigned n_shader, const class shader_core_config* config,
          const class gpgpu_sim* gpu);
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);

  void Advance();
  bool Busy() const { return false; };
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const;
  void DisplayStats() const { ; };
  void DisplayOverallStats() const { ; };
  unsigned GetFlitSize() const { return 1; };

  void DisplayState(FILE* fp) const { ; };

 private:
  struct Packet {
    Packet(void* m_data, unsigned m_output_deviceID) {
      data = m_data;
      output_deviceID = m_output_deviceID;
    }
    void* data;
    unsigned output_deviceID;
  };
  const unsigned m_ring_buffer_size = 1024;
  const unsigned m_in_out_buffer_size = 1024;
  bool m_bidirectional;
  std::array<std::vector<std::queue<Packet>>, 2> m_ring;
  std::array<std::vector<std::queue<Packet>>, 2> m_in;
  std::array<std::vector<std::queue<Packet>>, 2> m_out;
};

/// The SM-to-SM interconnect based on (micro-)benchmarks of the real H100.
class H100Model : public sm_2_sm_network {
 public:
  H100Model(unsigned n_shader, const class shader_core_config* config,
            const class gpgpu_sim* gpu);

  /// Due to manual memory management, explicitly define a destructor.
  ~H100Model();

  /// Rule of three: Disallow copy construction.
  H100Model(const H100Model& other) = delete;
  /// Rule of three: Disallow copy assignment.
  H100Model& operator=(const H100Model& other) = delete;

  /// Push a new packet onto the network.
  ///
  /// `input_deviceID` is the ID of the SM sending the packet
  /// `output_deviceID` is the ID of the SM where the packet is supposed to go
  /// `data` is the `cluster_shmem_request*` associated with this this packet.
  /// `size` TODO
  /// `network` indicates whether this packet is on the REQUEST_NET or
  /// REPLY_NET.
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);

  /// Retrieve a packet for the given deviceID.
  ///
  /// `output_deviceID` is the ID of the SM whose output-buffer we want to
  /// check. `network` indicates whether to query the REQUEST_NET or REPLY_NET.
  ///
  /// Returns: The `cluster_shmem_request*` associated with this packet.
  void* Pop(unsigned output_deviceID, Interconnect_type network);

  /// Advance the entire interconnect by one cycle.
  ///
  /// Will first advance all nodes (junctions) and then all pipes.
  void Advance();

  /// Returns true if there is at least one packet still in transit
  /// anywhere in the network, be it a junction or a pipe.
  bool Busy() const;

  /// Checks if the input buffer for SMID `deviceID` still has space for another
  /// packet.
  ///
  /// `size` indicates the size of the packet. Ignored in the H100Model as
  /// packets are discrete units. `network` differentates the REQUEST_NET and
  /// REPLY_NET, something the H100Model does not do.
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const;

 protected:
  // Forward delcarations.
  class Pipe;
  struct Packet;

  /// Class representing a single node within the interconnect.
  ///
  /// Can be either a processor or a junction.
  class Node {
   public:
    /// Initializes the node with the given ppc.
    Node(uint32_t packets_per_cycle);
    /// Requires a virtual destructor since it is abstract.
    virtual ~Node();

    /// Advance the node by one cycle. Implemented in subclass.
    virtual void Advance() = 0;

    /// All the pipes sending packets to this node.
    ///
    /// The order matters here as that is the order in which packets are
    /// processed.
    std::vector<Pipe*> incoming_pipes;
    /// All the pipes where this node can send packets.
    std::vector<Pipe*> outgoing_pipes;
    /// How many packets this node processes on each incoming pipe per cycle.
    uint32_t packets_per_cycle;
  };

  /// A junction within the interconnect.
  ///
  /// A junction is always an intermediate destination for any packet,
  /// and is used connect pipes in the interconnect and forward packets.
  class Junction : public Node {
   public:
    /// Advance the junction by one cycle.
    virtual void Advance();
    /// Constructor. Simply calls the base-class constructor.
    Junction(uint32_t packets_per_cycle);
  };

  /// A processor (SM) within the interconnect.
  ///
  /// Processors are the source / sink of packets, but do not perform any
  /// routing work. Only one bi-directional pipe should be connected to a
  /// processor, which connects it to the wider interconnect.
  class Processor : public Node {
   public:
    /// Advance the processor by one cycle.
    virtual void Advance();
    /// Constructor. Calls base-class constructor and copies over block_rank.
    Processor(uint32_t packets_per_cycle, uint32_t block_rank);
    /// The rank of this processor within the cluster. Within [0,16).
    uint32_t block_rank;
  };

  /// A single, uni-directional pipe within the interconnect.
  ///
  /// Forwards packets from the in_node to the out_node.
  class Pipe {
   public:
    Pipe(Node* in_node, Node* out_node, uint32_t packets_per_cycle,
         uint32_t buffer_capacity);

    /// Advance the pipe by one cycle,
    /// forwarind `packets_per_cycle` packets from the in_q to the out_q.
    void Advance();

    /// The node forwarding packets into this pipe.
    Node* in_node;
    /// The node to which packets are being forwarded to.
    Node* out_node;

    /// The queue of incoming packets limited by the pipe's `buffer_capacity`.
    std::deque<Packet> in_q;
    /// The queue of outgoing packets limited by the pipe's `buffer_capacity`.
    std::deque<Packet> out_q;

    /// Pointer to the pipe that points in the opposite direction.
    ///
    /// Pipes are conceptually single-directional. Bi-directional pipes are
    /// implemented by simply adding two pipes to the network, one for each
    /// direction. This pointer points to the reverse-direction counterpart, if
    /// it exists.
    Pipe* counterpart;

    /// How many packets are forwarded from the input to the output buffer each
    /// cycle.
    uint32_t packets_per_cycle;
    /// How many packets both the input and output buffer can hold.
    uint32_t buffer_capacity;

    /// This array contains routing hints for the network.
    /// If the boolean at the given index returns true,
    /// a packet can and should be forwarded through this pipe to reach the
    /// respective processor.
    bool reachable_processors[16];
  };

  /// A single packet being transmitted across the interconnect.
  struct Packet {
    /// Simple constructor
    Packet(uint32_t source_block_rank, uint32_t destination_block_rank,
           Interconnect_type associated_network, void* data);
    /// The block_rank (value in [0,16)) of the processor who sent this packet.
    uint32_t source_block_rank;
    /// The block_rank (value in [0,16)) of this packet's destination processor.
    uint32_t destination_block_rank;
    /// Whether this packet is associated with the REQUEST_NET or the REPLY_NET.
    /// The h100_model does not internally differentate between the two, every
    /// packet is sent across the same simulated network. Nonetheless, the code
    /// interfacing with the network expects to know whether a packet is
    /// associated with one or the other.
    Interconnect_type associated_network;
    /// The associated `cluster_shmem_request`, stored as a void* to be
    /// consistent with the interface provided by the sm_2_sm_network-baseclass.
    void* data;
  };

  /// List of all pipes in the interconnect.
  std::vector<Pipe*> pipe_list;
  /// List of all nodes (junctions and processors) in the interconnect.
  std::vector<Node*> node_list;

  /// Prints the entire network to stdout for debugging purposes.
  void print_network();
};

#endif  // SM_2_SM_NETWORK_H
