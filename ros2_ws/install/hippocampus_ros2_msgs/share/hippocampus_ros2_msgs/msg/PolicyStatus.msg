# Policy status message
# Diagnostics and status information for the policy system

# Status flags
bool is_active            # Policy is active and making decisions
bool graph_stale          # Topological graph is stale
bool using_snn            # Currently using SNN (vs heuristic)
bool hierarchical_enabled # Hierarchical planning enabled

# Performance metrics
float64 feature_compute_time_ms    # Feature computation time (ms)
float64 policy_decision_time_ms    # Policy decision time (ms)
float64 safety_filter_time_ms      # Safety filtering time (ms)
float64 total_latency_ms           # Total latency (ms)

# Graph information
uint32 graph_nodes        # Number of nodes in graph
uint32 graph_edges        # Number of edges in graph
float64 graph_staleness_s # Time since last graph update (seconds)

# Current state
float64 current_confidence # Current decision confidence
string current_reason      # Current decision reason
int32 current_waypoint     # Current waypoint node ID (-1 if none)

# Timestamp
builtin_interfaces/Time stamp

