# Graph snapshot message
# Represents a snapshot of the topological graph

# Graph metadata
uint32 epoch_id          # Graph epoch/version ID
string frame_id          # Coordinate frame
builtin_interfaces/Time stamp
builtin_interfaces/Time last_updated
float64 update_rate      # Graph update rate (Hz)
bool staleness_warning   # Staleness warning flag

# Nodes
hippocampus_ros2_msgs/GraphNode[] nodes

# Edges
hippocampus_ros2_msgs/GraphEdge[] edges
