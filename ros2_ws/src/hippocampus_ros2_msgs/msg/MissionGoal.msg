# Mission goal message
# Represents a navigation goal for the policy system

# Goal type
uint8 GOAL_TYPE_POINT = 0
uint8 GOAL_TYPE_NODE = 1
uint8 GOAL_TYPE_REGION = 2

uint8 goal_type           # Type of goal (POINT, NODE, REGION)

# Point goal (for GOAL_TYPE_POINT)
geometry_msgs/Point point_position
float64 point_tolerance  # Distance tolerance (m)

# Node goal (for GOAL_TYPE_NODE)
int32 node_id            # Graph node ID

# Region goal (for GOAL_TYPE_REGION)
geometry_msgs/Point region_center
float64 region_radius   # Region radius (m)

# Goal metadata
string frame_id          # Coordinate frame
float64 timeout          # Timeout in seconds (0.0 = no timeout)
bool is_reached          # Goal reached flag

# Timestamp
builtin_interfaces/Time stamp

