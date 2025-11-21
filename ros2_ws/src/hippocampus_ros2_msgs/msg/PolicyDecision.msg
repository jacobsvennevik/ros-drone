# Policy decision message
# Represents a decision made by the SNN Policy Service

# Action proposal
float64 linear_x          # Linear velocity (m/s)
float64 angular_z        # Angular velocity (rad/s)
float64 linear_z         # Vertical velocity (m/s) for 3D, 0.0 for 2D

# Decision metadata
float64 confidence        # Confidence score [0.0, 1.0]
string reason             # Decision reason ("heuristic", "snn", etc.)
int32 next_waypoint       # Next waypoint node ID (-1 if none)

# Timestamp
builtin_interfaces/Time stamp

