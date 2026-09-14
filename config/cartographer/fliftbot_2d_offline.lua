-- Standalone offline profile. The recorded wheel stream is the useful motion
-- prior for sparse/partial RPLIDAR packets, while local correlative matching
-- protects against small encoder quantization errors.
include "fliftbot_2d.lua"
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.30
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(25.)
POSE_GRAPH.optimization_problem.odometry_translation_weight = 5.0
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 5.0
return options
