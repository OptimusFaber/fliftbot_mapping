-- Pure localization against a frozen .pbstream.  The input must be a
-- LaserScan on /scan plus a continuous /wheel_odom and valid TF.  The
-- odometry messages use header.frame_id=odom (the topic name is independent).
include "fliftbot_2d.lua"

-- Runtime tuning is passed by the ROS launch file through per-process
-- environment variables.  Cartographer reads trajectory/pose-graph options
-- only once, when the trajectory is created; these are not dynamic ROS
-- parameters.
local function env_number(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  local parsed = tonumber(value)
  if parsed == nil then error(name .. " must be a number, got '" .. value .. "'") end
  return parsed
end

local function env_bool(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  value = string.lower(value)
  if value == "1" or value == "true" or value == "yes" or value == "on" then return true end
  if value == "0" or value == "false" or value == "no" or value == "off" then return false end
  error(name .. " must be true or false, got '" .. value .. "'")
end

-- Restore the frame contract used by the validated live version.
options.map_frame = os.getenv("FLIFT_MAP_FRAME") or "map"
options.tracking_frame = os.getenv("FLIFT_TRACKING_FRAME") or "base_link"
options.odom_frame = os.getenv("FLIFT_ODOM_FRAME") or "odom"
options.published_frame = os.getenv("FLIFT_PUBLISHED_FRAME") or "odom"
options.provide_odom_frame = false

TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}
-- Every new node is tested against the frozen map until a global constraint
-- is found.  The upstream default samples only a tiny fraction and can leave
-- a fresh trajectory close to identity for a long time.
-- Ten matched scans per second are sufficient with encoder motion prediction
-- and keep this Jetson able to process the stream in real time.
options.rangefinder_sampling_ratio = env_number("FLIFT_RANGEFINDER_SAMPLING_RATIO", 0.5)
options.publish_to_tf = env_bool("FLIFT_PUBLISH_TO_TF", true)
options.publish_tracked_pose = env_bool("FLIFT_PUBLISH_TRACKED_POSE", false)
options.pose_publish_period_sec = env_number("FLIFT_POSE_PUBLISH_PERIOD_SEC", 0.05)
-- There is no Cartographer switch for the trajectory/submap publishers.
-- A one-hour default makes their periodic work effectively disabled while
-- keeping them available through launch arguments for RViz sessions.
options.trajectory_publish_period_sec = env_number(
  "FLIFT_TRAJECTORY_PUBLISH_PERIOD_SEC", 3600.0)
options.submap_publish_period_sec = env_number(
  "FLIFT_SUBMAP_PUBLISH_PERIOD_SEC", 3600.0)
-- Force a node at least twice per second while stationary, so initial global
-- acquisition does not depend on robot motion.  Global matching and graph
-- optimization are sampled/batched to bound CPU use.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = env_number(
  "FLIFT_MOTION_FILTER_MAX_TIME_SEC", 0.5)
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = env_number(
  "FLIFT_MOTION_FILTER_MAX_DISTANCE_M", 0.2)
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(env_number(
  "FLIFT_MOTION_FILTER_MAX_ANGLE_DEG", 1.0))
POSE_GRAPH.global_sampling_ratio = env_number("FLIFT_GLOBAL_SAMPLING_RATIO", 0.2)
POSE_GRAPH.optimize_every_n_nodes = env_number("FLIFT_OPTIMIZE_EVERY_N_NODES", 10)
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
POSE_GRAPH.constraint_builder.log_matches = env_bool("FLIFT_LOG_MATCHES", false)
POSE_GRAPH.log_residual_histograms = env_bool("FLIFT_LOG_RESIDUAL_HISTOGRAMS", false)
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = env_bool(
  "FLIFT_USE_ONLINE_CORRELATIVE_SCAN_MATCHING", true)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = env_number(
  "FLIFT_CORRELATIVE_LINEAR_WINDOW_M", 0.15)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(env_number(
  "FLIFT_CORRELATIVE_ANGULAR_WINDOW_DEG", 10.0))

return options
