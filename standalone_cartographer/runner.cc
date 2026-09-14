// Offline Cartographer runner for the canonical fliftbot recording format.
// This executable intentionally has no ROS dependency: it feeds Cartographer's
// MapBuilder directly from horizon/*.npz and encoder.jsonl.
#include <zip.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "Eigen/Core"
#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/time.h"
#include "cartographer/io/proto_stream.h"
#include "cartographer/io/proto_stream_deserializer.h"
#include "cartographer/io/submap_painter.h"
#include "cartographer/mapping/map_builder.h"
#include "cartographer/mapping/map_builder_interface.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/timed_point_cloud_data.h"
#include "cartographer/transform/rigid_transform.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

namespace fs = std::filesystem;
using cartographer::common::FromSeconds;
using cartographer::common::FromUniversal;
using cartographer::common::Time;
using cartographer::mapping::TrajectoryBuilderInterface;

DEFINE_string(run_dir, "", "Recording directory containing horizon/ and encoder.jsonl.");
DEFINE_string(output, "", "Output directory (default: RUN_DIR/cartographer_standalone).");
DEFINE_string(configuration_dir, "", "Directory containing the Lua configuration.");
DEFINE_string(configuration_file, "fliftbot_2d.lua", "Lua configuration basename.");
DEFINE_double(resolution, 0.05, "Output map resolution in metres per pixel.");
DEFINE_bool(use_odometry, true, "Feed encoder-derived odometry to Cartographer.");
DEFINE_double(wheelbase, 1.410,
              "Distance in metres from the rotation axis to the driven/steered wheel.");
DEFINE_double(steering_offset_deg, 0.0,
              "Additive steering encoder zero offset in degrees.");
DEFINE_double(encoder_speed_sign, 0.0,
              "Encoder speed sign override (+1 or -1); 0 selects the "
              "recording/lidar-type default.");
DEFINE_double(odometry_time_offset, 0.0,
              "Seconds added to encoder timestamps when feeding odometry.");
DEFINE_double(max_range, 30.0,
              "Cartographer maximum LiDAR range in metres. Points beyond "
              "this range are retained until Cartographer crops them into "
              "misses at the configured range.");
DEFINE_int32(max_scans, 0, "Process at most this many scans; 0 means all.");
DEFINE_bool(skip_bad_frames, true, "Skip malformed NPZ files instead of aborting.");
DEFINE_string(points_frame, "auto", "NPZ xy frame: auto, sensor, or tracking (body/rotation_axis).");
DEFINE_string(tracking_frame, "auto",
              "Standalone Cartographer tracking frame: auto, lidar, or "
              "rotation_axis. Auto selects lidar for sensor-frame NPZ and "
              "rotation_axis for already transformed body clouds.");
DEFINE_bool(synthesize_rplidar_times, false,
            "Reconstruct per-beam RPLIDAR times when NPZ t is collapsed. "
            "Only collapsed arrays are changed; real per-point timestamps are "
            "always retained. Disabled by default for legacy HTTP files because "
            "their frame timestamp semantic is not recorded; enable it for a "
            "verified start/end timestamp contract.");
DEFINE_string(rplidar_timestamp_convention, "auto",
              "Collapsed RPLIDAR stamp convention: auto, start, or end. "
              "auto uses end (HTTP frame timestamp is observed after a full "
              "revolution); use start for a driver that timestamps the first ray.");

namespace {

struct NpyArray {
  std::string descr;
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
};

struct Scan {
  double stamp = 0.;
  int frame = 0;
  std::vector<float> xy;
  std::vector<double> point_times;
  bool point_times_collapsed = true;
};

struct RplidarTimingStats {
  std::string mode = "not_applicable";
  std::string convention = "not_applicable";
  size_t synthesized_scans = 0;
  double period_s = 0.;
  std::string direction = "unknown";
};

struct Encoder {
  double time = 0.;
  double velocity = 0.;
  double steering_deg = 0.;
};

struct Extrinsics {
  double x = 0., y = 0., z = 0., yaw = 0.;
  double encoder_speed_sign = 0.;
  bool apply = true;
  std::string point_frame = "sensor";
  std::string lidar_type;
};

struct Event {
  double unix_time;
  enum Kind { ODOM, RANGE } kind;
  size_t index;
  bool operator<(const Event& other) const {
    if (unix_time != other.unix_time) return unix_time < other.unix_time;
    return kind < other.kind;  // odometry first at an equal timestamp
  }
};

std::string ReadZipEntry(zip_t* archive, const std::string& name) {
  zip_stat_t stat;
  zip_stat_init(&stat);
  if (zip_stat(archive, name.c_str(), 0, &stat) != 0) {
    throw std::runtime_error("NPZ entry is missing: " + name);
  }
  zip_file_t* file = zip_fopen(archive, name.c_str(), 0);
  if (!file) throw std::runtime_error("cannot open NPZ entry: " + name);
  std::string result(static_cast<size_t>(stat.size), '\0');
  zip_int64_t got = zip_fread(file, result.data(), result.size());
  zip_fclose(file);
  if (got != static_cast<zip_int64_t>(result.size())) {
    throw std::runtime_error("short read from NPZ entry: " + name);
  }
  return result;
}

uint16_t U16(const std::string& s, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(s[offset])) |
         (static_cast<uint16_t>(static_cast<unsigned char>(s[offset + 1])) << 8);
}

NpyArray ParseNpy(const std::string& bytes) {
  if (bytes.size() < 12 || bytes.compare(0, 6, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("invalid NPY magic");
  const int major = static_cast<unsigned char>(bytes[6]);
  size_t header_size = 0, header_offset = 0;
  if (major == 1) {
    header_size = U16(bytes, 8); header_offset = 10;
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 16) throw std::runtime_error("short NPY v2 header");
    header_size = static_cast<uint32_t>(static_cast<unsigned char>(bytes[8])) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(bytes[9])) << 8) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(bytes[10])) << 16) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(bytes[11])) << 24);
    header_offset = 12;
  } else throw std::runtime_error("unsupported NPY version");
  if (header_offset + header_size > bytes.size()) throw std::runtime_error("short NPY header");
  const std::string header = bytes.substr(header_offset, header_size);
  std::smatch match;
  if (!std::regex_search(header, match, std::regex("'descr'\\s*:\\s*'([^']+)'")))
    throw std::runtime_error("NPY descr is missing");
  NpyArray out; out.descr = match[1].str();
  if (out.descr.size() < 3 || out.descr[0] == '>') throw std::runtime_error("big-endian NPY is unsupported");
  if (!std::regex_search(header, match, std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)")))
    throw std::runtime_error("NPY shape is missing");
  std::stringstream shape(match[1].str()); std::string part;
  while (std::getline(shape, part, ',')) {
    std::stringstream p(part); size_t value;
    if (p >> value) out.shape.push_back(value);
  }
  out.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(header_offset + header_size), bytes.end());
  size_t item_size = std::stoul(out.descr.substr(2));
  size_t count = 1; for (size_t n : out.shape) count *= n;
  if (item_size == 0 || count > std::numeric_limits<size_t>::max() / item_size || out.data.size() != count * item_size)
    throw std::runtime_error("NPY payload size does not match shape");
  return out;
}

NpyArray LoadNpzArray(const fs::path& path, const std::string& name) {
  int error = 0; zip_t* archive = zip_open(path.c_str(), ZIP_RDONLY, &error);
  if (!archive) throw std::runtime_error("cannot open NPZ: " + path.string());
  NpyArray array;
  try { array = ParseNpy(ReadZipEntry(archive, name + ".npy")); }
  catch (...) { zip_close(archive); throw; }
  zip_close(archive); return array;
}

template <typename T> T ReadScalar(const NpyArray& a, size_t i) {
  T value; std::memcpy(&value, a.data.data() + i * sizeof(T), sizeof(T)); return value;
}

double JsonNumber(const std::string& line, const std::string& key, double fallback = std::numeric_limits<double>::quiet_NaN()) {
  const std::string needle = "\"" + key + "\"";
  const size_t key_pos = line.find(needle);
  if (key_pos == std::string::npos) return fallback;
  const size_t colon = line.find(':', key_pos + needle.size());
  if (colon == std::string::npos) return fallback;
  const char* begin = line.c_str() + colon + 1;
  char* end = nullptr;
  while (*begin == ' ' || *begin == '\t') ++begin;
  const double value = std::strtod(begin, &end);
  return end == begin ? fallback : value;
}

std::vector<Encoder> LoadEncoders(const fs::path& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open " + path.string());
  std::vector<Encoder> result; std::string line;
  while (std::getline(in, line)) {
    Encoder e{JsonNumber(line, "enc_unix"), JsonNumber(line, "enc_v", 0.), JsonNumber(line, "enc_dgr", 0.)};
    if (std::isfinite(e.time) && std::isfinite(e.velocity) && std::isfinite(e.steering_deg)) result.push_back(e);
  }
  std::sort(result.begin(), result.end(), [](const Encoder& a, const Encoder& b) { return a.time < b.time; });
  // Some recorder sessions contain the same encoder sample twice. Cartographer
  // stores odometry in a strictly-increasing MapByTime, so equal timestamps
  // must be collapsed before feeding the trajectory.
  result.erase(std::unique(result.begin(), result.end(), [](const Encoder& a, const Encoder& b) {
    return std::abs(a.time - b.time) < 1e-7;
  }), result.end());
  return result;
}

Extrinsics LoadExtrinsics(const fs::path& path) {
  Extrinsics e;
  std::ifstream in(path);
  if (!in) return e;
  // Do not search the whole document for x/y/z: lidar.config also contains
  // coordinate_system.x/y/z strings and (for multi-lidar recordings) nested
  // sensors.  The canonical mount is explicitly position_m plus
  // orientation_deg at the document root.
  try {
    const nlohmann::json root = nlohmann::json::parse(in);
    const auto position = root.value("position_m", nlohmann::json::object());
    const auto orientation = root.value("orientation_deg", nlohmann::json::object());
    if (!position.is_object()) throw std::runtime_error("position_m must be an object");
    if (!orientation.is_object()) throw std::runtime_error("orientation_deg must be an object");
    e.x = position.value("x", 0.0);
    e.y = position.value("y", 0.0);
    e.z = position.value("z", 0.0);
    // v1 uses yaw, but accept yaw_deg as a backwards-compatible spelling.
    const double yaw_deg = orientation.contains("yaw_deg")
                               ? orientation.value("yaw_deg", 0.0)
                               : orientation.value("yaw", 0.0);
    e.yaw = yaw_deg * M_PI / 180.;
    const std::string configured_frame = root.value("points_frame", "");
    const std::string lidar_type = root.value("lidar_type", "");
    // The 2026-09 RPLIDAR HTTP recordings were validated against nav and use
    // enc_v directly (+1).  Older DePHAN/Pandar streams retain the project
    // body convention (-1) unless a recording explicitly overrides it.
    e.encoder_speed_sign = root.value(
        "encoder_speed_sign", lidar_type == "rplidar" ? 1.0 : -1.0);
    e.lidar_type = lidar_type;
    std::string frame = configured_frame;
    if (frame.empty()) {
      // Root `horizon/*.npz` produced by the robot runtime stores the
      // combined/body cloud.  The raw M360 converter stores sensor-local xy;
      // those DePHAN recordings have lidar_type=dephan_m360.  New recordings
      // should write points_frame explicitly; this fallback only preserves the
      // two existing canonical formats.
      frame = lidar_type == "rplidar" ? "tracking" : "sensor";
    }
    std::transform(frame.begin(), frame.end(), frame.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (FLAGS_points_frame != "auto") frame = FLAGS_points_frame;
    if (frame != "sensor" && frame != "tracking" && frame != "body" && frame != "rotation_axis") {
      throw std::runtime_error("points_frame must be auto, sensor, or tracking/body");
    }
    e.point_frame = frame;
    e.apply = frame == "sensor";
    if (FLAGS_encoder_speed_sign != 0.0) {
      e.encoder_speed_sign = FLAGS_encoder_speed_sign;
    }
    if (!std::isfinite(e.x) || !std::isfinite(e.y) ||
        !std::isfinite(e.z) || !std::isfinite(e.yaw) ||
        !std::isfinite(e.encoder_speed_sign) ||
        (std::abs(e.encoder_speed_sign) != 1.0)) {
      throw std::runtime_error("non-finite lidar extrinsics");
    }
  } catch (const std::exception& error) {
    throw std::runtime_error("cannot parse lidar.config " + path.string() + ": " + error.what());
  }
  return e;
}

Scan LoadScan(const fs::path& path) {
  const auto xy = LoadNpzArray(path, "xy");
  if (xy.descr != "<f4" || xy.shape.size() != 2 || xy.shape[1] != 2) throw std::runtime_error("xy must be little-endian float32 Nx2: " + path.string());
  NpyArray times; bool have_times = true;
  try { times = LoadNpzArray(path, "t"); } catch (...) { have_times = false; }
  const auto stamp_arr = LoadNpzArray(path, "stamp");
  if (stamp_arr.descr != "<f8" ||
      (stamp_arr.shape.size() == 1 && stamp_arr.shape[0] != 1) ||
      (stamp_arr.shape.size() > 1)) {
    throw std::runtime_error("stamp must be a scalar float64: " + path.string());
  }
  Scan scan; scan.stamp = ReadScalar<double>(stamp_arr, 0);
  const auto frame_arr = LoadNpzArray(path, "frame");
  if (frame_arr.descr == "<i4") scan.frame = ReadScalar<int32_t>(frame_arr, 0);
  const size_t n = xy.shape[0]; scan.xy.reserve(n * 2); scan.point_times.reserve(n);
  if (have_times && (times.descr != "<f8" || times.shape.size() != 1 || times.shape[0] != n)) {
    throw std::runtime_error("t must be a float64 vector with one value per xy point: " + path.string());
  }
  for (size_t i = 0; i < n; ++i) {
    const double lx = ReadScalar<float>(xy, i * 2), ly = ReadScalar<float>(xy, i * 2 + 1);
    const double range = std::hypot(lx, ly);
    // Do not pre-filter returns above max_range here. Cartographer's native
    // CropRangeData semantics preserve those beams as misses at max_range;
    // dropping them before AddSensorData changes the free-space evidence and
    // is especially harmful for recordings where valid room walls are 8--12m
    // away. The CLI max_range is applied to the trajectory options below.
    if (!std::isfinite(range) || range < 0.15) continue;
    // Keep the point in its recorded sensor frame.  Cartographer's
    // TimedPointCloudData contract is the same one used by cartographer_ros:
    // `origin` is the sensor origin in tracking frame and `ranges` are rotated
    // into tracking frame exactly once by the sensor bridge equivalent.  The
    // runner supplies that equivalent at AddSensorData time below.  Keeping
    // raw xy here also preserves the recorded order and bearing.
    scan.xy.push_back(static_cast<float>(lx));
    scan.xy.push_back(static_cast<float>(ly));
    const double pt = have_times && times.descr == "<f8" && times.shape.size() == 1 ? ReadScalar<double>(times, i) : scan.stamp;
    scan.point_times.push_back(std::isfinite(pt) ? pt : scan.stamp);
  }
  if (scan.xy.size() < 10) throw std::runtime_error("too few valid points: " + path.string());
  double minimum_time = std::numeric_limits<double>::infinity();
  double maximum_time = -std::numeric_limits<double>::infinity();
  size_t finite_time_count = 0;
  for (const double point_time : scan.point_times) {
    if (std::isfinite(point_time)) {
      minimum_time = std::min(minimum_time, point_time);
      maximum_time = std::max(maximum_time, point_time);
      ++finite_time_count;
    }
  }
  // A collapsed array is not evidence that all beams were acquired at the
  // same instant: the HTTP RPLIDAR endpoint writes one frame timestamp into
  // every point.  Keep this fact explicit so synthesis cannot accidentally
  // overwrite genuine Pandar/Hesai timestamps.
  scan.point_times_collapsed =
      finite_time_count < 2 ||
      !std::isfinite(minimum_time) ||
      maximum_time - minimum_time < 1.e-6;
  double latest = -std::numeric_limits<double>::infinity();
  for (const double point_time : scan.point_times) {
    if (std::isfinite(point_time)) latest = std::max(latest, point_time);
  }
  if (std::isfinite(latest)) scan.stamp = latest;
  if (!std::isfinite(scan.stamp)) throw std::runtime_error("scan has no finite timestamp: " + path.string());
  return scan;
}

// The robot HTTP RPLIDAR endpoint exposes a single timestamp for a complete
// revolution.  The recorder consequently writes identical values to NPZ[t],
// although the samples are still in firing order. Recover the timing model
// used by LaserScan: a revolution takes the median interval between successive
// frames and bearing phase determines each ray time. The legacy HTTP timestamp
// convention is selectable because old recordings did not say whether it
// denotes the first or last ray. Bearing differences are measured around the
// actual sensor origin, which matters for body-frame RPLIDAR clouds.
RplidarTimingStats SynthesizeRplidarTimes(
    std::vector<Scan>* scans, const Extrinsics& ext,
    const std::string& requested_convention) {
  RplidarTimingStats stats;
  if (scans->size() < 2 || ext.lidar_type != "rplidar") return stats;
  if (requested_convention != "auto" && requested_convention != "start" &&
      requested_convention != "end") {
    throw std::runtime_error(
        "--rplidar_timestamp_convention must be auto, start, or end");
  }
  // No timing can be recovered for a genuinely single-timestamp scan without
  // making up a period.  Synthesis is therefore restricted to the known HTTP
  // RPLIDAR contract, identified by collapsed NPZ[t].
  size_t collapsed_scans = 0;
  for (const auto& scan : *scans) collapsed_scans += scan.point_times_collapsed;
  if (collapsed_scans == 0) {
    stats.mode = "recorded_per_point";
    return stats;
  }
  std::vector<double> frame_deltas;
  for (size_t i = 1; i < scans->size(); ++i) {
    const double dt = (*scans)[i].stamp - (*scans)[i - 1].stamp;
    if (dt > 0.05 && dt < 1.0 && std::isfinite(dt)) frame_deltas.push_back(dt);
  }
  if (frame_deltas.empty()) return stats;
  std::nth_element(frame_deltas.begin(),
                   frame_deltas.begin() + frame_deltas.size() / 2,
                   frame_deltas.end());
  const double period = frame_deltas[frame_deltas.size() / 2];
  const bool stamp_is_end = requested_convention == "end" ||
                            requested_convention == "auto";
  stats.mode = "azimuth_period_synthesis";
  stats.convention = stamp_is_end ? "end" : "start";
  stats.period_s = period;
  double direction_sum = 0.;
  for (const auto& scan : *scans) {
    if (scan.xy.size() < 20) continue;
    double previous = 0.;
    bool have_previous = false;
    for (size_t i = 0; i < scan.xy.size() / 2; ++i) {
      const double sx = static_cast<double>(scan.xy[2 * i]) - ext.x;
      const double sy = static_cast<double>(scan.xy[2 * i + 1]) - ext.y;
      const double angle = std::atan2(
          -std::sin(ext.yaw) * sx + std::cos(ext.yaw) * sy,
          std::cos(ext.yaw) * sx + std::sin(ext.yaw) * sy);
      if (have_previous) {
        double delta = angle - previous;
        while (delta > M_PI) delta -= 2. * M_PI;
        while (delta < -M_PI) delta += 2. * M_PI;
        // Invalid returns create gaps, but not jumps near a half turn.
        if (std::abs(delta) < 0.5) direction_sum += delta;
      }
      previous = angle;
      have_previous = true;
    }
  }
  if (std::abs(direction_sum) < 1e-6) return stats;
  const double direction = direction_sum < 0. ? -1. : 1.;
  stats.direction = direction < 0. ? "clockwise" : "counter_clockwise";
  size_t synthesized = 0;
  for (auto& scan : *scans) {
    if (!scan.point_times_collapsed || scan.point_times.size() < 20) continue;
    const size_t scan_index = static_cast<size_t>(&scan - scans->data());
    const double frame_stamp = scan.stamp;
    // HTTP frames are not perfectly periodic.  Using one global median can
    // make neighbouring end-stamped sweeps overlap, which makes Cartographer
    // drop otherwise valid points as out-of-order.  Bound every reconstructed
    // sweep by both adjacent frame intervals; this preserves the measured
    // period while keeping successive sweep intervals non-overlapping.
    double scan_period = period;
    if (scan_index > 0) {
      const double previous_dt = frame_stamp - (*scans)[scan_index - 1].stamp;
      if (previous_dt > 0.01 && previous_dt < 1.0 &&
          std::isfinite(previous_dt)) {
        scan_period = std::min(scan_period, previous_dt);
      }
    }
    if (scan_index + 1 < scans->size()) {
      const double next_dt = (*scans)[scan_index + 1].stamp - frame_stamp;
      if (next_dt > 0.01 && next_dt < 1.0 && std::isfinite(next_dt)) {
        scan_period = std::min(scan_period, next_dt);
      }
    }
    double first_angle = 0., previous = 0., unwrapped = 0.;
    double previous_phase = 0.;
    bool have_previous = false;
    for (size_t i = 0; i < scan.point_times.size(); ++i) {
      const double sx = static_cast<double>(scan.xy[2 * i]) - ext.x;
      const double sy = static_cast<double>(scan.xy[2 * i + 1]) - ext.y;
      const double angle = std::atan2(
          -std::sin(ext.yaw) * sx + std::cos(ext.yaw) * sy,
          std::cos(ext.yaw) * sx + std::sin(ext.yaw) * sy);
      if (!have_previous) {
        first_angle = previous = angle;
      } else {
        double delta = angle - previous;
        while (delta > M_PI) delta -= 2. * M_PI;
        while (delta < -M_PI) delta += 2. * M_PI;
        if (std::abs(delta) >= 0.5) delta = 0.;
        unwrapped += delta;
        previous = angle;
      }
      (void)first_angle;
      double phase = std::clamp(direction * unwrapped / (2. * M_PI), 0., 1.);
      // Bearing quantisation and filtered returns can make adjacent azimuths
      // wobble by a few milliradians.  Cartographer's range-data collator
      // requires point times to be non-decreasing; without this clamp it
      // silently discards those beams as "earlier points" (often 5--10% of
      // an RPLIDAR frame), producing broken walls even though the global
      // sweep direction is correct.  Clamping only removes that measurement
      // jitter and preserves the deterministic azimuth/period model.
      phase = std::max(previous_phase, phase);
      previous_phase = phase;
      // Cartographer's TimedPointCloudData stamp is the latest point time.
      // If the HTTP timestamp marks the end of the revolution, the first ray
      // is one period earlier and all offsets are <= 0.  A start timestamp is
      // retained for explicit driver contracts and is advanced to the sweep
      // end after assigning the beam times.
      const double first_stamp = stamp_is_end
                                     ? frame_stamp - scan_period
                                     : frame_stamp;
      scan.point_times[i] = first_stamp + phase * scan_period;
      have_previous = true;
    }
    double latest = scan.point_times.front();
    for (const double t : scan.point_times) latest = std::max(latest, t);
    scan.stamp = latest;
    ++stats.synthesized_scans;
    ++synthesized;
  }
  std::cerr << "synthesized RPLIDAR per-beam times for " << synthesized
            << " scans (period=" << period << " s, direction="
            << (direction < 0. ? "clockwise" : "counter-clockwise") << ")\n";
  return stats;
}

bool IsForbiddenInputName(const fs::path& path) {
  std::string name = path.filename().string();
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  // The launcher performs the full recording audit.  Keep this small runtime
  // guard here as well so invoking the binary directly cannot accidentally
  // ingest an evaluation/navigation artifact copied into horizon/.
  return name.find("nav_") != std::string::npos ||
         name.find("_nav") != std::string::npos ||
         name.find("_gt") != std::string::npos ||
         name.find("gt_") != std::string::npos ||
         name.find("floorplan") != std::string::npos ||
         name.find("alignment") != std::string::npos ||
         name.find("groundtruth") != std::string::npos ||
         name.find("ground_truth") != std::string::npos;
}

Time CartoTime(double unix_seconds) {
  const double universal_seconds = unix_seconds + static_cast<double>(cartographer::common::kUtsEpochOffsetFromUnixEpochInSeconds);
  return FromUniversal(static_cast<int64_t>(std::llround(universal_seconds * 10000000.)));
}

// Distance from the rear rotation axis to the drive wheel. This is the
// project's calibrated WHEELBASE_M (robot_geometry.DRIVE_WHEEL_X_M).
std::vector<double> IntegrateOdom(const std::vector<Encoder>& encoders,
                                  double wheelbase,
                                  double steering_offset,
                                  double speed_sign) {
  std::vector<double> poses(encoders.size() * 3, 0.);
  constexpr double steering_sign = 1.;
  // Keep this aligned with ENCODER_STATIC_V in mapping/config.py.  The old
  // 0.08 m/s value silently discarded about 2 m of valid low-speed motion in
  // the 160617 recording and made the odometry prior frame-dependent.
  constexpr double static_speed = 0.008;
  for (size_t i = 1; i < encoders.size(); ++i) {
    double dt = encoders[i].time - encoders[i - 1].time;
    if (!(dt > 0. && dt <= .5)) continue;
    const double speed = speed_sign * .5 *
                         (encoders[i].velocity + encoders[i - 1].velocity);
    const double steering = steering_sign * .5 * (encoders[i].steering_deg + encoders[i - 1].steering_deg) * M_PI / 180. + steering_offset;
    const double wz = speed * std::sin(steering) / wheelbase, theta = wz * dt;
    const double vx = speed * std::cos(steering);
    double dx = std::abs(wz) < 1e-9 ? vx * dt : vx * std::sin(theta) / wz;
    double dy = std::abs(wz) < 1e-9 ? 0. : vx * (1. - std::cos(theta)) / wz;
    const double x = poses[(i - 1) * 3], y = poses[(i - 1) * 3 + 1], yaw = poses[(i - 1) * 3 + 2];
    if (std::abs(encoders[i].velocity) < static_speed &&
        std::abs(encoders[i - 1].velocity) < static_speed &&
        std::abs(theta) < M_PI / 3600.) {
      dx = dy = 0.;
    }
    poses[i * 3] = x + std::cos(yaw) * dx - std::sin(yaw) * dy;
    poses[i * 3 + 1] = y + std::sin(yaw) * dx + std::cos(yaw) * dy;
    poses[i * 3 + 2] = yaw + theta;
  }
  return poses;
}

std::unique_ptr<cartographer::common::LuaParameterDictionary> LoadLua(const fs::path& dir, const std::string& file) {
  std::vector<std::string> dirs{dir.string()};
  auto resolver = absl::make_unique<cartographer::common::ConfigurationFileResolver>(dirs);
  const std::string code = resolver->GetFileContentOrDie(file);
  return cartographer::common::LuaParameterDictionary::NonReferenceCounted(code, std::move(resolver));
}

void ExportMapAndTrajectory(const fs::path& output, double resolution,
                            const std::string& tracking_frame,
                            cartographer::mapping::MapBuilderInterface& builder) {
  cartographer::io::ProtoStreamReader reader((output / "map.pbstream").string());
  cartographer::io::ProtoStreamDeserializer deserializer(&reader);
  std::map<cartographer::mapping::SubmapId, cartographer::io::SubmapSlice> slices;
  cartographer::mapping::ValueConversionTables tables;
  cartographer::io::DeserializeAndFillSubmapSlices(&deserializer, &slices, &tables);
  if (slices.empty()) throw std::runtime_error("Cartographer produced no submap textures");
  auto painted = cartographer::io::PaintSubmapSlices(slices, resolution);
  cairo_surface_t* surface = painted.surface.get();
  cairo_surface_flush(surface);
  const int width = cairo_image_surface_get_width(surface), height = cairo_image_surface_get_height(surface), stride = cairo_image_surface_get_stride(surface);
  const auto* pixels = cairo_image_surface_get_data(surface);
  if (cairo_surface_write_to_png(surface, (output / "map.png").c_str()) != CAIRO_STATUS_SUCCESS) throw std::runtime_error("cannot write map.png");
  std::ofstream pgm(output / "map.pgm", std::ios::binary);
  pgm << "P5\n" << width << " " << height << "\n255\n";
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const uint8_t* p = pixels + y * stride + x * 4;
    const uint8_t alpha = p[3], intensity = p[2];
    const uint8_t value = alpha == 0 ? 205 : intensity;
    pgm.put(static_cast<char>(value));
  }
  // PaintSubmapSlices returns the top-left pixel in Cartographer's map
  // frame.  The PGM rows are written top-to-bottom, whereas a ROS map.yaml
  // origin is the lower-left cell.  The x origin therefore stays
  // -painted.origin.x() * resolution, but Y must account for the raster
  // height.  This is the same conversion used by Cartographer's canonical
  // pbstream_to_ros_map exporter:
  //   origin_y = (painted.origin.y() - height) * resolution.
  // Omitting the height term shifts a 671-pixel map by 13.85 m and makes the
  // otherwise same-frame global trajectory appear in unknown space.
  const double map_origin_x = -painted.origin.x() * resolution;
  const double map_origin_y = (painted.origin.y() - height) * resolution;
  std::ofstream yaml(output / "map.yaml");
  yaml << "image: map.pgm\nresolution: " << std::setprecision(9) << resolution << "\norigin: [" << map_origin_x << ", " << map_origin_y << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n";
  std::ofstream trajectory(output / "trajectory.json");
  trajectory << "{\n  \"schema\": \"fliftbot.cartographer_trajectory.v1\",\n"
              << "  \"tracking_frame\": \"" << tracking_frame << "\",\n"
              << "  \"map_frame\": {\n"
              << "    \"trajectory_pose_frame\": \"cartographer_map\",\n"
              << "    \"occupancy_grid_origin_xy_m\": [" << map_origin_x << ", " << map_origin_y << "],\n"
              << "    \"occupancy_grid_yaw_rad\": 0.0,\n"
              << "    \"occupancy_grid_resolution_m\": " << resolution << ",\n"
              << "    \"occupancy_grid_size_px\": [" << width << ", " << height << "],\n"
              << "    \"cartographer_map_to_occupancy_grid_se2\": {\n"
              << "      \"rotation_rad\": 0.0,\n"
              << "      \"translation_xy_m\": [" << -map_origin_x << ", " << -map_origin_y << "],\n"
              << "      \"meaning\": \"continuous occupancy-grid coordinates are R(-yaw) * (map_xy - origin_xy); map.yaml origin is the grid lower-left pose\"\n"
              << "    },\n"
              << "    \"raster_y_convention\": \"PGM row 0 is the top row; display/world conversion uses row = height - 1 - y_grid/resolution\"\n"
              << "  },\n"
              << "  \"poses\": [\n";
  bool first = true;
  for (const auto& entry : builder.pose_graph()->GetTrajectoryNodePoses()) {
    const auto& pose = entry.data;
    if (!pose.constant_pose_data) continue;
    const auto t = pose.global_pose.translation();
    const auto q = pose.global_pose.rotation();
    if (!first) trajectory << ",\n";
    first = false;
    trajectory << "    {\"trajectory_id\": " << entry.id.trajectory_id << ", \"node_index\": " << entry.id.node_index
               << ", \"timestamp_universal\": " << cartographer::common::ToUniversal(pose.constant_pose_data->time)
               << ", \"x\": " << t.x() << ", \"y\": " << t.y() << ", \"z\": " << t.z()
               << ", \"qx\": " << q.x() << ", \"qy\": " << q.y() << ", \"qz\": " << q.z() << ", \"qw\": " << q.w() << "}";
  }
  trajectory << "\n  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); FLAGS_logtostderr = true;
  google::SetUsageMessage("Build a Cartographer map directly from horizon NPZ + encoder.jsonl (no ROS/rosbag).\nExample: " + std::string(argv[0]) + " --run_dir RUN --output OUT --configuration_dir CONFIG");
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_run_dir.empty()) { std::cerr << "--run_dir is required\n"; google::ShowUsageWithFlagsRestrict(argv[0], ""); return 2; }
  const fs::path run = fs::absolute(FLAGS_run_dir), horizon = run / "horizon";
  const fs::path output = FLAGS_output.empty() ? run / "cartographer_standalone" : fs::absolute(FLAGS_output);
  const fs::path config_dir = FLAGS_configuration_dir.empty() ? fs::path("/home/optimus/ros-projects/fliftbot_mapping/config/cartographer") : fs::absolute(FLAGS_configuration_dir);
  try {
    if (!fs::is_directory(horizon)) throw std::runtime_error("missing horizon directory: " + horizon.string());
    fs::create_directories(output);
  const Extrinsics ext = LoadExtrinsics(run / "lidar.config");
    std::string tracking_frame = FLAGS_tracking_frame;
    std::transform(tracking_frame.begin(), tracking_frame.end(),
                   tracking_frame.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (tracking_frame == "auto") {
      tracking_frame = ext.apply ? "lidar" : "rotation_axis";
    }
    if (tracking_frame != "lidar" && tracking_frame != "rotation_axis") {
      throw std::runtime_error(
          "--tracking_frame must be auto, lidar, or rotation_axis");
    }
    const bool tracking_frame_is_lidar = tracking_frame == "lidar";
    std::vector<Scan> scans;
    std::vector<fs::path> files;
    for (const auto& item : fs::directory_iterator(horizon)) {
      if (item.path().extension() == ".npz") {
        if (IsForbiddenInputName(item.path())) {
          throw std::runtime_error("forbidden GT/navigation NPZ filename: " + item.path().string());
        }
        files.push_back(item.path());
      }
    }
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
      try {
        scans.push_back(LoadScan(file));
      }
      catch (const std::exception& e) { if (!FLAGS_skip_bad_frames) throw; LOG(WARNING) << e.what(); }
      if (FLAGS_max_scans > 0 && static_cast<int>(scans.size()) >= FLAGS_max_scans) break;
    }
    std::sort(scans.begin(), scans.end(), [](const Scan& a, const Scan& b) { return a.stamp < b.stamp; });
    scans.erase(std::unique(scans.begin(), scans.end(), [](const Scan& a, const Scan& b) {
      return std::abs(a.stamp - b.stamp) < 1e-7;
    }), scans.end());
    if (scans.empty()) throw std::runtime_error("no valid scans");
    RplidarTimingStats rplidar_timing;
    size_t collapsed_rplidar_scans = 0;
    if (ext.lidar_type == "rplidar") {
      for (const auto& scan : scans) {
        collapsed_rplidar_scans += scan.point_times_collapsed;
      }
      if (collapsed_rplidar_scans > 0 && !FLAGS_synthesize_rplidar_times) {
        std::cerr << "RPLIDAR NPZ has collapsed per-beam timestamps in "
                  << collapsed_rplidar_scans
                  << " scans; synthesis is disabled (use "
                     "--synthesize_rplidar_times=true only with a verified "
                     "timestamp convention)\n";
      }
    }
    if (FLAGS_synthesize_rplidar_times) {
      // The helper is a no-op for Pandar/DePHAN and for recordings which
      // already carry non-collapsed point timestamps.  This makes the safe
      // default automatic by lidar_type/metadata while keeping an explicit
      // --synthesize_rplidar_times=false escape hatch for A/B experiments.
      rplidar_timing = SynthesizeRplidarTimes(
          &scans, ext, FLAGS_rplidar_timestamp_convention);
    } else if (ext.lidar_type == "rplidar") {
      rplidar_timing.mode = "disabled";
      rplidar_timing.convention = FLAGS_rplidar_timestamp_convention;
    }
    // Synthesizing may move each frame timestamp from the first firing to the
    // last firing.  Keep the stream ordered before synchronizing encoders.
    std::sort(scans.begin(), scans.end(), [](const Scan& a, const Scan& b) { return a.stamp < b.stamp; });
    std::cerr << "loaded scans=" << scans.size() << "\n";
    std::vector<Encoder> encoders = LoadEncoders(run / "encoder.jsonl");
    std::cerr << "loaded encoders=" << encoders.size() << "\n";
    if (FLAGS_use_odometry && encoders.size() < 2) throw std::runtime_error("need at least two encoder samples");
    if (FLAGS_use_odometry && encoders.size() >= 2) {
      const double begin = encoders.front().time, end = encoders.back().time;
      scans.erase(std::remove_if(scans.begin(), scans.end(), [&](const Scan& scan) {
        return scan.stamp < begin || scan.stamp > end;
      }), scans.end());
      if (scans.empty()) throw std::runtime_error("no scans bracketed by encoder timestamps");
    }
    if (FLAGS_use_odometry && !scans.empty()) {
      // Do not feed unrelated samples outside the scan interval. Besides
      // reducing work for short smoke tests, this avoids leaving an odometry
      // queue far ahead of the final range measurement.
      const double begin = scans.front().stamp - 1.0, end = scans.back().stamp + 1.0;
      std::vector<Encoder> clipped;
      for (const auto& e : encoders) if (e.time >= begin && e.time <= end) clipped.push_back(e);
      if (clipped.size() >= 2) encoders.swap(clipped);
    }
    std::cerr << "clipped encoders=" << encoders.size() << "\n";
    if (!(FLAGS_wheelbase > 0.) || !std::isfinite(FLAGS_wheelbase) ||
        !std::isfinite(FLAGS_steering_offset_deg) ||
        !std::isfinite(FLAGS_odometry_time_offset)) {
      throw std::runtime_error("invalid odometry calibration flag");
    }
    const auto odom_poses = IntegrateOdom(
        encoders, FLAGS_wheelbase, FLAGS_steering_offset_deg * M_PI / 180.,
        ext.encoder_speed_sign);
    std::cerr << "integrated odom\n";
    auto top = LoadLua(config_dir, FLAGS_configuration_file);
    std::cerr << "loaded lua\n";
    std::cerr << "loaded recording scans=" << scans.size() << " encoders=" << encoders.size() << "\n";
    auto map_options = cartographer::mapping::CreateMapBuilderOptions(top->GetDictionary("map_builder").get());
    auto trajectory_options = cartographer::mapping::CreateTrajectoryBuilderOptions(top->GetDictionary("trajectory_builder").get());
    if (!(FLAGS_max_range > 0.0) || !std::isfinite(FLAGS_max_range)) {
      throw std::runtime_error("--max_range must be a finite positive value");
    }
    if (!trajectory_options.has_trajectory_builder_2d_options()) {
      throw std::runtime_error("trajectory builder has no 2D options");
    }
    trajectory_options.mutable_trajectory_builder_2d_options()->set_max_range(
        static_cast<float>(FLAGS_max_range));
    auto map_builder = cartographer::mapping::CreateMapBuilder(map_options);
    using SensorId = TrajectoryBuilderInterface::SensorId;
    const SensorId range{SensorId::SensorType::RANGE, "lidar"};
    const SensorId odom{SensorId::SensorType::ODOMETRY, "odom"};
    std::set<SensorId> ids{range}; if (FLAGS_use_odometry) ids.insert(odom);
    const int trajectory_id = map_builder->AddTrajectoryBuilder(ids, trajectory_options, nullptr);
    auto* builder = map_builder->GetTrajectoryBuilder(trajectory_id);
    std::cerr << "created trajectory\n";
    std::vector<Event> events; events.reserve(scans.size() + encoders.size());
    for (size_t i = 0; i < scans.size(); ++i) events.push_back({scans[i].stamp, Event::RANGE, i});
    if (FLAGS_use_odometry) for (size_t i = 0; i < encoders.size(); ++i) events.push_back({encoders[i].time + FLAGS_odometry_time_offset, Event::ODOM, i});
    std::sort(events.begin(), events.end());
    size_t fed_scans = 0, fed_odom = 0;
    std::cerr << "feeding events=" << events.size() << "\n";
    for (const auto& event : events) {
      if (event.kind == Event::ODOM) {
        const auto& e = encoders[event.index];
        const auto axis_pose = cartographer::transform::Rigid3d(
            Eigen::Vector3d(odom_poses[event.index * 3],
                            odom_poses[event.index * 3 + 1], 0.),
            Eigen::AngleAxisd(odom_poses[event.index * 3 + 2],
                              Eigen::Vector3d::UnitZ()));
        auto pose = axis_pose;
        if (tracking_frame_is_lidar) {
          const auto axis_to_lidar = cartographer::transform::Rigid3d(
              Eigen::Vector3d(ext.x, ext.y, ext.z),
              Eigen::AngleAxisd(ext.yaw, Eigen::Vector3d::UnitZ()));
          pose = axis_pose * axis_to_lidar;
        }
        builder->AddSensorData("odom", cartographer::sensor::OdometryData{
            CartoTime(e.time + FLAGS_odometry_time_offset), pose}); ++fed_odom;
      } else {
        const auto& scan = scans[event.index];
        cartographer::sensor::TimedPointCloud ranges;
        ranges.reserve(scan.point_times.size());
        const float c = static_cast<float>(std::cos(ext.yaw));
        const float s = static_cast<float>(std::sin(ext.yaw));
        for (size_t i = 0; i < scan.point_times.size(); ++i) {
          const float lx = scan.xy[i * 2], ly = scan.xy[i * 2 + 1];
          // Match cartographer_ros SensorBridge's sensor→tracking contract:
          // `origin` is the sensor origin in tracking coordinates and
          // `ranges` are endpoint coordinates after the complete fixed
          // transform.  Cartographer uses both: origin for ray tracing and
          // endpoint for scan matching.  For a body/tracking NPZ the endpoint
          // already includes the mount translation, so only sensor-local data
          // receives the transform below.
          Eigen::Vector3f point;
          if (tracking_frame_is_lidar) {
            // Tracking frame is the physical LiDAR.  Sensor-local NPZ is
            // already in this frame; a body/tracking NPZ is converted back
            // to sensor-local so the composed odometry pose supplies the
            // mount translation exactly once.
            if (ext.apply) {
              point = Eigen::Vector3f(lx, ly, 0.f);
            } else {
              point = Eigen::Vector3f(
                  c * (lx - static_cast<float>(ext.x)) +
                      s * (ly - static_cast<float>(ext.y)),
                  -s * (lx - static_cast<float>(ext.x)) +
                      c * (ly - static_cast<float>(ext.y)),
                  0.f);
            }
          } else if (ext.apply) {
            // Sensor-local cloud, tracking frame is the rotation axis.  Core
            // Cartographer expects ranges to be endpoint coordinates in the
            // tracking frame (SensorBridge applies the complete sensor→TF
            // transform), while `origin` separately carries the ray origin.
            // Keep both translation copies: this is not double counting.
            point = Eigen::Vector3f(
                c * lx - s * ly + static_cast<float>(ext.x),
                s * lx + c * ly + static_cast<float>(ext.y),
                static_cast<float>(ext.z));
          } else {
            // The canonical HTTP RPLIDAR root cloud is already in the
            // rotation-axis/body frame (mount translation included).  Do not
            // add the mount a second time; the endpoint is already expressed
            // in tracking coordinates.  `origin` still identifies the actual
            // sensor for free-space ray tracing.
            point = Eigen::Vector3f(lx, ly, 0.f);
          }
          ranges.push_back({point, static_cast<float>(scan.point_times[i] - scan.stamp)});
        }
        builder->AddSensorData(
            "lidar",
            cartographer::sensor::TimedPointCloudData{
                CartoTime(scan.stamp),
                tracking_frame_is_lidar
                    ? Eigen::Vector3f::Zero()
                    : Eigen::Vector3f(ext.x, ext.y, ext.z),
                std::move(ranges), {}});
        ++fed_scans;
      }
    }
    std::cerr << "fed scans=" << fed_scans << " odom=" << fed_odom << ", finishing\n";
    map_builder->FinishTrajectory(trajectory_id);
    std::cerr << "finished, optimizing\n";
    map_builder->pose_graph()->RunFinalOptimization();
    std::cerr << "optimized, serializing\n";
    if (!map_builder->SerializeStateToFile(true, (output / "map.pbstream").string())) throw std::runtime_error("failed to serialize map.pbstream");
    ExportMapAndTrajectory(output, FLAGS_resolution, tracking_frame,
                           *map_builder);
    std::cerr << "exported\n";
    std::ofstream stats(output / "run_stats.json"); stats
        << "{\n  \"scans\": " << fed_scans
        << ",\n  \"odom_samples\": " << fed_odom
        << ",\n  \"constraints\": " << map_builder->pose_graph()->constraints().size()
        << ",\n  \"tracking_frame\": \"" << tracking_frame
        << "\",\n  \"wheelbase_m\": " << FLAGS_wheelbase
        << ",\n  \"steering_offset_deg\": " << FLAGS_steering_offset_deg
        << ",\n  \"odometry_time_offset_s\": " << FLAGS_odometry_time_offset
        << ",\n  \"encoder_speed_sign\": " << ext.encoder_speed_sign
        << ",\n  \"rplidar_timing_mode\": \"" << rplidar_timing.mode
        << "\",\n  \"rplidar_timestamp_convention\": \""
        << rplidar_timing.convention
        << "\",\n  \"rplidar_synthesized_scans\": "
        << rplidar_timing.synthesized_scans
        << ",\n  \"rplidar_collapsed_scans\": " << collapsed_rplidar_scans
        << ",\n  \"rplidar_period_s\": " << rplidar_timing.period_s
        << ",\n  \"rplidar_direction\": \"" << rplidar_timing.direction << "\""
        << "\n}\n";
    std::cout << "Standalone Cartographer result: " << output << " (scans=" << fed_scans << ", odom=" << fed_odom << ")\n";
  } catch (const std::exception& e) { std::cerr << "standalone Cartographer failed: " << e.what() << "\n"; return 1; }
  return 0;
}
