#include "native_smi.hpp"

#include "backend_config.hpp"
#include "global_state.hpp"
#include "gpu_profile.hpp"
#include "logging.hpp"
#include "version.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include <unistd.h>

namespace fake_gpu {
namespace {

constexpr const char* kStateSchema = "fakegpu.smi_state.v2";
constexpr double kDefaultIntervalMs = 250.0;
constexpr double kMinimumIntervalMs = 50.0;

std::atomic<GlobalState*> g_publisher_owner{nullptr};

uint64_t saturating_add(uint64_t left, uint64_t right) {
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return left >= limit - right ? limit : left + right;
}

uint64_t current_timestamp_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

std::string environment_text(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(fallback);
}

bool fakecuda_runtime_selected() {
    const char* runtime = std::getenv("FAKEGPU_RUNTIME");
    if (!runtime) {
        return false;
    }
    std::string normalized(runtime);
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return normalized == "fakecuda";
}

std::filesystem::path configured_state_path() {
    if (fakecuda_runtime_selected()) {
        return {};
    }
    const char* explicit_path = std::getenv("FAKEGPU_SMI_STATE_PATH");
    if (explicit_path && *explicit_path) {
        return std::filesystem::path(explicit_path);
    }
    const char* directory = std::getenv("FAKEGPU_SMI_STATE_DIR");
    if (!directory || !*directory) {
        return {};
    }
    return std::filesystem::path(directory) /
        (std::to_string(static_cast<long long>(getpid())) + ".json");
}

std::chrono::milliseconds configured_interval() {
    double milliseconds = kDefaultIntervalMs;
    if (const char* value = std::getenv("FAKEGPU_SMI_INTERVAL_MS");
        value && *value) {
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end != value && end && *end == '\0' && std::isfinite(parsed)) {
            milliseconds = parsed;
        }
    }
    milliseconds = std::max(kMinimumIntervalMs, milliseconds);
    return std::chrono::milliseconds(
        static_cast<std::chrono::milliseconds::rep>(milliseconds));
}

std::string hostname() {
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) == 0 && buffer[0] != '\0') {
        return std::string(buffer);
    }
    return "localhost";
}

std::string process_name() {
    if (const char* configured = std::getenv("FAKEGPU_PROCESS_NAME");
        configured && *configured) {
        return configured;
    }
#ifdef __APPLE__
    if (const char* name = getprogname(); name && *name) {
        return name;
    }
#else
    std::ifstream command("/proc/self/cmdline", std::ios::binary);
    std::string name;
    if (command.good() && std::getline(command, name, '\0') && !name.empty()) {
        return name;
    }
#endif
    return "native-process";
}

const char* platform_name() {
#ifdef __APPLE__
    return "native-macos";
#elif defined(__linux__)
    return "native-linux";
#else
    return "native-unix";
#endif
}

void append_json_string(std::ostringstream& out, const std::string& value) {
    out << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (character < 0x20) {
                    out << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<unsigned int>(character)
                        << std::dec
                        << std::setfill(' ');
                } else {
                    out << static_cast<char>(character);
                }
        }
    }
    out << '"';
}

void append_string_array(
    std::ostringstream& out,
    const std::vector<std::string>& values) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        append_json_string(out, values[index]);
    }
    out << ']';
}

struct NativeActivity {
    uint64_t io_calls = 0;
    uint64_t io_bytes = 0;
    uint64_t kernel_launches = 0;
    uint64_t gemm_calls = 0;
    uint64_t gemm_flops = 0;
    uint64_t compatibility_events = 0;
    uint64_t unsupported_api_calls = 0;
};

NativeActivity native_activity(const DeviceReportStats& device) {
    NativeActivity activity;
    for (const auto value : {
             device.memcpy_h2d_calls,
             device.memcpy_d2h_calls,
             device.memcpy_d2d_calls,
             device.memcpy_peer_tx_calls,
             device.memcpy_peer_rx_calls,
             device.memset_calls,
         }) {
        activity.io_calls = saturating_add(activity.io_calls, value);
    }
    for (const auto value : {
             device.memcpy_h2d_bytes,
             device.memcpy_d2h_bytes,
             device.memcpy_d2d_bytes,
             device.memcpy_peer_tx_bytes,
             device.memcpy_peer_rx_bytes,
             device.memset_bytes,
         }) {
        activity.io_bytes = saturating_add(activity.io_bytes, value);
    }
    activity.kernel_launches = device.kernel_launch_total;
    activity.gemm_calls = saturating_add(
        device.cublas_gemm_calls,
        device.cublaslt_matmul_calls);
    activity.gemm_flops = saturating_add(
        device.cublas_gemm_flops,
        device.cublaslt_matmul_flops);
    for (const auto& event : device.compat_events) {
        activity.compatibility_events = saturating_add(
            activity.compatibility_events,
            std::get<2>(event));
    }
    for (const auto& event : device.unsupported_api_events) {
        activity.unsupported_api_calls = saturating_add(
            activity.unsupported_api_calls,
            std::get<3>(event));
    }
    return activity;
}

void append_kernel_counts(
    std::ostringstream& out,
    const DeviceReportStats& device) {
    out << '{';
    std::vector<std::pair<std::string, uint64_t>> kernels(
        device.kernel_launches.begin(),
        device.kernel_launches.end());
    std::sort(
        kernels.begin(),
        kernels.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        });
    for (std::size_t index = 0; index < kernels.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        append_json_string(out, kernels[index].first);
        out << ':' << kernels[index].second;
    }
    out << '}';
}

void append_unsupported_apis(
    std::ostringstream& out,
    const DeviceReportStats& device) {
    out << '[';
    for (std::size_t index = 0;
         index < device.unsupported_api_events.size();
         ++index) {
        if (index != 0) {
            out << ',';
        }
        const auto& [operation, behavior, policy, count] =
            device.unsupported_api_events[index];
        out << "{\"operation\":";
        append_json_string(out, operation);
        out << ",\"behavior\":";
        append_json_string(out, behavior);
        out << ",\"policy\":";
        append_json_string(out, policy);
        out << ",\"count\":" << count << '}';
    }
    out << ']';
}

void append_device(
    std::ostringstream& out,
    const DeviceReportStats& device) {
    const NativeActivity activity = native_activity(device);
    const uint64_t free_memory =
        device.total_memory > device.used_memory_current
        ? device.total_memory - device.used_memory_current
        : 0;
    const uint64_t headroom =
        device.total_memory > device.used_memory_peak
        ? device.total_memory - device.used_memory_peak
        : 0;

    out << "{\"index\":" << device.index;
    out << ",\"name\":";
    append_json_string(out, device.name);
    out << ",\"profile_id\":";
    append_json_string(
        out,
        device.profile_id.empty() ? "unknown" : device.profile_id);
    out << ",\"profile\":{\"id\":";
    append_json_string(
        out,
        device.profile_id.empty() ? "unknown" : device.profile_id);
    out << ",\"name\":";
    append_json_string(out, device.name);
    out << ",\"architecture\":";
    append_json_string(out, device.architecture);
    out << ",\"compute_capability\":";
    append_json_string(
        out,
        std::to_string(device.compute_major) + "." +
            std::to_string(device.compute_minor));
    out << ",\"compiler_target\":";
    append_json_string(
        out,
        "sm_" + std::to_string(device.compute_major) +
            std::to_string(device.compute_minor));
    out << ",\"memory_bytes\":" << device.total_memory;
    out << ",\"sm_count\":" << device.sm_count;
    out << ",\"memory_bus_width_bits\":"
        << device.memory_bus_width_bits;
    out << ",\"core_clock_mhz\":" << device.core_clock_mhz;
    out << ",\"memory_clock_mhz\":" << device.memory_clock_mhz;
    out << ",\"l2_cache_bytes\":" << device.l2_cache_bytes;
    out << ",\"typical_power_usage_mw\":"
        << device.typical_power_usage_mw;
    out << ",\"max_power_limit_mw\":"
        << device.max_power_limit_mw;
    out << ",\"supported_types\":";
    append_string_array(out, device.supported_types);
    out << '}';
    out << ",\"uuid\":";
    append_json_string(out, device.uuid);
    out << ",\"pci_bus_id\":";
    append_json_string(out, device.pci_bus_id);
    out << ",\"identity_source\":\"native_synthetic\"";
    out << ",\"architecture\":";
    append_json_string(out, device.architecture);
    out << ",\"compute_capability\":";
    append_json_string(
        out,
        std::to_string(device.compute_major) + "." +
            std::to_string(device.compute_minor));
    out << ",\"compiler_target\":";
    append_json_string(
        out,
        "sm_" + std::to_string(device.compute_major) +
            std::to_string(device.compute_minor));
    out << ",\"total_memory\":" << device.total_memory;
    out << ",\"free_memory\":" << free_memory;
    out << ",\"tracked_memory\":" << device.used_memory_current;
    out << ",\"peak_tracked_memory\":" << device.used_memory_peak;
    out << ",\"reserved_memory\":" << device.used_memory_current;
    out << ",\"peak_reserved_memory\":" << device.used_memory_peak;
    out << ",\"inactive_split_bytes\":0";
    out << ",\"segment_count\":"
        << (device.used_memory_current > 0 ? 1 : 0);
    out << ",\"reported_memory_source\":\"native_allocation\"";
    out << ",\"runtime_overhead_bytes\":0";
    out << ",\"reported_memory\":" << device.used_memory_current;
    out << ",\"reported_peak_memory\":" << device.used_memory_peak;
    out << ",\"headroom_bytes\":" << headroom;
    out << ",\"allocation_count\":" << device.alloc_calls;
    out << ",\"free_count\":" << device.free_calls;
    out << ",\"allocator_model\":\"direct_native_allocations.v1\"";
    out << ",\"current_bytes_by_category\":{\"native_allocation\":"
        << device.used_memory_current << '}';
    out << ",\"peak_by_stage\":{\"native\":"
        << device.used_memory_peak << '}';
    out << ",\"reserved_peak_by_stage\":{\"native\":"
        << device.used_memory_peak << '}';
    out << ",\"largest_allocations\":[]";
    out << ",\"native_activity\":{";
    out << "\"io_calls\":" << activity.io_calls;
    out << ",\"io_bytes\":" << activity.io_bytes;
    out << ",\"kernel_launches\":" << activity.kernel_launches;
    out << ",\"gemm_calls\":" << activity.gemm_calls;
    out << ",\"gemm_flops\":" << activity.gemm_flops;
    out << ",\"compatibility_events\":"
        << activity.compatibility_events;
    out << ",\"unsupported_api_calls\":"
        << activity.unsupported_api_calls;
    out << ",\"kernels\":";
    append_kernel_counts(out, device);
    out << ",\"unsupported_apis\":";
    append_unsupported_apis(out, device);
    out << '}';
    out << ",\"telemetry\":{"
        << "\"gpu_utilization_percent\":null,"
        << "\"temperature_c\":null,"
        << "\"fan_speed_percent\":null,"
        << "\"power_usage_mw\":null,"
        << "\"source\":\"hardware_telemetry_unavailable\"}";
    out << '}';
}

class NativeSmiPublisher {
public:
    NativeSmiPublisher(
        GlobalState& state,
        std::filesystem::path path,
        std::chrono::milliseconds interval)
        : state_(state),
          path_(std::move(path)),
          interval_(interval),
          hostname_(hostname()),
          process_name_(process_name()),
          profile_count_(builtin_profile_ids().size()) {
    }

    ~NativeSmiPublisher() {
        stop();
    }

    void start() {
        write_state(true);
        worker_ = std::thread([this]() { run(); });
    }

    void stop() noexcept {
        bool expected = false;
        if (!stopped_.compare_exchange_strong(expected, true)) {
            return;
        }
        wake_.notify_all();
        try {
            if (worker_.joinable()) {
                worker_.join();
            }
        } catch (...) {
            FGPU_LOG("[NativeSMI] Failed to join publisher thread\n");
        }
        write_state(false);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        while (!wake_.wait_for(
            lock,
            interval_,
            [this]() { return stopped_.load(); })) {
            lock.unlock();
            write_state(true);
            lock.lock();
        }
    }

    void write_state(bool running) noexcept {
        try {
            const std::vector<DeviceReportStats> devices =
                state_.snapshot_device_report();
            const BackendConfig& config = BackendConfig::instance();
            const double interval_seconds =
                static_cast<double>(interval_.count()) / 1000.0;

            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << "{\"schema_version\":\"" << kStateSchema << '"';
            out << ",\"timestamp_ns\":" << current_timestamp_ns();
            out << ",\"hostname\":";
            append_json_string(out, hostname_);
            out << ",\"pid\":" << static_cast<long long>(getpid());
            out << ",\"process_name\":";
            append_json_string(out, process_name_);
            out << ",\"runtime\":\"native\"";
            out << ",\"fakegpu\":{\"version\":";
            append_json_string(out, FAKEGPU_VERSION);
            out << ",\"runtime\":\"native\"";
            out << ",\"backend\":\"native_interception\"";
            out << ",\"mode\":";
            append_json_string(out, mode_name(config.mode()));
            out << ",\"oom_policy\":";
            append_json_string(out, policy_name(config.oom_policy()));
            out << ",\"unsupported_api_policy\":";
            append_json_string(
                out,
                unsupported_api_policy_name(
                    config.unsupported_api_policy()));
            out << ",\"distributed_mode\":";
            append_json_string(
                out,
                distributed::distributed_mode_name(
                    config.distributed_config().mode));
            out << ",\"memory_tracking_enabled\":true";
            out << ",\"dispatch_memory_tracking_enabled\":false";
            out << ",\"profile_catalog\":{\"profile_count\":"
                << profile_count_ << '}';
            out << ",\"native_capabilities\":{}}";
            out << ",\"software\":{\"python_version\":null";
            out << ",\"python_implementation\":null";
            out << ",\"python_executable\":null";
            out << ",\"platform\":";
            append_json_string(out, platform_name());
            out << ",\"torch_version\":null";
            out << ",\"torch_cuda_build\":null";
            out << ",\"cuda_version\":";
            append_json_string(
                out,
                environment_text(
                    "FAKEGPU_CUDA_VERSION",
                    "12.1"));
            out << ",\"cuda_version_source\":\"native_fakegpu_default\"";
            out << ",\"driver_version\":";
            append_json_string(
                out,
                environment_text(
                    "FAKEGPU_DRIVER_VERSION",
                    "simulated"));
            out << '}';
            out << ",\"publisher\":{\"interval_seconds\":"
                << interval_seconds;
            out << ",\"runtime_overhead_bytes\":0";
            out << ",\"source\":\"native_monitor\"}";
            out << ",\"running\":" << (running ? "true" : "false");
            out << ",\"tracking_confidence\":"
                << "\"C2_native_allocation_lifetime\"";
            out << ",\"stage\":";
            append_json_string(
                out,
                environment_text(
                    "FAKEGPU_PREFLIGHT_STAGE",
                    "native"));
            out << ",\"allocator_model\":"
                << "\"direct_native_allocations.v1\"";
            out << ",\"dispatch_tracking\":{"
                << "\"enabled\":false,"
                << "\"operator_calls\":0,"
                << "\"output_tensors\":0,"
                << "\"new_allocations\":0,"
                << "\"alias_outputs\":0,"
                << "\"inaccessible_outputs\":0,"
                << "\"operators\":{}}";
            out << ",\"devices\":[";
            for (std::size_t index = 0;
                 index < devices.size();
                 ++index) {
                if (index != 0) {
                    out << ',';
                }
                append_device(out, devices[index]);
            }
            out << "]}\n";

            const std::filesystem::path parent = path_.parent_path();
            std::error_code error;
            if (!parent.empty()) {
                std::filesystem::create_directories(parent, error);
                if (error) {
                    FGPU_LOG(
                        "[NativeSMI] Failed to create state directory: %s\n",
                        error.message().c_str());
                    return;
                }
            }

            const std::filesystem::path temporary =
                path_.parent_path() /
                ("." + path_.filename().string() + "." +
                 std::to_string(static_cast<long long>(getpid())) +
                 ".tmp");
            {
                std::ofstream stream(
                    temporary,
                    std::ios::binary | std::ios::trunc);
                stream << out.str();
                stream.flush();
                if (!stream.good()) {
                    stream.close();
                    std::filesystem::remove(temporary, error);
                    return;
                }
            }
            std::filesystem::rename(temporary, path_, error);
            if (error) {
                std::filesystem::remove(path_, error);
                error.clear();
                std::filesystem::rename(temporary, path_, error);
            }
            if (error) {
                FGPU_LOG(
                    "[NativeSMI] Failed to replace state file: %s\n",
                    error.message().c_str());
                std::filesystem::remove(temporary, error);
            }
        } catch (...) {
            FGPU_LOG("[NativeSMI] Exception while publishing state\n");
        }
    }

    GlobalState& state_;
    std::filesystem::path path_;
    std::chrono::milliseconds interval_;
    std::string hostname_;
    std::string process_name_;
    std::size_t profile_count_;
    std::atomic<bool> stopped_{false};
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    std::thread worker_;
};

void stop_registered_publisher() {
    GlobalState* owner = g_publisher_owner.exchange(nullptr);
    if (owner != nullptr) {
        owner->stop_native_smi_publisher();
    }
}

}  // namespace

void* create_native_smi_publisher(GlobalState& state) {
    const std::filesystem::path path = configured_state_path();
    if (path.empty()) {
        return nullptr;
    }
    try {
        auto publisher = std::make_unique<NativeSmiPublisher>(
            state,
            path,
            configured_interval());
        publisher->start();
        g_publisher_owner.store(&state);
        if (std::atexit(stop_registered_publisher) != 0) {
            g_publisher_owner.store(nullptr);
            publisher->stop();
            return nullptr;
        }
        return publisher.release();
    } catch (...) {
        FGPU_LOG("[NativeSMI] Failed to start state publisher\n");
        return nullptr;
    }
}

void destroy_native_smi_publisher(void* publisher) {
    delete static_cast<NativeSmiPublisher*>(publisher);
}

}  // namespace fake_gpu
