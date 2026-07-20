#include "cluster_report_writer.hpp"

#include "../core/version.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fake_gpu::distributed {

namespace {

using NodePairKey = std::pair<std::string, std::string>;

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char character : value) {
        switch (character) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (character < 0x20) {
                    char escaped[7] {};
                    std::snprintf(
                        escaped,
                        sizeof(escaped),
                        "\\u%04x",
                        static_cast<unsigned int>(character));
                    out << escaped;
                } else {
                    out << static_cast<char>(character);
                }
                break;
        }
    }
    return out.str();
}

std::string markdown_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char character : value) {
        if (character == '|' || character == '\\') {
            escaped.push_back('\\');
        }
        if (character == '\n' || character == '\r') {
            escaped.push_back(' ');
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

std::string lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

bool ensure_parent_directory(const std::string& path, std::string& error) {
    const std::filesystem::path report_path(path);
    const std::filesystem::path parent = report_path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(parent, filesystem_error);
    if (filesystem_error) {
        error =
            "failed to create report directory " + parent.string() + ": " +
            filesystem_error.message();
        return false;
    }
    return true;
}

std::size_t report_world_size(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot) {
    if (snapshot.world_size > 0) {
        return snapshot.world_size;
    }
    if (config.cluster_config.loaded()) {
        return config.cluster_config.world_size;
    }
    return snapshot.ranks.size();
}

std::size_t report_node_count(
    const DistributedConfig& config,
    std::size_t world_size) {
    if (config.cluster_config.loaded()) {
        return config.cluster_config.nodes.size();
    }
    return world_size > 0 ? 1U : 0U;
}

std::string node_name_for_rank(
    const ClusterConfigModel& config,
    int rank) {
    if (config.loaded()) {
        for (const ClusterNodeConfig& node : config.nodes) {
            for (int candidate : node.ranks) {
                if (candidate == rank) {
                    return node.id;
                }
            }
        }
    }
    return "node0";
}

NodePairKey make_node_pair_key(
    const std::string& first,
    const std::string& second) {
    return first < second
        ? NodePairKey{first, second}
        : NodePairKey{second, first};
}

std::vector<ClusterNodePairReportStats> complete_node_pairs(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot) {
    std::map<NodePairKey, ClusterNodePairReportStats> pairs;
    const ClusterConfigModel& cluster_config = config.cluster_config;

    if (cluster_config.loaded()) {
        for (std::size_t first = 0; first < cluster_config.nodes.size(); ++first) {
            for (std::size_t second = first + 1;
                 second < cluster_config.nodes.size();
                 ++second) {
                const NodePairKey key = make_node_pair_key(
                    cluster_config.nodes[first].id,
                    cluster_config.nodes[second].id);
                ClusterNodePairReportStats pair;
                pair.node_a = key.first;
                pair.node_b = key.second;
                pair.a_to_b.model_bandwidth_gbps =
                    cluster_config.inter_node_fabric.bandwidth_gbps;
                pair.b_to_a.model_bandwidth_gbps =
                    cluster_config.inter_node_fabric.bandwidth_gbps;
                pairs.emplace(key, pair);
            }
        }
    }

    for (const ClusterNodePairReportStats& snapshot_pair : snapshot.node_pairs) {
        const NodePairKey key =
            make_node_pair_key(snapshot_pair.node_a, snapshot_pair.node_b);
        ClusterNodePairReportStats pair = snapshot_pair;
        if (pair.node_a != key.first) {
            std::swap(pair.a_to_b, pair.b_to_a);
        }
        pair.node_a = key.first;
        pair.node_b = key.second;
        if (cluster_config.loaded()) {
            if (pair.a_to_b.model_bandwidth_gbps <= 0.0) {
                pair.a_to_b.model_bandwidth_gbps =
                    cluster_config.inter_node_fabric.bandwidth_gbps;
            }
            if (pair.b_to_a.model_bandwidth_gbps <= 0.0) {
                pair.b_to_a.model_bandwidth_gbps =
                    cluster_config.inter_node_fabric.bandwidth_gbps;
            }
        }
        pairs[key] = std::move(pair);
    }

    std::vector<ClusterNodePairReportStats> result;
    result.reserve(pairs.size());
    for (auto& entry : pairs) {
        result.push_back(std::move(entry.second));
    }
    return result;
}

std::uint64_t pair_total_bytes(const ClusterNodePairReportStats& pair) {
    return pair.a_to_b.bytes + pair.b_to_a.bytes;
}

std::uint64_t pair_total_transfers(const ClusterNodePairReportStats& pair) {
    return pair.a_to_b.transfers + pair.b_to_a.transfers;
}

double pair_estimated_time_us(const ClusterNodePairReportStats& pair) {
    return
        pair.a_to_b.estimated_time_us_total +
        pair.b_to_a.estimated_time_us_total;
}

double pair_contention_penalty_us(const ClusterNodePairReportStats& pair) {
    return
        pair.a_to_b.contention_penalty_us_total +
        pair.b_to_a.contention_penalty_us_total;
}

double estimated_throughput_gbps(
    std::uint64_t bytes,
    double estimated_time_us) {
    if (bytes == 0 || estimated_time_us <= 0.0) {
        return 0.0;
    }
    return
        (static_cast<double>(bytes) * 8.0) /
        (estimated_time_us * 1000.0);
}

std::string format_bytes(std::uint64_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream out;
    if (unit == 0) {
        out << bytes << " B";
    } else {
        out << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
    }
    return out.str();
}

std::string format_decimal(double value, int precision = 3) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

void write_collective_json(
    std::ostream& out,
    const char* name,
    const ClusterCollectiveReportStats& stats,
    bool trailing_comma) {
    out << "    \"" << name << "\": {"
        << "\"calls\": " << stats.calls
        << ", \"bytes\": " << stats.bytes
        << ", \"estimated_time_us_total\": "
        << format_decimal(stats.estimated_time_us_total)
        << ", \"contention_penalty_us_total\": "
        << format_decimal(stats.contention_penalty_us_total)
        << "}" << (trailing_comma ? "," : "") << "\n";
}

void write_direction_json(
    std::ostream& out,
    const ClusterNodePairDirectionReportStats& direction,
    const char* indent) {
    const double average_throughput = estimated_throughput_gbps(
        direction.bytes,
        direction.estimated_time_us_total);
    out << indent << "{\n";
    out << indent << "  \"transfers\": " << direction.transfers << ",\n";
    out << indent << "  \"total_bytes\": " << direction.bytes << ",\n";
    out << indent << "  \"peak_bytes_per_operation\": "
        << direction.peak_bytes_per_operation << ",\n";
    out << indent << "  \"model_bandwidth_gbps\": "
        << format_decimal(direction.model_bandwidth_gbps) << ",\n";
    out << indent << "  \"avg_latency_us\": "
        << format_decimal(direction.avg_latency_us) << ",\n";
    out << indent << "  \"estimated_time_us_total\": "
        << format_decimal(direction.estimated_time_us_total) << ",\n";
    out << indent << "  \"contention_penalty_us_total\": "
        << format_decimal(direction.contention_penalty_us_total) << ",\n";
    out << indent << "  \"average_estimated_throughput_gbps\": "
        << format_decimal(average_throughput) << ",\n";
    out << indent << "  \"peak_estimated_throughput_gbps\": "
        << format_decimal(direction.peak_estimated_throughput_gbps) << "\n";
    out << indent << "}";
}

bool write_json_report(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot,
    const std::vector<ClusterNodePairReportStats>& node_pairs,
    const std::string& json_report_path,
    const std::string& markdown_report_path,
    std::string& error) {
    if (!ensure_parent_directory(json_report_path, error)) {
        return false;
    }
    std::ofstream out(json_report_path);
    if (!out) {
        error = "failed to open cluster report: " + json_report_path;
        return false;
    }

    const ClusterConfigModel& cluster_config = config.cluster_config;
    const std::size_t world_size = report_world_size(config, snapshot);
    const std::size_t node_count = report_node_count(config, world_size);

    out << "{\n";
    out << "  \"report_version\": \"" << json_escape(FAKEGPU_VERSION) << "\",\n";
    out << "  \"schema\": \"experimental\",\n";
    out << "  \"cluster\": {\n";
    out << "    \"mode\": \""
        << distributed_mode_name(config.mode) << "\",\n";
    out << "    \"world_size\": " << world_size << ",\n";
    out << "    \"node_count\": " << node_count << ",\n";
    out << "    \"communicators\": " << snapshot.communicator_count << ",\n";
    out << "    \"coordinator_transport\": \""
        << coordinator_transport_name(config.coordinator_transport) << "\"";
    if (!markdown_report_path.empty()) {
        out << ",\n    \"markdown_report_path\": \""
            << json_escape(markdown_report_path) << "\"";
    }
    if (cluster_config.loaded()) {
        out << ",\n";
        out << "    \"name\": \"" << json_escape(cluster_config.name) << "\",\n";
        out << "    \"default_backend\": \""
            << json_escape(cluster_config.default_backend) << "\",\n";
        out << "    \"config_path\": \""
            << json_escape(cluster_config.source_path) << "\"\n";
    } else {
        out << "\n";
    }
    out << "  },\n";

    out << "  \"collectives\": {\n";
    write_collective_json(out, "all_reduce", snapshot.all_reduce, true);
    write_collective_json(out, "reduce", snapshot.reduce, true);
    write_collective_json(out, "broadcast", snapshot.broadcast, true);
    write_collective_json(out, "all_gather", snapshot.all_gather, true);
    write_collective_json(out, "reduce_scatter", snapshot.reduce_scatter, true);
    write_collective_json(out, "all_to_all", snapshot.all_to_all, true);
    write_collective_json(out, "barrier", snapshot.barrier, false);
    out << "  },\n";

    out << "  \"links\": [\n";
    for (std::size_t index = 0; index < snapshot.links.size(); ++index) {
        const ClusterLinkReportStats& link = snapshot.links[index];
        const double average_throughput = estimated_throughput_gbps(
            link.bytes,
            link.estimated_time_us_total);
        out << "    {\n";
        out << "      \"src\": \"" << json_escape(link.src_node) << "\",\n";
        out << "      \"dst\": \"" << json_escape(link.dst_node) << "\",\n";
        out << "      \"scope\": \"" << json_escape(link.scope) << "\",\n";
        out << "      \"samples\": " << link.samples << ",\n";
        out << "      \"bytes\": " << link.bytes << ",\n";
        out << "      \"peak_bytes_per_operation\": "
            << link.peak_bytes_per_operation << ",\n";
        out << "      \"bandwidth_gbps\": "
            << format_decimal(link.bandwidth_gbps) << ",\n";
        out << "      \"avg_latency_us\": "
            << format_decimal(link.avg_latency_us) << ",\n";
        out << "      \"estimated_time_us_total\": "
            << format_decimal(link.estimated_time_us_total) << ",\n";
        out << "      \"contention_penalty_us_total\": "
            << format_decimal(link.contention_penalty_us_total) << ",\n";
        out << "      \"average_estimated_throughput_gbps\": "
            << format_decimal(average_throughput) << ",\n";
        out << "      \"peak_estimated_throughput_gbps\": "
            << format_decimal(link.peak_estimated_throughput_gbps) << "\n";
        out << "    }"
            << (index + 1 < snapshot.links.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"node_pairs\": [\n";
    for (std::size_t index = 0; index < node_pairs.size(); ++index) {
        const ClusterNodePairReportStats& pair = node_pairs[index];
        const std::uint64_t total_bytes = pair_total_bytes(pair);
        const double estimated_time_us = pair_estimated_time_us(pair);
        out << "    {\n";
        out << "      \"node_a\": \"" << json_escape(pair.node_a) << "\",\n";
        out << "      \"node_b\": \"" << json_escape(pair.node_b) << "\",\n";
        out << "      \"scope\": \"inter_node\",\n";
        out << "      \"operations\": " << pair.operations << ",\n";
        out << "      \"a_to_b\": ";
        write_direction_json(out, pair.a_to_b, "      ");
        out << ",\n";
        out << "      \"b_to_a\": ";
        write_direction_json(out, pair.b_to_a, "      ");
        out << ",\n";
        out << "      \"total_bytes\": " << total_bytes << ",\n";
        out << "      \"peak_combined_bytes_per_operation\": "
            << pair.peak_combined_bytes_per_operation << ",\n";
        out << "      \"estimated_time_us_total\": "
            << format_decimal(estimated_time_us) << ",\n";
        out << "      \"contention_penalty_us_total\": "
            << format_decimal(pair_contention_penalty_us(pair)) << ",\n";
        out << "      \"average_estimated_throughput_gbps\": "
            << format_decimal(
                   estimated_throughput_gbps(total_bytes, estimated_time_us))
            << ",\n";
        out << "      \"peak_estimated_throughput_gbps\": "
            << format_decimal(pair.peak_estimated_throughput_gbps) << "\n";
        out << "    }" << (index + 1 < node_pairs.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"ranks\": [\n";
    for (std::size_t index = 0; index < snapshot.ranks.size(); ++index) {
        const ClusterRankReportStats& rank = snapshot.ranks[index];
        out << "    {\n";
        out << "      \"rank\": " << rank.rank << ",\n";
        out << "      \"node\": \""
            << json_escape(node_name_for_rank(cluster_config, rank.rank)) << "\",\n";
        out << "      \"wait_time_ms\": "
            << format_decimal(rank.wait_time_ms) << ",\n";
        out << "      \"timeouts\": " << rank.timeouts << ",\n";
        out << "      \"communicator_inits\": "
            << rank.communicator_inits << ",\n";
        out << "      \"collective_calls\": "
            << rank.collective_calls << ",\n";
        out << "      \"barrier_calls\": "
            << rank.barrier_calls << ",\n";
        out << "      \"group_prepares\": "
            << rank.group_prepares << "\n";
        out << "    }" << (index + 1 < snapshot.ranks.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    if (!out) {
        error = "failed to write cluster report: " + json_report_path;
        return false;
    }
    return true;
}

bool write_markdown_report(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot,
    const std::vector<ClusterNodePairReportStats>& node_pairs,
    const std::string& json_report_path,
    const std::string& markdown_report_path,
    std::string& error) {
    if (markdown_report_path.empty()) {
        return true;
    }
    if (markdown_report_path == json_report_path) {
        error = "cluster JSON and Markdown report paths must be different";
        return false;
    }
    if (!ensure_parent_directory(markdown_report_path, error)) {
        return false;
    }
    std::ofstream out(markdown_report_path);
    if (!out) {
        error = "failed to open cluster Markdown report: " + markdown_report_path;
        return false;
    }

    const ClusterConfigModel& cluster_config = config.cluster_config;
    const std::size_t world_size = report_world_size(config, snapshot);
    const std::size_t node_count = report_node_count(config, world_size);
    const std::string cluster_name = cluster_config.loaded()
        ? cluster_config.name
        : "unconfigured-cluster";

    out << "# FakeGPU Cluster Communication Report\n\n";
    out << "- Cluster: `" << markdown_escape(cluster_name) << "`\n";
    out << "- World size: " << world_size << "\n";
    out << "- Nodes: " << node_count << "\n";
    out << "- Communicators: " << snapshot.communicator_count << "\n";
    out << "- Coordinator transport: `"
        << coordinator_transport_name(config.coordinator_transport) << "`\n";
    out << "- JSON source: `" << markdown_escape(json_report_path) << "`\n\n";

    out << "## Collective Summary\n\n";
    out << "| Collective | Calls | Total payload | Estimated time | Contention penalty |\n";
    out << "|---|---:|---:|---:|---:|\n";
    const std::vector<std::pair<const char*, const ClusterCollectiveReportStats*>>
        collectives = {
            {"all_reduce", &snapshot.all_reduce},
            {"reduce", &snapshot.reduce},
            {"broadcast", &snapshot.broadcast},
            {"all_gather", &snapshot.all_gather},
            {"reduce_scatter", &snapshot.reduce_scatter},
            {"all_to_all", &snapshot.all_to_all},
            {"barrier", &snapshot.barrier},
        };
    for (const auto& entry : collectives) {
        out << "| `" << entry.first << "`"
            << " | " << entry.second->calls
            << " | " << format_bytes(entry.second->bytes)
            << " | " << format_decimal(entry.second->estimated_time_us_total)
            << " us"
            << " | " << format_decimal(
                   entry.second->contention_penalty_us_total)
            << " us |\n";
    }

    out << "\n## Node-Pair Communication\n\n";
    out << "| Node A | Node B | A → B total | B → A total | Combined total"
        << " | A → B peak/op | B → A peak/op | Pair peak/op | Operations"
        << " | Transfers | Avg est. Gbit/s | Peak est. Gbit/s"
        << " | Est. time | Contention |\n";
    out << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const ClusterNodePairReportStats& pair : node_pairs) {
        const std::uint64_t total_bytes = pair_total_bytes(pair);
        const double estimated_time_us = pair_estimated_time_us(pair);
        out << "| `" << markdown_escape(pair.node_a)
            << "` | `" << markdown_escape(pair.node_b)
            << "` | " << format_bytes(pair.a_to_b.bytes)
            << " | " << format_bytes(pair.b_to_a.bytes)
            << " | " << format_bytes(total_bytes)
            << " | " << format_bytes(pair.a_to_b.peak_bytes_per_operation)
            << " | " << format_bytes(pair.b_to_a.peak_bytes_per_operation)
            << " | " << format_bytes(pair.peak_combined_bytes_per_operation)
            << " | " << pair.operations
            << " | " << pair_total_transfers(pair)
            << " | " << format_decimal(
                   estimated_throughput_gbps(total_bytes, estimated_time_us))
            << " | " << format_decimal(pair.peak_estimated_throughput_gbps)
            << " | " << format_decimal(estimated_time_us) << " us"
            << " | " << format_decimal(pair_contention_penalty_us(pair))
            << " us |\n";
    }
    if (node_pairs.empty()) {
        out << "| _No distinct node pairs_ |  | 0 B | 0 B | 0 B"
            << " | 0 B | 0 B | 0 B | 0 | 0 | 0.000 | 0.000"
            << " | 0.000 us | 0.000 us |\n";
    }

    out << "\n## Rank Summary\n\n";
    out << "| Rank | Node | Wait time | Timeouts | Communicator inits"
        << " | Collective calls | Barrier calls | Group prepares |\n";
    out << "|---:|---|---:|---:|---:|---:|---:|---:|\n";
    for (const ClusterRankReportStats& rank : snapshot.ranks) {
        out << "| " << rank.rank
            << " | `" << markdown_escape(
                   node_name_for_rank(cluster_config, rank.rank))
            << "` | " << format_decimal(rank.wait_time_ms) << " ms"
            << " | " << rank.timeouts
            << " | " << rank.communicator_inits
            << " | " << rank.collective_calls
            << " | " << rank.barrier_calls
            << " | " << rank.group_prepares
            << " |\n";
    }

    out << "\n## Metric Notes\n\n";
    out << "- Every distinct node pair from the cluster configuration appears in"
        << " the table, including pairs with zero traffic.\n";
    out << "- `peak/op` is the largest payload attributed to that direction or"
        << " node pair during one completed communication operation.\n";
    out << "- Estimated throughput, time, and contention come from the configured"
        << " topology model. They are not packet captures or measured NIC/NCCL"
        << " bandwidth.\n";
    out << "- The JSON report retains exact byte counters and directional details"
        << " for automated analysis.\n";

    if (!out) {
        error =
            "failed to write cluster Markdown report: " + markdown_report_path;
        return false;
    }
    return true;
}

}  // namespace

std::string resolve_cluster_markdown_report_path(
    const std::string& json_report_path) {
    const char* configured =
        std::getenv("FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH");
    if (configured && *configured) {
        const std::string value(configured);
        const std::string normalized = lower_copy(value);
        if (normalized == "0" || normalized == "off" || normalized == "none") {
            return "";
        }
        return value;
    }

    std::filesystem::path markdown_path(json_report_path);
    const std::string extension = lower_copy(markdown_path.extension().string());
    if (extension == ".json") {
        markdown_path.replace_extension(".md");
    } else {
        markdown_path += ".md";
    }
    return markdown_path.string();
}

bool write_cluster_report_files(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot,
    const std::string& json_report_path,
    std::string& error) {
    error.clear();
    if (json_report_path.empty()) {
        error = "cluster JSON report path must not be empty";
        return false;
    }
    if (!snapshot.has_data) {
        return true;
    }

    const std::vector<ClusterNodePairReportStats> node_pairs =
        complete_node_pairs(config, snapshot);
    const std::string markdown_report_path =
        resolve_cluster_markdown_report_path(json_report_path);
    if (!write_json_report(
            config,
            snapshot,
            node_pairs,
            json_report_path,
            markdown_report_path,
            error)) {
        return false;
    }
    return write_markdown_report(
        config,
        snapshot,
        node_pairs,
        json_report_path,
        markdown_report_path,
        error);
}

}  // namespace fake_gpu::distributed
