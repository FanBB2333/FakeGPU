#pragma once

#include "cluster_config.hpp"
#include "communicator.hpp"

#include <string>

namespace fake_gpu::distributed {

std::string resolve_cluster_markdown_report_path(
    const std::string& json_report_path);

bool write_cluster_report_files(
    const DistributedConfig& config,
    const ClusterReportSnapshot& snapshot,
    const std::string& json_report_path,
    std::string& error);

}  // namespace fake_gpu::distributed
