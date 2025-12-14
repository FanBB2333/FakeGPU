#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <atomic>
#include "../core/global_state.hpp"

namespace fake_gpu {

// Global flag to track if we've already dumped the report
static std::atomic<bool> g_report_dumped{false};
static std::atomic<bool> g_shutting_down{false};

class ResourceMonitor {
public:
    static ResourceMonitor& instance() {
        static ResourceMonitor s_instance;
        return s_instance;
    }

    ResourceMonitor() {
        printf("[Monitor] Initialized\n");
    }

    ~ResourceMonitor() {
        // Try to dump report if not already done
        if (!g_report_dumped.load()) {
            dump_report_internal();
        }
        // Mark as shutting down to prevent further operations
        g_shutting_down.store(true);
    }

    void dump_report() {
        // Only dump report once, and not during shutdown
        if (g_shutting_down.load() || g_report_dumped.exchange(true)) {
            return;
        }
        dump_report_internal();
    }

private:
    void dump_report_internal() {
        try {
            const char* report_path = "fake_gpu_report.json";
            GlobalState& gs = GlobalState::instance();
            int count = gs.get_device_count();
            printf("[Monitor] Dumping report to %s. GlobalState Addr: %p, Device Count: %d\n", report_path, (void*)&gs, count);

            FILE* out = fopen(report_path, "w");
            if (!out) {
                printf("[Monitor] Failed to open report file\n");
                return;
            }

            fprintf(out, "{\n");
            fprintf(out, "  \"devices\": [\n");

            for (int i = 0; i < count; ++i) {
                Device& dev = gs.get_device(i);
                fprintf(out, "    {\n");
                fprintf(out, "      \"name\": \"%s\",\n", dev.name.c_str());
                fprintf(out, "      \"uuid\": \"%s\",\n", dev.uuid.c_str());
                fprintf(out, "      \"total_memory\": %lu,\n", (unsigned long)dev.total_memory);
                fprintf(out, "      \"used_memory_peak\": %lu,\n", (unsigned long)dev.used_memory_peak);
                fprintf(out, "      \"used_memory_current\": %lu\n", (unsigned long)dev.used_memory);
                fprintf(out, "    }%s\n", (i < count - 1 ? "," : ""));
            }

            fprintf(out, "  ]\n");
            fprintf(out, "}\n");
            fclose(out);
            g_report_dumped.store(true);
        } catch (...) {
            printf("[Monitor] Exception during report dump\n");
        }
    }
};

// Static instance to force construction/destruction
static ResourceMonitor& s_monitor = ResourceMonitor::instance();

// Public function to dump report
void dump_monitor_report() {
    ResourceMonitor::instance().dump_report();
}

} // namespace fake_gpu
