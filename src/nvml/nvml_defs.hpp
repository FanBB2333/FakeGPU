#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct nvmlMemory_st {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct nvmlMemory_v2_st {
    unsigned int version;
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_v2_t;

#define nvmlMemory_v2 2

typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

// Field value queries (nvmlDeviceGetFieldValues)
typedef enum {
    NVML_VALUE_TYPE_DOUBLE = 0,
    NVML_VALUE_TYPE_UNSIGNED_INT = 1,
    NVML_VALUE_TYPE_UNSIGNED_LONG = 2,
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
    NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4,
    NVML_VALUE_TYPE_SIGNED_INT = 5,
    NVML_VALUE_TYPE_UNSIGNED_SHORT = 6,
    NVML_VALUE_TYPE_COUNT = 7,
} nvmlValueType_t;

typedef union nvmlValue_st {
    double dVal;
    unsigned int uiVal;
    unsigned long ulVal;
    unsigned long long ullVal;
    long long sllVal;
    int siVal;
    unsigned short usVal;
} nvmlValue_t;

typedef struct nvmlFieldValue_st {
    uint32_t fieldId;
    uint32_t scopeId;
    int64_t timestamp;
    int64_t latencyUsec;
    nvmlValueType_t valueType;
    nvmlReturn_t nvmlReturn;
    nvmlValue_t value;
} nvmlFieldValue_t;

typedef struct nvmlPciInfo_st {
    // pynvml's nvmlPciInfo_t expects:
    // busIdLegacy (16 bytes) + domain + bus + device + pciDeviceId + pciSubSystemId + busId (32 bytes)
    char busIdLegacy[16];    // Legacy bus ID (16 bytes)
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];          // Current bus ID (32 bytes)
} nvmlPciInfo_t;

typedef struct nvmlPciInfo_v3_st {
    char busIdLegacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} nvmlPciInfo_v3_t;

// Constants
#define NVML_DEVICE_NAME_BUFFER_SIZE 64
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE 16
#define NVML_DEVICE_PART_NUMBER_BUFFER_SIZE 80
#define NVML_GSP_FIRMWARE_VERSION_BUF_SIZE 0x40

// Versioned structs/constants used by nvidia-smi and pynvml.
#define nvmlDeviceAddressingMode_v1 0x1000008

typedef struct nvmlDeviceAddressingMode_st {
    unsigned int version;
    unsigned int value;
} nvmlDeviceAddressingMode_t;

// Addressing mode values
#define NVML_DEVICE_ADDRESSING_MODE_NONE 0
#define NVML_DEVICE_ADDRESSING_MODE_HMM 1
#define NVML_DEVICE_ADDRESSING_MODE_ATS 2

#define nvmlPciInfoExt_v1 0x1000040

typedef struct nvmlPciInfoExt_v1_st {
    unsigned int version;
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    unsigned int baseClass;
    unsigned int subClass;
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
} nvmlPciInfoExt_v1_t;

typedef struct nvmlFBCStats_st {
    unsigned int sessionsCount;
    unsigned int averageFPS;
    unsigned int averageLatency;
} nvmlFBCStats_t;

typedef struct nvmlRowRemapperHistogramValues_st {
    unsigned int max;
    unsigned int high;
    unsigned int partial;
    unsigned int low;
    unsigned int none;
} nvmlRowRemapperHistogramValues_t;

typedef struct nvmlBridgeChipInfo_st {
    unsigned int type;
    unsigned int fwVersion;
} nvmlBridgeChipInfo_t;

typedef struct nvmlBridgeChipHierarchy_st {
    unsigned int bridgeCount;
    nvmlBridgeChipInfo_t bridgeChipInfo[128];
} nvmlBridgeChipHierarchy_t;

typedef struct nvmlC2cModeInfo_v1_st {
    unsigned int isC2cEnabled;
} nvmlC2cModeInfo_v1_t;

#define nvmlDramEncryptionInfo_v1 0x01000008

typedef struct nvmlDramEncryptionInfo_st {
    unsigned int version;
    unsigned int encryptionState;
} nvmlDramEncryptionInfo_t;

#define nvmlMarginTemperature_v1 0x1000008

typedef struct nvmlMarginTemperature_v1_st {
    unsigned int version;
    int marginTemperature;
} nvmlMarginTemperature_v1_t;

#define VgpuHeterogeneousMode_v1 0x1000008

typedef struct nvmlVgpuHeterogeneousMode_v1_st {
    unsigned int version;
    unsigned int mode;
} nvmlVgpuHeterogeneousMode_v1_t;

#define NVML_GPU_FABRIC_UUID_LEN 16
#define nvmlGpuFabricInfo_v2 0x02000024
#define nvmlGpuFabricInfo_v3 0x3000028

typedef struct nvmlGpuFabricInfo_v2_st {
    unsigned int version;
    unsigned char clusterUuid[NVML_GPU_FABRIC_UUID_LEN];
    nvmlReturn_t status;
    unsigned int cliqueId;
    unsigned int state;
    unsigned int healthMask;
} nvmlGpuFabricInfo_v2_t;

typedef struct nvmlGpuFabricInfo_v3_st {
    unsigned int version;
    unsigned char clusterUuid[NVML_GPU_FABRIC_UUID_LEN];
    nvmlReturn_t status;
    unsigned int cliqueId;
    unsigned int state;
    unsigned int healthMask;
    unsigned char healthSummary;
    unsigned char reserved[3];
} nvmlGpuFabricInfo_v3_t;

// Virtualization mode
typedef enum {
    NVML_GPU_VIRTUALIZATION_MODE_NONE = 0,
    NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1,
    NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2,
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3,
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA = 4
} nvmlGpuVirtualizationMode_t;

#ifdef __cplusplus
}
#endif
