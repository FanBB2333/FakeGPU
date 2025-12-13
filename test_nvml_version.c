#include <stdio.h>
#include <dlfcn.h>

typedef int (*nvmlInit_t)();
typedef int (*nvmlSystemGetNVMLVersion_t)(char*, unsigned int);
typedef int (*nvmlSystemGetDriverVersion_t)(char*, unsigned int);
typedef int (*nvmlShutdown_t)();

int main() {
    void* handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!handle) {
        printf("Failed to load library: %s\n", dlerror());
        return 1;
    }

    nvmlInit_t nvmlInit = (nvmlInit_t)dlsym(handle, "nvmlInit");
    nvmlSystemGetNVMLVersion_t nvmlSystemGetNVMLVersion =
        (nvmlSystemGetNVMLVersion_t)dlsym(handle, "nvmlSystemGetNVMLVersion");
    nvmlSystemGetDriverVersion_t nvmlSystemGetDriverVersion =
        (nvmlSystemGetDriverVersion_t)dlsym(handle, "nvmlSystemGetDriverVersion");
    nvmlShutdown_t nvmlShutdown = (nvmlShutdown_t)dlsym(handle, "nvmlShutdown");

    if (!nvmlInit || !nvmlSystemGetNVMLVersion || !nvmlSystemGetDriverVersion || !nvmlShutdown) {
        printf("Failed to find functions\n");
        return 1;
    }

    printf("Calling nvmlInit...\n");
    int ret = nvmlInit();
    printf("nvmlInit returned: %d\n", ret);

    char version[80];
    printf("Calling nvmlSystemGetNVMLVersion...\n");
    ret = nvmlSystemGetNVMLVersion(version, sizeof(version));
    printf("nvmlSystemGetNVMLVersion returned: %d, version: %s\n", ret, version);

    printf("Calling nvmlSystemGetDriverVersion...\n");
    ret = nvmlSystemGetDriverVersion(version, sizeof(version));
    printf("nvmlSystemGetDriverVersion returned: %d, version: %s\n", ret, version);

    printf("Calling nvmlShutdown...\n");
    nvmlShutdown();

    dlclose(handle);
    return 0;
}
