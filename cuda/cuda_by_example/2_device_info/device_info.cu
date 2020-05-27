#include "../_common_libs/includes.h"
#include "../_common_libs/handlers.h"
#include "../_common_libs/helpers.h"

__global__ void kernel() {

}

int main(void) {
    cudaDeviceProp dev_prop;
    int driverVersion = 0, runtimeVersion = 0;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Number of GPU Devices: %d\n", count);

    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaGetDeviceProperties(&dev_prop, i));

        printf("Device properties for device %d - %s:\n", i, dev_prop.name);

        // Driver:
        HANDLE_ERROR(cudaDriverGetVersion(&driverVersion));
        HANDLE_ERROR(cudaRuntimeGetVersion(&runtimeVersion));
        printf("CUDA Driver Version / Runtime Version	%d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("\n");

        printf("Memory: %.0f MBytes (%llu bytes)\n", (float)dev_prop.totalGlobalMem / 1048576.0f, (unsigned long long) dev_prop.totalGlobalMem);
        printf("\n");

        printf("Multiprocessors: %d\n", dev_prop.multiProcessorCount);
        printf("Cores per multiprocessor: %d\n", _ConvertSMVer2Cores(dev_prop.major, dev_prop.minor));
        printf("Total cores: %d\n", _ConvertSMVer2Cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
        printf("\n");

        printf("GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n", dev_prop.clockRate * 1e-3f, dev_prop.clockRate * 1e-6f);
        printf("\n");

        printf("Memory Clock rate: %.0f Mhz\n", dev_prop.memoryClockRate * 1e-3f);
        printf("Memory Bus Width: %d-bit\n", dev_prop.memoryBusWidth);
        if (dev_prop.l2CacheSize)
        {
            printf("  L2 Cache Size: %d bytes\n", dev_prop.l2CacheSize);
        }
        printf("\n");

        printf("Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
            dev_prop.maxTexture1D, dev_prop.maxTexture2D[0], dev_prop.maxTexture2D[1],
            dev_prop.maxTexture3D[0], dev_prop.maxTexture3D[1], dev_prop.maxTexture3D[2]);
        printf("Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
            dev_prop.maxTexture1DLayered[0], dev_prop.maxTexture1DLayered[1]);
        printf("Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
            dev_prop.maxTexture2DLayered[0], dev_prop.maxTexture2DLayered[1], dev_prop.maxTexture2DLayered[2]);
        printf("\n");

        printf("Total amount of constant memory:               %lu bytes\n", dev_prop.totalConstMem);
        printf("Total amount of shared memory per block:       %lu bytes\n", dev_prop.sharedMemPerBlock);
        printf("Total number of registers available per block: %d\n", dev_prop.regsPerBlock);
        printf("Warp size:                                     %d\n", dev_prop.warpSize);
        printf("Maximum number of threads per multiprocessor:  %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:           %d\n", dev_prop.maxThreadsPerBlock);
        printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
            dev_prop.maxThreadsDim[0],
            dev_prop.maxThreadsDim[1],
            dev_prop.maxThreadsDim[2]);
        printf("Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
            dev_prop.maxGridSize[0],
            dev_prop.maxGridSize[1],
            dev_prop.maxGridSize[2]);
        printf("Maximum memory pitch:                          %lu bytes\n", dev_prop.memPitch);
        printf("Texture alignment:                             %lu bytes\n", dev_prop.textureAlignment);
        printf("Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (dev_prop.deviceOverlap ? "Yes" : "No"), dev_prop.asyncEngineCount);
        printf("Run time limit on kernels:                     %s\n", dev_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("Integrated GPU sharing Host Memory:            %s\n", dev_prop.integrated ? "Yes" : "No");
        printf("Support host page-locked memory mapping:       %s\n", dev_prop.canMapHostMemory ? "Yes" : "No");
        printf("Alignment requirement for Surfaces:            %s\n", dev_prop.surfaceAlignment ? "Yes" : "No");
        printf("Device has ECC support:                        %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("\n");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("CUDA Device Driver Mode (TCC or WDDM):         %s\n", dev_prop.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("Device supports Unified Addressing (UVA):      %s\n", dev_prop.unifiedAddressing ? "Yes" : "No");
        printf("Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", dev_prop.pciDomainID, dev_prop.pciBusID, dev_prop.pciDeviceID);

    }

    printf("Press any key to continue...");
    getchar();

    return 0;
}
