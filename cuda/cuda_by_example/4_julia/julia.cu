#include "../_common_libs/includes.h"
#include "../_common_libs/handlers.h"
#include "../_common_libs/helpers.h"
#include "../_common_libs/bitmap.h"

#define SIZE_XY 1000

// CPU
struct complexNumber {
    double real, imagenary;
    complexNumber(double a, double b) : real(a), imagenary(b) {}
    double testMagnitude(void) {
        return real * real + imagenary * imagenary;
    }
    double testMagnitude2(void) {
        return real * real + imagenary * imagenary;
    }
    complexNumber operator+(const complexNumber& a) {
        return complexNumber(real + a.real, imagenary + a.imagenary);
    }
    complexNumber operator*(const complexNumber& a) {
        return complexNumber(real * a.real - imagenary * a.imagenary, imagenary * a.real + real * a.imagenary);
    }
};

int inSet(int x, int y) {
    const double scale = 1.5;
    double setX = scale * (double)(SIZE_XY / 2 - x) / (SIZE_XY / 2);
    double setY = scale * (double)(SIZE_XY / 2 - y) / (SIZE_XY / 2);

    complexNumber c(-0.8, 0.156);
    //complexNumber c(-0.5, 0.6);
    //complexNumber c(-0.5, 0);
    //complexNumber c(-0.835, -0.2321);
    //complexNumber c(0, 0.156);

    complexNumber z(setX, setY);

    for (int i = 0; i < 200; i++) {
        z = z * z + c;
        if (z.testMagnitude() > 1000)
            return 0;
    }
    return 1;
}

void mainCPU(unsigned char *bitmapPointer) {
    for (int y = 0; y < SIZE_XY; y++) {
        for (int x = 0; x < SIZE_XY; x++) {
            int offset = x + y * SIZE_XY;
            int fractalValue = inSet(x, y);
            bitmapPointer[offset * 4 + 0] = 132 * fractalValue;
            bitmapPointer[offset * 4 + 1] = 117 * fractalValue;
            bitmapPointer[offset * 4 + 2] = 172 * fractalValue;
            bitmapPointer[offset * 4 + 3] = 255;
        }
    }
}

// GPU
struct complexNumberGPU {
    double real, imagenary;
    __device__ complexNumberGPU(double a, double b) : real(a), imagenary(b) {}
    __device__ double testMagnitude(void) {
        return real * real + imagenary * imagenary;
    }
    __device__ complexNumberGPU operator+(const complexNumberGPU& a) {
        return complexNumberGPU(real + a.real, imagenary + a.imagenary);
    }
    __device__ complexNumberGPU operator*(const complexNumberGPU& a) {
        return complexNumberGPU(real * a.real - imagenary * a.imagenary, imagenary * a.real + real * a.imagenary);
    }
};

__device__ int inSetGPU(int x, int y) {
    const double scale = 1.5;
    double setX = scale * (double)(SIZE_XY / 2 - x) / (SIZE_XY / 2);
    double setY = scale * (double)(SIZE_XY / 2 - y) / (SIZE_XY / 2);

    //complexNumberGPU c(-0.8, 0.156);
    complexNumberGPU c(0, 0.156);
    complexNumberGPU z(setX, setY);

    for (int i = 0; i < 200; i++) {
        z = z * z + c;
        if (z.testMagnitude() > 1000)
            return 0;
    }
    return 1;
}

__global__ void mainGPU(unsigned char *bitmapPointer) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int fractalValue = inSetGPU(x, y);
    bitmapPointer[offset * 4 + 0] = 132 * fractalValue;
    bitmapPointer[offset * 4 + 1] = 117 * fractalValue;
    bitmapPointer[offset * 4 + 2] = 172 * fractalValue;
    bitmapPointer[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *devBitmapPointer;
};

int main(void) {
    DataBlock   data;
    CPUBitmap bitmap(SIZE_XY, SIZE_XY, &data);

    // CPU
    //unsigned char *bitmapPointer;
    //mainCPU(bitmapPointer);

    // GPU
    unsigned char *devBitmapPointer;
    HANDLE_ERROR(cudaMalloc((void**)&devBitmapPointer, bitmap.image_size()));
    data.devBitmapPointer = devBitmapPointer;
    dim3 grid(SIZE_XY, SIZE_XY);
    mainGPU<<<grid, 1>>>(devBitmapPointer);
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), devBitmapPointer,bitmap.image_size(),cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(devBitmapPointer));

    bitmap.display_and_exit();

    //printf("\n");
    //printf("Press any key to continue...");
    //getchar();

    return 0;
}
