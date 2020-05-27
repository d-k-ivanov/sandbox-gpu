#include "lib/includes.h"
#include "lib/handlers.h"
#include "lib/helpers.h"
#include "lib/bitmap.h"

// In Debug mode image size is restricted (I don't know why)
//#define SIZE_XY 885
//#define SIZE_XY 1000
#define SIZE_X 1920
#define SIZE_Y 1080

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
    // double setX = scale * (double)(SIZE_XY / 2 - x) / (SIZE_XY / 2);
    // double setY = scale * (double)(SIZE_XY / 2 - y) / (SIZE_XY / 2);
    double setX = scale * (double)(SIZE_X / 2 - x) / (SIZE_X / 2);
    double setY = scale * (double)(SIZE_Y / 2 - y) / (SIZE_Y / 2);

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
    for (int y = 0; y < SIZE_Y; y++) {
        for (int x = 0; x < SIZE_X; x++) {
            int offset = x + y * SIZE_X;
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
    //double setX = scale * (double)(SIZE_XY / 2 - x) / (SIZE_XY / 2);
    //double setY = scale * (double)(SIZE_XY / 2 - y) / (SIZE_XY / 2);
    double setX = scale * (double)(SIZE_X / 2 - x) / (SIZE_X / 2);
    double setY = scale * (double)(SIZE_Y / 2 - y) / (SIZE_Y / 2);

    complexNumberGPU c(-0.8, 0.156);
    //complexNumberGPU c(0, 0.156);
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
__global__ struct DataBlock {
    unsigned char   *devBitmapPointer;
};

int main(void) {
    // --------------- CPU Version ---------------
    // CPUBitmap bitmap(SIZE_XY, SIZE_XY);
    //CPUBitmap bitmap(SIZE_X, SIZE_Y);
    //unsigned char *bitmapPointer = bitmap.get_ptr();
    //mainCPU(bitmapPointer);
    //bitmap.display_and_exit();

    // --------------- GPU Version ---------------
    DataBlock data;
    // CPUBitmap bitmap(SIZE_XY, SIZE_XY, &data);
    CPUBitmap bitmap(SIZE_X, SIZE_Y, &data);
    //dim3 grid(SIZE_XY, SIZE_XY);
    dim3 grid(SIZE_X, SIZE_Y);

    HANDLE_ERROR(cudaMalloc((void**)&data.devBitmapPointer, bitmap.image_size()));
    mainGPU<<<grid,1>>>(data.devBitmapPointer);

    printf("Bitmap: %d\n", bitmap.image_size());
    printf("Press any key to continue...");
    // getchar(); // View console
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), data.devBitmapPointer, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(data.devBitmapPointer));
    bitmap.display_and_exit();

    return 0;
}
