#include "../_common_libs/includes.h"
#include "../_common_libs/handlers.h"
#include "../_common_libs/helpers.h"

//#define vecSize 50000
#define vecSize 50000
void addVectorCPU(int *a, int *b, int *c) {

    int threadID = 0;
    while (threadID < vecSize) {
        c[threadID] = a[threadID] + b[threadID];
        if (threadID % 1000 == 0)
            printf("%d elements\n", threadID);
        //printf("%d : %d : %d | ", a[threadID], b[threadID], c[threadID]);
        threadID += 1;
    }

}

__global__ void addVectorGPU(int *a, int *b, int *c) {
    int threadID = blockIdx.x;
    if (threadID < vecSize)
        c[threadID] = a[threadID] + b[threadID];
}

void mainCPU(int *a, int *b, int *c) {
    addVectorCPU(a, b, c);
    printf("CPU version: \n");
    /*for (int i = 0; i < vecSize; i++) {
        printf("%6d + %9d = %d\n", a[i], b[i], c[i]);
    }*/
}

void mainGPU(int *a, int *b, int *c) {
    int *dev_a, *dev_b, *dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, vecSize * sizeof(vecSize)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, vecSize * sizeof(vecSize)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, vecSize * sizeof(vecSize)));
    HANDLE_ERROR(cudaMemcpy(dev_a, a, vecSize * sizeof(vecSize), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, vecSize * sizeof(vecSize), cudaMemcpyHostToDevice));
    addVectorGPU<<<vecSize, 1>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaMemcpy(c, dev_c, vecSize * sizeof(vecSize), cudaMemcpyDeviceToHost));
    printf("GPU version: \n");
    /*for (int i = 0; i < vecSize; i++) {
        printf("%6d + %9d = %d\n", a[i], b[i], c[i]);
    }*/
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}


int main(void) {


    // Generate a vector
    int a[vecSize], b[vecSize], cCPU[vecSize], cGPU[vecSize];
    srand(time(NULL));
    for (int i = 0; i < vecSize; i++) {
        a[i] = rand() % 200001;
        b[i] = rand() % 999999999;
        //a[i] = i;
        //b[i] = i * i;
        //printf("%d : %d | ", a[i], b[i]);
    }
    printf("\n");

    clock_t start = clock();
    // CPU
    mainCPU(a, b, cCPU);
    // GPU
    //mainGPU(a, b, cGPU);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Extcution tine: %f seconds\n", seconds);

    printf("Press any key to continue...");
    getchar();

    return 0;
}
