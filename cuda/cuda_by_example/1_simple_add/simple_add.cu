#include "../_common_libs/includes.h"
#include "../_common_libs/handlers.h"

// 0 - Hello.World
//__global__ void kernel(void) {
//
//}
//
//int main(void)
//{
//    kernel<<<1, 1>>>();
//    printf("Hello, world!\n");
//
//    return 0;
//}

// 1 Add
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(c)));
    add<<<1, 1 >>> (100, 66, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(dev_c), cudaMemcpyDeviceToHost));

    printf("100 + 66 = %d\n", c);
    cudaFree(dev_c);

    printf("Press any key to continue...");
    getchar();

    return 0;
}
