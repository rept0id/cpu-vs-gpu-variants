#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda.h>

// Fast Inverse Square Root

// Fast Inverse Square Root : CPU
void cpuFisqrt(const std::vector<float>& _arr, std::vector<float>& _result) {
    for (size_t idx = 0; idx < _arr.size(); ++idx) {
        float num = _arr[idx];

        /*** * * ***/

        // FISQRT

        int32_t i;
        float x2, y;
        const float threehalfs = 1.5f;

        x2 = num * 0.5f;
        y = num;
        // FISQRT : Bit-level floating-point hack
        i = *reinterpret_cast<int32_t*>(&y);
        // FISQRT : Magic number for the approximation
        i = 0x5f3759df - (i >> 1);
        y = *reinterpret_cast<float*>(&i);
        
        // 1st iteration of Newton's method (FISQRT)
        y = y * (threehalfs - (x2 * y * y));     
        
        // 2nd iteration of Newton's method for better accuracy (FISQRT)
        y = y * (threehalfs - (x2 * y * y));   

        /*** * * ***/  

        _result[idx] = y; // Store the result
    }
}

// Fast Inverse Square Root : GPU 

// Fast Inverse Square Root : GPU : CUDA

// Fast Inverse Square Root : GPU : CUDA : kernel
__global__ void CudaGpuFisqrt(float* _arr, float* _result, size_t size) {
    int idx;

    /*** * * ***/

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*** * * ***/

    if (idx < size) {
        float num = _arr[idx];

        /*** * * ***/

        // FISQRT

        int32_t i;
        float x2, y;
        const float threehalfs = 1.5f;

        x2 = num * 0.5f;
        y = num;
        // FISQRT : Bit-level floating-point hack
        i = *reinterpret_cast<int32_t*>(&y);
        // FISQRT : Magic number for the approximation
        i = 0x5f3759df - (i >> 1);
        y = *reinterpret_cast<float*>(&i);
        
        // 1st iteration of Newton's method (FISQRT)
        y = y * (threehalfs - (x2 * y * y));                
        
        // 2nd iteration of Newton's method for better accuracy (FISQRT)
        y = y * (threehalfs - (x2 * y * y));                

        /*** * * ***/

        _result[idx] = y;
    }
}


// Benchmark

// Benchmark : CPU

// Benchmark : CPU : Fast Inverse Square Root

int cpuFisqrtBenchmark(const std::vector<float>& _req, const int timeoutSec) {
    int count = 0;

    std::vector<float> _res(_req.size());

    std::chrono::_V2::system_clock::time_point startChrono;

    /*** * * ***/

    startChrono = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - startChrono < std::chrono::seconds(timeoutSec)) {
        cpuFisqrt(_req, _res);
        count++;
    }

    /*** * * ***/

    return count;
}

int cudaGpuFisqrtBenchmark(const std::vector<float>& _req, const int timeoutSec) {
    int count = 0;

    float *_reqGpu;
    float *_resGpu;

    std::chrono::_V2::system_clock::time_point startChrono;

    int blockSize;
    int gridSize;

    /*** * * ***/

    // Allocate memory on GPU
    cudaMalloc(&_reqGpu, _req.size() * sizeof(float));
    cudaMalloc(&_resGpu, _req.size() * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(_reqGpu, _req.data(), _req.size() * sizeof(float), cudaMemcpyHostToDevice);

    startChrono = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - startChrono < std::chrono::seconds(timeoutSec)) {
        blockSize = 256;
        gridSize = (_req.size() + blockSize - 1) / blockSize;

        CudaGpuFisqrt<<<gridSize, blockSize>>>(_reqGpu, _resGpu, _req.size());
        cudaDeviceSynchronize();

        count++;
    }

    // Free GPU memory
    cudaFree(_reqGpu);
    cudaFree(_resGpu);

    return count;
}

/*** * * ***/

int main() {
    std::vector<float> _req(10'000'000);

    /*** * * ***/

    for (size_t i = 0; i < _req.size(); ++i) {
        _req[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    /*** * * ***/

    // Benchmarks

    // Benchmarks : CPU

    std::cout << "CPU calculations in one second: ";
    std::cout << cpuFisqrtBenchmark(_req, 1);
    std::cout << std::endl;

    // Benchmarks : GPU (CUDA)

    std::cout << "CUDA calculations in one second: ";
    std::cout << cudaGpuFisqrtBenchmark(_req, 1);
    std::cout << std::endl;

    /*** * * ***/

    return 0;
}

