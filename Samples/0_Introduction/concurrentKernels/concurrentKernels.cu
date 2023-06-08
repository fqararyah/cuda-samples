/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// This sample demonstrates the use of streams for concurrent execution. It also
// illustrates how to introduce dependencies between CUDA streams with the
// cudaStreamWaitEvent function.
//

// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <helper_functions.h>

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
__global__ void clock_block(clock_t *d_o, clock_t clock_count)
{
  unsigned int start_clock = (unsigned int)clock();

  clock_t clock_offset = 0;

  while (clock_offset < clock_count)
  {
    unsigned int end_clock = (unsigned int)clock();

    // The code below should work like
    // this (thanks to modular arithmetics):
    //
    // clock_offset = (clock_t) (end_clock > start_clock ?
    //                           end_clock - start_clock :
    //                           end_clock + (0xffffffffu - start_clock));
    //
    // Indeed, let m = 2^32 then
    // end - start = end + m - start (mod m).

    clock_offset = (clock_t)(end_clock - start_clock);
  }

  d_o[0] = clock_offset;
}

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

// Single warp reduction kernel
__global__ void sum(clock_t *d_clocks, int N)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ clock_t s_clocks[32];

  clock_t my_sum = 0;

  for (int i = threadIdx.x; i < N; i += blockDim.x)
  {
    my_sum += d_clocks[i];
  }

  s_clocks[threadIdx.x] = my_sum;
  cg::sync(cta);

  for (int i = 16; i > 0; i /= 2)
  {
    if (threadIdx.x < i)
    {
      s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
    }

    cg::sync(cta);
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv)
{
  int nkernels = 7;                        // number of concurrent kernels
  int nstreams = nkernels + 1;             // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(clock_t); // number of data bytes
  float kernel_time = 10;                  // time the kernel should run in ms
  float elapsed_time;                      // timing variables
  int cuda_device = 0;

  printf("[%s] - Starting...\n", argv[0]);

  // get number of kernels if overridden on the command line
  if (checkCmdLineFlag(argc, (const char **)argv, "nkernels"))
  {
    nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
    nstreams = nkernels + 1;
  }

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  cuda_device = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDevice(&cuda_device));

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

  if ((deviceProp.concurrentKernels == 0))
  {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

  // allocate host memory
  clock_t *a = 0; // pointer to the array data in host memory
  checkCudaErrors(cudaMallocHost((void **)&a, nbytes));

  // allocate device memory
  clock_t *d_a = 0; // pointers to data and init value in the device memory
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

  // allocate and initialize an array of stream handles
  cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  for (int i = 0; i < nstreams; i++)
  {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

  // create CUDA event handles
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  cudaEvent_t *kernelEvent;
  kernelEvent = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));

  for (int i = 0; i < nkernels; i++)
  {
    checkCudaErrors(
        cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
  }

  //////////////////////////////////////////////////////////////////////
  // time execution with nkernels streams
  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to
  // prevent hangs reduce time_clocks.
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif

  //**************************************************************************************************
  // cudaEventRecord(start_event, 0);
  // // queue nkernels in separate streams and record when they are done
  // for (int i = 0; i < nkernels; ++i) {
  //   clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
  //   total_clocks += time_clocks;
  //   checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));

  //   // make the last stream wait for the kernel event to be recorded
  //   checkCudaErrors(
  //       cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0));
  // }
  int numElements = 50000;
  cudaError_t err = cudaSuccess;

  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 1024;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  cudaEventRecord(start_event, 0);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(d_A, d_B, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(d_C, d_A, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[2]>>>(d_C, d_B, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[3]>>>(d_C, d_A, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[4]>>>(d_C, d_B, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[5]>>>(d_C, d_A, d_C, numElements);
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[6]>>>(d_C, d_B, d_C, numElements);

  // make the last stream wait for the kernel event to be recorded
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[0], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[1], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[2], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[3], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[4], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[5], 0));
  checkCudaErrors(
      cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[6], 0));

  // in this sample we just wait until the GPU is done
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  bool vector_test_passed = true;
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(4 * h_A[i] + 4 * h_B[i] - h_C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      vector_test_passed = false;
      // exit(EXIT_FAILURE);
    }
  }

  if (vector_test_passed)
  {
    printf("vector add Test PASSED\n");
  }

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  //**************************************************************************************************

  printf("Expected time for serial execution of %d kernels = %.3fms\n", nkernels,
         nkernels * kernel_time);
  printf("Expected time for concurrent execution of %d kernels = %.3fms\n",
         nkernels, kernel_time);
  printf("Measured time for sample = %.3fms\n", elapsed_time);

  bool bTestResult = (a[0] > total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++)
  {
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(kernelEvent[i]);
  }

  free(streams);
  free(kernelEvent);

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFreeHost(a);
  cudaFree(d_a);

  if (!bTestResult)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
