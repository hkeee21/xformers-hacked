#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_reduce.cuh>

namespace {
template <typename T>
__global__ void rms_norm_kernel(T *input, T *gamma, T *output, const int depth,
                                          const float epsilon);

template <>
__global__ void rms_norm_kernel<float>(
                                    float* input,
                                    float* gamma,
                                    float* output,
                                    const int depth,
                                    const float epsilon) {
      typedef cub::BlockReduce<float, 512> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float s_inv_rms;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float sum_squares = 0;
      for (int i = threadIdx.x; i < depth; i += blockDim.x)
        sum_squares += input[i] * input[i];
      sum_squares = BlockReduce(temp_storage).Sum(sum_squares);

      if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_squares / depth + epsilon);

      __syncthreads();

      for (int i = threadIdx.x; i < depth; i += blockDim.x)
        output[i] = input[i] * s_inv_rms * gamma[i];
    }

template <>
__global__ void rms_norm_kernel<half>(
                                    half* input,
                                    half* gamma,
                                    half* output,
                                    const int depth,
                                    const float epsilon) {
      typedef cub::BlockReduce<float, 512> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float s_inv_rms;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float sum_squares = 0.0f;
      for (int i = threadIdx.x; i < depth; i += blockDim.x)
      {
        float temp = __half2float(input[i]);
        sum_squares += temp * temp;
      }
      sum_squares = BlockReduce(temp_storage).Sum(sum_squares);

      if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_squares / depth + epsilon);

      __syncthreads();

      for (int i = threadIdx.x; i < depth; i += blockDim.x)
        output[i] = __hmul(__hmul(input[i], __float2half(s_inv_rms)), gamma[i]);
    }

at::Tensor rms_norm(at::Tensor input,
                    at::Tensor weight, 
                    double eps){

  const int depth = input.size(2);
  const int batch_size = input.size(0) * input.size(1);

  bool is_half = input.scalar_type() == at::ScalarType::Half;
  auto options =
      torch::TensorOptions().dtype(input.dtype()).device(input.device());
  at::Tensor output = torch::empty_like(input);

  if (is_half){
    rms_norm_kernel<half><<<batch_size, 512, 0, 0>>>(
        reinterpret_cast<half *>(input.data_ptr<at::Half>()),
        reinterpret_cast<half *>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        depth,
        eps);
  }
  else{
    rms_norm_kernel<float><<<batch_size, 512, 0, 0>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        depth,
        eps);
  }

  return output;
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::rms_norm"), TORCH_FN(rms_norm));
}