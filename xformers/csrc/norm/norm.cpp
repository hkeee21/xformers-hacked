#include <torch/types.h>
#include <ATen/ATen.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::rms_norm(Tensor input, Tensor weight, float eps) -> Tensor"));
}