#include "dwconv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dwconv_forward_backward", &dwconv_forward_backward, "forward_backward");
  m.def("dwconv_weight_backward", &dwconv_weight_backward, "weight_backward");
  m.def("inverse_neigh", &inverse_neigh, "inverse_neigh");
    m.def("dwconv_forward_backward_t", &dwconv_forward_backward_t, "forward_backward_t");
  m.def("dwconv_weight_backward_t", &dwconv_weight_backward_t, "weight_backward_t");
}
