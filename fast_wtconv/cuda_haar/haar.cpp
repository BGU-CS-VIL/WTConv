#include <torch/extension.h>
#include <vector>

// Fused Haar -> depthwise conv -> scale (fused_haar_conv.cu)
void fused_haar_conv_forward(torch::Tensor input, torch::Tensor fused_weight,
                             torch::Tensor output, c10::optional<torch::Tensor> ll_output);
void fused_haar_conv_backward(torch::Tensor grad_output, torch::Tensor fused_weight,
                              torch::Tensor grad_input, c10::optional<torch::Tensor> grad_ll);
void fused_haar_grad_weight(torch::Tensor input, torch::Tensor grad_output,
                            torch::Tensor grad_fused_weight);
void haar_coeffs(torch::Tensor input, torch::Tensor output);

// Depthwise conv weight gradient for the base-conv path (depthwise_grad.cu)
void depthwise_grad_weight(torch::Tensor input, torch::Tensor grad_output,
                           torch::Tensor grad_weight);

// Fused inverse Haar cascade with optional fused add (ihaar_cascade.cu)
void ihaar_cascade(std::vector<torch::Tensor> levels, torch::Tensor output,
                   c10::optional<torch::Tensor> add);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_haar_conv_forward", &fused_haar_conv_forward,
          "Fused Haar + depthwise conv + scale, forward (CUDA)",
          py::arg("input"), py::arg("fused_weight"), py::arg("output"),
          py::arg("ll_output") = py::none());
    m.def("fused_haar_conv_backward", &fused_haar_conv_backward,
          "Fused Haar + depthwise conv + scale, grad w.r.t. input (CUDA)",
          py::arg("grad_output"), py::arg("fused_weight"), py::arg("grad_input"),
          py::arg("grad_ll") = py::none());
    m.def("fused_haar_grad_weight", &fused_haar_grad_weight,
          "Weight gradient straight from the level input, no coefficients (CUDA)");
    m.def("haar_coeffs", &haar_coeffs,
          "Single-level Haar coefficients, (B,C,H,W) -> (B,C,4,H2,W2) (CUDA)");
    m.def("depthwise_grad_weight", &depthwise_grad_weight,
          "Depthwise conv weight gradient, stride 1, 'same' padding (CUDA)");
    m.def("ihaar_cascade", &ihaar_cascade,
          "Fused 1-5 level inverse Haar cascade with optional fused add (CUDA)",
          py::arg("levels"), py::arg("output"), py::arg("add") = py::none());
}
