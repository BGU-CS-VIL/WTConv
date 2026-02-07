#include <torch/extension.h>

// Single level forward/backward (haar_single.cu)
void haar2d_forward(torch::Tensor input, torch::Tensor output);
void haar2d_backward(torch::Tensor grad_output, torch::Tensor grad_input);

// Single level inverse (haar_inverse.cu)
void haar2d_inverse(torch::Tensor input, torch::Tensor output);
void haar2d_inverse_backward(torch::Tensor grad_output, torch::Tensor grad_input);

// Fused forward cascade (haar_forward_cascade.cu)
void haar2d_double_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2);
void haar2d_triple_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2, torch::Tensor level3);
void haar2d_quad_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4);
void haar2d_quint_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor level5);

// Fused forward cascade backward (haar_forward_cascade.cu)
void haar2d_double_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_input);
void haar2d_triple_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_level3, torch::Tensor grad_input);
void haar2d_quad_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_level3, torch::Tensor grad_level4, torch::Tensor grad_input);
void haar2d_quint_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_level3, torch::Tensor grad_level4, torch::Tensor grad_level5, torch::Tensor grad_input);

// Fused inverse cascade (haar_inverse_cascade.cu)
void ihaar2d_double_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor output);
void ihaar2d_triple_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor output);
void ihaar2d_quad_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor output);
void ihaar2d_quint_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor level5, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Single level
    m.def("haar2d_forward", &haar2d_forward, "Haar 2D forward (CUDA)");
    m.def("haar2d_backward", &haar2d_backward, "Haar 2D backward (CUDA)");
    m.def("haar2d_inverse", &haar2d_inverse, "Haar 2D inverse (CUDA)");
    m.def("haar2d_inverse_backward", &haar2d_inverse_backward, "Haar 2D inverse backward (CUDA)");
    
    // Fused forward cascade
    m.def("haar2d_double_cascade", &haar2d_double_cascade, "2-level fused forward Haar (CUDA)");
    m.def("haar2d_triple_cascade", &haar2d_triple_cascade, "3-level fused forward Haar (CUDA)");
    m.def("haar2d_quad_cascade", &haar2d_quad_cascade, "4-level fused forward Haar (CUDA)");
    m.def("haar2d_quint_cascade", &haar2d_quint_cascade, "5-level fused forward Haar (CUDA)");
    
    // Fused forward cascade backward
    m.def("haar2d_double_cascade_backward", &haar2d_double_cascade_backward, "2-level fused forward Haar backward (CUDA)");
    m.def("haar2d_triple_cascade_backward", &haar2d_triple_cascade_backward, "3-level fused forward Haar backward (CUDA)");
    m.def("haar2d_quad_cascade_backward", &haar2d_quad_cascade_backward, "4-level fused forward Haar backward (CUDA)");
    m.def("haar2d_quint_cascade_backward", &haar2d_quint_cascade_backward, "5-level fused forward Haar backward (CUDA)");
    
    // Fused inverse cascade
    m.def("ihaar2d_double_cascade", &ihaar2d_double_cascade, "2-level fused inverse Haar (CUDA)");
    m.def("ihaar2d_triple_cascade", &ihaar2d_triple_cascade, "3-level fused inverse Haar (CUDA)");
    m.def("ihaar2d_quad_cascade", &ihaar2d_quad_cascade, "4-level fused inverse Haar (CUDA)");
    m.def("ihaar2d_quint_cascade", &ihaar2d_quint_cascade, "5-level fused inverse Haar (CUDA)");
}
