#include "ggml.h"
#include <iostream>
#include <vector>

int main() {
    struct ggml_init_params params = {
        /* .mem_size = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // A: 2x3 matrix (GGML ne[0]=2, ne[1]=3 -> 3 rows, 2 cols)
    struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    float A_data[6] = {
        1, 2,  // row 0
        3, 4,  // row 1
        5, 6   // row 2
    };
    memcpy(A->data, A_data, sizeof(A_data));
    
    // B: 2x4 matrix (GGML ne[0]=2, ne[1]=4 -> 4 rows, 2 cols)
    struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    float B_data[8] = {
        1, 1,  // row 0
        0, 1,  // row 1
        1, 0,  // row 2
        0, 0   // row 3
    };
    memcpy(B->data, B_data, sizeof(B_data));
    
    // C = A * B
    struct ggml_tensor* C = ggml_mul_mat(ctx, A, B);
    
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(gf, C);
    
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    std::cout << "C shape: ne[0]=" << C->ne[0] << ", ne[1]=" << C->ne[1] << std::endl;
    float* C_data = (float*)C->data;
    for (int i = 0; i < C->ne[1]; i++) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < C->ne[0]; j++) {
            std::cout << C_data[i * C->ne[0] + j] << " ";
        }
        std::cout << std::endl;
    }
    
    ggml_free(ctx);
    return 0;
}
