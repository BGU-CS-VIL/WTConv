// haar.mm - Objective-C++ bindings for Metal Haar kernels
// Properly integrates with PyTorch MPS backend using torch::mps APIs

#import <Metal/Metal.h>
#import <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <string>
#include <unordered_map>

// Helper to get MTLBuffer from PyTorch tensor
// This is the proper way to access the Metal buffer from a PyTorch MPS tensor
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Synchronize MPS stream before dispatching custom kernels
static inline void sync_mps() {
    torch::mps::synchronize();
}

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;
static NSString* g_sourcePath = nil;

void set_metal_source_path(std::string path) {
    g_sourcePath = [NSString stringWithUTF8String:path.c_str()];
}

// Initialize Metal device and load library
static void init_metal() {
    if (g_device != nil) return;
    
    g_device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(g_device != nil, "No Metal device found");
    
    // Try to load precompiled metallib
    NSError* error = nil;
    NSString* bundlePath = [[NSBundle mainBundle] bundlePath];
    NSArray* searchPaths = @[
        @"metal_haar/haar_kernels.metallib",
        [bundlePath stringByAppendingPathComponent:@"haar_kernels.metallib"],
        [[NSFileManager defaultManager] currentDirectoryPath]
    ];
    
    if (g_sourcePath != nil) {
        searchPaths = @[[g_sourcePath stringByAppendingPathComponent:@"haar_kernels.metallib"]];
    }
    
    for (NSString* basePath in searchPaths) {
        NSString* fullPath = basePath;
        if (![basePath hasSuffix:@".metallib"]) {
            fullPath = [basePath stringByAppendingPathComponent:@"haar_kernels.metallib"];
        }
        if ([[NSFileManager defaultManager] fileExistsAtPath:fullPath]) {
            NSURL* libURL = [NSURL fileURLWithPath:fullPath];
            g_library = [g_device newLibraryWithURL:libURL error:&error];
            if (g_library != nil) {
                NSLog(@"Loaded Metal library from: %@", fullPath);
                break;
            }
        }
    }
    
    // If metallib not found, compile from source
    if (g_library == nil) {
        NSString* currentDir = [[NSFileManager defaultManager] currentDirectoryPath];
        NSArray* metalFiles = @[
            @"haar_single.metal",
            @"haar_inverse.metal",
            @"haar_forward_cascade.metal",
            @"haar_inverse_cascade.metal"
        ];
        
        NSMutableString* allSource = [NSMutableString string];
        for (NSString* file in metalFiles) {
            NSString* path;
            if (g_sourcePath != nil) {
                path = [g_sourcePath stringByAppendingPathComponent:file];
            } else {
                NSString* currentDir = [[NSFileManager defaultManager] currentDirectoryPath];
                path = [[currentDir stringByAppendingPathComponent:@"metal_haar"] 
                                  stringByAppendingPathComponent:file];
            }
            NSString* source = [NSString stringWithContentsOfFile:path 
                                                         encoding:NSUTF8StringEncoding 
                                                            error:&error];
            if (source != nil) {
                [allSource appendString:source];
                [allSource appendString:@"\n"];
            }
        }
        
        if ([allSource length] > 0) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            g_library = [g_device newLibraryWithSource:allSource options:options error:&error];
        }
    }
    
    TORCH_CHECK(g_library != nil, "Failed to load Metal library: ",
                error ? [[error localizedDescription] UTF8String] : "library not found");
}

// Get or create compute pipeline
static id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    init_metal();
    
    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) {
        return it->second;
    }
    
    NSString* funcName = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> function = [g_library newFunctionWithName:funcName];
    TORCH_CHECK(function != nil, "Metal function not found: ", name);
    
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pipeline != nil, "Failed to create pipeline: ", [[error localizedDescription] UTF8String]);
    
    g_pipelines[name] = pipeline;
    return pipeline;
}

// Get kernel name based on dtype
static std::string kernel_name(const std::string& base, const torch::Tensor& t) {
    if (t.scalar_type() == torch::kFloat16) {
        return base + "_kernel_half";
    }
    return base + "_kernel";
}

// Get cascade kernel name based on dtype
static std::string cascade_kernel_name(const std::string& base, const torch::Tensor& t) {
    if (t.scalar_type() == torch::kFloat16) {
        return base + "_half";
    }
    return base;
}

// =============================================================================
// Single Level Forward
// =============================================================================

void haar2d_forward_metal(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_mps(), "Input must be on MPS device");
    TORCH_CHECK(output.is_mps(), "Output must be on MPS device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");
    
    init_metal();
    sync_mps();  // Ensure previous work is complete
    
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int H2 = (int)((H + 1) / 2);
    int W2 = (int)((W + 1) / 2);
    
    auto pipeline = get_pipeline(kernel_name("haar2d_forward", input));
    
    // Get the MPS stream's command buffer
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(cmdBuffer != nil, "Failed to get MPS command buffer");
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
    
    [encoder setComputePipelineState:pipeline];
    
    // Set buffers
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input);
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
    
    [encoder setBuffer:inputBuffer offset:input.storage_offset() * input.element_size() atIndex:0];
    [encoder setBuffer:outputBuffer offset:output.storage_offset() * output.element_size() atIndex:1];
    
    // Set constants
    int Ci = (int)C, Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Ci length:sizeof(int) atIndex:2];
    [encoder setBytes:&Hi length:sizeof(int) atIndex:3];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:4];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:5];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:6];
    
    // Dispatch
    MTLSize gridSize = MTLSizeMake((W2 + 15) / 16, (H2 + 15) / 16, B * C);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    // Commit through PyTorch's MPS stream
    torch::mps::commit();
}

void haar2d_backward_metal(torch::Tensor grad_output, torch::Tensor grad_input) {
    TORCH_CHECK(grad_output.is_mps(), "grad_output must be on MPS device");
    TORCH_CHECK(grad_input.is_mps(), "grad_input must be on MPS device");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
    
    init_metal();
    sync_mps();
    
    int64_t B = grad_input.size(0);
    int64_t C = grad_input.size(1);
    int64_t H = grad_input.size(2);
    int64_t W = grad_input.size(3);
    int H2 = (int)((H + 1) / 2);
    int W2 = (int)((W + 1) / 2);
    
    auto pipeline = get_pipeline(kernel_name("haar2d_backward", grad_output));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    id<MTLBuffer> gradOutBuffer = getMTLBufferStorage(grad_output);
    id<MTLBuffer> gradInBuffer = getMTLBufferStorage(grad_input);
    
    [encoder setBuffer:gradOutBuffer offset:grad_output.storage_offset() * grad_output.element_size() atIndex:0];
    [encoder setBuffer:gradInBuffer offset:grad_input.storage_offset() * grad_input.element_size() atIndex:1];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:2];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:3];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:4];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake((W2 + 15) / 16, (H2 + 15) / 16, B * C);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// Inverse Haar
// =============================================================================

void haar2d_inverse_metal(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_mps(), "Input must be on MPS device");
    TORCH_CHECK(output.is_mps(), "Output must be on MPS device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");
    
    init_metal();
    sync_mps();
    
    int64_t B = output.size(0);
    int64_t C = output.size(1);
    int64_t H = output.size(2);
    int64_t W = output.size(3);
    int H2 = (int)input.size(3);
    int W2 = (int)input.size(4);
    
    auto pipeline = get_pipeline(kernel_name("haar2d_inverse", input));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input);
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
    
    [encoder setBuffer:inputBuffer offset:input.storage_offset() * input.element_size() atIndex:0];
    [encoder setBuffer:outputBuffer offset:output.storage_offset() * output.element_size() atIndex:1];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:2];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:3];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:4];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake((W2 + 15) / 16, (H2 + 15) / 16, B * C);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

void haar2d_inverse_backward_metal(torch::Tensor grad_output, torch::Tensor grad_input) {
    TORCH_CHECK(grad_output.is_mps(), "grad_output must be on MPS device");
    TORCH_CHECK(grad_input.is_mps(), "grad_input must be on MPS device");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
    
    init_metal();
    sync_mps();
    
    int64_t B = grad_output.size(0);
    int64_t C = grad_output.size(1);
    int64_t H = grad_output.size(2);
    int64_t W = grad_output.size(3);
    int H2 = (int)grad_input.size(3);
    int W2 = (int)grad_input.size(4);
    
    auto pipeline = get_pipeline(kernel_name("haar2d_inverse_backward", grad_output));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    id<MTLBuffer> gradOutBuffer = getMTLBufferStorage(grad_output);
    id<MTLBuffer> gradInBuffer = getMTLBufferStorage(grad_input);
    
    [encoder setBuffer:gradOutBuffer offset:grad_output.storage_offset() * grad_output.element_size() atIndex:0];
    [encoder setBuffer:gradInBuffer offset:grad_input.storage_offset() * grad_input.element_size() atIndex:1];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:2];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:3];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:4];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake((W2 + 15) / 16, (H2 + 15) / 16, B * C);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// 2-Level Cascade
// =============================================================================

void haar2d_double_forward_metal(torch::Tensor input, torch::Tensor output_level1, torch::Tensor output_level2) {
    TORCH_CHECK(input.is_mps() && output_level1.is_mps() && output_level2.is_mps(), "All tensors must be on MPS");
    
    init_metal();
    sync_mps();
    
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    
    auto pipeline = get_pipeline(cascade_kernel_name("haar2d_double_cascade_kernel", input));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(input) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(output_level1) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(output_level2) offset:0 atIndex:2];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:3];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:4];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:5];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:6];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:7];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:8];
    
    MTLSize gridSize = MTLSizeMake(W4, H4, B * C);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

void haar2d_double_backward_metal(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_input) {
    TORCH_CHECK(grad_level2.is_mps() && grad_input.is_mps(), "Tensors must be on MPS");
    
    init_metal();
    sync_mps();
    
    int64_t B = grad_input.size(0);
    int64_t C = grad_input.size(1);
    int64_t H = grad_input.size(2);
    int64_t W = grad_input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    
    auto pipeline = get_pipeline(cascade_kernel_name("ihaar2d_double_cascade_kernel", grad_level1));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(grad_level1) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(grad_level2) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(grad_input) offset:0 atIndex:2];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:3];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:4];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:5];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:6];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:7];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:8];
    
    MTLSize gridSize = MTLSizeMake(W2, H2, B * C);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// 3-Level Cascade
// =============================================================================

void haar2d_triple_forward_metal(torch::Tensor input, torch::Tensor output_level1, 
                                  torch::Tensor output_level2, torch::Tensor output_level3) {
    TORCH_CHECK(input.is_mps(), "Input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    
    auto pipeline = get_pipeline(cascade_kernel_name("haar2d_triple_cascade_kernel", input));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(input) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(output_level1) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(output_level2) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(output_level3) offset:0 atIndex:3];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:4];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:5];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:6];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:7];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:8];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:9];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:10];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:11];
    
    // Triple cascade uses 16x16 threadgroups, each processing 32x32 input tiles
    // Grid is number of 32x32 tiles
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(tiles_x * 16, tiles_y * 16, B * C);
    
    // Allocate shared memory for LL values (16x16 floats)
    [encoder setThreadgroupMemoryLength:16 * 16 * sizeof(float) atIndex:0];
    
    [encoder dispatchThreadgroups:MTLSizeMake(tiles_x, tiles_y, B * C) threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

void haar2d_triple_backward_metal(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                   torch::Tensor grad_level3, torch::Tensor grad_input) {
    TORCH_CHECK(grad_input.is_mps(), "grad_input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = grad_input.size(0);
    int64_t C = grad_input.size(1);
    int64_t H = grad_input.size(2);
    int64_t W = grad_input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    
    auto pipeline = get_pipeline(cascade_kernel_name("ihaar2d_triple_cascade_kernel", grad_level1));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(grad_level1) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(grad_level2) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(grad_level3) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(grad_input) offset:0 atIndex:3];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:4];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:5];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:6];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:7];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:8];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:9];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:10];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:11];
    
    MTLSize gridSize = MTLSizeMake(W2, H2, B * C);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// 4-Level Cascade
// =============================================================================

void haar2d_quad_forward_metal(torch::Tensor input, torch::Tensor output_level1, 
                                torch::Tensor output_level2, torch::Tensor output_level3,
                                torch::Tensor output_level4) {
    TORCH_CHECK(input.is_mps(), "Input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    int H16 = (int)((H + 15) / 16), W16 = (int)((W + 15) / 16);
    
    auto pipeline = get_pipeline(cascade_kernel_name("haar2d_quad_cascade_kernel", input));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(input) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(output_level1) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(output_level2) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(output_level3) offset:0 atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(output_level4) offset:0 atIndex:4];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:5];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:6];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:7];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:8];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:9];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:10];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:11];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:12];
    [encoder setBytes:&H16 length:sizeof(int) atIndex:13];
    [encoder setBytes:&W16 length:sizeof(int) atIndex:14];
    
    // Quad cascade uses 16x16 threadgroups, each processing 32x32 input tiles
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    // Allocate shared memory for LL values (16x16 floats)
    [encoder setThreadgroupMemoryLength:16 * 16 * sizeof(float) atIndex:0];
    
    [encoder dispatchThreadgroups:MTLSizeMake(tiles_x, tiles_y, B * C) threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

void haar2d_quad_backward_metal(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                 torch::Tensor grad_level3, torch::Tensor grad_level4,
                                 torch::Tensor grad_input) {
    TORCH_CHECK(grad_input.is_mps(), "grad_input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = grad_input.size(0);
    int64_t C = grad_input.size(1);
    int64_t H = grad_input.size(2);
    int64_t W = grad_input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    int H16 = (int)((H + 15) / 16), W16 = (int)((W + 15) / 16);
    
    auto pipeline = get_pipeline(cascade_kernel_name("ihaar2d_quad_cascade_kernel", grad_level1));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(grad_level1) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(grad_level2) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(grad_level3) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(grad_level4) offset:0 atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(grad_input) offset:0 atIndex:4];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:5];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:6];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:7];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:8];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:9];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:10];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:11];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:12];
    [encoder setBytes:&H16 length:sizeof(int) atIndex:13];
    [encoder setBytes:&W16 length:sizeof(int) atIndex:14];
    
    MTLSize gridSize = MTLSizeMake(W2, H2, B * C);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// 5-Level Cascade
// =============================================================================

void haar2d_quint_forward_metal(torch::Tensor input, torch::Tensor output_level1, 
                                 torch::Tensor output_level2, torch::Tensor output_level3,
                                 torch::Tensor output_level4, torch::Tensor output_level5) {
    TORCH_CHECK(input.is_mps(), "Input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    int H16 = (int)((H + 15) / 16), W16 = (int)((W + 15) / 16);
    int H32 = (int)((H + 31) / 32), W32 = (int)((W + 31) / 32);
    
    auto pipeline = get_pipeline(cascade_kernel_name("haar2d_quint_cascade_kernel", input));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(input) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(output_level1) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(output_level2) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(output_level3) offset:0 atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(output_level4) offset:0 atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(output_level5) offset:0 atIndex:5];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:6];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:7];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:8];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:9];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:10];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:11];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:12];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:13];
    [encoder setBytes:&H16 length:sizeof(int) atIndex:14];
    [encoder setBytes:&W16 length:sizeof(int) atIndex:15];
    [encoder setBytes:&H32 length:sizeof(int) atIndex:16];
    [encoder setBytes:&W32 length:sizeof(int) atIndex:17];
    
    // Quint cascade uses 16x16 threadgroups, each processing 32x32 input tiles
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    // Allocate shared memory for LL values (16x16 floats)
    [encoder setThreadgroupMemoryLength:16 * 16 * sizeof(float) atIndex:0];
    
    [encoder dispatchThreadgroups:MTLSizeMake(tiles_x, tiles_y, B * C) threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

void haar2d_quint_backward_metal(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                  torch::Tensor grad_level3, torch::Tensor grad_level4,
                                  torch::Tensor grad_level5, torch::Tensor grad_input) {
    TORCH_CHECK(grad_input.is_mps(), "grad_input must be on MPS");
    init_metal();
    sync_mps();
    
    int64_t B = grad_input.size(0);
    int64_t C = grad_input.size(1);
    int64_t H = grad_input.size(2);
    int64_t W = grad_input.size(3);
    int H2 = (int)((H + 1) / 2), W2 = (int)((W + 1) / 2);
    int H4 = (int)((H + 3) / 4), W4 = (int)((W + 3) / 4);
    int H8 = (int)((H + 7) / 8), W8 = (int)((W + 7) / 8);
    int H16 = (int)((H + 15) / 16), W16 = (int)((W + 15) / 16);
    int H32 = (int)((H + 31) / 32), W32 = (int)((W + 31) / 32);
    
    auto pipeline = get_pipeline(cascade_kernel_name("ihaar2d_quint_cascade_kernel", grad_level1));
    
    id<MTLCommandBuffer> cmdBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBuffer:getMTLBufferStorage(grad_level1) offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(grad_level2) offset:0 atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(grad_level3) offset:0 atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(grad_level4) offset:0 atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(grad_level5) offset:0 atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(grad_input) offset:0 atIndex:5];
    
    int Hi = (int)H, Wi = (int)W;
    [encoder setBytes:&Hi length:sizeof(int) atIndex:6];
    [encoder setBytes:&Wi length:sizeof(int) atIndex:7];
    [encoder setBytes:&H2 length:sizeof(int) atIndex:8];
    [encoder setBytes:&W2 length:sizeof(int) atIndex:9];
    [encoder setBytes:&H4 length:sizeof(int) atIndex:10];
    [encoder setBytes:&W4 length:sizeof(int) atIndex:11];
    [encoder setBytes:&H8 length:sizeof(int) atIndex:12];
    [encoder setBytes:&W8 length:sizeof(int) atIndex:13];
    [encoder setBytes:&H16 length:sizeof(int) atIndex:14];
    [encoder setBytes:&W16 length:sizeof(int) atIndex:15];
    [encoder setBytes:&H32 length:sizeof(int) atIndex:16];
    [encoder setBytes:&W32 length:sizeof(int) atIndex:17];
    
    MTLSize gridSize = MTLSizeMake(W2, H2, B * C);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    torch::mps::commit();
}

// =============================================================================
// PyBind11 Module Definition
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Configuration
    m.def("set_metal_source_path", &set_metal_source_path, "Set path to Metal source files");

    // Single level
    m.def("haar2d_forward", &haar2d_forward_metal, "Haar 2D forward (Metal)");
    m.def("haar2d_backward", &haar2d_backward_metal, "Haar 2D backward (Metal)");
    m.def("haar2d_inverse", &haar2d_inverse_metal, "Haar 2D inverse (Metal)");
    m.def("haar2d_inverse_backward", &haar2d_inverse_backward_metal, "Haar 2D inverse backward (Metal)");
    
    // 2-level cascade
    m.def("haar2d_double_forward", &haar2d_double_forward_metal, "2-level cascade Haar forward (Metal)");
    m.def("haar2d_double_backward", &haar2d_double_backward_metal, "2-level cascade Haar backward (Metal)");
    
    // 3-level cascade
    m.def("haar2d_triple_forward", &haar2d_triple_forward_metal, "3-level cascade Haar forward (Metal)");
    m.def("haar2d_triple_backward", &haar2d_triple_backward_metal, "3-level cascade Haar backward (Metal)");
    
    // 4-level cascade
    m.def("haar2d_quad_forward", &haar2d_quad_forward_metal, "4-level cascade Haar forward (Metal)");
    m.def("haar2d_quad_backward", &haar2d_quad_backward_metal, "4-level cascade Haar backward (Metal)");
    
    // 5-level cascade
    m.def("haar2d_quint_forward", &haar2d_quint_forward_metal, "5-level cascade Haar forward (Metal)");
    m.def("haar2d_quint_backward", &haar2d_quint_backward_metal, "5-level cascade Haar backward (Metal)");
}
