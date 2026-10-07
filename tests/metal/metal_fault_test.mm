// tests/metal/metal_fault_test.mm - verify Metal command-buffer fault reporting and recovery.
#include "strata/metal/context.hpp"
#include "../../src/metal/engine/metal_fence.hpp"

#include "metallib_sha256.inc"

#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

using namespace strata::metal;
namespace fs = std::filesystem;

static int failures = 0;

static void check(bool ok, const char* name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    failures += !ok;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                                 NSString* name, std::string& err) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        err = std::string("fault test library is missing ") + name.UTF8String;
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        err = error != nil ? error.localizedDescription.UTF8String : "unknown pipeline error";
    }
    return pipeline;
}

static bool dispatch(Context& context, id<MTLComputePipelineState> pipeline, BufferRef buffer,
                     const std::string& label, std::string& err, MTLCommandBufferStatus* status,
                     bool* validation_log_describes_fault = nullptr) {
    auto command = context.make_command_buffer(label, err);
    if (command == nullptr) return false;
    id<MTLCommandBuffer> command_buffer =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    check((command_buffer.errorOptions & MTLCommandBufferErrorOptionEncoderExecutionStatus) != 0,
          "encoder execution status is enabled");
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, command_buffer);
    if (encoder == nil) {
        err = "cannot create compute encoder for " + label;
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffer.handle offset:0 atIndex:0];
    encoder.label = @(label.c_str());
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    end_compute_encoder(context, encoder);
    if (!command->commit(err)) return false;
    [command_buffer waitUntilCompleted];
    if (validation_log_describes_fault != nullptr) {
        *validation_log_describes_fault = false;
        for (id<MTLFunctionLog> log in command_buffer.logs) {
            if (log.type != MTLFunctionLogTypeValidation) continue;
            const char* text = log.description.UTF8String;
            const std::string detail = text != nullptr ? text : "";
            const bool has_faulting_offset = detail.find("1099511627776") != std::string::npos;
            const bool names_faulting_kernel = detail.find("fault_kernel") != std::string::npos;
            const bool identifies_load = detail.find("Invalid load") != std::string::npos ||
                                         detail.find("out-of-bounds") != std::string::npos;
            *validation_log_describes_fault =
                *validation_log_describes_fault || (has_faulting_offset && names_faulting_kernel && identifies_load);
        }
    }
    *status = command_buffer.status;
    return true;
}

static int run_child(const char* metallib) {
    ContextOptions options;
    options.metallib = metallib;
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string err;
    auto context = Context::open(options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    const char* source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void fault_kernel(device uint* p [[buffer(0)]]) {\n"
        "  p[0] = p[0x4000000000ul];\n"
        "}\n"
        "kernel void recovery_kernel(device uint* p [[buffer(0)]]) { p[0] = 0x1234u; }\n";
    NSError* compile_error = nil;
    id<MTLLibrary> fault_library =
        [(__bridge id<MTLDevice>)context->device() newLibraryWithSource:@(source) options:nil error:&compile_error];
    check(fault_library != nil, "compile fault and recovery kernels",
          compile_error != nil ? compile_error.localizedDescription.UTF8String : "");
    if (fault_library == nil) return 1;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device();
    id<MTLComputePipelineState> fault_pipeline =
        make_pipeline(device, fault_library, @"fault_kernel", err);
    check(fault_pipeline != nil, "build fault pipeline", err);
    id<MTLComputePipelineState> recovery_pipeline =
        make_pipeline(device, fault_library, @"recovery_kernel", err);
    check(recovery_pipeline != nil, "build recovery pipeline", err);
    if (fault_pipeline == nil || recovery_pipeline == nil) return 1;

    BufferRef buffer =
        context->buffers().allocate("strata fault test", 4096, err);
    check(static_cast<bool>(buffer), "allocate fault test buffer", err);
    if (!buffer) {
        return 1;
    }
    *static_cast<std::uint32_t*>(buffer.contents) = 0;

    MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
    bool validation_log_describes_fault = false;
    check(dispatch(*context, fault_pipeline, buffer, "metal fault injection", err, &status,
                   &validation_log_describes_fault),
          "submit faulting kernel", err);
    check((status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError) &&
              validation_log_describes_fault,
          "shader validation logs the 1 TiB out-of-bounds load");

    status = MTLCommandBufferStatusNotEnqueued;
    check(dispatch(*context, recovery_pipeline, buffer, "metal fault recovery", err, &status),
          "submit recovery kernel", err);
    check(status == MTLCommandBufferStatusCompleted, "command buffer after the fault completes");
    check(*static_cast<std::uint32_t*>(buffer.contents) == 0x1234u, "recovery kernel writes its result");
    return failures != 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc == 3 && std::string(argv[1]) == "--child") return run_child(argv[2]);
    if (argc != 2) {
        std::fprintf(stderr, "usage: metal_fault_test strata.metallib\n");
        return 2;
    }

    int output_pipe[2];
    if (pipe(output_pipe) != 0) {
        std::perror("pipe");
        return 1;
    }
    const std::string executable = fs::absolute(argv[0]).string();
    char* child_argv[] = {const_cast<char*>(executable.c_str()), const_cast<char*>("--child"), argv[1], nullptr};
    posix_spawn_file_actions_t actions;
    int spawn_error = posix_spawn_file_actions_init(&actions);
    const bool actions_ready = spawn_error == 0;
    if (spawn_error == 0) spawn_error = posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
    if (spawn_error == 0) spawn_error = posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDERR_FILENO);
    if (spawn_error == 0) spawn_error = posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
    if (spawn_error == 0) spawn_error = posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
    pid_t child = -1;
    if (spawn_error == 0) spawn_error = posix_spawn(&child, executable.c_str(), &actions, nullptr, child_argv, environ);
    if (actions_ready) posix_spawn_file_actions_destroy(&actions);
    close(output_pipe[1]);
    if (spawn_error != 0) {
        close(output_pipe[0]);
        std::fprintf(stderr, "cannot start the Metal fault test child: %s\n", std::strerror(spawn_error));
        return 1;
    }

    std::string output;
    char chunk[4096];
    ssize_t count = 0;
    while ((count = read(output_pipe[0], chunk, sizeof(chunk))) > 0) output.append(chunk, count);
    close(output_pipe[0]);
    int wait_status = 0;
    if (waitpid(child, &wait_status, 0) < 0) {
        std::perror("waitpid");
        return 1;
    }
    std::fwrite(output.data(), 1, output.size(), stdout);
    const bool child_ok = WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0;
    const bool error_reported = output.find("strata: GPU error in \"metal fault injection\":") != std::string::npos;
    const bool out_of_bounds_reported =
        output.find("Invalid load") != std::string::npos &&
        output.find("1099511627776") != std::string::npos &&
        output.find("fault_kernel") != std::string::npos;
    check(child_ok, "faulting command buffer is reported and the GPU recovers");
    check(error_reported, "diagnostic names the faulting command buffer");
    check(out_of_bounds_reported, "diagnostic identifies the out-of-bounds load");
    std::printf("metal_fault_test: %s\n", failures ? "FAILED" : "OK");
    return failures != 0;
}
