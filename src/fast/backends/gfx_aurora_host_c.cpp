// The plain-C host table (gfx_aurora_host_c.h) turned into the backend's AuroraGfxHost (gfx_aurora_host.h).
#include "fast/backends/gfx_aurora_host_c.h"

#ifdef ENABLE_AURORA

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include "fast/backends/gfx_aurora_host.h"

#ifdef LUS_AURORA_HOST_PROCS
extern "C" int LusWgpuHostProcs_Load(void* (*getProc)(void* user, const char* name), void* user);
#endif

namespace Fast {

namespace {

LusAuroraHostV1 sC{};
std::atomic<float> sZNear{ 0.0f }, sZFar{ 0.0f };

struct DrawHolder {
    aurora::gfx::DrawCallback fn;
    void* userdata;
};
struct TaskHolder {
    aurora::gfx::EncoderTaskCallback fn;
    void* userdata;
};
// Live for the process: the host may call a callback until it has processed an unregister.
std::vector<std::unique_ptr<DrawHolder>> sDraws;
std::vector<std::unique_ptr<TaskHolder>> sTasks;

aurora::gfx::RenderTargetLayout ToLayout(const LusAuroraTargetLayout& c) {
    aurora::gfx::RenderTargetLayout l{};
    l.key = c.key;
    l.colorAttachmentCount = std::min<uint32_t>(c.color_count, aurora::gfx::MaxColorAttachments);
    for (uint32_t i = 0; i < l.colorAttachmentCount; ++i) {
        l.colorAttachments[i].semantic = i == aurora::gfx::SceneColorAttachmentIndex
                                             ? aurora::gfx::ColorAttachmentSemantic::SceneColor
                                             : aurora::gfx::ColorAttachmentSemantic::Auxiliary;
        l.colorAttachments[i].format = static_cast<wgpu::TextureFormat>(c.color_formats[i]);
        l.colorAttachments[i].width = c.width;
        l.colorAttachments[i].height = c.height;
    }
    l.depthStencilFormat = static_cast<wgpu::TextureFormat>(c.depth_format);
    l.sampleCount = c.sample_count == 0 ? 1 : c.sample_count;
    return l;
}

void DrawTramp(void* fnUser, const LusAuroraDrawContext* ctx, const void* payload, size_t size) {
    const auto* h = static_cast<const DrawHolder*>(fnUser);
    aurora::gfx::DrawContext dc{};
    dc.device = wgpu::Device(static_cast<WGPUDevice>(ctx->device));
    dc.queue = wgpu::Queue(static_cast<WGPUQueue>(ctx->queue));
    dc.layout = ToLayout(ctx->layout);
    const wgpu::RenderPassEncoder pass(static_cast<WGPURenderPassEncoder>(ctx->pass));
    h->fn(dc, pass, payload, size, h->userdata);
}

void TaskTramp(void* fnUser, void* device, void* queue, void* encoder, const void* payload, size_t size) {
    const auto* h = static_cast<const TaskHolder*>(fnUser);
    aurora::gfx::EncoderTaskContext tc{};
    tc.device = wgpu::Device(static_cast<WGPUDevice>(device));
    tc.queue = wgpu::Queue(static_cast<WGPUQueue>(queue));
    const wgpu::CommandEncoder cmd(static_cast<WGPUCommandEncoder>(encoder));
    h->fn(tc, cmd, payload, size, h->userdata);
}

const AuroraGfxHost kFromC = {
    [] { return wgpu::Device(static_cast<WGPUDevice>(sC.device)); },
    [] { return wgpu::Queue(static_cast<WGPUQueue>(sC.queue)); },
    [] {
        LusAuroraTargetLayout c{};
        return sC.scene_layout(sC.user, &c) ? ToLayout(c) : aurora::gfx::RenderTargetLayout{};
    },
    [] {
        LusAuroraTargetLayout c{};
        return sC.scene_layout(sC.user, &c) && c.reversed_z != 0;
    },
    [](const aurora::gfx::DrawTypeDescriptor& d) -> aurora::gfx::DrawTypeId {
        sDraws.push_back(std::make_unique<DrawHolder>(DrawHolder{ d.draw, d.userdata }));
        return sC.register_draw(sC.user, d.label, DrawTramp, sDraws.back().get());
    },
    [](aurora::gfx::DrawTypeId t) { sC.unregister_draw(sC.user, t); },
    [](aurora::gfx::DrawTypeId t, const void* p, size_t n) { return sC.push_draw(sC.user, t, p, n) != 0; },
    [](const aurora::gfx::EncoderTaskDescriptor& d) -> aurora::gfx::EncoderTaskId {
        // afterSubmit has no host form; only the standalone self-test uses it.
        sTasks.push_back(std::make_unique<TaskHolder>(TaskHolder{ d.callback, d.userdata }));
        return sC.register_task(sC.user, d.label, TaskTramp, sTasks.back().get());
    },
    [](aurora::gfx::EncoderTaskId t) { sC.unregister_task(sC.user, t); },
    [](aurora::gfx::EncoderTaskId t, const void* p, size_t n) { return sC.push_task(sC.user, t, p, n) != 0; },
    // Hosted, the backend replays from its own buffers: the host's streaming buffers are not used.
    [](const uint8_t*, size_t, size_t) { return aurora::gfx::Range{}; },
    [](const uint8_t*, size_t, size_t) { return aurora::gfx::Range{}; },
    [](const uint8_t*, size_t) { return aurora::gfx::Range{}; },
    [](const uint8_t*, size_t) { return aurora::gfx::Range{}; },
    [](const aurora::gfx::ResolveDesc&, aurora::gfx::ResolvedTargets&) { return false; },
    [] {},
    [](float projZ[4], float* worldScale) {
        LusAuroraWorldDepth d{};
        if (sC.world_depth == nullptr || sC.world_depth(sC.user, &d) != 1 || !d.enabled) {
            return false;
        }
        std::copy(d.proj_z, d.proj_z + 4, projZ);
        *worldScale = d.world_scale;
        return true;
    },
};

} // namespace

void SetAuroraWorldDepthRange(float zNear, float zFar) {
    sZNear = zNear;
    sZFar = zFar;
}

bool AuroraWorldDepthRange(float* zNear, float* zFar) {
    *zNear = sZNear;
    *zFar = sZFar;
    return *zNear > 0.0f && *zFar > *zNear;
}

bool AuroraGfxHostCSupported() {
#ifdef LUS_AURORA_HOST_PROCS
    return true;
#else
    return false;
#endif
}

bool SetAuroraGfxHostC(const LusAuroraHostV1* host) {
    if (host == nullptr) {
        SetAuroraGfxHost(nullptr);
        return true;
    }
    if (host->struct_size < LUS_AURORA_HOST_V1_BASE_SIZE || host->version != LUS_AURORA_HOST_VERSION ||
        host->get_proc == nullptr || host->device == nullptr || host->scene_layout == nullptr ||
        host->register_draw == nullptr || host->push_draw == nullptr || host->register_task == nullptr ||
        host->push_task == nullptr) {
        SPDLOG_ERROR("Aurora host: table refused (size {}, version {})", host->struct_size, host->version);
        return false;
    }
#ifdef LUS_AURORA_HOST_PROCS
    const int missing = LusWgpuHostProcs_Load(host->get_proc, host->user);
    // A host may not offer every Dawn extension; libultraship uses the core set.
    SPDLOG_INFO("Aurora host: WebGPU from the host ({} functions not offered)", missing);
    sC = LusAuroraHostV1{};
    std::memcpy(&sC, host, std::min<size_t>(host->struct_size, sizeof(sC))); // an older table has no world_depth
    SetAuroraGfxHost(&kFromC);
    return true;
#else
    SPDLOG_ERROR("Aurora host: this libultraship was built without LUS_AURORA_HOST_PROCS (its own Dawn cannot use the "
                 "host's device)");
    return false;
#endif
}

} // namespace Fast

#else

namespace Fast {
void SetAuroraWorldDepthRange(float, float) {
}
bool AuroraGfxHostCSupported() {
    return false;
}
bool SetAuroraGfxHostC(const LusAuroraHostV1*) {
    return false; // no Aurora backend in this build
}
} // namespace Fast

#endif
