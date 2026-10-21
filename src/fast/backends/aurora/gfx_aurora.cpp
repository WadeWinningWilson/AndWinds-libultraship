#ifdef ENABLE_AURORA

#include "fast/backends/gfx_aurora.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <spdlog/spdlog.h>

#include "fast/Fast3dGui.h"
#include "fast/Fast3dWindow.h"
#include "fast/interpreter.h"
#include "ship/Context.h"

namespace Fast {

namespace {

namespace gfx = aurora::gfx;

constexpr const char* CompositeWgsl = R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct VO { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO; o.p = vec4f(xy, 0.0, 1.0); o.uv = vec2f(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5); return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return textureSample(t, s, i.uv); }
)";

struct ReplayPayload {
    GfxRenderingAPIAurora* self;
    uint32_t slot;
};

struct CompositePayload {
    GfxRenderingAPIAurora* self;
    WGPUTextureView view;
};

struct SelfTestPayload {
    GfxRenderingAPIAurora* self;
    WGPUTexture texture;
};

void ReplayTrampoline(const gfx::EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void* payload, size_t,
                      void*) {
    const auto& p = *static_cast<const ReplayPayload*>(payload);
    p.self->ReplayFrame(cmd, p.slot);
}

void CompositeTrampoline(const gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void* payload,
                         size_t, void*) {
    const auto& p = *static_cast<const CompositePayload*>(payload);
    p.self->DrawComposite(ctx, pass, p.view);
}

void SelfTestTrampoline(const gfx::EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void* payload, size_t,
                        void*) {
    const auto& p = *static_cast<const SelfTestPayload*>(payload);
    p.self->SelfTestCopy(cmd, p.texture);
}

void SelfTestAfterSubmit(const gfx::EncoderTaskCompletionContext&, const void* payload, size_t, void*) {
    static_cast<const SelfTestPayload*>(payload)->self->SelfTestMapped();
}

wgpu::ShaderModule MakeModule(const wgpu::Device& device, const char* wgsl, const char* label) {
    wgpu::ShaderSourceWGSL source{};
    source.code = wgsl;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &source;
    desc.label = label;
    return device.CreateShaderModule(&desc);
}

} // namespace

GfxRenderingAPIAurora::GfxRenderingAPIAurora(GfxWindowBackendAurora* window) : mWindow(window) {
    mFramebuffers.resize(1); // framebuffer 0: the screen
}

GfxRenderingAPIAurora::~GfxRenderingAPIAurora() {
    if (mDevice) {
        AuroraBridge::Gfx::Synchronize();
        AuroraBridge::Gfx::UnregisterEncoderTaskType(mReplayTask);
        AuroraBridge::Gfx::UnregisterDrawType(mCompositeDraw);
        AuroraBridge::Gfx::UnregisterEncoderTaskType(mSelfTestTask);
    }
}

const char* GfxRenderingAPIAurora::GetName() {
    return "Aurora (WebGPU)";
}

int GfxRenderingAPIAurora::GetMaxTextureSize() {
    wgpu::Limits limits{};
    if (mDevice && mDevice.GetLimits(&limits)) {
        return static_cast<int>(limits.maxTextureDimension2D);
    }
    return 8192;
}

GfxClipParameters GfxRenderingAPIAurora::GetClipParameters() {
    // WebGPU: NDC z in [0, 1], texture origin top-left (as D3D11).
    return { true, false };
}

void GfxRenderingAPIAurora::UnloadShader(ShaderProgram*) {
    mCurrentShader = nullptr;
}

void GfxRenderingAPIAurora::LoadShader(ShaderProgram* newPrg) {
    mCurrentShader = reinterpret_cast<ShaderProgramAurora*>(newPrg);
}

void GfxRenderingAPIAurora::ClearShaderCache() {
    mShaders.clear();
    mCurrentShader = nullptr;
}

ShaderProgram* GfxRenderingAPIAurora::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    CCFeatures cc{};
    gfx_cc_get_features(shaderId0, shaderId1, &cc);
    auto& prg = mShaders[{ shaderId0, shaderId1 }];
    prg.id0 = shaderId0;
    prg.id1 = shaderId1;
    prg.numInputs = static_cast<uint8_t>(cc.numInputs);
    prg.usedTextures[0] = cc.usedTextures[0];
    prg.usedTextures[1] = cc.usedTextures[1];
    // Step 2 builds the WGSL program and pipelines here.
    LoadShader(reinterpret_cast<ShaderProgram*>(&prg));
    return reinterpret_cast<ShaderProgram*>(&prg);
}

ShaderProgram* GfxRenderingAPIAurora::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto it = mShaders.find({ shaderId0, shaderId1 });
    return it == mShaders.end() ? nullptr : reinterpret_cast<ShaderProgram*>(&it->second);
}

void GfxRenderingAPIAurora::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    const auto* p = reinterpret_cast<ShaderProgramAurora*>(prg);
    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

uint32_t GfxRenderingAPIAurora::NewTexture() {
    return mTextureCount++;
}

void GfxRenderingAPIAurora::SelectTexture(int, uint32_t) {
}

void GfxRenderingAPIAurora::UploadTexture(const uint8_t*, uint32_t, uint32_t) {
}

void GfxRenderingAPIAurora::SetSamplerParameters(int, bool, uint32_t, uint32_t) {
}

void GfxRenderingAPIAurora::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = z_upd;
}

void GfxRenderingAPIAurora::SetZmodeDecal(bool decal) {
    mCurrentZmodeDecal = decal;
}

void GfxRenderingAPIAurora::SetViewport(int, int, int, int) {
}

void GfxRenderingAPIAurora::SetScissor(int, int, int, int) {
}

void GfxRenderingAPIAurora::SetUseAlpha(bool) {
}

void GfxRenderingAPIAurora::DrawTriangles(float[], size_t, size_t buf_vbo_num_tris) {
    ++mStatDraws;
    mStatTris += buf_vbo_num_tris;
    if (mRecording && !mFrames[mRecordingSlot].empty()) {
        ++mFrames[mRecordingSlot].back().draws;
    }
}

void GfxRenderingAPIAurora::Init() {
    AuroraTrace("[lus-aurora] rendering: init");
    mDevice = AuroraBridge::Gfx::Device();
    if (!mDevice) {
        SPDLOG_ERROR("Aurora backend: no WebGPU device");
        return;
    }
    mReplayTask = AuroraBridge::Gfx::RegisterEncoderTaskType(
        { .label = "Fast3D frame", .callback = ReplayTrampoline, .userdata = this });
    mCompositeDraw =
        AuroraBridge::Gfx::RegisterDrawType({ .label = "Fast3D composite", .draw = CompositeTrampoline, .userdata = this });
    mSelfTestTask = AuroraBridge::Gfx::RegisterEncoderTaskType({ .label = "Fast3D self-test",
                                                                 .callback = SelfTestTrampoline,
                                                                 .userdata = this,
                                                                 .afterSubmit = SelfTestAfterSubmit });

    if (const char* v = std::getenv("LUS_AURORA_DEBUG_CLEAR")) {
        int r = 0, g = 0, b = 0;
        if (std::sscanf(v, "%d,%d,%d", &r, &g, &b) == 3) {
            mDebugClear = true;
            mDebugClearColor = { r / 255.0f, g / 255.0f, b / 255.0f, 1.0f };
        }
    }
    if (const char* v = std::getenv("LUS_AURORA_SELFTEST")) {
        mSelfTestFrame = static_cast<uint32_t>(std::atoi(v));
        wgpu::BufferDescriptor bd{};
        bd.label = "Fast3D self-test readback";
        bd.size = 256;
        bd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        mSelfTestBuffer = mDevice.CreateBuffer(&bd);
    }

    uint32_t w = 0, h = 0;
    int32_t x = 0, y = 0;
    mWindow->GetDimensions(&w, &h, &x, &y);
    GuiWindowInitData impl{};
    impl.Aurora = { w, h };
    impl.Backend = WindowBackend::FAST3D_AURORA;
    auto gui = std::dynamic_pointer_cast<Fast3dGui>(Ship::Context::GetRawInstance()->GetWindow()->GetGui());
    if (gui) {
        AuroraTrace("[lus-aurora] rendering: GUI init");
        gui->Init(impl);
    }
    AuroraTrace("[lus-aurora] rendering: ready");
    SPDLOG_INFO("Aurora backend: rendering API ready (debug clear {}, self-test frame {})", mDebugClear,
                mSelfTestFrame);
}

void GfxRenderingAPIAurora::OnResize() {
}

void GfxRenderingAPIAurora::StartFrame() {
    mRecording = mWindow->BeginFrameIfNeeded();
    static uint64_t sStartCalls = 0, sSkipped = 0;
    ++sStartCalls;
    if (!mRecording) {
        ++sSkipped;
    }
    if (sStartCalls <= 3 || sStartCalls == 60 || sStartCalls == 600) {
        AuroraTrace("[lus-aurora] StartFrame #%llu: Aurora frame %s (%llu skipped so far)",
                    static_cast<unsigned long long>(sStartCalls), mRecording ? "open" : "NOT open",
                    static_cast<unsigned long long>(sSkipped));
    }
    if (!mRecording) {
        return;
    }
    mRecordingSlot = mFrameCounter % FrameSlots;
    mFrames[mRecordingSlot].clear();
    RetireResources();
}

void GfxRenderingAPIAurora::BeginPass(int fb) {
    if (!mRecording || fb < 0 || fb >= static_cast<int>(mFramebuffers.size())) {
        return;
    }
    const Framebuffer& f = mFramebuffers[fb];
    if (!f.colorView) {
        return;
    }
    PassOp op;
    op.fb = fb;
    op.colorView = f.colorView;
    op.depthView = f.depthView;
    mFrames[mRecordingSlot].push_back(std::move(op));
    ++mStatPasses;
}

void GfxRenderingAPIAurora::EndFrame() {
    if (!mRecording) {
        return;
    }
    ++mStatFrames;
    if (mStatFrames == 1 || mStatFrames == 60 || mStatFrames == 600) {
        AuroraTrace("[lus-aurora] frame %llu: %zu pass(es) recorded, %llu draws so far",
                    static_cast<unsigned long long>(mStatFrames), mFrames[mRecordingSlot].size(),
                    static_cast<unsigned long long>(mStatDraws));
    }
    const ReplayPayload rp{ this, mRecordingSlot };
    AuroraBridge::Gfx::PushEncoderTask(mReplayTask, &rp, sizeof(rp));
    mCompositeViews[mRecordingSlot] = mFramebuffers[0].colorView;
    if (mCompositeViews[mRecordingSlot]) {
        const CompositePayload cp{ this, mCompositeViews[mRecordingSlot].Get() };
        AuroraBridge::Gfx::PushCustomDraw(mCompositeDraw, &cp, sizeof(cp));
    }

    if (mSelfTestFrame != 0 && mStatFrames == mSelfTestFrame) {
        gfx::ResolvedTargets snap{};
        if (AuroraBridge::Gfx::ResolvePass({ .color = true }, snap) && snap.colorTexture) {
            mSelfTestTexture = snap.colorTexture;
            mSelfTestFormat = snap.colorFormat;
            mSelfTestX = snap.width / 2;
            mSelfTestY = snap.height / 2;
            const SelfTestPayload sp{ this, mSelfTestTexture.Get() };
            AuroraBridge::Gfx::PushEncoderTask(mSelfTestTask, &sp, sizeof(sp));
        }
    }

    const double now = mWindow->GetTime();
    if (now - mStatLastLog >= 1.0) {
        SPDLOG_INFO("Aurora backend: {} frames, {} passes, {} clears, {} draws ({} tris) not yet rendered",
                    mStatFrames, mStatPasses, mStatClears, mStatDraws, mStatTris);
        mStatLastLog = now;
    }
    ++mFrameCounter;
    mRecording = false;
}

void GfxRenderingAPIAurora::FinishRender() {
}

int GfxRenderingAPIAurora::CreateFramebuffer() {
    mFramebuffers.emplace_back();
    return static_cast<int>(mFramebuffers.size() - 1);
}

void GfxRenderingAPIAurora::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height,
                                                        uint32_t msaa_level, bool, bool, bool has_depth_buffer, bool) {
    if (!mDevice || fb_id < 0 || fb_id >= static_cast<int>(mFramebuffers.size()) || width == 0 || height == 0) {
        return;
    }
    Framebuffer& f = mFramebuffers[fb_id];
    // E6: WebGPU has 1x and 4x only. Step 1 renders single-sample.
    const uint32_t samples = 1;
    (void)msaa_level;
    if (f.width == width && f.height == height && f.samples == samples && f.hasDepth == has_depth_buffer &&
        f.color) {
        return;
    }
    if (f.color) {
        mRetired.emplace_back(mFrameCounter, f);
    }
    f.width = width;
    f.height = height;
    f.samples = samples;
    f.hasDepth = has_depth_buffer;

    wgpu::TextureDescriptor td{};
    td.label = "Fast3D framebuffer";
    td.size = { width, height, 1 };
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
               wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    f.color = mDevice.CreateTexture(&td);
    f.colorView = f.color.CreateView();
    f.depth = nullptr;
    f.depthView = nullptr;
    if (has_depth_buffer) {
        wgpu::TextureDescriptor dd{};
        dd.label = "Fast3D depth";
        dd.size = { width, height, 1 };
        dd.format = wgpu::TextureFormat::Depth32Float;
        dd.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        f.depth = mDevice.CreateTexture(&dd);
        f.depthView = f.depth.CreateView();
    }
}

void GfxRenderingAPIAurora::RetireResources() {
    // A slot is reused after FrameSlots frames; by then no frame in flight uses the old textures.
    std::erase_if(mRetired, [&](const auto& r) { return mFrameCounter - r.first >= FrameSlots + 1; });
}

void GfxRenderingAPIAurora::StartDrawToFramebuffer(int fbId, float) {
    mCurrentFb = fbId;
    BeginPass(fbId);
}

void GfxRenderingAPIAurora::CopyFramebuffer(int, int, int, int, int, int, int, int, int, int) {
}

void GfxRenderingAPIAurora::ClearFramebuffer(bool color, bool depth) {
    if (!mRecording) {
        return;
    }
    ++mStatClears;
    auto& ops = mFrames[mRecordingSlot];
    // A clear before any draw in the pass becomes the pass's load op; otherwise a
    // new pass on the same target starts with the clear (step 1: full-target clears).
    if (ops.empty() || ops.back().fb != mCurrentFb || ops.back().draws > 0) {
        BeginPass(mCurrentFb);
    }
    if (ops.empty()) {
        return;
    }
    PassOp& op = ops.back();
    op.clearColor = op.clearColor || color;
    op.clearDepth = op.clearDepth || depth;
    op.color = (op.fb == 0 && mDebugClear) ? mDebugClearColor : std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f };
}

void GfxRenderingAPIAurora::ClearDepthRegion(int, int, int, int) {
    ClearFramebuffer(false, true);
}

void GfxRenderingAPIAurora::ReadFramebufferToCPU(int, uint32_t width, uint32_t height, uint16_t* rgba16Buf) {
    std::memset(rgba16Buf, 0, static_cast<size_t>(width) * height * sizeof(uint16_t));
}

void GfxRenderingAPIAurora::ResolveMSAAColorBuffer(int, int) {
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIAurora::GetPixelDepth(int, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    for (const auto& c : coordinates) {
        res.emplace(c, 0);
    }
    return res;
}

void* GfxRenderingAPIAurora::GetFramebufferTextureId(int fbId) {
    if (fbId < 0 || fbId >= static_cast<int>(mFramebuffers.size())) {
        return nullptr;
    }
    return mFramebuffers[fbId].colorView.Get();
}

void GfxRenderingAPIAurora::SelectTextureFb(int) {
}

void GfxRenderingAPIAurora::DeleteTexture(uint32_t) {
}

void GfxRenderingAPIAurora::SetTextureFilter(FilteringMode mode) {
    mFilter = mode;
}

FilteringMode GfxRenderingAPIAurora::GetTextureFilter() {
    return mFilter;
}

void GfxRenderingAPIAurora::SetSrgbMode() {
    mSrgbMode = true;
}

ImTextureID GfxRenderingAPIAurora::GetTextureById(int) {
    return 0;
}

void GfxRenderingAPIAurora::SetCurrentPrimDepth(float depth) {
    mCurrentPrimDepth = depth;
}

// --- render worker ------------------------------------------------------------

void GfxRenderingAPIAurora::ReplayFrame(const wgpu::CommandEncoder& cmd, uint32_t slot) {
    for (const PassOp& op : mFrames[slot]) {
        wgpu::RenderPassColorAttachment ca{};
        ca.view = op.colorView;
        ca.loadOp = op.clearColor ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
        ca.storeOp = wgpu::StoreOp::Store;
        ca.clearValue = { op.color[0], op.color[1], op.color[2], op.color[3] };
        wgpu::RenderPassDepthStencilAttachment da{};
        wgpu::RenderPassDescriptor rp{};
        rp.label = "Fast3D pass";
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &ca;
        if (op.depthView) {
            da.view = op.depthView;
            da.depthLoadOp = op.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
            da.depthStoreOp = wgpu::StoreOp::Store;
            da.depthClearValue = 1.0f;
            rp.depthStencilAttachment = &da;
        }
        cmd.BeginRenderPass(&rp).End();
    }
}

void GfxRenderingAPIAurora::DrawComposite(const gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass,
                                          WGPUTextureView view) {
    if (!mCompositePipeline || mCompositeLayoutKey != ctx.layout.key) {
        if (!mCompositeBgl) {
            wgpu::BindGroupLayoutEntry e[2]{};
            e[0].binding = 0;
            e[0].visibility = wgpu::ShaderStage::Fragment;
            e[0].texture.sampleType = wgpu::TextureSampleType::Float;
            e[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
            e[1].binding = 1;
            e[1].visibility = wgpu::ShaderStage::Fragment;
            e[1].sampler.type = wgpu::SamplerBindingType::Filtering;
            wgpu::BindGroupLayoutDescriptor bld{};
            bld.entryCount = 2;
            bld.entries = e;
            mCompositeBgl = ctx.device.CreateBindGroupLayout(&bld);
            wgpu::SamplerDescriptor sd{};
            sd.magFilter = wgpu::FilterMode::Linear;
            sd.minFilter = wgpu::FilterMode::Linear;
            mCompositeSampler = ctx.device.CreateSampler(&sd);
        }
        auto module = MakeModule(ctx.device, CompositeWgsl, "Fast3D composite");
        wgpu::ColorTargetState cts[gfx::MaxColorAttachments]{};
        for (uint32_t i = 0; i < ctx.layout.colorAttachmentCount; ++i) {
            cts[i].format = ctx.layout.colorAttachments[i].format;
            cts[i].writeMask =
                i == gfx::SceneColorAttachmentIndex ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None;
        }
        wgpu::FragmentState fs{};
        fs.module = module;
        fs.entryPoint = "fs";
        fs.targetCount = ctx.layout.colorAttachmentCount;
        fs.targets = cts;
        wgpu::DepthStencilState ds{};
        ds.format = ctx.layout.depthStencilFormat;
        ds.depthWriteEnabled = false;
        ds.depthCompare = wgpu::CompareFunction::Always;
        wgpu::PipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &mCompositeBgl;
        wgpu::RenderPipelineDescriptor pd{};
        pd.label = "Fast3D composite";
        pd.layout = ctx.device.CreatePipelineLayout(&pld);
        pd.vertex.module = module;
        pd.vertex.entryPoint = "vs";
        pd.fragment = &fs;
        pd.depthStencil = ctx.layout.depthStencilFormat == wgpu::TextureFormat::Undefined ? nullptr : &ds;
        pd.multisample.count = ctx.layout.sampleCount;
        mCompositePipeline = ctx.device.CreateRenderPipeline(&pd);
        mCompositeLayoutKey = ctx.layout.key;
    }
    if (view != mCompositeBoundView || !mCompositeBg) {
        wgpu::BindGroupEntry be[2]{};
        be[0].binding = 0;
        be[0].textureView = wgpu::TextureView(view);
        be[1].binding = 1;
        be[1].sampler = mCompositeSampler;
        wgpu::BindGroupDescriptor bgd{};
        bgd.layout = mCompositeBgl;
        bgd.entryCount = 2;
        bgd.entries = be;
        mCompositeBg = ctx.device.CreateBindGroup(&bgd);
        mCompositeBoundView = view;
    }
    pass.SetPipeline(mCompositePipeline);
    pass.SetBindGroup(0, mCompositeBg);
    pass.Draw(3);
}

void GfxRenderingAPIAurora::SelfTestCopy(const wgpu::CommandEncoder& cmd, WGPUTexture texture) {
    wgpu::TexelCopyTextureInfo src{};
    src.texture = wgpu::Texture(texture);
    src.origin = { mSelfTestX, mSelfTestY, 0 };
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = mSelfTestBuffer;
    dst.layout.bytesPerRow = 256;
    dst.layout.rowsPerImage = 1;
    wgpu::Extent3D one{ 1, 1, 1 };
    cmd.CopyTextureToBuffer(&src, &dst, &one);
}

void GfxRenderingAPIAurora::SelfTestMapped() {
    mSelfTestBuffer.MapAsync(
        wgpu::MapMode::Read, 0, 256, wgpu::CallbackMode::AllowSpontaneous,
        [this](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status != wgpu::MapAsyncStatus::Success) {
                SPDLOG_ERROR("Aurora backend self-test: readback failed");
                return;
            }
            uint8_t px[4];
            std::memcpy(px, mSelfTestBuffer.GetConstMappedRange(0, 256), 4);
            mSelfTestBuffer.Unmap();
            const bool bgra = mSelfTestFormat == wgpu::TextureFormat::BGRA8Unorm ||
                              mSelfTestFormat == wgpu::TextureFormat::BGRA8UnormSrgb;
            const int r = bgra ? px[2] : px[0], g = px[1], b = bgra ? px[0] : px[2];
            SPDLOG_INFO("Aurora backend self-test: scene centre pixel at frame {} = {},{},{}", mSelfTestFrame, r, g,
                        b);
            AuroraTrace("[lus-aurora] self-test: scene centre pixel at frame %u = %d,%d,%d", mSelfTestFrame, r, g, b);
        });
}

} // namespace Fast

#endif
