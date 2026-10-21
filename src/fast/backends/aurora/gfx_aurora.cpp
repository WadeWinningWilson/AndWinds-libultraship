#ifdef ENABLE_AURORA

#include "fast/backends/gfx_aurora.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <spdlog/spdlog.h>

#include "fast/Fast3dGui.h"
#include "fast/Fast3dWindow.h"
#include "fast/interpreter.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

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

// Texture upload: words[0] = width, words[1] = height, then RGBA8 pixels.
constexpr const char* UploadWgsl = R"(
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let w = src[0];
  let h = src[1];
  if (id.x >= w || id.y >= h) { return; }
  textureStore(dst, vec2i(id.xy), unpack4x8unorm(src[2u + id.y * w + id.x]));
}
)";

// Matches struct Uniforms in the WGSL template.
struct DrawUniforms {
    int32_t frameCount;
    float noiseScale;
    float primDepth;
    float decalSlope;
    float decalUnits;
    float pad[3];
    float viewport[4]; // clip-space scale.xy, offset.xy
    float texSize[4];  // w0, h0, w1, h1
    int32_t texFilter[4];
};
static_assert(sizeof(DrawUniforms) == 80);

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

void ReplayTrampoline(const gfx::EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload,
                      size_t, void*) {
    const auto& p = *static_cast<const ReplayPayload*>(payload);
    p.self->ReplayFrame(ctx, cmd, p.slot);
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

wgpu::AddressMode AddressMode(uint32_t cm) {
    switch (cm) {
        case G_TX_NOMIRROR | G_TX_CLAMP:
            return wgpu::AddressMode::ClampToEdge;
        case G_TX_MIRROR | G_TX_WRAP:
            return wgpu::AddressMode::MirrorRepeat;
        case G_TX_MIRROR | G_TX_CLAMP:
            // No mirror-clamp in WebGPU; Fast3D's clamp attributes clamp the coordinates.
            return wgpu::AddressMode::MirrorRepeat;
        case G_TX_NOMIRROR | G_TX_WRAP:
        default:
            return wgpu::AddressMode::Repeat;
    }
}

wgpu::VertexFormat FloatFormat(uint32_t n) {
    switch (n) {
        case 1:
            return wgpu::VertexFormat::Float32;
        case 2:
            return wgpu::VertexFormat::Float32x2;
        case 3:
            return wgpu::VertexFormat::Float32x3;
        default:
            return wgpu::VertexFormat::Float32x4;
    }
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

// --- shaders -------------------------------------------------------------------

void GfxRenderingAPIAurora::UnloadShader(ShaderProgram*) {
    mCurrentShader = nullptr;
}

void GfxRenderingAPIAurora::LoadShader(ShaderProgram* newPrg) {
    mCurrentShader = reinterpret_cast<ShaderProgramAurora*>(newPrg);
}

void GfxRenderingAPIAurora::ClearShaderCache() {
    mShaders.clear();
    mPipelines.clear();
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
    prg.alpha = cc.opt_alpha;

    // The vertex layout Fast3D packs (same order as the OpenGL backend's attributes).
    prg.attributes.clear();
    uint32_t offset = 0;
    auto add = [&](uint32_t floats) {
        wgpu::VertexAttribute a{};
        a.format = FloatFormat(floats);
        a.offset = offset * 4;
        a.shaderLocation = static_cast<uint32_t>(prg.attributes.size());
        prg.attributes.push_back(a);
        offset += floats;
    };
    add(4); // aVtxPos
    for (int i = 0; i < 2; i++) {
        if (cc.usedTextures[i]) {
            add(2);
            for (int j = 0; j < 2; j++) {
                if (cc.clamp[i][j]) {
                    add(1);
                }
            }
        }
    }
    if (cc.opt_fog) {
        add(4);
    }
    if (cc.opt_grayscale) {
        add(4);
    }
    for (int i = 0; i < cc.numInputs; i++) {
        add(cc.opt_alpha ? 4 : 3);
    }
    prg.stride = offset * 4;

    const std::string wgsl = BuildAuroraShader(cc, mFilter == FILTER_THREE_POINT, mSrgbMode);
    if (mSelfTestFrame != 0) {
        std::error_code ec;
        std::filesystem::create_directories("lus_aurora_shaders", ec);
        char name[96];
        std::snprintf(name, sizeof(name), "lus_aurora_shaders/%016llx_%016llx.wgsl",
                      static_cast<unsigned long long>(shaderId0), static_cast<unsigned long long>(shaderId1));
        std::ofstream(name) << wgsl;
        AuroraTrace("[lus-aurora] shader %s (%u floats per vertex)", name, offset);
    }
    prg.module = MakeModule(mDevice, wgsl.c_str(), "Fast3D combiner");
    ++mStatShaders;

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

wgpu::RenderPipeline GfxRenderingAPIAurora::GetPipeline(ShaderProgramAurora* prg, bool hasDepth) {
    const bool depthTest = mCurrentDepthTest != 0;
    const bool depthMask = mCurrentDepthMask != 0;
    const bool decal = mCurrentZmodeDecal != 0;
    // Key: program, depth test/mask/decal, blend, target has depth.
    uint64_t key = reinterpret_cast<uintptr_t>(prg);
    key = key * 31 + (depthTest ? 1 : 0) + (depthMask ? 2 : 0) + (decal ? 4 : 0) + (mUseAlpha ? 8 : 0) +
          (hasDepth ? 16 : 0);
    auto it = mPipelines.find(key);
    if (it != mPipelines.end()) {
        return it->second;
    }

    wgpu::VertexBufferLayout vbl{};
    vbl.arrayStride = prg->stride;
    vbl.attributeCount = prg->attributes.size();
    vbl.attributes = prg->attributes.data();

    wgpu::BlendState blend{};
    blend.color = { wgpu::BlendOperation::Add, wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::OneMinusSrcAlpha };
    blend.alpha = { wgpu::BlendOperation::Add, wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::OneMinusSrcAlpha };
    wgpu::ColorTargetState ct{};
    ct.format = wgpu::TextureFormat::RGBA8Unorm;
    ct.blend = mUseAlpha ? &blend : nullptr;
    wgpu::FragmentState fs{};
    fs.module = prg->module;
    fs.entryPoint = "fs_main";
    fs.targetCount = 1;
    fs.targets = &ct;

    wgpu::DepthStencilState ds{};
    ds.format = wgpu::TextureFormat::Depth32Float;
    if (depthTest || depthMask) {
        ds.depthWriteEnabled = depthMask;
        ds.depthCompare = depthTest ? (decal ? wgpu::CompareFunction::LessEqual : wgpu::CompareFunction::Less)
                                    : wgpu::CompareFunction::Always;
    } else {
        ds.depthWriteEnabled = false;
        ds.depthCompare = wgpu::CompareFunction::Always;
    }

    wgpu::RenderPipelineDescriptor pd{};
    pd.label = "Fast3D";
    pd.layout = mPipelineLayout;
    pd.vertex.module = prg->module;
    pd.vertex.entryPoint = "vs_main";
    pd.vertex.bufferCount = 1;
    pd.vertex.buffers = &vbl;
    pd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pd.primitive.cullMode = wgpu::CullMode::None; // Fast3D culls on the CPU
    pd.fragment = &fs;
    pd.depthStencil = hasDepth ? &ds : nullptr;
    auto pipeline = mDevice.CreateRenderPipeline(&pd);
    mPipelines.emplace(key, pipeline);
    return pipeline;
}

// --- textures --------------------------------------------------------------------

uint32_t GfxRenderingAPIAurora::NewTexture() {
    mTextures.emplace_back();
    return static_cast<uint32_t>(mTextures.size() - 1);
}

void GfxRenderingAPIAurora::SelectTexture(int tile, uint32_t textureId) {
    if (tile < 0 || tile >= static_cast<int>(TextureSlots)) {
        return;
    }
    mSlotTextures[tile] = textureId;
    mSlotFbViews[tile] = nullptr;
    mCurrentTile = tile;
}

void GfxRenderingAPIAurora::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    if (!mDevice || width == 0 || height == 0) {
        return;
    }
    const uint32_t id = mSlotTextures[mCurrentTile];
    if (id >= mTextures.size()) {
        return;
    }
    TextureAurora& t = mTextures[id];
    // A new texture object per upload: draws already recorded this frame keep
    // the old contents, as they would on a GPU that executes in order.
    if (t.texture) {
        mRetiredTextures.emplace_back(mFrameCounter, t.texture);
    }
    wgpu::TextureDescriptor td{};
    td.label = "Fast3D texture";
    td.size = { width, height, 1 };
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding;
    t.texture = mDevice.CreateTexture(&td);
    t.view = t.texture.CreateView();
    t.width = width;
    t.height = height;

    PendingUpload up;
    up.storageView = t.view;
    up.width = width;
    up.height = height;
    up.words.resize(2 + static_cast<size_t>(width) * height);
    up.words[0] = width;
    up.words[1] = height;
    std::memcpy(up.words.data() + 2, rgba32Buf, static_cast<size_t>(width) * height * 4);
    mPendingUploads.push_back(std::move(up));
    ++mStatUploads;
}

void GfxRenderingAPIAurora::FlushUploads() {
    // Push pending pixels into Aurora's storage stream, within this frame's budget.
    size_t budget = UploadBudgetPerFrame;
    auto& ups = mUploads[mRecordingSlot];
    while (!mPendingUploads.empty()) {
        PendingUpload& p = mPendingUploads.front();
        const size_t bytes = p.words.size() * 4;
        if (bytes > budget && !ups.empty()) {
            break; // the rest goes next frame
        }
        UploadOp op;
        op.storageView = p.storageView;
        op.width = p.width;
        op.height = p.height;
        op.range = AuroraBridge::Gfx::PushStorage(reinterpret_cast<const uint8_t*>(p.words.data()), bytes);
        ups.push_back(std::move(op));
        budget -= std::min(budget, bytes);
        mPendingUploads.pop_front();
    }
}

void GfxRenderingAPIAurora::SetSamplerParameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    if (tile < 0 || tile >= static_cast<int>(TextureSlots)) {
        return;
    }
    const uint32_t id = mSlotTextures[tile];
    if (id >= mTextures.size()) {
        return;
    }
    TextureAurora& t = mTextures[id];
    t.linear = linear_filter;
    t.cms = cms;
    t.cmt = cmt;
    // As the OpenGL backend: the shader's three-point path is for textures the
    // game asked to filter.
    t.filtering = !linear_filter ? FILTER_LINEAR : FILTER_THREE_POINT;
}

wgpu::Sampler GfxRenderingAPIAurora::GetSampler(bool linear, uint32_t cms, uint32_t cmt) {
    const bool useLinear = linear && mFilter == FILTER_LINEAR;
    const uint32_t key = (useLinear ? 1u : 0u) | (cms << 4) | (cmt << 8);
    auto it = mSamplers.find(key);
    if (it != mSamplers.end()) {
        return it->second;
    }
    wgpu::SamplerDescriptor sd{};
    sd.addressModeU = AddressMode(cms);
    sd.addressModeV = AddressMode(cmt);
    sd.magFilter = useLinear ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
    sd.minFilter = sd.magFilter;
    auto sampler = mDevice.CreateSampler(&sd);
    mSamplers.emplace(key, sampler);
    return sampler;
}

wgpu::BindGroup GfxRenderingAPIAurora::GetTextureBindGroup() {
    std::array<wgpu::TextureView, TextureSlots> views;
    for (uint32_t i = 0; i < TextureSlots; ++i) {
        if (mSlotFbViews[i]) {
            views[i] = mSlotFbViews[i];
        } else if (mSlotTextures[i] < mTextures.size() && mTextures[mSlotTextures[i]].view) {
            views[i] = mTextures[mSlotTextures[i]].view;
        } else {
            views[i] = mDummyView;
        }
    }
    std::array<wgpu::Sampler, 2> samplers;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t id = mSlotTextures[i];
        if (id < mTextures.size()) {
            const TextureAurora& t = mTextures[id];
            samplers[i] = GetSampler(t.linear, t.cms, t.cmt);
        } else {
            samplers[i] = GetSampler(false, G_TX_CLAMP, G_TX_CLAMP);
        }
    }
    std::array<void*, TextureSlots + 2> key{};
    for (uint32_t i = 0; i < TextureSlots; ++i) {
        key[i] = views[i].Get();
    }
    key[TextureSlots] = samplers[0].Get();
    key[TextureSlots + 1] = samplers[1].Get();
    auto it = mTextureBindGroups.find(key);
    if (it != mTextureBindGroups.end()) {
        it->second.second = mFrameCounter;
        return it->second.first;
    }
    wgpu::BindGroupEntry e[TextureSlots + 2]{};
    for (uint32_t i = 0; i < TextureSlots; ++i) {
        e[i].binding = i;
        e[i].textureView = views[i];
    }
    e[TextureSlots].binding = TextureSlots;
    e[TextureSlots].sampler = samplers[0];
    e[TextureSlots + 1].binding = TextureSlots + 1;
    e[TextureSlots + 1].sampler = samplers[1];
    wgpu::BindGroupDescriptor bgd{};
    bgd.label = "Fast3D textures";
    bgd.layout = mTextureBgl;
    bgd.entryCount = TextureSlots + 2;
    bgd.entries = e;
    auto bg = mDevice.CreateBindGroup(&bgd);
    mTextureBindGroups.emplace(key, std::make_pair(bg, mFrameCounter));
    return bg;
}

void GfxRenderingAPIAurora::DeleteTexture(uint32_t texId) {
    if (texId < mTextures.size() && mTextures[texId].texture) {
        mRetiredTextures.emplace_back(mFrameCounter, mTextures[texId].texture);
        mTextures[texId] = TextureAurora{};
    }
}

void GfxRenderingAPIAurora::SetTextureFilter(FilteringMode mode) {
    if (mode != mFilter) {
        mFilter = mode;
        ClearShaderCache(); // the three-point path is compiled in or out
    }
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

// --- state ----------------------------------------------------------------------

void GfxRenderingAPIAurora::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = z_upd;
}

void GfxRenderingAPIAurora::SetZmodeDecal(bool decal) {
    mCurrentZmodeDecal = decal;
}

void GfxRenderingAPIAurora::SetViewport(int x, int y, int width, int height) {
    mViewport = { x, y, width, height };
}

void GfxRenderingAPIAurora::SetScissor(int x, int y, int width, int height) {
    mScissor = { x, y, width, height };
}

void GfxRenderingAPIAurora::SetUseAlpha(bool useAlpha) {
    mUseAlpha = useAlpha;
}

void GfxRenderingAPIAurora::SetCurrentPrimDepth(float depth) {
    mCurrentPrimDepth = depth;
}

// --- drawing ---------------------------------------------------------------------

void GfxRenderingAPIAurora::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    ++mStatDraws;
    mStatTris += buf_vbo_num_tris;
    if (!mRecording || mCurrentShader == nullptr || !mCurrentShader->module) {
        ++mStatSkippedDraws;
        return;
    }
    auto& ops = mFrames[mRecordingSlot];
    if (ops.empty() || ops.back().fb != mCurrentFb) {
        BeginPass(mCurrentFb);
    }
    if (ops.empty()) {
        ++mStatSkippedDraws;
        return;
    }
    PassOp& pass = ops.back();
    const Framebuffer& fb = mFramebuffers[pass.fb];
    const float tw = static_cast<float>(fb.width), th = static_cast<float>(fb.height);

    // Fast3D gives viewport/scissor bottom-left; WebGPU wants top-left and inside the target.
    const float vx = static_cast<float>(mViewport[0]);
    const float vw = static_cast<float>(mViewport[2]);
    const float vh = static_cast<float>(mViewport[3]);
    const float vy = th - static_cast<float>(mViewport[1]) - vh;
    const float cx = std::clamp(vx, 0.0f, tw), cy = std::clamp(vy, 0.0f, th);
    const float cw = std::clamp(vx + vw, 0.0f, tw) - cx, ch = std::clamp(vy + vh, 0.0f, th) - cy;
    int sx = mScissor[0], sw = mScissor[2], sh = mScissor[3];
    int sy = static_cast<int>(fb.height) - mScissor[1] - sh;
    const int sx0 = std::clamp(sx, 0, static_cast<int>(fb.width)), sy0 = std::clamp(sy, 0, static_cast<int>(fb.height));
    const int sx1 = std::clamp(sx + sw, 0, static_cast<int>(fb.width)), sy1 = std::clamp(sy + sh, 0, static_cast<int>(fb.height));
    if (cw <= 0.0f || ch <= 0.0f || sx1 <= sx0 || sy1 <= sy0 || vw <= 0.0f || vh <= 0.0f) {
        ++mStatSkippedDraws;
        return;
    }

    DrawUniforms u{};
    u.frameCount = mFrameCountUniform;
    u.noiseScale = mNoiseScale;
    u.primDepth = mCurrentPrimDepth;
    if (mCurrentZmodeDecal) {
        // As the OpenGL backend's glPolygonOffset(SSDB, -2), applied in the shader
        // because hardware bias does not apply to a written frag_depth.
        float ssdb = -2.0f;
        switch (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(CVAR_Z_FIGHTING_MODE, 0)) {
            case 1:
                ssdb = -1.0f * th / 120.0f;
                break;
            case 2:
                ssdb = -1.0f * th / 100.0f;
                break;
            default:
                break;
        }
        u.decalSlope = ssdb;
        u.decalUnits = -2.0f / 16777216.0f;
    }
    // Clip-space correction from the requested viewport to the clamped one.
    u.viewport[0] = vw / cw;
    u.viewport[1] = vh / ch;
    u.viewport[2] = (2.0f * (vx - cx) + vw - cw) / cw;
    u.viewport[3] = (ch - 2.0f * (vy - cy) - vh) / ch;
    for (int i = 0; i < 2; ++i) {
        const uint32_t id = mSlotTextures[i];
        if (id < mTextures.size()) {
            u.texSize[i * 2] = static_cast<float>(std::max(1u, mTextures[id].width));
            u.texSize[i * 2 + 1] = static_cast<float>(std::max(1u, mTextures[id].height));
            u.texFilter[i] = mTextures[id].filtering;
        } else {
            u.texSize[i * 2] = 1.0f;
            u.texSize[i * 2 + 1] = 1.0f;
        }
    }

    DrawOp d;
    d.pipeline = GetPipeline(mCurrentShader, static_cast<bool>(pass.depthView));
    d.textures = GetTextureBindGroup();
    d.verts = AuroraBridge::Gfx::PushVerts(reinterpret_cast<const uint8_t*>(buf_vbo), buf_vbo_len * sizeof(float), 4);
    d.uniformOffset = AuroraBridge::Gfx::PushUniform(reinterpret_cast<const uint8_t*>(&u), sizeof(u)).offset;
    d.vertexCount = static_cast<uint32_t>(buf_vbo_num_tris * 3);
    d.viewport = { cx, cy, cw, ch };
    d.scissor = { static_cast<uint32_t>(sx0), static_cast<uint32_t>(sy0), static_cast<uint32_t>(sx1 - sx0),
                  static_cast<uint32_t>(sy1 - sy0) };
    pass.drawOps.push_back(std::move(d));
    ++pass.draws;
}

// --- frame ------------------------------------------------------------------------

void GfxRenderingAPIAurora::CreateSharedObjects() {
    // Group 0: per-draw uniforms (dynamic offset into Aurora's uniform stream).
    wgpu::BindGroupLayoutEntry ue{};
    ue.binding = 0;
    ue.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    ue.buffer.type = wgpu::BufferBindingType::Uniform;
    ue.buffer.hasDynamicOffset = true;
    ue.buffer.minBindingSize = sizeof(DrawUniforms);
    wgpu::BindGroupLayoutDescriptor ubd{};
    ubd.entryCount = 1;
    ubd.entries = &ue;
    mUniformBgl = mDevice.CreateBindGroupLayout(&ubd);

    // Group 1: six textures, two samplers.
    wgpu::BindGroupLayoutEntry te[TextureSlots + 2]{};
    for (uint32_t i = 0; i < TextureSlots; ++i) {
        te[i].binding = i;
        te[i].visibility = wgpu::ShaderStage::Fragment;
        te[i].texture.sampleType = wgpu::TextureSampleType::Float;
        te[i].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        te[TextureSlots + i].binding = TextureSlots + i;
        te[TextureSlots + i].visibility = wgpu::ShaderStage::Fragment;
        te[TextureSlots + i].sampler.type = wgpu::SamplerBindingType::Filtering;
    }
    wgpu::BindGroupLayoutDescriptor tbd{};
    tbd.entryCount = TextureSlots + 2;
    tbd.entries = te;
    mTextureBgl = mDevice.CreateBindGroupLayout(&tbd);

    wgpu::BindGroupLayout layouts[2] = { mUniformBgl, mTextureBgl };
    wgpu::PipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = 2;
    pld.bindGroupLayouts = layouts;
    mPipelineLayout = mDevice.CreatePipelineLayout(&pld);

    // A 1x1 texture for unused slots.
    wgpu::TextureDescriptor td{};
    td.label = "Fast3D dummy texture";
    td.size = { 1, 1, 1 };
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::TextureBinding;
    mDummyTexture = mDevice.CreateTexture(&td);
    mDummyView = mDummyTexture.CreateView();

    // Texture uploads.
    wgpu::BindGroupLayoutEntry ce[2]{};
    ce[0].binding = 0;
    ce[0].visibility = wgpu::ShaderStage::Compute;
    ce[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    ce[1].binding = 1;
    ce[1].visibility = wgpu::ShaderStage::Compute;
    ce[1].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
    ce[1].storageTexture.format = wgpu::TextureFormat::RGBA8Unorm;
    ce[1].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;
    wgpu::BindGroupLayoutDescriptor cbd{};
    cbd.entryCount = 2;
    cbd.entries = ce;
    mUploadBgl = mDevice.CreateBindGroupLayout(&cbd);
    wgpu::PipelineLayoutDescriptor cpl{};
    cpl.bindGroupLayoutCount = 1;
    cpl.bindGroupLayouts = &mUploadBgl;
    wgpu::ComputePipelineDescriptor cpd{};
    cpd.label = "Fast3D texture upload";
    cpd.layout = mDevice.CreatePipelineLayout(&cpl);
    cpd.compute.module = MakeModule(mDevice, UploadWgsl, "Fast3D texture upload");
    cpd.compute.entryPoint = "main";
    mUploadPipeline = mDevice.CreateComputePipeline(&cpd);

    mSlotTextures.fill(0xFFFFFFFFu);
}

void GfxRenderingAPIAurora::Init() {
    AuroraTrace("[lus-aurora] rendering: init");
    mDevice = AuroraBridge::Gfx::Device();
    if (!mDevice) {
        SPDLOG_ERROR("Aurora backend: no WebGPU device");
        return;
    }
    CreateSharedObjects();
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
    mUploads[mRecordingSlot].clear();
    ++mFrameCountUniform;
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
    FlushUploads();
    if (mStatFrames == 1 || mStatFrames == 60 || mStatFrames == 600) {
        size_t drawOps = 0;
        for (const auto& p : mFrames[mRecordingSlot]) {
            drawOps += p.drawOps.size();
        }
        AuroraTrace("[lus-aurora] frame %llu: %zu pass(es), %zu draw(s) this frame; %llu shaders, %llu uploads, "
                    "%llu draws skipped so far",
                    static_cast<unsigned long long>(mStatFrames), mFrames[mRecordingSlot].size(), drawOps,
                    static_cast<unsigned long long>(mStatShaders), static_cast<unsigned long long>(mStatUploads),
                    static_cast<unsigned long long>(mStatSkippedDraws));
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
            // The whole on-screen frame: written to lus_aurora_frame.png when mapped.
            mSelfTestX = snap.width;
            mSelfTestY = snap.height;
            wgpu::BufferDescriptor bd{};
            bd.label = "Fast3D self-test frame";
            bd.size = static_cast<uint64_t>((snap.width * 4 + 255) / 256 * 256) * snap.height;
            bd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
            mSelfTestBuffer = mDevice.CreateBuffer(&bd);
            const SelfTestPayload sp{ this, mSelfTestTexture.Get() };
            AuroraBridge::Gfx::PushEncoderTask(mSelfTestTask, &sp, sizeof(sp));
        }
    }

    const double now = mWindow->GetTime();
    if (now - mStatLastLog >= 1.0) {
        SPDLOG_INFO("Aurora backend: {} frames, {} passes, {} clears, {} draws ({} tris, {} skipped), {} shaders, "
                    "{} uploads",
                    mStatFrames, mStatPasses, mStatClears, mStatDraws, mStatTris, mStatSkippedDraws, mStatShaders,
                    mStatUploads);
        mStatLastLog = now;
    }
    ++mFrameCounter;
    mRecording = false;
}

void GfxRenderingAPIAurora::FinishRender() {
}

void GfxRenderingAPIAurora::RetireResources() {
    // A slot is reused after FrameSlots frames; by then no frame in flight uses the old objects.
    std::erase_if(mRetired, [&](const auto& r) { return mFrameCounter - r.first >= FrameSlots + 1; });
    std::erase_if(mRetiredTextures, [&](const auto& r) { return mFrameCounter - r.first >= FrameSlots + 1; });
    if (mFrameCounter % 120 == 0) {
        std::erase_if(mTextureBindGroups,
                      [&](const auto& e) { return mFrameCounter - e.second.second >= FrameSlots + 120; });
    }
}

// --- framebuffers -----------------------------------------------------------------

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
    // E6: WebGPU has 1x and 4x only. Rendering is single-sample for now.
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

void GfxRenderingAPIAurora::StartDrawToFramebuffer(int fbId, float noiseScale) {
    mCurrentFb = fbId;
    if (noiseScale != 0.0f) {
        mNoiseScale = 1.0f / noiseScale;
    }
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
    // new pass on the same target starts with the clear (full-target clears).
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

void GfxRenderingAPIAurora::SelectTextureFb(int fbId) {
    if (fbId < 0 || fbId >= static_cast<int>(mFramebuffers.size()) || fbId == mCurrentFb) {
        // Sampling the target of the open pass is not allowed; copies come in a later step.
        return;
    }
    mSlotFbViews[0] = mFramebuffers[fbId].colorView;
}

// --- render worker ------------------------------------------------------------------

void GfxRenderingAPIAurora::ReplayFrame(const gfx::EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd,
                                        uint32_t slot) {
    // Texture uploads first: pixels from Aurora's storage stream into the textures.
    const auto& ups = mUploads[slot];
    if (!ups.empty()) {
        auto cpass = cmd.BeginComputePass();
        cpass.SetPipeline(mUploadPipeline);
        for (const UploadOp& up : ups) {
            if (up.range.size == 0) {
                continue;
            }
            wgpu::BindGroupEntry e[2]{};
            e[0].binding = 0;
            e[0].buffer = ctx.storageBuffer;
            e[0].offset = up.range.offset;
            e[0].size = up.range.size;
            e[1].binding = 1;
            e[1].textureView = up.storageView;
            wgpu::BindGroupDescriptor bgd{};
            bgd.layout = mUploadBgl;
            bgd.entryCount = 2;
            bgd.entries = e;
            cpass.SetBindGroup(0, ctx.device.CreateBindGroup(&bgd));
            cpass.DispatchWorkgroups((up.width + 7) / 8, (up.height + 7) / 8);
        }
        cpass.End();
    }

    // The uniform bind group for Aurora's uniform buffer (one per buffer object).
    auto ubIt = mUniformBindGroups.find(ctx.uniformBuffer.Get());
    if (ubIt == mUniformBindGroups.end()) {
        wgpu::BindGroupEntry e{};
        e.binding = 0;
        e.buffer = ctx.uniformBuffer;
        e.size = sizeof(DrawUniforms);
        wgpu::BindGroupDescriptor bgd{};
        bgd.layout = mUniformBgl;
        bgd.entryCount = 1;
        bgd.entries = &e;
        ubIt = mUniformBindGroups.emplace(ctx.uniformBuffer.Get(), ctx.device.CreateBindGroup(&bgd)).first;
    }

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
        auto pass = cmd.BeginRenderPass(&rp);
        for (const DrawOp& d : op.drawOps) {
            if (d.verts.size == 0) {
                continue;
            }
            pass.SetPipeline(d.pipeline);
            pass.SetBindGroup(0, ubIt->second, 1, &d.uniformOffset);
            pass.SetBindGroup(1, d.textures);
            pass.SetVertexBuffer(0, ctx.vertexBuffer, d.verts.offset, d.verts.size);
            pass.SetViewport(d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3], 0.0f, 1.0f);
            pass.SetScissorRect(d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3]);
            pass.Draw(d.vertexCount);
        }
        pass.End();
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
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = mSelfTestBuffer;
    dst.layout.bytesPerRow = (mSelfTestX * 4 + 255) / 256 * 256;
    dst.layout.rowsPerImage = mSelfTestY;
    wgpu::Extent3D size{ mSelfTestX, mSelfTestY, 1 };
    cmd.CopyTextureToBuffer(&src, &dst, &size);
}

namespace {

// Minimal PNG writer for the diagnostic frame grab (stored deflate, no compression).
uint32_t Crc32(const uint8_t* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc;
}

void PutBE32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24);
    v.push_back((x >> 16) & 0xFF);
    v.push_back((x >> 8) & 0xFF);
    v.push_back(x & 0xFF);
}

void PngChunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    PutBE32(out, static_cast<uint32_t>(data.size()));
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    PutBE32(out, Crc32(td.data(), td.size()) ^ 0xFFFFFFFFu);
}

bool WritePng(const char* path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> raw;
    raw.reserve((w * 3 + 1) * h);
    for (uint32_t y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + static_cast<size_t>(y) * w * 3,
                   rgb.begin() + static_cast<size_t>(y + 1) * w * 3);
    }
    std::vector<uint8_t> z{ 0x78, 0x01 };
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    for (size_t pos = 0; pos < raw.size();) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);
        z.push_back(n & 0xFF);
        z.push_back(n >> 8);
        z.push_back(~n & 0xFF);
        z.push_back((~n >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    PutBE32(z, (b << 16) | a);
    std::vector<uint8_t> png{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };
    std::vector<uint8_t> ihdr;
    PutBE32(ihdr, w);
    PutBE32(ihdr, h);
    ihdr.insert(ihdr.end(), { 8, 2, 0, 0, 0 });
    PngChunk(png, "IHDR", ihdr);
    PngChunk(png, "IDAT", z);
    PngChunk(png, "IEND", {});
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return f.good();
}

} // namespace

void GfxRenderingAPIAurora::SelfTestMapped() {
    const uint64_t size = mSelfTestBuffer.GetSize();
    mSelfTestBuffer.MapAsync(
        wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowSpontaneous,
        [this, size](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status != wgpu::MapAsyncStatus::Success) {
                SPDLOG_ERROR("Aurora backend self-test: readback failed");
                return;
            }
            const auto* data = static_cast<const uint8_t*>(mSelfTestBuffer.GetConstMappedRange(0, size));
            const uint32_t w = mSelfTestX, h = mSelfTestY, row = (w * 4 + 255) / 256 * 256;
            const bool bgra = mSelfTestFormat == wgpu::TextureFormat::BGRA8Unorm ||
                              mSelfTestFormat == wgpu::TextureFormat::BGRA8UnormSrgb;
            std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) {
                    const uint8_t* p = data + static_cast<size_t>(y) * row + x * 4;
                    uint8_t* q = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
                    q[0] = bgra ? p[2] : p[0];
                    q[1] = p[1];
                    q[2] = bgra ? p[0] : p[2];
                }
            }
            mSelfTestBuffer.Unmap();
            const uint8_t* px = rgb.data() + (static_cast<size_t>(h / 2) * w + w / 2) * 3;
            const int r = px[0], g = px[1], b = px[2];
            const bool wrote = WritePng("lus_aurora_frame.png", w, h, rgb);
            AuroraTrace("[lus-aurora] self-test: frame %u (%ux%u) %s lus_aurora_frame.png", mSelfTestFrame, w, h,
                        wrote ? "saved to" : "could NOT be saved to");
            SPDLOG_INFO("Aurora backend self-test: scene centre pixel at frame {} = {},{},{}", mSelfTestFrame, r, g,
                        b);
            AuroraTrace("[lus-aurora] self-test: scene centre pixel at frame %u = %d,%d,%d", mSelfTestFrame, r, g, b);
        });
}

} // namespace Fast

#endif
