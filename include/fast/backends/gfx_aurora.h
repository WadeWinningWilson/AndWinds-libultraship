#pragma once

#ifdef ENABLE_AURORA

// Fast3D on Aurora's lower WebGPU layer (not GX), through aurora_bridge.dll.
// Design: ALBT&M DOCS/Fast3D_Aurora_Backend_Gap_Map.md (model B): the backend
// owns every Fast3D framebuffer as its own wgpu::Texture, records its passes
// for the frame and replays them in one Aurora encoder task, then composites
// framebuffer 0 into Aurora's scene pass with one custom draw. Aurora owns the
// window, device, queue, frame encoder, submit and present.
//
// Step 1: window, frame lifecycle, framebuffers, clears and the composite.
// Step 2: WGSL combiners, pipelines, textures, samplers and draws. Copies,
// resolves and readbacks come in later steps.

#include <array>
#include <atomic>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <aurora_bridge.h>

#include "fast/backends/gfx_rendering_api.h"
#include "fast/backends/gfx_window_manager_api.h"

struct CCFeatures;
struct ImDrawData;

namespace Fast {

// Builds the WGSL module for one Fast3D combiner (gfx_aurora_shader.cpp).
std::string BuildAuroraShader(const ::CCFeatures& cc, bool threePointFiltering, bool srgbMode);

// Writes to lus_aurora_trace.log (working directory) and stderr. Aurora aborts on
// a fatal error before spdlog flushes, and on Windows libultraship points stderr
// at its own console, so the file is what survives.
void AuroraTrace(const char* fmt, ...);

class GfxWindowBackendAurora final : public GfxWindowBackend {
  public:
    void Init(const char* gameName, const char* apiName, bool startFullScreen, uint32_t width, uint32_t height,
              int32_t posX, int32_t posY) override;
    void Close() override;
    void SetKeyboardCallbacks(bool (*onKeyDown)(int scancode), bool (*onKeyUp)(int scancode),
                              void (*onAllKeysUp)()) override;
    void SetMouseCallbacks(bool (*onMouseButtonDown)(int btn), bool (*onMouseButtonUp)(int btn)) override;
    void SetFullscreenChangedCallback(void (*onFullscreenChanged)(bool isNowFullscreen)) override;
    void SetFullscreen(bool fullscreen) override;
    void GetActiveWindowRefreshRate(uint32_t* refreshRate) override;
    void SetCursorVisibility(bool visible) override;
    void SetMousePos(int32_t posX, int32_t posY) override;
    void GetMousePos(int32_t* x, int32_t* y) override;
    void GetMouseDelta(int32_t* x, int32_t* y) override;
    void GetMouseWheel(float* x, float* y) override;
    bool GetMouseState(uint32_t btn) override;
    void SetMouseCapture(bool capture) override;
    bool IsMouseCaptured() override;
    void GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) override;
    void SetDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) override;
    Ship::WindowRect GetPrimaryMonitorRect() override;
    void HandleEvents() override;
    bool IsFrameReady() override;
    void SwapBuffersBegin() override;
    void SwapBuffersEnd() override;
    double GetTime() override;
    int GetTargetFps() override;
    void SetTargetFps(int fps) override;
    void SetMaxFrameLatency(int latency) override;
    const char* GetKeyName(int scancode) override;
    bool CanDisableVsync() override;
    bool IsRunning() override;
    void Destroy() override;
    bool IsFullscreen() override;

    // Aurora's frame: opened by the rendering API's StartFrame, closed (and
    // presented) by SwapBuffersBegin.
    bool BeginFrameIfNeeded();
    bool IsFrameOpen() const {
        return mFrameOpen;
    }

  private:
    bool mInitialized = false;
    bool mFrameOpen = false;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    int32_t mPosX = 0;
    int32_t mPosY = 0;
    uint32_t mMouseButtons = 0;
    // Frame pacing to the target FPS (as GfxWindowBackendDXGI::SwapBuffersBegin).
    void WaitForNextFrame();
    double mPreviousPresentTime = 0.0;
    void* mTimer = nullptr; // Windows high-resolution waitable timer
    void (*mOnAllKeysUp)() = nullptr;
};

class GfxRenderingAPIAurora final : public GfxRenderingAPI {
  public:
    explicit GfxRenderingAPIAurora(GfxWindowBackendAurora* window);
    ~GfxRenderingAPIAurora() override;

    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    void ClearShaderCache() override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ClearDepthRegion(int x, int y, int w, int h) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;
    void SetCurrentPrimDepth(float depth) override;

    // Render-worker callbacks (Aurora encoder task and custom draw).
    void ReplayFrame(const aurora::gfx::EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, uint32_t slot);
    void DrawComposite(const aurora::gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass,
                       WGPUTextureView view);
    void SelfTestCopy(const wgpu::CommandEncoder& cmd, WGPUTexture texture);
    // libultraship's ImGui: copied on the game thread (Fast3dGui::ImGuiRenderDrawData),
    // drawn by imgui_impl_wgpu into Aurora's scene pass after the game composite.
    void RenderImGui(ImDrawData* data);
    void DrawImGui(const wgpu::RenderPassEncoder& pass, uint32_t slot);
    void SelfTestMapped();

    struct Framebuffer {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t samples = 1;
        bool hasDepth = false;
        wgpu::Texture color;     // single-sample; with MSAA, the resolved image
        wgpu::TextureView colorView;
        wgpu::Texture msaaColor; // 4x when the framebuffer asked for MSAA (E6), resolved into color
        wgpu::TextureView msaaView;
        wgpu::Texture depth;     // same sample count as the pass
        wgpu::TextureView depthView;
        // Used as a texture (SelectTextureFb), a framebuffer takes sampler state like one.
        bool linear = false;
        uint32_t cms = 0;
        uint32_t cmt = 0;
        int filtering = FILTER_LINEAR;
        // Copy of this framebuffer for sampling it while it is the open target.
        wgpu::Texture snapshot;
        wgpu::TextureView snapshotView;
    };

    // One recorded draw: everything the worker needs, captured at record time.
    struct DrawOp {
        wgpu::RenderPipeline pipeline;
        wgpu::BindGroup textures;
        aurora::gfx::Range verts;
        uint32_t uniformOffset = 0;
        uint32_t vertexCount = 0;
        std::array<float, 4> viewport{}; // x, y, w, h (top-left origin, clamped to the target)
        std::array<uint32_t, 4> scissor{}; // x, y, w, h (clamped to the target)
        // CPU copies, for replaying mid-frame through the backend's own buffers
        // (ReadFramebufferToCPU): Aurora's streams reach the GPU only at end of frame.
        std::vector<float> vertsCpu;
        std::array<uint8_t, 80> uniformsCpu{};
    };

    // One recorded pass on a backend-owned framebuffer.
    // One recorded op: a pass on a backend-owned framebuffer, or (fb == -1) a copy
    // between framebuffers, which WebGPU only allows outside a pass.
    struct CopyOp {
        wgpu::Texture src;
        wgpu::Texture dst;
        std::array<uint32_t, 2> srcOrigin{};
        std::array<uint32_t, 2> dstOrigin{};
        std::array<uint32_t, 2> size{};
    };

    struct PassOp {
        int fb = 0;
        bool isCopy = false;
        CopyOp copy;
        bool clearColor = false;
        bool clearDepth = false;
        std::array<float, 4> color{ 0.0f, 0.0f, 0.0f, 1.0f };
        uint32_t draws = 0;
        // Targets captured at record time, so the worker never reads mFramebuffers.
        wgpu::TextureView colorView;
        wgpu::TextureView msaaView; // set: render here, resolve into colorView
        wgpu::TextureView depthView;
        std::vector<DrawOp> drawOps;
    };

    // A texture upload: pixels in Aurora's storage stream, written by a compute
    // pass in the frame's encoder task (no queue writes from the game thread, E3).
    struct UploadOp {
        wgpu::TextureView storageView;
        uint32_t width = 0;
        uint32_t height = 0;
        aurora::gfx::Range range;
    };

    struct PendingUpload {
        wgpu::TextureView storageView;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<uint32_t> words; // width, height, then RGBA8 pixels
    };

    struct TextureAurora {
        wgpu::Texture texture;
        wgpu::TextureView view;
        uint32_t width = 0;
        uint32_t height = 0;
        bool linear = false;
        uint32_t cms = 0;
        uint32_t cmt = 0;
        int filtering = FILTER_THREE_POINT;
    };

    struct ShaderProgramAurora {
        uint64_t id0 = 0;
        uint64_t id1 = 0;
        uint8_t numInputs = 0;
        bool usedTextures[2]{};
        bool alpha = false;
        wgpu::ShaderModule module;
        std::vector<wgpu::VertexAttribute> attributes;
        uint32_t stride = 0;
    };

  private:
    static constexpr uint32_t FrameSlots = 3; // Aurora keeps 2 frames in flight; one more for recording.
    static constexpr uint32_t TextureSlots = 6; // tex0, tex1, mask0, mask1, blend0, blend1
    static constexpr size_t UploadBudgetPerFrame = 6u << 20; // of Aurora's 8 MiB storage stream

    void BeginPass(int fb);
    void RetireResources();
    void CreateSharedObjects();
    wgpu::RenderPipeline GetPipeline(ShaderProgramAurora* prg, bool hasDepth, uint32_t samples);
    wgpu::Sampler GetSampler(bool linear, uint32_t cms, uint32_t cmt);
    wgpu::BindGroup GetTextureBindGroup();
    void FlushUploads();
    // Records into cmd: uploads (from storageBuffer ranges), then passes and copies.
    void ReplayOps(const wgpu::Device& device, const wgpu::CommandEncoder& cmd, const std::vector<UploadOp>& ups,
                   const std::vector<PassOp>& ops, const wgpu::Buffer& storageBuffer, const wgpu::Buffer& vertexBuffer,
                   const wgpu::BindGroup& uniforms);
    // Executes everything recorded so far this frame on the GPU now (after
    // gfx::synchronize(), through the backend's own encoder and buffers) and
    // drops it from the frame. For mid-frame reads (E2).
    void ExecuteRecordedNow();
    bool WaitForMap(const wgpu::Buffer& buffer, uint64_t size);

    GfxWindowBackendAurora* mWindow;
    wgpu::Device mDevice;
    aurora::gfx::EncoderTaskId mReplayTask = aurora::gfx::InvalidEncoderTask;
    aurora::gfx::DrawTypeId mCompositeDraw = aurora::gfx::InvalidDrawType;
    aurora::gfx::EncoderTaskId mSelfTestTask = aurora::gfx::InvalidEncoderTask;

    std::vector<Framebuffer> mFramebuffers;
    int mCurrentFb = 0;
    std::array<std::vector<PassOp>, FrameSlots> mFrames;
    std::array<std::vector<UploadOp>, FrameSlots> mUploads;
    // Framebuffer 0's view per slot: holds the reference the composite draw samples.
    std::array<wgpu::TextureView, FrameSlots> mCompositeViews;
    uint32_t mFrameCounter = 0;
    uint32_t mRecordingSlot = 0;
    bool mRecording = false;
    // Objects replaced mid-run (resized framebuffers), kept until no frame in flight can use them.
    std::vector<std::pair<uint32_t, Framebuffer>> mRetired;
    std::vector<std::pair<uint32_t, wgpu::Texture>> mRetiredTextures;

    // Shaders and pipelines.
    std::map<std::pair<uint64_t, uint64_t>, ShaderProgramAurora> mShaders;
    ShaderProgramAurora* mCurrentShader = nullptr;
    std::unordered_map<uint64_t, wgpu::RenderPipeline> mPipelines;
    wgpu::BindGroupLayout mUniformBgl;
    wgpu::BindGroupLayout mTextureBgl;
    wgpu::PipelineLayout mPipelineLayout;
    bool mUseAlpha = false;

    // Textures and samplers.
    std::vector<TextureAurora> mTextures;
    std::array<uint32_t, TextureSlots> mSlotTextures{};
    std::array<wgpu::TextureView, TextureSlots> mSlotFbViews; // SelectTextureFb overrides a slot
    std::array<int, TextureSlots> mSlotFbIds{};                 // ... with this framebuffer (-1: none)
    int mCurrentTile = 0;
    std::map<uint32_t, wgpu::Sampler> mSamplers;
    std::map<std::array<void*, TextureSlots + 2>, std::pair<wgpu::BindGroup, uint32_t>> mTextureBindGroups;
    wgpu::Texture mDummyTexture;
    wgpu::TextureView mDummyView;
    std::deque<PendingUpload> mPendingUploads;
    FilteringMode mFilter = FILTER_THREE_POINT;

    // Upload compute pipeline (worker uses it).
    wgpu::ComputePipeline mUploadPipeline;
    wgpu::Instance mInstance;
    aurora::gfx::DrawTypeId mImGuiDraw = aurora::gfx::InvalidDrawType;
    std::array<ImDrawData*, FrameSlots> mImGuiFrames{};
    std::array<bool, FrameSlots> mImGuiThisFrame{};
    bool mImGuiReady = false;
    uint64_t mImGuiLayoutKey = 0;
    // GetPixelDepth: a compute pass loads depth at the queried points (E4).
    wgpu::ComputePipeline mDepthPipeline;
    wgpu::ComputePipeline mDepthPipelineMs; // multisampled depth: sample 0
    uint64_t mStatReads = 0;
    uint64_t mStatDepthReads = 0;
    wgpu::BindGroupLayout mUploadBgl;

    // Per-draw state.
    std::array<int, 4> mViewport{}; // x, y, w, h as Fast3D gives them (bottom-left origin)
    std::array<int, 4> mScissor{};
    float mNoiseScale = 1.0f;
    int32_t mFrameCountUniform = 0;

    // Worker-side: uniform bind group per Aurora uniform buffer.
    std::map<WGPUBuffer, wgpu::BindGroup> mUniformBindGroups;

    // Composite (worker thread only).
    wgpu::RenderPipeline mCompositePipeline;
    uint64_t mCompositeLayoutKey = ~0ull;
    wgpu::BindGroupLayout mCompositeBgl;
    wgpu::Sampler mCompositeSampler;
    WGPUTextureView mCompositeBoundView = nullptr;
    wgpu::BindGroup mCompositeBg;

    // Diagnostics: LUS_AURORA_DEBUG_CLEAR=r,g,b (0-255) overrides framebuffer 0's
    // clear colour; LUS_AURORA_SELFTEST=<frame> logs the scene's centre pixel then;
    // with LUS_AURORA_SELFTEST set, each new WGSL module is written to lus_aurora_shaders/.
    bool mDebugClear = false;
    std::array<float, 4> mDebugClearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
    uint32_t mSelfTestFrame = 0;
    double mSelfTestAtTime = 0.0;
    double mStartTime = 0.0;
    bool mSelfTestTaken = false;
    // LUS_AURORA_SELFTEST_BURST=<n>: after the first capture, n more, each as soon as the
    // previous one is written, to lus_aurora_frame_<1..n>.png (short effects such as sword trails).
    uint32_t mSelfTestBurst = 0;
    uint32_t mSelfTestShot = 0;
    std::atomic<bool> mSelfTestBusy{ false };
    // LUS_AURORA_READBACK_TEST=t<seconds>: at that time, ReadFramebufferToCPU(0, 320, 240)
    // mid-frame (exercising the flush path) and save it to lus_aurora_readback.png.
    double mReadbackTestAt = 0.0;
    bool mReadbackTestDone = false;
    wgpu::Buffer mSelfTestBuffer;
    wgpu::Texture mSelfTestTexture;
    wgpu::TextureFormat mSelfTestFormat = wgpu::TextureFormat::Undefined;
    uint32_t mSelfTestX = 0;
    uint32_t mSelfTestY = 0;

    // Counters, logged once a second.
    uint64_t mStatFrames = 0;
    uint64_t mStatDraws = 0;
    uint64_t mStatTris = 0;
    uint64_t mStatClears = 0;
    uint64_t mStatPasses = 0;
    uint64_t mStatUploads = 0;
    uint64_t mStatShaders = 0;
    uint64_t mStatSkippedDraws = 0;
    uint64_t mStatCopies = 0;
    double mStatLastLog = 0;
};

} // namespace Fast

#endif
