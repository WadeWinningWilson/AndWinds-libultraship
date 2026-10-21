#pragma once

#ifdef ENABLE_AURORA

// Fast3D on Aurora's lower WebGPU layer (not GX), through aurora_bridge.dll.
// Design: ALBT&M DOCS/Fast3D_Aurora_Backend_Gap_Map.md (model B): the backend
// owns every Fast3D framebuffer as its own wgpu::Texture, records its passes
// for the frame and replays them in one Aurora encoder task, then composites
// framebuffer 0 into Aurora's scene pass with one custom draw. Aurora owns the
// window, device, queue, frame encoder, submit and present.
//
// Step 1 (this version): window, frame lifecycle, framebuffers, clears and the
// composite. Draws, textures and readbacks are accepted and counted but not
// rendered yet.

#include <array>
#include <map>
#include <utility>
#include <vector>

#include <aurora_bridge.h>

#include "fast/backends/gfx_rendering_api.h"
#include "fast/backends/gfx_window_manager_api.h"

namespace Fast {

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
    void ReplayFrame(const wgpu::CommandEncoder& cmd, uint32_t slot);
    void DrawComposite(const aurora::gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass,
                       WGPUTextureView view);
    void SelfTestCopy(const wgpu::CommandEncoder& cmd, WGPUTexture texture);
    void SelfTestMapped();

    struct Framebuffer {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t samples = 1;
        bool hasDepth = false;
        wgpu::Texture color;
        wgpu::TextureView colorView;
        wgpu::Texture depth;
        wgpu::TextureView depthView;
    };

    // One recorded pass (step 1: begin/clear only).
    struct PassOp {
        int fb = 0;
        bool clearColor = false;
        bool clearDepth = false;
        std::array<float, 4> color{ 0.0f, 0.0f, 0.0f, 1.0f };
        uint32_t draws = 0;
        // Targets captured at record time, so the worker never reads mFramebuffers.
        wgpu::TextureView colorView;
        wgpu::TextureView depthView;
    };

    struct ShaderProgramAurora {
        uint64_t id0 = 0;
        uint64_t id1 = 0;
        uint8_t numInputs = 0;
        bool usedTextures[2]{};
    };

  private:
    static constexpr uint32_t FrameSlots = 3; // Aurora keeps 2 frames in flight; one more for recording.

    void BeginPass(int fb);
    void RetireResources();

    GfxWindowBackendAurora* mWindow;
    wgpu::Device mDevice;
    aurora::gfx::EncoderTaskId mReplayTask = aurora::gfx::InvalidEncoderTask;
    aurora::gfx::DrawTypeId mCompositeDraw = aurora::gfx::InvalidDrawType;
    aurora::gfx::EncoderTaskId mSelfTestTask = aurora::gfx::InvalidEncoderTask;

    std::vector<Framebuffer> mFramebuffers;
    int mCurrentFb = 0;
    std::array<std::vector<PassOp>, FrameSlots> mFrames;
    // Framebuffer 0's view per slot: holds the reference the composite draw samples.
    std::array<wgpu::TextureView, FrameSlots> mCompositeViews;
    uint32_t mFrameCounter = 0;
    uint32_t mRecordingSlot = 0;
    bool mRecording = false;
    // Textures replaced by a resize, kept until no frame in flight can use them.
    std::vector<std::pair<uint32_t, Framebuffer>> mRetired;

    std::map<std::pair<uint64_t, uint64_t>, ShaderProgramAurora> mShaders;
    ShaderProgramAurora* mCurrentShader = nullptr;
    uint32_t mTextureCount = 0;
    FilteringMode mFilter = FILTER_THREE_POINT;

    // Composite (worker thread only).
    wgpu::RenderPipeline mCompositePipeline;
    uint64_t mCompositeLayoutKey = ~0ull;
    wgpu::BindGroupLayout mCompositeBgl;
    wgpu::Sampler mCompositeSampler;
    WGPUTextureView mCompositeBoundView = nullptr;
    wgpu::BindGroup mCompositeBg;

    // Diagnostics: LUS_AURORA_DEBUG_CLEAR=r,g,b (0-255) overrides framebuffer 0's
    // clear colour; LUS_AURORA_SELFTEST=<frame> logs the scene's centre pixel then.
    bool mDebugClear = false;
    std::array<float, 4> mDebugClearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
    uint32_t mSelfTestFrame = 0;
    wgpu::Buffer mSelfTestBuffer;
    wgpu::Texture mSelfTestTexture;
    wgpu::TextureFormat mSelfTestFormat = wgpu::TextureFormat::Undefined;
    uint32_t mSelfTestX = 0;
    uint32_t mSelfTestY = 0;

    // Step-1 counters, logged once a second.
    uint64_t mStatFrames = 0;
    uint64_t mStatDraws = 0;
    uint64_t mStatTris = 0;
    uint64_t mStatClears = 0;
    uint64_t mStatPasses = 0;
    double mStatLastLog = 0;
};

} // namespace Fast

#endif
