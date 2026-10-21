#pragma once

// A DLL boundary around Aurora for libultraship's Aurora backend.
//
// Aurora owns an SDL3 window, events and device; libultraship links SDL2
// statically. Linked into one module, the two SDLs share C symbol names and
// calls silently cross over (Fast3D_Aurora_Backend_Gap_Map.md §6.6). So
// Aurora and its SDL3 live inside aurora_bridge.dll, and the backend talks
// to them only through this header: plain values, WebGPU handles (from the
// one shared webgpu_dawn.dll), and Aurora's own gfx extension types.

#include <cstdint>

#include <aurora/gfx.hpp>

#if defined(_WIN32)
#if defined(AURORA_BRIDGE_BUILD)
#define AURORA_BRIDGE_API __declspec(dllexport)
#else
#define AURORA_BRIDGE_API __declspec(dllimport)
#endif
#else
#define AURORA_BRIDGE_API __attribute__((visibility("default")))
#endif

namespace AuroraBridge {

enum class LogLevel : int { Debug, Info, Warning, Error, Fatal };
using LogCallback = void (*)(LogLevel level, const char* module, const char* message);

enum class Backend : int { Auto, D3D11, D3D12, Metal, Vulkan, OpenGL, OpenGLES, WebGPU, Null };

struct Config {
    const char* appName = "libultraship";
    uint32_t width = 640;
    uint32_t height = 480;
    int32_t posX = -1;
    int32_t posY = -1;
    bool fullscreen = false;
    bool vsync = true;
    uint32_t msaa = 1;
    Backend backend = Backend::Auto;
    LogCallback log = nullptr;
};

struct Info {
    Backend backend = Backend::Null;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fbWidth = 0;
    uint32_t fbHeight = 0;
};

struct Event {
    enum class Type : int {
        None,
        Exit,
        Resized,     // a = width, b = height (pixels)
        KeyDown,     // a = SDL scancode (same values in SDL2 and SDL3), b = platform scancode (SDL3 'raw')
        KeyUp,       // a = SDL scancode, b = platform scancode
        MouseDown,   // a = button (1 left, 2 middle, 3 right)
        MouseUp,     // a = button
        FocusLost,
        FocusGained,
    };
    Type type = Type::None;
    int32_t a = 0;
    int32_t b = 0;
};

// --- application ------------------------------------------------------------
AURORA_BRIDGE_API bool Initialize(const Config& config, Info& info);
AURORA_BRIDGE_API void Shutdown();
// Pumps Aurora (aurora_update) and returns up to `max` translated events.
AURORA_BRIDGE_API int Poll(Event* events, int max);
AURORA_BRIDGE_API bool BeginFrame();
AURORA_BRIDGE_API void EndFrame();
AURORA_BRIDGE_API Backend GetBackend();

// --- window (Aurora's SDL3 window) -----------------------------------------
AURORA_BRIDGE_API void GetWindowSize(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY);
AURORA_BRIDGE_API void SetWindowSize(uint32_t width, uint32_t height, int32_t posX, int32_t posY);
AURORA_BRIDGE_API void SetWindowTitle(const char* title);
AURORA_BRIDGE_API void SetFullscreen(bool fullscreen);
AURORA_BRIDGE_API bool IsFullscreen();
AURORA_BRIDGE_API void SetCursorVisible(bool visible);
AURORA_BRIDGE_API void GetMousePos(int32_t* x, int32_t* y);
AURORA_BRIDGE_API uint32_t GetRefreshRate();
AURORA_BRIDGE_API double GetTime();

// --- gfx (aurora::gfx, forwarded) -------------------------------------------
namespace Gfx {
AURORA_BRIDGE_API wgpu::Device Device();
AURORA_BRIDGE_API wgpu::Queue Queue();
AURORA_BRIDGE_API aurora::gfx::RenderTargetLayout SceneLayout();
AURORA_BRIDGE_API bool UsesReversedZ();
AURORA_BRIDGE_API aurora::gfx::DrawTypeId RegisterDrawType(const aurora::gfx::DrawTypeDescriptor& desc);
AURORA_BRIDGE_API void UnregisterDrawType(aurora::gfx::DrawTypeId type);
AURORA_BRIDGE_API bool PushCustomDraw(aurora::gfx::DrawTypeId type, const void* payload, size_t payloadSize);
AURORA_BRIDGE_API aurora::gfx::EncoderTaskId RegisterEncoderTaskType(const aurora::gfx::EncoderTaskDescriptor& desc);
AURORA_BRIDGE_API void UnregisterEncoderTaskType(aurora::gfx::EncoderTaskId type);
AURORA_BRIDGE_API bool PushEncoderTask(aurora::gfx::EncoderTaskId type, const void* payload, size_t payloadSize);
AURORA_BRIDGE_API aurora::gfx::Range PushVerts(const uint8_t* data, size_t length, size_t alignment);
AURORA_BRIDGE_API aurora::gfx::Range PushIndices(const uint8_t* data, size_t length, size_t alignment);
AURORA_BRIDGE_API aurora::gfx::Range PushUniform(const uint8_t* data, size_t length);
AURORA_BRIDGE_API aurora::gfx::Range PushStorage(const uint8_t* data, size_t length);
AURORA_BRIDGE_API bool ResolvePass(const aurora::gfx::ResolveDesc& desc, aurora::gfx::ResolvedTargets& out);
AURORA_BRIDGE_API uint32_t CurrentFrame();
// Drains the render worker. The game thread may use Queue() until it records again.
AURORA_BRIDGE_API void Synchronize();
} // namespace Gfx

} // namespace AuroraBridge
