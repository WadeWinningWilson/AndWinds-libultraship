#ifdef ENABLE_AURORA

#include "fast/backends/gfx_aurora.h"
#include "ship/Context.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <thread>
#include <string>

#include <csignal>
#include <exception>

#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace Fast {

void AuroraTrace(const char* fmt, ...) {
    static std::mutex sMutex;
    char line[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    std::lock_guard lock{ sMutex };
    if (FILE* f = std::fopen("lus_aurora_trace.log", "a")) {
        std::fputs(line, f);
        std::fputc('\n', f);
        std::fclose(f);
    }
    std::fprintf(stderr, "%s\n", line);
    std::fflush(stderr);
}

namespace {

#ifdef _WIN32
// Diagnostics (LUS_AURORA_SELFTEST set): record why the process ends instead of
// stopping on the CRT's "abort() has been called" dialog.
void TraceStack() {
    void* frames[48];
    const USHORT n = CaptureStackBackTrace(0, 48, frames, nullptr);
    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(process, nullptr, TRUE);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    for (USHORT i = 0; i < n; ++i) {
        auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        DWORD lineDisp = 0;
        const DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
        const bool hasSym = SymFromAddr(process, addr, &disp, sym);
        const bool hasLine = SymGetLineFromAddr64(process, addr, &lineDisp, &line);
        AuroraTrace("[lus-aurora]   #%u %s %s:%lu", i, hasSym ? sym->Name : "?", hasLine ? line.FileName : "?",
                    hasLine ? line.LineNumber : 0UL);
    }
}

void OnAbort(int) {
    AuroraTrace("[lus-aurora] abort() called; stack:");
    TraceStack();
    std::_Exit(3);
}

void OnTerminate() {
    const char* what = "(no active exception)";
    std::string message;
    if (auto ex = std::current_exception()) {
        try {
            std::rethrow_exception(ex);
        } catch (const std::exception& e) {
            message = e.what();
            what = message.c_str();
        } catch (...) {
            what = "(non-std exception)";
        }
    }
    AuroraTrace("[lus-aurora] std::terminate: %s; stack:", what);
    TraceStack();
    std::_Exit(3);
}

void InstallCrashTrace() {
    if (std::getenv("LUS_AURORA_SELFTEST") == nullptr) {
        return;
    }
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    std::signal(SIGABRT, OnAbort);
    std::set_terminate(OnTerminate);
}
#else
void InstallCrashTrace() {
}
#endif

void BridgeLog(AuroraBridge::LogLevel level, const char* module, const char* message) {
    // Aurora aborts right after a fatal message, before spdlog would flush.
    if (level >= AuroraBridge::LogLevel::Warning) {
        AuroraTrace("[aurora:%s] %s", module, message);
    }
    switch (level) {
        case AuroraBridge::LogLevel::Debug:
            SPDLOG_DEBUG("[aurora:{}] {}", module, message);
            break;
        case AuroraBridge::LogLevel::Info:
            SPDLOG_INFO("[aurora:{}] {}", module, message);
            break;
        case AuroraBridge::LogLevel::Warning:
            SPDLOG_WARN("[aurora:{}] {}", module, message);
            break;
        default:
            SPDLOG_ERROR("[aurora:{}] {}", module, message);
            break;
    }
}

// libultraship's keyboard scancodes are the DXGI backend's: the platform scan
// code with 0x100 for extended keys ((lParam >> 16) & 0x1ff). SDL3's 'raw' key
// field carries the platform scan code; accept either extended form (0x1xx or
// 0xE0xx). Other platforms need a table (gfx_sdl2.cpp's is file-local).
int LusScancode(int raw) {
#ifdef _WIN32
    if ((raw & 0xFF00) == 0xE000) {
        return (raw & 0xFF) | 0x100;
    }
    return raw & 0x1FF;
#else
    (void)raw;
    return 0;
#endif
}

// LUS_AURORA_BACKEND=d3d12|vulkan|d3d11|opengl picks Aurora's GPU API (default: Aurora's choice).
AuroraBridge::Backend BackendFromEnv() {
    const char* v = std::getenv("LUS_AURORA_BACKEND");
    if (v == nullptr) {
        return AuroraBridge::Backend::Auto;
    }
    const std::string s = v;
    if (s == "d3d12") return AuroraBridge::Backend::D3D12;
    if (s == "vulkan") return AuroraBridge::Backend::Vulkan;
    if (s == "d3d11") return AuroraBridge::Backend::D3D11;
    if (s == "opengl") return AuroraBridge::Backend::OpenGL;
    if (s == "metal") return AuroraBridge::Backend::Metal;
    return AuroraBridge::Backend::Auto;
}

} // namespace

void GfxWindowBackendAurora::Init(const char* gameName, const char* apiName, bool startFullScreen, uint32_t width,
                                  uint32_t height, int32_t posX, int32_t posY) {
    InstallCrashTrace();
    AuroraBridge::Config config;
    config.appName = gameName;
    config.width = width;
    config.height = height;
    config.posX = posX;
    config.posY = posY;
    config.fullscreen = startFullScreen;
    config.vsync = mVsyncEnabled;
    config.msaa = 1;
    config.backend = BackendFromEnv();
    config.log = &BridgeLog;
    AuroraBridge::Info info;
    AuroraTrace("[lus-aurora] window: initialising Aurora (GPU API %d)", static_cast<int>(config.backend));
    mInitialized = AuroraBridge::Initialize(config, info);
    AuroraTrace("[lus-aurora] window: Aurora initialised = %d", int(mInitialized));
    if (!mInitialized) {
        SPDLOG_ERROR("Aurora backend: Aurora failed to initialise");
        mIsRunning = false;
        return;
    }
    mFullScreen = startFullScreen;
    const std::string title = std::string(gameName) + " (" + apiName + ")";
    AuroraBridge::SetWindowTitle(title.c_str());
    AuroraBridge::GetWindowSize(&mWidth, &mHeight, &mPosX, &mPosY);
    SPDLOG_INFO("Aurora backend: window {}x{}, GPU API {}", mWidth, mHeight, static_cast<int>(info.backend));
}

void GfxWindowBackendAurora::Close() {
    mIsRunning = false;
}

void GfxWindowBackendAurora::SetKeyboardCallbacks(bool (*onKeyDown)(int scancode), bool (*onKeyUp)(int scancode),
                                                  void (*onAllKeysUp)()) {
    mOnKeyDown = onKeyDown;
    mOnKeyUp = onKeyUp;
    mOnAllKeysUp = onAllKeysUp;
}

void GfxWindowBackendAurora::SetMouseCallbacks(bool (*onMouseButtonDown)(int btn), bool (*onMouseButtonUp)(int btn)) {
    mOnMouseButtonDown = onMouseButtonDown;
    mOnMouseButtonUp = onMouseButtonUp;
}

void GfxWindowBackendAurora::SetFullscreenChangedCallback(void (*onFullscreenChanged)(bool isNowFullscreen)) {
    mOnFullscreenChanged = onFullscreenChanged;
}

void GfxWindowBackendAurora::SetFullscreen(bool fullscreen) {
    if (fullscreen == mFullScreen) {
        return;
    }
    AuroraBridge::SetFullscreen(fullscreen);
    mFullScreen = fullscreen;
    if (mOnFullscreenChanged != nullptr) {
        mOnFullscreenChanged(fullscreen);
    }
}

void GfxWindowBackendAurora::GetActiveWindowRefreshRate(uint32_t* refreshRate) {
    *refreshRate = AuroraBridge::GetRefreshRate();
}

void GfxWindowBackendAurora::SetCursorVisibility(bool visible) {
    AuroraBridge::SetCursorVisible(visible);
}

void GfxWindowBackendAurora::SetMousePos(int32_t, int32_t) {
}

void GfxWindowBackendAurora::GetMousePos(int32_t* x, int32_t* y) {
    AuroraBridge::GetMousePos(x, y);
}

void GfxWindowBackendAurora::GetMouseDelta(int32_t* x, int32_t* y) {
    *x = 0;
    *y = 0;
}

void GfxWindowBackendAurora::GetMouseWheel(float* x, float* y) {
    *x = 0;
    *y = 0;
}

bool GfxWindowBackendAurora::GetMouseState(uint32_t btn) {
    return btn < 32 && (mMouseButtons & (1u << btn)) != 0;
}

void GfxWindowBackendAurora::SetMouseCapture(bool) {
}

bool GfxWindowBackendAurora::IsMouseCaptured() {
    return false;
}

void GfxWindowBackendAurora::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    *width = mWidth;
    *height = mHeight;
    *posX = mPosX;
    *posY = mPosY;
}

void GfxWindowBackendAurora::SetDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) {
    AuroraBridge::SetWindowSize(width, height, posX, posY);
    AuroraBridge::GetWindowSize(&mWidth, &mHeight, &mPosX, &mPosY);
}

Ship::WindowRect GfxWindowBackendAurora::GetPrimaryMonitorRect() {
    return { 0, 0, static_cast<int32_t>(mWidth), static_cast<int32_t>(mHeight) };
}

void GfxWindowBackendAurora::HandleEvents() {
    if (!mInitialized) {
        return;
    }
    AuroraBridge::Event events[64];
    const int n = AuroraBridge::Poll(events, 64);
    static uint64_t sCalls = 0, sEvents = 0;
    ++sCalls;
    sEvents += static_cast<uint64_t>(n);
    if (sCalls == 1 || sCalls == 60 || sCalls == 600) {
        AuroraTrace("[lus-aurora] HandleEvents #%llu: %llu translated events so far",
                    static_cast<unsigned long long>(sCalls), static_cast<unsigned long long>(sEvents));
    }
    for (int i = 0; i < n; ++i) {
        const auto& ev = events[i];
        switch (ev.type) {
            case AuroraBridge::Event::Type::Exit:
                mIsRunning = false;
                break;
            case AuroraBridge::Event::Type::Resized:
                AuroraBridge::GetWindowSize(&mWidth, &mHeight, &mPosX, &mPosY);
                break;
            case AuroraBridge::Event::Type::MouseDown:
                if (ev.a >= 0 && ev.a < 32) mMouseButtons |= 1u << ev.a;
                if (mOnMouseButtonDown != nullptr) mOnMouseButtonDown(ev.a - 1);
                break;
            case AuroraBridge::Event::Type::MouseUp:
                if (ev.a >= 0 && ev.a < 32) mMouseButtons &= ~(1u << ev.a);
                if (mOnMouseButtonUp != nullptr) mOnMouseButtonUp(ev.a - 1);
                break;
            case AuroraBridge::Event::Type::FocusLost:
                if (mOnAllKeysUp != nullptr) mOnAllKeysUp();
                break;
            case AuroraBridge::Event::Type::KeyDown:
            case AuroraBridge::Event::Type::KeyUp: {
                const bool down = ev.type == AuroraBridge::Event::Type::KeyDown;
                const int key = LusScancode(ev.b);
                static int sTraced = 0;
                if (sTraced < 10) {
                    ++sTraced;
                    AuroraTrace("[lus-aurora] key %s: SDL %d, raw 0x%x -> libultraship 0x%x", down ? "down" : "up",
                                ev.a, ev.b, key);
                }
                if (key != 0) {
                    if (down && mOnKeyDown != nullptr) mOnKeyDown(key);
                    if (!down && mOnKeyUp != nullptr) mOnKeyUp(key);
                }
                break;
            }
            default:
                break;
        }
    }
    AuroraBridge::GetWindowSize(&mWidth, &mHeight, &mPosX, &mPosY);
}

bool GfxWindowBackendAurora::IsFrameReady() {
    // Aurora paces frames itself (vsync in present).
    return true;
}

bool GfxWindowBackendAurora::BeginFrameIfNeeded() {
    if (!mInitialized) {
        return false;
    }
    if (!mFrameOpen) {
        mFrameOpen = AuroraBridge::BeginFrame();
    }
    return mFrameOpen;
}

void GfxWindowBackendAurora::WaitForNextFrame() {
    // Hold each frame to 1 / target FPS after the previous one, like the DXGI
    // backend: a coarse wait, then a short spin for the last 1.5 ms. Aurora's
    // present adds vsync on top when enabled.
    if (Ship::Context::HostPacesFrames() || mTargetFps <= 0) {
        mPreviousPresentTime = AuroraBridge::GetTime();
        return;
    }
    const double next = mPreviousPresentTime + 1.0 / static_cast<double>(mTargetFps);
    double now = AuroraBridge::GetTime();
    const double coarse = next - now - 0.0015;
    if (coarse > 0) {
#ifdef _WIN32
        if (mTimer == nullptr) {
            mTimer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (mTimer == nullptr) {
                mTimer = CreateWaitableTimer(nullptr, FALSE, nullptr);
            }
        }
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(coarse * 1e7); // 100 ns units, relative
        SetWaitableTimer(static_cast<HANDLE>(mTimer), &due, 0, nullptr, nullptr, false);
        WaitForSingleObject(static_cast<HANDLE>(mTimer), INFINITE);
#else
        std::this_thread::sleep_for(std::chrono::duration<double>(coarse));
#endif
    }
    do {
        now = AuroraBridge::GetTime();
    } while (now < next);
    // A frame that ran long starts the schedule again rather than racing to catch up.
    mPreviousPresentTime = (now - next > 1.0 / static_cast<double>(mTargetFps)) ? now : next;
}

void GfxWindowBackendAurora::SwapBuffersBegin() {
    if (mFrameOpen) {
        WaitForNextFrame();
        AuroraBridge::EndFrame();
        mFrameOpen = false;
    }
}

void GfxWindowBackendAurora::SwapBuffersEnd() {
}

double GfxWindowBackendAurora::GetTime() {
    return AuroraBridge::GetTime();
}

int GfxWindowBackendAurora::GetTargetFps() {
    return mTargetFps;
}

void GfxWindowBackendAurora::SetTargetFps(int fps) {
    mTargetFps = fps;
}

void GfxWindowBackendAurora::SetMaxFrameLatency(int) {
}

const char* GfxWindowBackendAurora::GetKeyName(int) {
    return "";
}

bool GfxWindowBackendAurora::CanDisableVsync() {
    return false;
}

bool GfxWindowBackendAurora::IsRunning() {
    return mIsRunning;
}

void GfxWindowBackendAurora::Destroy() {
    if (mInitialized) {
        AuroraBridge::Shutdown();
        mInitialized = false;
    }
#ifdef _WIN32
    if (mTimer != nullptr) {
        CloseHandle(static_cast<HANDLE>(mTimer));
        mTimer = nullptr;
    }
#endif
}

bool GfxWindowBackendAurora::IsFullscreen() {
    return mFullScreen;
}

} // namespace Fast

#endif
