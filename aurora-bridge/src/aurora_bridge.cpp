#include "aurora_bridge.h"

#include <aurora/aurora.h>
#include <aurora/event.h>

#include <SDL3/SDL.h>

#include <cstring>

namespace AuroraBridge {

namespace {

LogCallback sLog = nullptr;
SDL_Window* sWindow = nullptr;
Backend sBackend = Backend::Null;

void AuroraLog(AuroraLogLevel level, const char* module, const char* message, unsigned int) {
    if (sLog != nullptr) {
        sLog(static_cast<LogLevel>(level), module, message);
    }
}

int32_t MouseButton(uint8_t sdlButton) {
    switch (sdlButton) {
        case SDL_BUTTON_LEFT:
            return 1;
        case SDL_BUTTON_MIDDLE:
            return 2;
        case SDL_BUTTON_RIGHT:
            return 3;
        default:
            return sdlButton;
    }
}

} // namespace

bool Initialize(const Config& config, Info& info) {
    sLog = config.log;
    AuroraConfig ac{};
    ac.appName = config.appName;
    ac.desiredBackend = static_cast<AuroraBackend>(config.backend);
    ac.msaa = config.msaa;
    ac.vsync = config.vsync;
    ac.startFullscreen = config.fullscreen;
    ac.windowPosX = config.posX;
    ac.windowPosY = config.posY;
    ac.windowWidth = config.width;
    ac.windowHeight = config.height;
    ac.allowJoystickBackgroundEvents = true;
    ac.logCallback = &AuroraLog;
    ac.logLevel = LOG_INFO;
    // No GameCube memory: libultraship games do not use Aurora's MEM1/ARAM.
    ac.mem1Size = 0;
    ac.mem2Size = 0;
    const AuroraInfo ai = aurora_initialize(0, nullptr, &ac);
    sWindow = ai.window;
    if (sWindow != nullptr) {
        // Text events for libultraship's ImGui (SDL3 delivers them only when enabled).
        SDL_StartTextInput(sWindow);
    }
    sBackend = static_cast<Backend>(ai.backend);
    info.backend = sBackend;
    info.width = ai.windowSize.width;
    info.height = ai.windowSize.height;
    info.fbWidth = ai.windowSize.fb_width;
    info.fbHeight = ai.windowSize.fb_height;
    return sWindow != nullptr;
}

void Shutdown() {
    aurora::gfx::synchronize();
    aurora_shutdown();
    sWindow = nullptr;
}

int Poll(Event* events, int max) {
    int n = 0;
    for (const AuroraEvent* ev = aurora_update(); ev != nullptr && ev->type != AURORA_NONE; ++ev) {
        if (n >= max) {
            break;
        }
        Event out{};
        switch (ev->type) {
            case AURORA_EXIT:
                out.type = Event::Type::Exit;
                break;
            case AURORA_WINDOW_RESIZED:
                out.type = Event::Type::Resized;
                out.a = static_cast<int32_t>(ev->windowSize.fb_width);
                out.b = static_cast<int32_t>(ev->windowSize.fb_height);
                break;
            case AURORA_SDL_EVENT:
                switch (ev->sdl.type) {
                    case SDL_EVENT_KEY_DOWN:
                        out.type = Event::Type::KeyDown;
                        out.a = static_cast<int32_t>(ev->sdl.key.scancode);
                        out.b = static_cast<int32_t>(ev->sdl.key.raw);
                        break;
                    case SDL_EVENT_KEY_UP:
                        out.type = Event::Type::KeyUp;
                        out.a = static_cast<int32_t>(ev->sdl.key.scancode);
                        out.b = static_cast<int32_t>(ev->sdl.key.raw);
                        break;
                    case SDL_EVENT_MOUSE_BUTTON_DOWN:
                        out.type = Event::Type::MouseDown;
                        out.a = MouseButton(ev->sdl.button.button);
                        break;
                    case SDL_EVENT_MOUSE_BUTTON_UP:
                        out.type = Event::Type::MouseUp;
                        out.a = MouseButton(ev->sdl.button.button);
                        break;
                    case SDL_EVENT_WINDOW_FOCUS_LOST:
                        out.type = Event::Type::FocusLost;
                        break;
                    case SDL_EVENT_WINDOW_FOCUS_GAINED:
                        out.type = Event::Type::FocusGained;
                        break;
                    case SDL_EVENT_MOUSE_MOTION: {
                        // SDL3 reports window coordinates; libultraship's sizes are pixels. Scale by
                        // the window's own pixel/coordinate ratio (on Windows with per-monitor DPI
                        // awareness the two are the same, whatever the display scale).
                        float d = 1.0f;
                        if (sWindow != nullptr) {
                            int lw = 0, lh = 0, pw = 0, ph = 0;
                            SDL_GetWindowSize(sWindow, &lw, &lh);
                            SDL_GetWindowSizeInPixels(sWindow, &pw, &ph);
                            if (lw > 0 && pw > 0) {
                                d = static_cast<float>(pw) / static_cast<float>(lw);
                            }
                        }
                        static int sMotionLogged = 0;
                        if (sMotionLogged < 3 && sLog != nullptr) {
                            ++sMotionLogged;
                            int lw = 0, lh = 0, pw = 0, ph = 0;
                            SDL_GetWindowSize(sWindow, &lw, &lh);
                            SDL_GetWindowSizeInPixels(sWindow, &pw, &ph);
                            char msg[160];
                            SDL_snprintf(msg, sizeof(msg), "mouse motion raw (%.1f, %.1f); window %dx%d, pixels %dx%d, density %.3f",
                                         ev->sdl.motion.x, ev->sdl.motion.y, lw, lh, pw, ph, SDL_GetWindowPixelDensity(sWindow));
                            sLog(LogLevel::Warning, "bridge", msg);
                        }
                        out.type = Event::Type::MouseMove;
                        out.a = static_cast<int32_t>(ev->sdl.motion.x * d);
                        out.b = static_cast<int32_t>(ev->sdl.motion.y * d);
                        break;
                    }
                    case SDL_EVENT_MOUSE_WHEEL:
                        out.type = Event::Type::MouseWheel;
                        out.a = static_cast<int32_t>(ev->sdl.wheel.x * 120.0f);
                        out.b = static_cast<int32_t>(ev->sdl.wheel.y * 120.0f);
                        break;
                    case SDL_EVENT_TEXT_INPUT: {
                        // UTF-8 -> one event per code point.
                        const auto* p = reinterpret_cast<const unsigned char*>(ev->sdl.text.text);
                        while (p != nullptr && *p != 0 && n < max) {
                            uint32_t cp = *p;
                            int extra = 0;
                            if (cp >= 0xF0) { cp &= 0x07; extra = 3; }
                            else if (cp >= 0xE0) { cp &= 0x0F; extra = 2; }
                            else if (cp >= 0xC0) { cp &= 0x1F; extra = 1; }
                            ++p;
                            for (int k = 0; k < extra && *p != 0; ++k, ++p) {
                                cp = (cp << 6) | (*p & 0x3F);
                            }
                            Event t{};
                            t.type = Event::Type::Text;
                            t.a = static_cast<int32_t>(cp);
                            events[n++] = t;
                        }
                        break;
                    }
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        if (out.type != Event::Type::None) {
            events[n++] = out;
        }
    }
    return n;
}

bool BeginFrame() {
    return aurora_begin_frame();
}

void EndFrame() {
    aurora_end_frame();
}

Backend GetBackend() {
    return sBackend;
}

void GetWindowSize(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    int w = 0, h = 0, x = 0, y = 0;
    if (sWindow != nullptr) {
        SDL_GetWindowSizeInPixels(sWindow, &w, &h);
        SDL_GetWindowPosition(sWindow, &x, &y);
    }
    if (width) *width = static_cast<uint32_t>(w);
    if (height) *height = static_cast<uint32_t>(h);
    if (posX) *posX = x;
    if (posY) *posY = y;
}

void SetWindowSize(uint32_t width, uint32_t height, int32_t posX, int32_t posY) {
    if (sWindow != nullptr) {
        SDL_SetWindowSize(sWindow, static_cast<int>(width), static_cast<int>(height));
        SDL_SetWindowPosition(sWindow, posX, posY);
    }
}

void SetWindowTitle(const char* title) {
    if (sWindow != nullptr) {
        SDL_SetWindowTitle(sWindow, title);
    }
}

void SetFullscreen(bool fullscreen) {
    if (sWindow != nullptr) {
        SDL_SetWindowFullscreen(sWindow, fullscreen);
    }
}

bool IsFullscreen() {
    return sWindow != nullptr && (SDL_GetWindowFlags(sWindow) & SDL_WINDOW_FULLSCREEN) != 0;
}

void SetCursorVisible(bool visible) {
    if (visible) {
        SDL_ShowCursor();
    } else {
        SDL_HideCursor();
    }
}

void GetMousePos(int32_t* x, int32_t* y) {
    float fx = 0, fy = 0;
    SDL_GetMouseState(&fx, &fy);
    if (x) *x = static_cast<int32_t>(fx);
    if (y) *y = static_cast<int32_t>(fy);
}

uint32_t GetRefreshRate() {
    if (sWindow == nullptr) {
        return 60;
    }
    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(sWindow));
    return mode != nullptr && mode->refresh_rate > 0 ? static_cast<uint32_t>(mode->refresh_rate) : 60;
}

double GetTime() {
    return static_cast<double>(SDL_GetPerformanceCounter()) / static_cast<double>(SDL_GetPerformanceFrequency());
}

namespace Gfx {
wgpu::Device Device() {
    return aurora::gfx::device();
}
wgpu::Queue Queue() {
    return aurora::gfx::queue();
}
aurora::gfx::RenderTargetLayout SceneLayout() {
    return aurora::gfx::scene_render_target_layout();
}
bool UsesReversedZ() {
    return aurora::gfx::uses_reversed_z();
}
aurora::gfx::DrawTypeId RegisterDrawType(const aurora::gfx::DrawTypeDescriptor& desc) {
    return aurora::gfx::register_draw_type(desc);
}
void UnregisterDrawType(aurora::gfx::DrawTypeId type) {
    aurora::gfx::unregister_draw_type(type);
}
bool PushCustomDraw(aurora::gfx::DrawTypeId type, const void* payload, size_t payloadSize) {
    return aurora::gfx::push_custom_draw(type, payload, payloadSize);
}
aurora::gfx::EncoderTaskId RegisterEncoderTaskType(const aurora::gfx::EncoderTaskDescriptor& desc) {
    return aurora::gfx::register_encoder_task_type(desc);
}
void UnregisterEncoderTaskType(aurora::gfx::EncoderTaskId type) {
    aurora::gfx::unregister_encoder_task_type(type);
}
bool PushEncoderTask(aurora::gfx::EncoderTaskId type, const void* payload, size_t payloadSize) {
    return aurora::gfx::push_encoder_task(type, payload, payloadSize);
}
aurora::gfx::Range PushVerts(const uint8_t* data, size_t length, size_t alignment) {
    return aurora::gfx::push_verts(data, length, alignment);
}
aurora::gfx::Range PushIndices(const uint8_t* data, size_t length, size_t alignment) {
    return aurora::gfx::push_indices(data, length, alignment);
}
aurora::gfx::Range PushUniform(const uint8_t* data, size_t length) {
    return aurora::gfx::push_uniform(data, length);
}
aurora::gfx::Range PushStorage(const uint8_t* data, size_t length) {
    return aurora::gfx::push_storage(data, length);
}
bool ResolvePass(const aurora::gfx::ResolveDesc& desc, aurora::gfx::ResolvedTargets& out) {
    return aurora::gfx::resolve_pass(desc, out);
}
uint32_t CurrentFrame() {
    return aurora::gfx::current_frame();
}
void Synchronize() {
    aurora::gfx::synchronize();
}
} // namespace Gfx

} // namespace AuroraBridge
