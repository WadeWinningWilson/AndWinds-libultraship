#pragma once

#include <stdint.h>

namespace Fast {

/**
 * Receives every finished frame just before it is presented (game rendering plus GUI overlays), for a frontend that
 * shows the game somewhere other than its own window, e.g. inside a host program.
 *
 * Called on the render thread. rgba is R8G8B8A8 (sRGB-encoded), top row first, valid only during the call.
 * Implemented by the DXGI (Direct3D 11) window backend; other backends never call it.
 */
typedef void (*PresentCaptureCallback)(const uint8_t* rgba, uint32_t width, uint32_t height, uint32_t pitchBytes,
                                       void* user);

/** Register (or with nullptr, remove) the capture callback. Set it before the window is created. */
void SetPresentCaptureCallback(PresentCaptureCallback callback, void* user);

} // namespace Fast
