#pragma once

#ifdef ENABLE_AURORA

#include <cstddef>
#include <cstdint>

#include <aurora/gfx.hpp>

namespace Fast {

// The Aurora that GfxRenderingAPIAurora renders through. Standalone, that is aurora_bridge.dll (the default table).
// Hosted, it is the host's own Aurora: a frontend that embeds libultraship in a host built on Aurora (ALBT&W's gfx
// service, svc/gfx.h, has the same calls) installs a table before the window is created, and libultraship then
// draws into the host's frame with the host's device.
//
// Every entry mirrors the aurora::gfx call of the same name. Two notes for a host table:
// - push*, register* and resolvePass are called where libultraship records its frame. A host whose service must be
//   called on its own thread defers them to that thread itself.
// - encoder-task descriptors' afterSubmit may be ignored (only the standalone self-test uses it).
struct AuroraGfxHost {
    wgpu::Device (*device)();
    wgpu::Queue (*queue)();
    aurora::gfx::RenderTargetLayout (*sceneLayout)();
    bool (*usesReversedZ)();
    aurora::gfx::DrawTypeId (*registerDrawType)(const aurora::gfx::DrawTypeDescriptor& desc);
    void (*unregisterDrawType)(aurora::gfx::DrawTypeId type);
    bool (*pushCustomDraw)(aurora::gfx::DrawTypeId type, const void* payload, size_t payloadSize);
    aurora::gfx::EncoderTaskId (*registerEncoderTaskType)(const aurora::gfx::EncoderTaskDescriptor& desc);
    void (*unregisterEncoderTaskType)(aurora::gfx::EncoderTaskId type);
    bool (*pushEncoderTask)(aurora::gfx::EncoderTaskId type, const void* payload, size_t payloadSize);
    aurora::gfx::Range (*pushVerts)(const uint8_t* data, size_t length, size_t alignment);
    aurora::gfx::Range (*pushIndices)(const uint8_t* data, size_t length, size_t alignment);
    aurora::gfx::Range (*pushUniform)(const uint8_t* data, size_t length);
    aurora::gfx::Range (*pushStorage)(const uint8_t* data, size_t length);
    bool (*resolvePass)(const aurora::gfx::ResolveDesc& desc, aurora::gfx::ResolvedTargets& out);
    // Drains the render worker so the caller may use queue() directly until it records again.
    void (*synchronize)();
    // The host camera's depth for framebuffer 0's composite (render worker); false = composite without depth.
    // proj_z / world_scale as LusAuroraWorldDepth (gfx_aurora_host_c.h).
    bool (*worldDepth)(float projZ[4], float* worldScale);
};

// The game's view planes for its depth (SetAuroraWorldDepthRange); false while unknown.
bool AuroraWorldDepthRange(float* zNear, float* zFar);

// Installs a host's table (copied), or restores the bridge with nullptr. Call before the Aurora backend initialises.
void SetAuroraGfxHost(const AuroraGfxHost* host);
// True while a host's table is installed.
bool AuroraGfxIsHosted();
// The table in use.
const AuroraGfxHost& AuroraGfx();

} // namespace Fast

#endif
