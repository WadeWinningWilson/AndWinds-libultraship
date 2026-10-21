#ifdef ENABLE_AURORA

// WGSL for Fast3D combiners: the same prism template inputs as the OpenGL
// backend (gfx_opengl.cpp BuildFsShader/BuildVsShader), one module with both
// entry points. Differences from the GLSL template:
//  - depth clamp is emulated (Fast3D_Aurora_Backend_Gap_Map.md E5): the vertex
//    stage keeps z inside the clip volume and the fragment stage writes
//    frag_depth = clamp(z / w), plus the decal bias the hardware cannot add to a
//    written depth;
//  - the viewport is clamped to the target on the CPU and the difference is a
//    clip-space scale/offset (WebGPU requires viewports inside the target);
//  - textures are sampled with textureSampleLevel (no mips; no uniformity rules).

#include "fast/backends/gfx_aurora.h"

#include <string>

#include <prism/processor.h>

#include "fast/interpreter.h"

namespace Fast {

namespace {

#define AURORA_RAND_NOISE "((random(vec3f(floor(fragCoord.xy * u.noise_scale), f32(u.frame_count))) + 1.0) / 2.0)"

const char* ShaderItemToWgsl(uint32_t item, bool with_alpha, bool only_alpha, bool inputs_have_alpha, bool first_cycle,
                             bool hint_single_element) {
    if (!only_alpha) {
        switch (item) {
            case SHADER_0:
                return with_alpha ? "vec4f(0.0, 0.0, 0.0, 0.0)" : "vec3f(0.0, 0.0, 0.0)";
            case SHADER_1:
                return with_alpha ? "vec4f(1.0, 1.0, 1.0, 1.0)" : "vec3f(1.0, 1.0, 1.0)";
            case SHADER_INPUT_1:
                return with_alpha || !inputs_have_alpha ? "vInput1" : "vInput1.rgb";
            case SHADER_INPUT_2:
                return with_alpha || !inputs_have_alpha ? "vInput2" : "vInput2.rgb";
            case SHADER_INPUT_3:
                return with_alpha || !inputs_have_alpha ? "vInput3" : "vInput3.rgb";
            case SHADER_INPUT_4:
                return with_alpha || !inputs_have_alpha ? "vInput4" : "vInput4.rgb";
            case SHADER_TEXEL0:
                return first_cycle ? (with_alpha ? "texVal0" : "texVal0.rgb")
                                   : (with_alpha ? "texVal1" : "texVal1.rgb");
            case SHADER_TEXEL0A:
                return first_cycle ? (hint_single_element ? "texVal0.a"
                                                          : (with_alpha ? "vec4f(texVal0.a)" : "vec3f(texVal0.a)"))
                                   : (hint_single_element ? "texVal1.a"
                                                          : (with_alpha ? "vec4f(texVal1.a)" : "vec3f(texVal1.a)"));
            case SHADER_TEXEL1A:
                return first_cycle ? (hint_single_element ? "texVal1.a"
                                                          : (with_alpha ? "vec4f(texVal1.a)" : "vec3f(texVal1.a)"))
                                   : (hint_single_element ? "texVal0.a"
                                                          : (with_alpha ? "vec4f(texVal0.a)" : "vec3f(texVal0.a)"));
            case SHADER_TEXEL1:
                return first_cycle ? (with_alpha ? "texVal1" : "texVal1.rgb")
                                   : (with_alpha ? "texVal0" : "texVal0.rgb");
            case SHADER_COMBINED:
                return with_alpha ? "texel" : "texel.rgb";
            case SHADER_NOISE:
                return with_alpha ? "vec4f(" AURORA_RAND_NOISE ")" : "vec3f(" AURORA_RAND_NOISE ")";
        }
    } else {
        switch (item) {
            case SHADER_0:
                return "0.0";
            case SHADER_1:
                return "1.0";
            case SHADER_INPUT_1:
                return "vInput1.a";
            case SHADER_INPUT_2:
                return "vInput2.a";
            case SHADER_INPUT_3:
                return "vInput3.a";
            case SHADER_INPUT_4:
                return "vInput4.a";
            case SHADER_TEXEL0:
            case SHADER_TEXEL0A:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL1A:
            case SHADER_TEXEL1:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_COMBINED:
                return "texel.a";
            case SHADER_NOISE:
                return AURORA_RAND_NOISE;
        }
    }
    // Unknown items read as zero, so a new combiner input degrades instead of
    // producing invalid WGSL.
    return only_alpha ? "0.0" : (with_alpha ? "vec4f(0.0)" : "vec3f(0.0)");
}

bool GetBool(prism::ContextTypes* value) {
    return std::holds_alternative<int>(*value) && std::get<int>(*value) == 1;
}

prism::ContextTypes* AppendFormulaWgsl(prism::ContextTypes*, prism::ContextTypes* a_arg, prism::ContextTypes* a_single,
                                       prism::ContextTypes* a_mult, prism::ContextTypes* a_mix,
                                       prism::ContextTypes* a_with_alpha, prism::ContextTypes* a_only_alpha,
                                       prism::ContextTypes* a_alpha, prism::ContextTypes* a_first_cycle) {
    auto c = std::get<prism::MTDArray<int>>(*a_arg);
    const bool doSingle = GetBool(a_single);
    const bool doMultiply = GetBool(a_mult);
    const bool doMix = GetBool(a_mix);
    const bool withAlpha = GetBool(a_with_alpha);
    const bool onlyAlpha = GetBool(a_only_alpha);
    const bool optAlpha = GetBool(a_alpha);
    const bool firstCycle = GetBool(a_first_cycle);
    auto item = [&](int k, bool single) {
        return std::string(ShaderItemToWgsl(c.at(onlyAlpha, k), withAlpha, onlyAlpha, optAlpha, firstCycle, single));
    };
    std::string out;
    if (doSingle) {
        out = item(3, false);
    } else if (doMultiply) {
        out = item(0, false) + " * " + item(2, true);
    } else if (doMix) {
        out = "mix(" + item(1, false) + ", " + item(0, false) + ", " + item(2, true) + ")";
    } else {
        out = "(" + item(0, false) + " - " + item(1, false) + ") * " + item(2, true) + " + " + item(3, false);
    }
    return new prism::ContextTypes{ out };
}

int sVertexLocation = 0;
int sVaryingLocation = 0;

prism::ContextTypes* NextVertexLocation(prism::ContextTypes*, prism::ContextTypes*) {
    return new prism::ContextTypes{ std::to_string(sVertexLocation++) };
}

prism::ContextTypes* NextVaryingLocation(prism::ContextTypes*, prism::ContextTypes*) {
    return new prism::ContextTypes{ std::to_string(sVaryingLocation++) };
}

constexpr const char* kTemplate = R"(@prism(type='fragment', name='Fast3D WGSL Shader', version='1.0.0', description='Fast3D combiner on Aurora (WebGPU)', author='libultraship')
struct Uniforms {
    frame_count: i32,
    noise_scale: f32,
    prim_depth: f32,
    decal_slope: f32,
    decal_units: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    viewport: vec4f,
    tex_size: vec4f,
    tex_filter: vec4i,
};

@{AT}group(0) @{AT}binding(0) var<uniform> u: Uniforms;
@{AT}group(1) @{AT}binding(0) var uTex0: texture_2d<f32>;
@{AT}group(1) @{AT}binding(1) var uTex1: texture_2d<f32>;
@{AT}group(1) @{AT}binding(2) var uTexMask0: texture_2d<f32>;
@{AT}group(1) @{AT}binding(3) var uTexMask1: texture_2d<f32>;
@{AT}group(1) @{AT}binding(4) var uTexBlend0: texture_2d<f32>;
@{AT}group(1) @{AT}binding(5) var uTexBlend1: texture_2d<f32>;
@{AT}group(1) @{AT}binding(6) var uSampler0: sampler;
@{AT}group(1) @{AT}binding(7) var uSampler1: sampler;

struct VertexIn {
    @{AT}location(@{vloc(0)}) aVtxPos: vec4f,
@for(i in 0..2)
@if(o_textures[i])
    @{AT}location(@{vloc(0)}) aTexCoord@{i}: vec2f,
@for(j in 0..2)
@if(o_clamp[i][j])
@if(j == 0)
    @{AT}location(@{vloc(0)}) aTexClampS@{i}: f32,
@else
    @{AT}location(@{vloc(0)}) aTexClampT@{i}: f32,
@end
@end
@end
@end
@end
@if(o_fog)
    @{AT}location(@{vloc(0)}) aFog: vec4f,
@end
@if(o_grayscale)
    @{AT}location(@{vloc(0)}) aGrayscaleColor: vec4f,
@end
@for(i in 0..o_inputs)
@if(o_alpha)
    @{AT}location(@{vloc(0)}) aInput@{i + 1}: vec4f,
@else
    @{AT}location(@{vloc(0)}) aInput@{i + 1}: vec3f,
@end
@end
};

struct VSOut {
    @{AT}builtin(position) position: vec4f,
    @{AT}location(@{oloc(0)}) vDepthZW: vec2f,
@for(i in 0..2)
@if(o_textures[i])
    @{AT}location(@{oloc(0)}) vTexCoord@{i}: vec2f,
@for(j in 0..2)
@if(o_clamp[i][j])
@if(j == 0)
    @{AT}location(@{oloc(0)}) vTexClampS@{i}: f32,
@else
    @{AT}location(@{oloc(0)}) vTexClampT@{i}: f32,
@end
@end
@end
@end
@end
@if(o_fog)
    @{AT}location(@{oloc(0)}) vFog: vec4f,
@end
@if(o_grayscale)
    @{AT}location(@{oloc(0)}) vGrayscaleColor: vec4f,
@end
@for(i in 0..o_inputs)
@if(o_alpha)
    @{AT}location(@{oloc(0)}) vInput@{i + 1}: vec4f,
@else
    @{AT}location(@{oloc(0)}) vInput@{i + 1}: vec3f,
@end
@end
};

struct FSOut {
    @{AT}location(0) color: vec4f,
    @{AT}builtin(frag_depth) depth: f32,
};

@{AT}vertex
fn vs_main(vin: VertexIn) -> VSOut {
    var o: VSOut;
    let p = vec4f(vin.aVtxPos.xy * u.viewport.xy + u.viewport.zw * vin.aVtxPos.w, vin.aVtxPos.z, vin.aVtxPos.w);
    o.position = vec4f(p.x, p.y, 0.5 * p.w, p.w);
    o.vDepthZW = vec2f(p.z, p.w);
@for(i in 0..2)
@if(o_textures[i])
    o.vTexCoord@{i} = vin.aTexCoord@{i};
@for(j in 0..2)
@if(o_clamp[i][j])
@if(j == 0)
    o.vTexClampS@{i} = vin.aTexClampS@{i};
@else
    o.vTexClampT@{i} = vin.aTexClampT@{i};
@end
@end
@end
@end
@end
@if(o_fog)
    o.vFog = vin.aFog;
@end
@if(o_grayscale)
    o.vGrayscaleColor = vin.aGrayscaleColor;
@end
@for(i in 0..o_inputs)
    o.vInput@{i + 1} = vin.aInput@{i + 1};
@end
    return o;
}

fn wrap1(x: f32, lo: f32, hi: f32) -> f32 {
    return x - (hi - lo) * floor((x - lo) / (hi - lo));
}
fn wrap3(x: vec3f, lo: f32, hi: f32) -> vec3f {
    return x - (hi - lo) * floor((x - vec3f(lo)) / (hi - lo));
}
fn wrap4(x: vec4f, lo: f32, hi: f32) -> vec4f {
    return x - (hi - lo) * floor((x - vec4f(lo)) / (hi - lo));
}

fn random(value: vec3f) -> f32 {
    let r = dot(sin(value), vec3f(12.9898, 78.233, 37.719));
    return fract(sin(r) * 143758.5453);
}

fn fromLinear(c: vec4f) -> vec4f {
    let cutoff = c.rgb < vec3f(0.0031308);
    let higher = vec3f(1.055) * pow(c.rgb, vec3f(1.0 / 2.4)) - vec3f(0.055);
    let lower = c.rgb * vec3f(12.92);
    return vec4f(select(higher, lower, cutoff), c.a);
}

fn filter3point(t: texture_2d<f32>, s: sampler, uv: vec2f, texSize: vec2f) -> vec4f {
    var offset = fract(uv * texSize - vec2f(0.5));
    offset -= vec2f(step(1.0, offset.x + offset.y));
    let c0 = textureSampleLevel(t, s, uv - offset / texSize, 0.0);
    let c1 = textureSampleLevel(t, s, uv - vec2f(offset.x - sign(offset.x), offset.y) / texSize, 0.0);
    let c2 = textureSampleLevel(t, s, uv - vec2f(offset.x, offset.y - sign(offset.y)) / texSize, 0.0);
    return c0 + abs(offset.x) * (c1 - c0) + abs(offset.y) * (c2 - c0);
}

fn hookTexture2D(id: i32, t: texture_2d<f32>, s: sampler, uv: vec2f, texSize: vec2f) -> vec4f {
@if(o_three_point_filtering)
    if (u.tex_filter[id] == @{FILTER_THREE_POINT}) {
        return filter3point(t, s, uv, texSize);
    }
@end
    return textureSampleLevel(t, s, uv, 0.0);
}

@{AT}fragment
fn fs_main(fin: VSOut) -> FSOut {
    // Derivatives first, while control flow is still uniform.
    let depthLinear = fin.vDepthZW.x / fin.vDepthZW.y;
    let depthSlope = max(abs(dpdx(depthLinear)), abs(dpdy(depthLinear)));
    let fragCoord = fin.position;
@for(i in 0..2)
@if(o_textures[i])
    let vTexCoord@{i} = fin.vTexCoord@{i};
@for(j in 0..2)
@if(o_clamp[i][j])
@if(j == 0)
    let vTexClampS@{i} = fin.vTexClampS@{i};
@else
    let vTexClampT@{i} = fin.vTexClampT@{i};
@end
@end
@end
@end
@end
@if(o_fog)
    let vFog = fin.vFog;
@end
@if(o_grayscale)
    let vGrayscaleColor = fin.vGrayscaleColor;
@end
@for(i in 0..o_inputs)
    let vInput@{i + 1} = fin.vInput@{i + 1};
@end

@for(i in 0..2)
@if(o_textures[i])
@{s = o_clamp[i][0]}
@{t = o_clamp[i][1]}
    let texSize@{i} = vec2f(u.tex_size[@{i * 2}], u.tex_size[@{i * 2 + 1}]);
@if(!s && !t)
    let vTexCoordAdj@{i} = vTexCoord@{i};
@else
@if(s && t)
    let vTexCoordAdj@{i} = clamp(vTexCoord@{i}, vec2f(0.5) / texSize@{i}, vec2f(vTexClampS@{i}, vTexClampT@{i}));
@elseif(s)
    let vTexCoordAdj@{i} = vec2f(clamp(vTexCoord@{i}.x, 0.5 / texSize@{i}.x, vTexClampS@{i}), vTexCoord@{i}.y);
@else
    let vTexCoordAdj@{i} = vec2f(vTexCoord@{i}.x, clamp(vTexCoord@{i}.y, 0.5 / texSize@{i}.y, vTexClampT@{i}));
@end
@end
    var texVal@{i} = hookTexture2D(@{i}, uTex@{i}, uSampler@{i}, vTexCoordAdj@{i}, texSize@{i});
@if(o_masks[i])
    let maskSize@{i} = vec2f(textureDimensions(uTexMask@{i}));
    let maskVal@{i} = hookTexture2D(@{i}, uTexMask@{i}, uSampler@{i}, vTexCoordAdj@{i}, maskSize@{i});
@if(o_blend[i])
    let blendVal@{i} = hookTexture2D(@{i}, uTexBlend@{i}, uSampler@{i}, vTexCoordAdj@{i}, texSize@{i});
@else
    let blendVal@{i} = vec4f(0.0);
@end
    texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
@end
@end
@end

@if(o_alpha)
    var texel: vec4f;
@else
    var texel: vec3f;
@end

@if(o_2cyc)
@{f_range = 2}
@else
@{f_range = 1}
@end

@for(c in 0..f_range)
@if(c == 1)
@if(o_alpha)
@if(o_c[c][1][2] == SHADER_COMBINED)
    texel.a = wrap1(texel.a, -1.01, 1.01);
@else
    texel.a = wrap1(texel.a, -0.51, 1.51);
@end
@if(o_c[c][0][2] == SHADER_COMBINED)
    texel = vec4f(wrap3(texel.rgb, -1.01, 1.01), texel.a);
@else
    texel = vec4f(wrap3(texel.rgb, -0.51, 1.51), texel.a);
@end
@else
@if(o_c[c][0][2] == SHADER_COMBINED)
    texel = wrap3(texel, -1.01, 1.01);
@else
    texel = wrap3(texel, -0.51, 1.51);
@end
@end
@end
@if(!o_color_alpha_same[c] && o_alpha)
    texel = vec4f(@{append_formula(o_c[c], o_do_single[c][0], o_do_multiply[c][0], o_do_mix[c][0], false, false, true, c == 0)}, @{append_formula(o_c[c], o_do_single[c][1], o_do_multiply[c][1], o_do_mix[c][1], true, true, true, c == 0)});
@else
    texel = @{append_formula(o_c[c], o_do_single[c][0], o_do_multiply[c][0], o_do_mix[c][0], o_alpha, false, o_alpha, c == 0)};
@end
@end

@if(o_alpha)
    texel = clamp(wrap4(texel, -0.51, 1.51), vec4f(0.0), vec4f(1.0));
@else
    texel = clamp(wrap3(texel, -0.51, 1.51), vec3f(0.0), vec3f(1.0));
@end

@if(o_fog)
@if(o_alpha)
    texel = vec4f(mix(texel.rgb, vFog.rgb, vFog.a), texel.a);
@else
    texel = mix(texel, vFog.rgb, vFog.a);
@end
@end

@if(o_texture_edge && o_alpha)
    if (texel.a > 0.19) {
        texel.a = 1.0;
    } else {
        discard;
    }
@end

@if(o_alpha && o_noise)
    texel.a *= floor(clamp(random(vec3f(floor(fragCoord.xy * u.noise_scale), f32(u.frame_count))) + texel.a, 0.0, 1.0));
@end

@if(o_grayscale)
    let intensity = (texel.r + texel.g + texel.b) / 3.0;
    let newTexel = vGrayscaleColor.rgb * intensity;
@if(o_alpha)
    texel = vec4f(mix(texel.rgb, newTexel, vGrayscaleColor.a), texel.a);
@else
    texel = mix(texel, newTexel, vGrayscaleColor.a);
@end
@end

    var res: FSOut;
@if(o_alpha)
@if(o_alpha_threshold)
    if (texel.a < 8.0 / 256.0) {
        discard;
    }
@end
@if(o_invisible)
    texel.a = 0.0;
@end
    res.color = texel;
@else
    res.color = vec4f(texel, 1.0);
@end
@if(srgb_mode)
    res.color = fromLinear(res.color);
@end
@if(o_prim_depth)
    res.depth = u.prim_depth;
@else
    res.depth = clamp(clamp(depthLinear, 0.0, 1.0) + u.decal_slope * depthSlope + u.decal_units, 0.0, 1.0);
@end
    return res;
}
)";

} // namespace

std::string BuildAuroraShader(const ::CCFeatures& cc, bool threePointFiltering, bool srgbMode) {
    sVertexLocation = 0;
    sVaryingLocation = 0;
    prism::Processor processor;
    prism::ContextItems context = {
        { "o_c", M_ARRAY(cc.c, int, 2, 2, 4) },
        { "o_alpha", cc.opt_alpha },
        { "o_fog", cc.opt_fog },
        { "o_texture_edge", cc.opt_texture_edge },
        { "o_noise", cc.opt_noise },
        { "o_2cyc", cc.opt_2cyc },
        { "o_alpha_threshold", cc.opt_alpha_threshold },
        { "o_invisible", cc.opt_invisible },
        { "o_grayscale", cc.opt_grayscale },
        { "o_prim_depth", cc.opt_prim_depth },
        { "o_textures", M_ARRAY(cc.usedTextures, bool, 2) },
        { "o_masks", M_ARRAY(cc.used_masks, bool, 2) },
        { "o_blend", M_ARRAY(cc.used_blend, bool, 2) },
        { "o_clamp", M_ARRAY(cc.clamp, bool, 2, 2) },
        { "o_inputs", cc.numInputs },
        { "o_do_mix", M_ARRAY(cc.do_mix, bool, 2, 2) },
        { "o_do_single", M_ARRAY(cc.do_single, bool, 2, 2) },
        { "o_do_multiply", M_ARRAY(cc.do_multiply, bool, 2, 2) },
        { "o_color_alpha_same", M_ARRAY(cc.color_alpha_same, bool, 2) },
        { "FILTER_THREE_POINT", FILTER_THREE_POINT },
        { "FILTER_LINEAR", FILTER_LINEAR },
        { "FILTER_NONE", FILTER_NONE },
        { "srgb_mode", srgbMode },
        { "SHADER_0", SHADER_0 },
        { "SHADER_INPUT_1", SHADER_INPUT_1 },
        { "SHADER_INPUT_2", SHADER_INPUT_2 },
        { "SHADER_INPUT_3", SHADER_INPUT_3 },
        { "SHADER_INPUT_4", SHADER_INPUT_4 },
        { "SHADER_INPUT_5", SHADER_INPUT_5 },
        { "SHADER_INPUT_6", SHADER_INPUT_6 },
        { "SHADER_INPUT_7", SHADER_INPUT_7 },
        { "SHADER_TEXEL0", SHADER_TEXEL0 },
        { "SHADER_TEXEL0A", SHADER_TEXEL0A },
        { "SHADER_TEXEL1", SHADER_TEXEL1 },
        { "SHADER_TEXEL1A", SHADER_TEXEL1A },
        { "SHADER_1", SHADER_1 },
        { "SHADER_COMBINED", SHADER_COMBINED },
        { "SHADER_NOISE", SHADER_NOISE },
        { "o_three_point_filtering", threePointFiltering },
        { "append_formula", (InvokeFunc)AppendFormulaWgsl },
        { "vloc", (InvokeFunc)NextVertexLocation },
        { "oloc", (InvokeFunc)NextVaryingLocation },
        // WGSL attributes start with @, which prism reads as its own directives.
        { "AT", "@" },
    };
    processor.populate(context);
    processor.load(kTemplate);
    return processor.process();
}

} // namespace Fast

#endif
