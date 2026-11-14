#pragma once

#include <cstdint>
#include <cstdlib>

#include "dxso_common.h"
#include "dxso_decoder.h"

namespace dxvk {

  enum class DxsoBindingType : uint32_t {
    ConstantBuffer,
    Image,
  };

  enum DxsoConstantBuffers : uint32_t {
    VSConstantBuffer = 0,
    VSClipPlanes     = 1,
    VSFixedFunction  = 2,
    VSVertexBlendData = 3,
    VSCount,

    PSConstantBuffer = 0,
    PSFixedFunction  = 1,
    PSShared         = 2,
    PSCount
  };

  constexpr uint32_t computeResourceSlotId(
        DxsoProgramType shaderStage,
        DxsoBindingType bindingType,
        uint32_t        bindingIndex) {
    const uint32_t stageOffset = 8 * uint32_t(shaderStage);

    if (bindingType == DxsoBindingType::ConstantBuffer)
      return bindingIndex + stageOffset;
    else // if (bindingType == DxsoBindingType::Image)
      return bindingIndex + stageOffset + (shaderStage == DxsoProgramType::PixelShader ? PSCount : VSCount);
  }

  // TODO: Intergrate into compute resource slot ID/refactor all of this?
  constexpr uint32_t getSWVPBufferSlot() {
    return 27; // From last pixel shader slot, above.
  }

  uint32_t RegisterLinkerSlot(DxsoSemantic semantic);

  /* Stage 0 of the true-stereo work (twcclegg/skyrim-apple#23), a timing experiment: LB_STEREO_DUP=<k>, k a
   * non-zero float, issues every main-scene draw (the passes the eye shift below applies to) twice on the DXVK
   * command thread, the second time with the opposite-sign shift, into the same render targets. It measures
   * what a second eye costs in draw submission; the picture is wrong (blended draws land twice).
   *
   * k is the clip-space x shift used while the runtime has set no eye offset (dxvk_stereo_set_clip_offset),
   * so an ordinary mono bench run works: the main draw gets pos.x -= k, the copy pos.x += k (perspective
   * vertices only). LB_STEREO_DUP=1 thus gives +/-1.0 clip units, an eye roughly one Skyrim unit (~1.4 cm)
   * to the side at the usual field of view (k = P00 * e). With an offset set by the runtime, that offset and
   * its negation are used instead. Turns the shader code on as LB_STEREO_EYE_UNITS does. 0 (off) when unset. */
  inline float DxsoStereoDupOffset() {
    static const float offset = [] {
      const char* e = std::getenv("LB_STEREO_DUP");
      return e && *e ? float(std::atof(e)) : 0.0f;
    }();
    return offset;
  }

  inline bool DxsoStereoDupEnabled() {
    return DxsoStereoDupOffset() != 0.0f;
  }

  /* Apple port experiment (true two-eye stereo captures, include/native/dxvk_stereo.h).
   *
   * When LB_STEREO_EYE_UNITS (or LB_STEREO_DUP, above) is set to a non-zero value, every programmable D3D9
   * vertex shader gets
   *   pos.x -= k * (abs(pos.w - 1) > 1e-5)
   * appended after vs_main, with k read from one extra vec4 that sits right after the six clip planes in the
   * VS clip-plane uniform buffer (DxsoConstantBuffers::VSClipPlanes, binding 1 of the VS stage). That buffer
   * is already bound to every programmable and fixed-function VS, so no new binding or pipeline layout change
   * is needed, and the CPU can change k between any two draws with an ordinary buffer slice upload.
   * Read once per process; without the variable shaders are byte-for-byte what they were. */
  inline bool DxsoStereoClipShiftEnabled() {
    static const bool enabled = [] {
      const char* e = std::getenv("LB_STEREO_EYE_UNITS");
      return (e && *e && std::atof(e) != 0.0) || DxsoStereoDupEnabled();
    }();
    return enabled;
  }

  /* vec4 index of the stereo shift within the clip-plane buffer: right after the clip planes (6). */
  constexpr uint32_t DxsoStereoClipShiftIndex = 6;

}