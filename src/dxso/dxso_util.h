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

  /* Apple port experiment (true two-eye stereo captures, include/native/dxvk_stereo.h).
   *
   * When LB_STEREO_EYE_UNITS is set to a non-zero value, every programmable D3D9 vertex shader gets
   *   pos.x -= k * (abs(pos.w - 1) > 1e-5)
   * appended after vs_main, with k read from one extra vec4 that sits right after the six clip planes in the
   * VS clip-plane uniform buffer (DxsoConstantBuffers::VSClipPlanes, binding 1 of the VS stage). That buffer
   * is already bound to every programmable and fixed-function VS, so no new binding or pipeline layout change
   * is needed, and the CPU can change k between any two draws with an ordinary buffer slice upload.
   * Read once per process; without the variable shaders are byte-for-byte what they were. */
  inline bool DxsoStereoClipShiftEnabled() {
    static const bool enabled = [] {
      const char* e = std::getenv("LB_STEREO_EYE_UNITS");
      return e && *e && std::atof(e) != 0.0;
    }();
    return enabled;
  }

  /* vec4 index of the stereo shift within the clip-plane buffer: right after the clip planes (6). */
  constexpr uint32_t DxsoStereoClipShiftIndex = 6;

}