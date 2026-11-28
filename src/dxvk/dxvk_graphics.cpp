#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "../util/util_time.h"

#include "dxvk_device.h"
#include "dxvk_graphics.h"
#include "dxvk_pipemanager.h"
#include "dxvk_spec_const.h"
#include "dxvk_state_cache.h"

// Apple port, LB_PIPE_STAGES: MoltenVK's performance statistics, in the layout of the MoltenVK built against
#if defined(__APPLE__) && defined(__has_include)
#  if __has_include(<MoltenVK/mvk_private_api.h>)
#    include <MoltenVK/mvk_private_api.h>
#    define DXVK_HAS_MVK_STATS 1
#  endif
#endif

namespace dxvk {

  DxvkGraphicsPipeline::DxvkGraphicsPipeline(
          DxvkPipelineManager*        pipeMgr,
          DxvkGraphicsPipelineShaders shaders)
  : m_vkd(pipeMgr->m_device->vkd()), m_pipeMgr(pipeMgr),
    m_shaders(std::move(shaders)) {
    if (m_shaders.vs  != nullptr) m_shaders.vs ->defineResourceSlots(m_slotMapping);
    if (m_shaders.tcs != nullptr) m_shaders.tcs->defineResourceSlots(m_slotMapping);
    if (m_shaders.tes != nullptr) m_shaders.tes->defineResourceSlots(m_slotMapping);
    if (m_shaders.gs  != nullptr) m_shaders.gs ->defineResourceSlots(m_slotMapping);
    if (m_shaders.fs  != nullptr) m_shaders.fs ->defineResourceSlots(m_slotMapping);
    
    m_slotMapping.makeDescriptorsDynamic(
      pipeMgr->m_device->options().maxNumDynamicUniformBuffers,
      pipeMgr->m_device->options().maxNumDynamicStorageBuffers);
    
    m_layout = new DxvkPipelineLayout(m_vkd,
      m_slotMapping, VK_PIPELINE_BIND_POINT_GRAPHICS);
    
    m_vsIn  = m_shaders.vs != nullptr ? m_shaders.vs->interfaceSlots().inputSlots  : 0;
    m_fsOut = m_shaders.fs != nullptr ? m_shaders.fs->interfaceSlots().outputSlots : 0;

    if (m_shaders.gs != nullptr && m_shaders.gs->flags().test(DxvkShaderFlag::HasTransformFeedback))
      m_flags.set(DxvkGraphicsPipelineFlag::HasTransformFeedback);
    
    if (m_layout->getStorageDescriptorStages())
      m_flags.set(DxvkGraphicsPipelineFlag::HasStorageDescriptors);
    
    m_common.msSampleShadingEnable = m_shaders.fs != nullptr && m_shaders.fs->flags().test(DxvkShaderFlag::HasSampleRateShading);
    m_common.msSampleShadingFactor = 1.0f;
  }
  
  
  DxvkGraphicsPipeline::~DxvkGraphicsPipeline() {
    for (const auto& instance : m_pipelines)
      this->destroyPipeline(instance.pipeline());
  }
  
  
  Rc<DxvkShader> DxvkGraphicsPipeline::getShader(
          VkShaderStageFlagBits             stage) const {
    switch (stage) {
      case VK_SHADER_STAGE_VERTEX_BIT:                  return m_shaders.vs;
      case VK_SHADER_STAGE_GEOMETRY_BIT:                return m_shaders.gs;
      case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:    return m_shaders.tcs;
      case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return m_shaders.tes;
      case VK_SHADER_STAGE_FRAGMENT_BIT:                return m_shaders.fs;
      default:
        return nullptr;
    }
  }


  /* Apple port, LB_PIPE_LOG=1: one stderr line per graphics pipeline compiled, with how long it took and whether
   * a draw waited for it ("at draw", on the CS thread) or the state cache's workers compiled it ahead. Lines land
   * between the frame-time lines of the frame they belong to, so hitches can be matched to compiles. An at-draw
   * time includes the wait for the shader set's lock, which is also given on its own; a draw that waited 1 ms or
   * more for that lock also gets a "pipe lock wait at draw" line, even when a compiler thread had just compiled
   * the state it needed. */
  static bool pipeLogEnabled() {
    return dxvkPipeLogLevel() >= 1;
  }

  static double pipeNowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  /* Apple port, LB_PIPE_STAGES=1 (diagnostic, off by default): where one graphics pipeline compile's time goes.
   * One stderr line per pipeline compile: DXVK's own work (shader modules: SPIR-V patching plus
   * vkCreateShaderModule), the whole vkCreateGraphicsPipelines call, and MoltenVK's own activity counters
   * (MVK_CONFIG_PERFORMANCE_TRACKING=1) diffed across the shader modules and the call: shader hashing, SPIR-V to
   * MSL, MSL to MTLLibrary, shader library lookup in the VkPipelineCache (recorded only on a hit, and including
   * the wait for that cache's lock), MTLFunction retrieval and specialization, render pipeline state. The line
   * ends with the pipeline's LB_PIPE_LOG=2 key. MoltenVK's counters are per device, and DXVK's meta and compute
   * pipelines and MoltenVK's internal ones feed them too, so a line says "solo" only when no other graphics
   * pipeline compile overlapped it and its counts fit one pipeline with its stages; otherwise "overlap". */
  static bool pipeStagesEnabled() {
    static const bool on = env::getEnvVar("LB_PIPE_STAGES") == "1";
    return on;
  }

#ifdef DXVK_HAS_MVK_STATS
  // MoltenVK registers vkGetPerformanceStatisticsMVK as an instance-level entry point
  static PFN_vkGetPerformanceStatisticsMVK mvkGetPerf(const DxvkDevice* device) {
    static PFN_vkGetPerformanceStatisticsMVK fn = reinterpret_cast<PFN_vkGetPerformanceStatisticsMVK>(
      device->adapter()->vki()->sym("vkGetPerformanceStatisticsMVK"));
    return fn;
  }

  // False when MoltenVK gives no statistics, or not in the layout this was built against (VK_INCOMPLETE)
  static bool mvkReadStats(const DxvkDevice* device, VkDevice vkDevice, MVKPerformanceStatistics* stats) {
    auto fn = mvkGetPerf(device);
    if (!fn)
      return false;

    size_t size = sizeof(*stats);  // MoltenVK writes back the size it copied
    VkResult vr = fn(vkDevice, stats, &size);

    if (vr != VK_SUCCESS) {
      static std::atomic<bool> warned = { false };
      if (!warned.exchange(true))
        std::fprintf(stderr, "pipe stages: MoltenVK statistics unavailable (VkResult %d), lines say mvk n/a\n", int(vr));
      return false;
    }

    return true;
  }

  static double mvkTotal(const MVKPerformanceTracker& t) { return t.average * double(t.count); }
#endif

  // Graphics pipeline compiles started (high 32 bits) and in flight (low 32 bits), in one atomic so that a
  // compile reads both at the same instant
  static std::atomic<uint64_t> g_stageCounts = { 0u };
  static constexpr uint64_t StageStart = (uint64_t(1) << 32) | 1u;

  // Ends a compile's in-flight count however createPipeline leaves (createShaderModule can throw)
  struct StageInflightGuard {
    bool active = false;

    uint64_t end() {
      active = false;
      return g_stageCounts.fetch_sub(1u);
    }

    ~StageInflightGuard() {
      if (active)
        g_stageCounts.fetch_sub(1u);
    }
  };

  static thread_local bool t_stageAtDraw = false;


  bool dxvkPipelineNormalizeEnabled() {
    static const bool on = env::getEnvVar("DXVK_PIPELINE_NORMALIZE") != "0";
    return on;
  }


  void dxvkNormalizePipelineState(
          DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPassFormat&          format,
          uint32_t                       fsOutMask) {
    // Colour attachments: no blend state where the fragment shader writes nothing or the pass has no attachment
    // (createPipeline then writes nothing there), no factors or ops where blending is off. The swizzle only where
    // the shader writes nothing: for a written output it is a spec constant applied to the value, and output 0's
    // alpha still feeds alpha-to-coverage when the pass has no attachment there.
    for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
      const bool written = fsOutMask & (1u << i);

      if (!written)
        std::memset(&state.omSwizzle[i], 0, sizeof(state.omSwizzle[i]));

      if (!written || format.color[i].format == VK_FORMAT_UNDEFINED) {
        std::memset(&state.omBlend[i], 0, sizeof(state.omBlend[i]));
      } else if (!state.omBlend[i].blendEnable()) {
        state.omBlend[i] = DxvkOmAttachmentBlend(VK_FALSE,
          VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
          VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
          state.omBlend[i].colorWriteMask());
      }
    }

    // Depth and stencil: nothing without a depth-stencil attachment; with the depth test off Vulkan neither
    // compares nor writes depth; stencil ops only matter with the stencil test on
    if (format.depth.format == VK_FORMAT_UNDEFINED) {
      state.ds = DxvkDsInfo(VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE, VK_COMPARE_OP_NEVER);
    } else if (!state.ds.enableDepthTest()) {
      state.ds = DxvkDsInfo(VK_FALSE, VK_FALSE,
        state.ds.enableDepthBoundsTest(), state.ds.enableStencilTest(), VK_COMPARE_OP_NEVER);
    }

    if (!state.ds.enableStencilTest()) {
      std::memset(&state.dsFront, 0, sizeof(state.dsFront));
      std::memset(&state.dsBack,  0, sizeof(state.dsBack));
    }

    if (!state.om.enableLogicOp())
      state.om = DxvkOmInfo(VK_FALSE, VkLogicOp(0));
  }


  const DxvkGraphicsPipelineStateInfo& DxvkGraphicsPipeline::normalizeState(
    const DxvkGraphicsPipelineStateInfo&          state,
    const DxvkRenderPass*                         renderPass,
          std::optional<DxvkGraphicsPipelineStateInfo>& scratch) const {
    if (!dxvkPipelineNormalizeEnabled())
      return state;

    scratch.emplace(state);
    dxvkNormalizePipelineState(*scratch, renderPass->format(), m_fsOut);
    return *scratch;
  }


  VkPipeline DxvkGraphicsPipeline::getPipelineHandle(
    const DxvkGraphicsPipelineStateInfo& stateIn,
    const DxvkRenderPass*                renderPass,
          DxvkGraphicsPipelineDynamicState* dynamicState) {
    // Apple port: pipelines are looked up, compiled and cached by their normalized state
    // (DXVK_PIPELINE_NORMALIZE), so draws that differ only in state Vulkan ignores share one
    std::optional<DxvkGraphicsPipelineStateInfo> scratch;
    const DxvkGraphicsPipelineStateInfo& state = this->normalizeState(stateIn, renderPass, scratch);

    // The pipeline's dynamic states follow the state it is built with: normalization can turn off
    // the stencil test, the depth bounds test or constant-factor blending that the caller's state has
    if (dynamicState != nullptr) {
      dynamicState->blendConstants = state.useDynamicBlendConstants();
      dynamicState->depthBias      = state.useDynamicDepthBias();
      dynamicState->depthBounds    = state.useDynamicDepthBounds();
      dynamicState->stencilRef     = state.useDynamicStencilRef();
    }

    DxvkGraphicsPipelineInstance* instance = this->findInstance(state, renderPass);

    if (unlikely(!instance)) {
      // Exit early if the state vector is invalid
      if (!this->validatePipelineState(state, true))
        return VK_NULL_HANDLE;

      // Prevent other threads from adding new instances and check again. LB_PIPE_LOG times the draw from
      // here, so the time includes waiting for a compiler thread that holds this shader set's lock.
      const double t0 = pipeLogEnabled() ? pipeNowMs() : 0.0;
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      instance = this->findInstance(state, renderPass);
      const double lockWaitMs = pipeLogEnabled() ? pipeNowMs() - t0 : 0.0;

      // LB_PIPE_LOG: a compiler thread held the lock, compiling this very state (found) or another one
      if (pipeLogEnabled() && lockWaitMs >= 1.0)
        std::fprintf(stderr, "pipe lock wait at draw: %.1f ms, %s\n", lockWaitMs, instance ? "found" : "compiling");

      if (!instance) {
        // Keep pipeline object locked, at worst we're going to stall
        // a state cache worker and the current thread needs priority.
        t_stageAtDraw = true;
        instance = this->createInstance(state, renderPass);
        t_stageAtDraw = false;
        if (pipeLogEnabled()) {
          const double ms = pipeNowMs() - t0;
          std::fprintf(stderr, "pipe compiled at draw: %.1f ms (lock wait %.1f ms) %s\n", ms, lockWaitMs,
            this->describeCacheMiss(state, renderPass->format()).c_str());
        }
        bool queued = this->writePipelineStateToCache(state, renderPass->format());

        // LB_PIPE_LOG=2: the pipeline's key, the closest cache entry field by field, and whether it was queued
        // for writing to the cache file (it is not when an identical entry was read from the file)
        if (dxvkPipeLogLevel() >= 2 && m_pipeMgr->m_stateCache != nullptr) {
          size_t instances = 0;
          for (const auto& i : m_pipelines) { (void) i; instances += 1; }

          std::fprintf(stderr, "%s; %s; instances %zu\n",
            m_pipeMgr->m_stateCache->describeMissFields(getStateCacheKey(), state, renderPass->format(), m_fsOut).c_str(),
            queued ? "write queued" : "write skipped", instances);
        }
      }
    }

    return instance->pipeline();
  }


  void DxvkGraphicsPipeline::compilePipeline(
    const DxvkGraphicsPipelineStateInfo& stateIn,
    const DxvkRenderPass*                renderPass) {
    // Cache entries written before normalization (or by a build without it) compile in normalized form too
    std::optional<DxvkGraphicsPipelineStateInfo> scratch;
    const DxvkGraphicsPipelineStateInfo& state = this->normalizeState(stateIn, renderPass, scratch);

    // Exit early if the state vector is invalid
    if (!this->validatePipelineState(state, false))
      return;

    // Keep the object locked while compiling a pipeline since compiling
    // similar pipelines concurrently is fragile on some drivers
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (!this->findInstance(state, renderPass)) {
      const double t0 = pipeLogEnabled() ? pipeNowMs() : 0.0;
      this->createInstance(state, renderPass);
      if (pipeLogEnabled())
        std::fprintf(stderr, "pipe compiled ahead: %.1f ms\n", pipeNowMs() - t0);
    }
  }


  DxvkGraphicsPipelineInstance* DxvkGraphicsPipeline::createInstance(
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPass*                renderPass) {
    VkPipeline pipeline = this->createPipeline(state, renderPass);

    m_pipeMgr->m_numGraphicsPipelines += 1;
    return &(*m_pipelines.emplace(state, renderPass, pipeline));
  }
  
  
  DxvkGraphicsPipelineInstance* DxvkGraphicsPipeline::findInstance(
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPass*                renderPass) {
    for (auto& instance : m_pipelines) {
      if (instance.isCompatible(state, renderPass))
        return &instance;
    }
    
    return nullptr;
  }
  
  
  VkPipeline DxvkGraphicsPipeline::createPipeline(
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPass*                renderPass) const {
    if (Logger::logLevel() <= LogLevel::Debug) {
      Logger::debug("Compiling graphics pipeline...");
      this->logPipelineState(LogLevel::Debug, state);
    }

    // Render pass format and image layouts
    DxvkRenderPassFormat passFormat = renderPass->format();
    
    // Set up dynamic states as needed
    std::array<VkDynamicState, 6> dynamicStates;
    uint32_t                      dynamicStateCount = 0;
    
    dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
    dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;

    if (state.useDynamicDepthBias())
      dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_BIAS;
    
    if (state.useDynamicDepthBounds())
      dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_BOUNDS;
    
    if (state.useDynamicBlendConstants())
      dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
    
    if (state.useDynamicStencilRef())
      dynamicStates[dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;

    // Figure out the actual sample count to use
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;

    if (state.ms.sampleCount())
      sampleCount = VkSampleCountFlagBits(state.ms.sampleCount());
    else if (state.rs.sampleCount())
      sampleCount = VkSampleCountFlagBits(state.rs.sampleCount());
    
    // Set up some specialization constants
    DxvkSpecConstants specData;
    specData.set(uint32_t(DxvkSpecConstantId::RasterizerSampleCount), sampleCount, VK_SAMPLE_COUNT_1_BIT);
    
    for (uint32_t i = 0; i < m_layout->bindingCount(); i++)
      specData.set(i, state.bsBindingMask.test(i), true);
    
    for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
      if ((m_fsOut & (1 << i)) != 0) {
        specData.set(uint32_t(DxvkSpecConstantId::ColorComponentMappings) + i,
          state.omSwizzle[i].rIndex() << 0 | state.omSwizzle[i].gIndex() << 4 |
          state.omSwizzle[i].bIndex() << 8 | state.omSwizzle[i].aIndex() << 12, 0x3210u);
      }
    }

    for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
      specData.set(getSpecId(i), state.sc.specConstants[i], 0u);
    
    VkSpecializationInfo specInfo = specData.getSpecInfo();
    
    const bool stages = pipeStagesEnabled();
    StageInflightGuard stageGuard;
    uint64_t stageCounts0 = 0;
    double ts0 = 0.0, ts1 = 0.0;
#ifdef DXVK_HAS_MVK_STATS
    MVKPerformanceStatistics mvk0 = { }, mvk1 = { };
    bool mvkOk = false;
#endif

    if (stages) {
      stageCounts0 = g_stageCounts.fetch_add(StageStart);
      stageGuard.active = true;
#ifdef DXVK_HAS_MVK_STATS
      mvkOk = mvkReadStats(m_pipeMgr->m_device, m_vkd->device(), &mvk0);
#endif
      ts0 = pipeNowMs();
    }

    auto vsm  = createShaderModule(m_shaders.vs,  state, passFormat.viewMask);
    auto tcsm = createShaderModule(m_shaders.tcs, state, passFormat.viewMask);
    auto tesm = createShaderModule(m_shaders.tes, state, passFormat.viewMask);
    auto gsm  = createShaderModule(m_shaders.gs,  state, passFormat.viewMask);
    auto fsm  = createShaderModule(m_shaders.fs,  state, passFormat.viewMask);

    if (stages)
      ts1 = pipeNowMs();

    std::vector<VkPipelineShaderStageCreateInfo> stageInfos;
    if (vsm)  stageInfos.push_back(vsm.stageInfo(&specInfo));
    if (tcsm) stageInfos.push_back(tcsm.stageInfo(&specInfo));
    if (tesm) stageInfos.push_back(tesm.stageInfo(&specInfo));
    if (gsm)  stageInfos.push_back(gsm.stageInfo(&specInfo));
    if (fsm)  stageInfos.push_back(fsm.stageInfo(&specInfo));

    // Fix up color write masks using the component mappings
    std::array<VkPipelineColorBlendAttachmentState, MaxNumRenderTargets> omBlendAttachments;

    const VkColorComponentFlags fullMask
      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
      | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
      omBlendAttachments[i] = state.omBlend[i].state();

      if (omBlendAttachments[i].colorWriteMask != fullMask) {
        omBlendAttachments[i].colorWriteMask = util::remapComponentMask(
          state.omBlend[i].colorWriteMask(), state.omSwizzle[i].mapping());
      }
      
      if ((m_fsOut & (1 << i)) == 0)
        omBlendAttachments[i].colorWriteMask = 0;
    }

    // Generate per-instance attribute divisors
    std::array<VkVertexInputBindingDivisorDescriptionEXT, MaxNumVertexBindings> viDivisorDesc;
    uint32_t                                                                    viDivisorCount = 0;

    for (uint32_t i = 0; i < state.il.bindingCount(); i++) {
      if (state.ilBindings[i].inputRate() == VK_VERTEX_INPUT_RATE_INSTANCE
       && state.ilBindings[i].divisor()   != 1) {
        const uint32_t id = viDivisorCount++;
        
        viDivisorDesc[id].binding = i; /* see below */
        viDivisorDesc[id].divisor = state.ilBindings[i].divisor();
      }
    }

    int32_t rasterizedStream = m_shaders.gs != nullptr
      ? m_shaders.gs->shaderOptions().rasterizedStream
      : 0;
    
    // Compact vertex bindings so that we can more easily update vertex buffers
    std::array<VkVertexInputAttributeDescription, MaxNumVertexAttributes> viAttribs;
    std::array<VkVertexInputBindingDescription,   MaxNumVertexBindings>   viBindings;
    std::array<uint32_t,                          MaxNumVertexBindings>   viBindingMap = { };

    for (uint32_t i = 0; i < state.il.bindingCount(); i++) {
      viBindings[i] = state.ilBindings[i].description();
      viBindings[i].binding = i;
      viBindingMap[state.ilBindings[i].binding()] = i;
    }

    for (uint32_t i = 0; i < state.il.attributeCount(); i++) {
      viAttribs[i] = state.ilAttributes[i].description();
      viAttribs[i].binding = viBindingMap[state.ilAttributes[i].binding()];
    }

    VkPipelineVertexInputDivisorStateCreateInfoEXT viDivisorInfo;
    viDivisorInfo.sType                     = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT;
    viDivisorInfo.pNext                     = nullptr;
    viDivisorInfo.vertexBindingDivisorCount = viDivisorCount;
    viDivisorInfo.pVertexBindingDivisors    = viDivisorDesc.data();
    
    VkPipelineVertexInputStateCreateInfo viInfo;
    viInfo.sType                            = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viInfo.pNext                            = &viDivisorInfo;
    viInfo.flags                            = 0;
    viInfo.vertexBindingDescriptionCount    = state.il.bindingCount();
    viInfo.pVertexBindingDescriptions       = viBindings.data();
    viInfo.vertexAttributeDescriptionCount  = state.il.attributeCount();
    viInfo.pVertexAttributeDescriptions     = viAttribs.data();
    
    if (viDivisorCount == 0)
      viInfo.pNext = viDivisorInfo.pNext;
    
    // TODO remove this once the extension is widely supported
    if (!m_pipeMgr->m_device->features().extVertexAttributeDivisor.vertexAttributeInstanceRateDivisor)
      viInfo.pNext = viDivisorInfo.pNext;
    
    VkPipelineInputAssemblyStateCreateInfo iaInfo;
    iaInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaInfo.pNext                  = nullptr;
    iaInfo.flags                  = 0;
    iaInfo.topology               = state.ia.primitiveTopology();
    iaInfo.primitiveRestartEnable = state.ia.primitiveRestart();
    
    VkPipelineTessellationStateCreateInfo tsInfo;
    tsInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tsInfo.pNext                  = nullptr;
    tsInfo.flags                  = 0;
    tsInfo.patchControlPoints     = state.ia.patchVertexCount();
    
    VkPipelineViewportStateCreateInfo vpInfo;
    vpInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpInfo.pNext                  = nullptr;
    vpInfo.flags                  = 0;
    vpInfo.viewportCount          = state.rs.viewportCount();
    vpInfo.pViewports             = nullptr;
    vpInfo.scissorCount           = state.rs.viewportCount();
    vpInfo.pScissors              = nullptr;
    
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeInfo;
    conservativeInfo.sType        = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
    conservativeInfo.pNext        = nullptr;
    conservativeInfo.flags        = 0;
    conservativeInfo.conservativeRasterizationMode = state.rs.conservativeMode();
    conservativeInfo.extraPrimitiveOverestimationSize = 0.0f;

    VkPipelineRasterizationStateStreamCreateInfoEXT xfbStreamInfo;
    xfbStreamInfo.sType           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT;
    xfbStreamInfo.pNext           = nullptr;
    xfbStreamInfo.flags           = 0;
    xfbStreamInfo.rasterizationStream = uint32_t(rasterizedStream);

    VkPipelineRasterizationDepthClipStateCreateInfoEXT rsDepthClipInfo;
    rsDepthClipInfo.sType         = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT;
    rsDepthClipInfo.pNext         = nullptr;
    rsDepthClipInfo.flags         = 0;
    rsDepthClipInfo.depthClipEnable = state.rs.depthClipEnable();

    VkPipelineRasterizationStateCreateInfo rsInfo;
    rsInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsInfo.pNext                  = nullptr;
    rsInfo.flags                  = 0;
    rsInfo.depthClampEnable       = m_pipeMgr->m_device->features().core.features.depthClamp;
    rsInfo.rasterizerDiscardEnable = rasterizedStream < 0;
    rsInfo.polygonMode            = state.rs.polygonMode();
    rsInfo.cullMode               = state.rs.cullMode();
    rsInfo.frontFace              = state.rs.frontFace();
    rsInfo.depthBiasEnable        = state.rs.depthBiasEnable();
    rsInfo.depthBiasConstantFactor= 0.0f;
    rsInfo.depthBiasClamp         = 0.0f;
    rsInfo.depthBiasSlopeFactor   = 0.0f;
    rsInfo.lineWidth              = 1.0f;
    
    if (rasterizedStream > 0)
      xfbStreamInfo.pNext = std::exchange(rsInfo.pNext, &xfbStreamInfo);

    if (conservativeInfo.conservativeRasterizationMode != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT)
      conservativeInfo.pNext = std::exchange(rsInfo.pNext, &conservativeInfo);

    if (m_pipeMgr->m_device->features().extDepthClipEnable.depthClipEnable)
      rsDepthClipInfo.pNext = std::exchange(rsInfo.pNext, &rsDepthClipInfo);
    else
      rsInfo.depthClampEnable = !state.rs.depthClipEnable()
        && m_pipeMgr->m_device->features().core.features.depthClamp;

    uint32_t sampleMask = state.ms.sampleMask();

    VkPipelineMultisampleStateCreateInfo msInfo;
    msInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msInfo.pNext                  = nullptr;
    msInfo.flags                  = 0;
    msInfo.rasterizationSamples   = sampleCount;
    msInfo.sampleShadingEnable    = m_common.msSampleShadingEnable;
    msInfo.minSampleShading       = m_common.msSampleShadingFactor;
    msInfo.pSampleMask            = &sampleMask;
    msInfo.alphaToCoverageEnable  = state.ms.enableAlphaToCoverage();
    msInfo.alphaToOneEnable       = VK_FALSE;
    
    VkPipelineDepthStencilStateCreateInfo dsInfo;
    dsInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsInfo.pNext                  = nullptr;
    dsInfo.flags                  = 0;
    dsInfo.depthTestEnable        = state.ds.enableDepthTest();
    dsInfo.depthWriteEnable       = state.ds.enableDepthWrite() && !util::isDepthReadOnlyLayout(passFormat.depth.layout);
    dsInfo.depthCompareOp         = state.ds.depthCompareOp();
    dsInfo.depthBoundsTestEnable  = state.ds.enableDepthBoundsTest();
    dsInfo.stencilTestEnable      = state.ds.enableStencilTest();
    dsInfo.front                  = state.dsFront.state();
    dsInfo.back                   = state.dsBack.state();
    dsInfo.minDepthBounds         = 0.0f;
    dsInfo.maxDepthBounds         = 1.0f;
    
    VkPipelineColorBlendStateCreateInfo cbInfo;
    cbInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbInfo.pNext                  = nullptr;
    cbInfo.flags                  = 0;
    cbInfo.logicOpEnable          = state.om.enableLogicOp();
    cbInfo.logicOp                = state.om.logicOp();
    cbInfo.attachmentCount        = DxvkLimits::MaxNumRenderTargets;
    cbInfo.pAttachments           = omBlendAttachments.data();
    
    for (uint32_t i = 0; i < 4; i++)
      cbInfo.blendConstants[i] = 0.0f;
    
    VkPipelineDynamicStateCreateInfo dyInfo;
    dyInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyInfo.pNext                  = nullptr;
    dyInfo.flags                  = 0;
    dyInfo.dynamicStateCount      = dynamicStateCount;
    dyInfo.pDynamicStates         = dynamicStates.data();
    
    VkGraphicsPipelineCreateInfo info;
    info.sType                    = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext                    = nullptr;
    info.flags                    = 0;
    info.stageCount               = stageInfos.size();
    info.pStages                  = stageInfos.data();
    info.pVertexInputState        = &viInfo;
    info.pInputAssemblyState      = &iaInfo;
    info.pTessellationState       = &tsInfo;
    info.pViewportState           = &vpInfo;
    info.pRasterizationState      = &rsInfo;
    info.pMultisampleState        = &msInfo;
    info.pDepthStencilState       = &dsInfo;
    info.pColorBlendState         = &cbInfo;
    info.pDynamicState            = &dyInfo;
    info.layout                   = m_layout->pipelineLayout();
    info.renderPass               = renderPass->getDefaultHandle();
    info.subpass                  = 0;
    info.basePipelineHandle       = VK_NULL_HANDLE;
    info.basePipelineIndex        = -1;
    
    if (tsInfo.patchControlPoints == 0)
      info.pTessellationState = nullptr;
    
    // Time pipeline compilation for debugging purposes
    dxvk::high_resolution_clock::time_point t0, t1;

    if (Logger::logLevel() <= LogLevel::Debug)
      t0 = dxvk::high_resolution_clock::now();
    
    VkPipeline pipeline = VK_NULL_HANDLE;
    const double ts2 = stages ? pipeNowMs() : 0.0;
    VkResult vr = m_vkd->vkCreateGraphicsPipelines(m_vkd->device(),
      m_pipeMgr->m_cache->handle(), 1, &info, nullptr, &pipeline);

    if (stages) {
      const double ts3 = pipeNowMs();
#ifdef DXVK_HAS_MVK_STATS
      mvkOk = mvkOk && mvkReadStats(m_pipeMgr->m_device, m_vkd->device(), &mvk1);
#endif
      const uint64_t stageCounts1 = stageGuard.end();

      // No other graphics pipeline compile was in flight when this one started, and none started since
      bool solo = uint32_t(stageCounts0) == 0 && (stageCounts1 >> 32) == (stageCounts0 >> 32) + 1;
      std::string mvk = "mvk n/a";

#ifdef DXVK_HAS_MVK_STATS
      if (mvkOk) {
        const MVKShaderCompilationPerformance& a = mvk0.shaderCompilation;
        const MVKShaderCompilationPerformance& b = mvk1.shaderCompilation;
        auto d  = [] (const MVKPerformanceTracker& x, const MVKPerformanceTracker& y) { return mvkTotal(y) - mvkTotal(x); };
        auto dn = [] (const MVKPerformanceTracker& x, const MVKPerformanceTracker& y) { return y.count - x.count; };

        // More MoltenVK work than one pipeline with these stages needs: something else compiled meanwhile
        const uint32_t n = uint32_t(stageInfos.size());
        if (dn(a.spirvToMSL, b.spirvToMSL) > n || dn(a.mslCompile, b.mslCompile) > n
         || dn(a.functionRetrieval, b.functionRetrieval) > n || dn(a.pipelineCompile, b.pipelineCompile) > 1)
          solo = false;

        char buf[256];
        std::snprintf(buf, sizeof(buf), "hash %.2f s2m %.2f/%u msl %.2f/%u lib %.2f/%u fret %.2f/%u fspec %.2f/%u pso %.2f/%u",
          d(a.hashShaderCode, b.hashShaderCode),
          d(a.spirvToMSL, b.spirvToMSL), dn(a.spirvToMSL, b.spirvToMSL),
          d(a.mslCompile, b.mslCompile), dn(a.mslCompile, b.mslCompile),
          d(a.shaderLibraryFromCache, b.shaderLibraryFromCache), dn(a.shaderLibraryFromCache, b.shaderLibraryFromCache),
          d(a.functionRetrieval, b.functionRetrieval), dn(a.functionRetrieval, b.functionRetrieval),
          d(a.functionSpecialization, b.functionSpecialization), dn(a.functionSpecialization, b.functionSpecialization),
          d(a.pipelineCompile, b.pipelineCompile), dn(a.pipelineCompile, b.pipelineCompile));
        mvk = buf;
      }
#endif

      std::fprintf(stderr, "pipe stages: %s t %.0f total %.2f spv %.2f vk %.2f | %s | %s vm %u key %s\n",
        t_stageAtDraw ? "draw" : "ahead", ts0, ts3 - ts0, ts1 - ts0, ts3 - ts2, mvk.c_str(),
        solo ? "solo" : "overlap", passFormat.viewMask,
        DxvkStateCache::pipelineKey(getStateCacheKey(), state, passFormat).c_str());
    }

    if (vr != VK_SUCCESS) {
      Logger::err("DxvkGraphicsPipeline: Failed to compile pipeline");
      this->logPipelineState(LogLevel::Error, state);
      return VK_NULL_HANDLE;
    }
    
    if (Logger::logLevel() <= LogLevel::Debug) {
      t1 = dxvk::high_resolution_clock::now();
      auto td = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      Logger::debug(str::format("DxvkGraphicsPipeline: Finished in ", td.count(), " ms"));
    }

    return pipeline;
  }
  
  
  void DxvkGraphicsPipeline::destroyPipeline(VkPipeline pipeline) const {
    m_vkd->vkDestroyPipeline(m_vkd->device(), pipeline, nullptr);
  }


  DxvkShaderModule DxvkGraphicsPipeline::createShaderModule(
    const Rc<DxvkShader>&                shader,
    const DxvkGraphicsPipelineStateInfo& state,
          uint32_t                       viewMask) const {
    if (shader == nullptr)
      return DxvkShaderModule();

    DxvkShaderModuleCreateInfo info;
    info.noViewIndex = shader->stage() == VK_SHADER_STAGE_FRAGMENT_BIT && viewMask == 0;

    // Fix up fragment shader outputs for dual-source blending
    if (shader->stage() == VK_SHADER_STAGE_FRAGMENT_BIT) {
      info.fsDualSrcBlend = state.omBlend[0].blendEnable() && (
        util::isDualSourceBlendFactor(state.omBlend[0].srcColorBlendFactor()) ||
        util::isDualSourceBlendFactor(state.omBlend[0].dstColorBlendFactor()) ||
        util::isDualSourceBlendFactor(state.omBlend[0].srcAlphaBlendFactor()) ||
        util::isDualSourceBlendFactor(state.omBlend[0].dstAlphaBlendFactor()));
    }

    // Deal with undefined shader inputs
    uint32_t consumedInputs = shader->interfaceSlots().inputSlots;
    uint32_t providedInputs = 0;

    if (shader->stage() == VK_SHADER_STAGE_VERTEX_BIT) {
      for (uint32_t i = 0; i < state.il.attributeCount(); i++)
        providedInputs |= 1u << state.ilAttributes[i].location();
    } else if (shader->stage() != VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
      auto prevStage = getPrevStageShader(shader->stage());
      providedInputs = prevStage->interfaceSlots().outputSlots;
    } else {
      // Technically not correct, but this
      // would need a lot of extra care
      providedInputs = consumedInputs;
    }

    info.undefinedInputs = (providedInputs & consumedInputs) ^ consumedInputs;
    return shader->createShaderModule(m_vkd, m_slotMapping, info);
  }


  Rc<DxvkShader> DxvkGraphicsPipeline::getPrevStageShader(VkShaderStageFlagBits stage) const {
    if (stage == VK_SHADER_STAGE_VERTEX_BIT)
      return nullptr;

    if (stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
      return m_shaders.tcs;

    Rc<DxvkShader> result = m_shaders.vs;

    if (stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
      return result;

    if (m_shaders.tes != nullptr)
      result = m_shaders.tes;

    if (stage == VK_SHADER_STAGE_GEOMETRY_BIT)
      return result;

    if (m_shaders.gs != nullptr)
      result = m_shaders.gs;

    return result;
  }


  bool DxvkGraphicsPipeline::validatePipelineState(
    const DxvkGraphicsPipelineStateInfo&  state,
          bool                            trusted) const {
    // Tessellation shaders and patches must be used together
    bool hasPatches = state.ia.primitiveTopology() == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;

    bool hasTcs = m_shaders.tcs != nullptr;
    bool hasTes = m_shaders.tes != nullptr;

    if (hasPatches != hasTcs || hasPatches != hasTes)
      return false;
    
    // Filter out undefined primitive topologies
    if (state.ia.primitiveTopology() == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM)
      return false;
    
    // Prevent unintended out-of-bounds access to the IL arrays
    if (state.il.attributeCount() > DxvkLimits::MaxNumVertexAttributes
     || state.il.bindingCount()   > DxvkLimits::MaxNumVertexBindings)
      return false;

    // Exit here on the fast path, perform more thorough validation if
    // the state vector comes from an untrusted source (i.e. the cache)
    if (trusted)
      return true;

    // Validate shaders
    if (!m_shaders.validate()) {
      Logger::err("Invalid pipeline: Shader types do not match stage");
      return false;
    }

    // Validate vertex input layout
    const DxvkDevice* device = m_pipeMgr->m_device;
    uint32_t ilLocationMask = 0;
    uint32_t ilBindingMask = 0;

    for (uint32_t i = 0; i < state.il.bindingCount(); i++)
      ilBindingMask |= 1u << state.ilBindings[i].binding();

    for (uint32_t i = 0; i < state.il.attributeCount(); i++) {
      const DxvkIlAttribute& attribute = state.ilAttributes[i];

      if (ilLocationMask & (1u << attribute.location())) {
        Logger::err(str::format("Invalid pipeline: Vertex location ", attribute.location(), " defined twice"));
        return false;
      }

      if (!(ilBindingMask & (1u << attribute.binding()))) {
        Logger::err(str::format("Invalid pipeline: Vertex binding ", attribute.binding(), " not defined"));
        return false;
      }

      VkFormatProperties formatInfo = device->adapter()->formatProperties(attribute.format());

      if (!(formatInfo.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
        Logger::err(str::format("Invalid pipeline: Format ", attribute.format(), " not supported for vertex buffers"));
        return false;
      }

      ilLocationMask |= 1u << attribute.location();
    }

    // Validate rasterization state
    if (state.rs.conservativeMode() != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT) {
      if (!device->extensions().extConservativeRasterization) {
        Logger::err("Conservative rasterization not supported by device");
        return false;
      }

      if (state.rs.conservativeMode() == VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT
       && !device->properties().extConservativeRasterization.primitiveUnderestimation) {
        Logger::err("Primitive underestimation not supported by device");
        return false;
      }
    }

    // Validate depth-stencil state
    if (state.ds.enableDepthBoundsTest() && !device->features().core.features.depthBounds) {
      Logger::err("Depth bounds not supported by device");
      return false;
    }

    return true;
  }
  
  
  DxvkStateCacheKey DxvkGraphicsPipeline::getStateCacheKey() const {
    DxvkStateCacheKey key;
    if (m_shaders.vs  != nullptr) key.vs = m_shaders.vs->getShaderKey();
    if (m_shaders.tcs != nullptr) key.tcs = m_shaders.tcs->getShaderKey();
    if (m_shaders.tes != nullptr) key.tes = m_shaders.tes->getShaderKey();
    if (m_shaders.gs  != nullptr) key.gs = m_shaders.gs->getShaderKey();
    if (m_shaders.fs  != nullptr) key.fs = m_shaders.fs->getShaderKey();
    return key;
  }


  bool DxvkGraphicsPipeline::writePipelineStateToCache(
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPassFormat&          format) const {
    if (m_pipeMgr->m_stateCache == nullptr)
      return false;
    
    return m_pipeMgr->m_stateCache->addGraphicsPipeline(getStateCacheKey(), state, format, m_fsOut);
  }
  
  
  std::string DxvkGraphicsPipeline::describeCacheMiss(
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPassFormat&          format) const {
    if (m_pipeMgr->m_stateCache == nullptr)
      return "cache: off";

    return m_pipeMgr->m_stateCache->describeMiss(getStateCacheKey(), state, format, m_fsOut);
  }


  void DxvkGraphicsPipeline::logPipelineState(
          LogLevel                       level,
    const DxvkGraphicsPipelineStateInfo& state) const {
    if (m_shaders.vs  != nullptr) Logger::log(level, str::format("  vs  : ", m_shaders.vs ->debugName()));
    if (m_shaders.tcs != nullptr) Logger::log(level, str::format("  tcs : ", m_shaders.tcs->debugName()));
    if (m_shaders.tes != nullptr) Logger::log(level, str::format("  tes : ", m_shaders.tes->debugName()));
    if (m_shaders.gs  != nullptr) Logger::log(level, str::format("  gs  : ", m_shaders.gs ->debugName()));
    if (m_shaders.fs  != nullptr) Logger::log(level, str::format("  fs  : ", m_shaders.fs ->debugName()));

    for (uint32_t i = 0; i < state.il.attributeCount(); i++) {
      const auto& attr = state.ilAttributes[i];
      Logger::log(level, str::format("  attr ", i, " : location ", attr.location(), ", binding ", attr.binding(), ", format ", attr.format(), ", offset ", attr.offset()));
    }
    for (uint32_t i = 0; i < state.il.bindingCount(); i++) {
      const auto& bind = state.ilBindings[i];
      Logger::log(level, str::format("  binding ", i, " : binding ", bind.binding(), ", stride ", bind.stride(), ", rate ", bind.inputRate(), ", divisor ", bind.divisor()));
    }
    
    // TODO log more pipeline state
  }
  
}
