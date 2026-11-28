
#pragma once

#include <mutex>
#include <unordered_map>

#include "dxvk_compute.h"
#include "dxvk_graphics.h"

namespace dxvk {

  class DxvkStateCache;

  /**
   * \brief Pipeline count
   * 
   * Stores number of graphics and
   * compute pipelines, individually.
   */
  struct DxvkPipelineCount {
    uint32_t numGraphicsPipelines;
    uint32_t numComputePipelines;
  };
  
  
  /**
   * \brief Pipeline manager
   * 
   * Creates and stores graphics pipelines and compute
   * pipelines for each combination of shaders that is
   * used within the application. This is necessary
   * because DXVK does not expose the concept of shader
   * pipeline objects to the client API.
   */
  class DxvkPipelineManager {
    friend class DxvkComputePipeline;
    friend class DxvkGraphicsPipeline;
  public:
    
    DxvkPipelineManager(
      const DxvkDevice*         device,
            DxvkRenderPassPool* passManager);
    
    ~DxvkPipelineManager();
    
    /**
     * \brief Retrieves a compute pipeline object
     * 
     * If a pipeline for the given shader stage object
     * already exists, it will be returned. Otherwise,
     * a new pipeline will be created.
     * \param [in] shaders Shaders for the pipeline
     * \returns Compute pipeline object
     */
    DxvkComputePipeline* createComputePipeline(
      const DxvkComputePipelineShaders& shaders);
    
    /**
     * \brief Retrieves a graphics pipeline object
     * 
     * If a pipeline for the given shader stage objects
     * already exists, it will be returned. Otherwise,
     * a new pipeline will be created.
     * \param [in] shaders Shaders for the pipeline
     * \returns Graphics pipeline object
     */
    DxvkGraphicsPipeline* createGraphicsPipeline(
      const DxvkGraphicsPipelineShaders& shaders);
    
    /*
     * \brief Registers a shader
     * 
     * Starts compiling pipelines asynchronously
     * in case the state cache contains state
     * vectors for this shader.
     * \param [in] shader Newly compiled shader
     */
    void registerShader(
      const Rc<DxvkShader>&         shader);
    
    /**
     * \brief Retrieves total pipeline count
     * \returns Number of compute/graphics pipelines
     */
    DxvkPipelineCount getPipelineCount() const;

    /**
     * \brief Checks whether async compiler is busy
     * \returns \c true if shaders are being compiled
     */
    bool isCompilingShaders() const;

    /**
     * \brief Stops async compiler threads
     */
    void stopWorkerThreads() const;

    /**
     * \brief Vulkan pipeline cache to compile a pipeline with
     *
     * With \c dxvk.drawPipelineCache, pipelines that a draw
     * waits for get a cache of their own, so they don't wait
     * for the compiler threads on a driver that locks the
     * cache while compiling. Otherwise all use one cache.
     *
     * The draw cache must stay internally synchronized (never
     * created with VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT):
     * besides the CS thread, the present path compiles through it
     * on the app thread (swapchain blitter, HUD), at the same time.
     * \param [in] atDraw Whether a draw waits for the pipeline
     * \returns Pipeline cache handle
     */
    VkPipelineCache getVkPipelineCache(bool atDraw) const {
      return atDraw && m_drawCache != nullptr
        ? m_drawCache->handle()
        : m_cache->handle();
    }
    
  private:
    
    const DxvkDevice*         m_device;
    Rc<DxvkPipelineCache>     m_cache;
    Rc<DxvkPipelineCache>     m_drawCache;
    Rc<DxvkStateCache>        m_stateCache;

    std::atomic<uint32_t>     m_numComputePipelines  = { 0 };
    std::atomic<uint32_t>     m_numGraphicsPipelines = { 0 };
    
    dxvk::mutex m_mutex;
    
    std::unordered_map<
      DxvkComputePipelineShaders,
      DxvkComputePipeline,
      DxvkHash, DxvkEq> m_computePipelines;
    
    std::unordered_map<
      DxvkGraphicsPipelineShaders,
      DxvkGraphicsPipeline,
      DxvkHash, DxvkEq> m_graphicsPipelines;
    
  };
  
}