#include "dxvk_device.h"
#include "dxvk_pipemanager.h"
#include "dxvk_state_cache.h"

namespace dxvk {
  
  DxvkPipelineManager::DxvkPipelineManager(
    const DxvkDevice*         device,
          DxvkRenderPassPool* passManager)
  : m_device    (device),
    m_cache     (new DxvkPipelineCache(device->vkd())) {
    std::string useStateCache = env::getEnvVar("DXVK_STATE_CACHE");
    
    if (useStateCache != "0" && device->config().enableStateCache)
      m_stateCache = new DxvkStateCache(device, this, passManager);

    if (env::getEnvVar("DXVK_ASYNC") == "1") {
      m_async = true;
      for (uint32_t i = 0; i < 2; i++)
        m_asyncThreads.emplace_back([this] () { asyncWorker(); });
      Logger::info("DXVK: draw-time pipelines compile in the background (DXVK_ASYNC=1)");
    }
  }
  
  
  DxvkPipelineManager::~DxvkPipelineManager() {
    this->stopWorkerThreads();
  }


  void DxvkPipelineManager::compileAsync(
          DxvkGraphicsPipeline*          pipeline,
    const DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPass*                renderPass) {
    { std::lock_guard<dxvk::mutex> lock(m_asyncMutex);
      m_asyncQueue.push({ pipeline, state, renderPass });
    }
    m_asyncCond.notify_one();
  }


  void DxvkPipelineManager::asyncWorker() {
    env::setThreadName("dxvk-async");

    for (;;) {
      AsyncItem item;

      { std::unique_lock<dxvk::mutex> lock(m_asyncMutex);
        m_asyncCond.wait(lock, [this] { return m_asyncStop || !m_asyncQueue.empty(); });

        if (m_asyncStop)
          return;

        item = m_asyncQueue.front();
        m_asyncQueue.pop();
      }

      item.pipeline->compileQueued(item.state, item.renderPass);
    }
  }
  
  
  DxvkComputePipeline* DxvkPipelineManager::createComputePipeline(
    const DxvkComputePipelineShaders& shaders) {
    if (shaders.cs == nullptr)
      return nullptr;
    
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    
    auto pair = m_computePipelines.find(shaders);
    if (pair != m_computePipelines.end())
      return &pair->second;
    
    auto iter = m_computePipelines.emplace(
      std::piecewise_construct,
      std::tuple(shaders),
      std::tuple(this, shaders));
    return &iter.first->second;
  }
  
  
  DxvkGraphicsPipeline* DxvkPipelineManager::createGraphicsPipeline(
    const DxvkGraphicsPipelineShaders& shaders) {
    if (shaders.vs == nullptr)
      return nullptr;
    
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    
    auto pair = m_graphicsPipelines.find(shaders);
    if (pair != m_graphicsPipelines.end())
      return &pair->second;
    
    auto iter = m_graphicsPipelines.emplace(
      std::piecewise_construct,
      std::tuple(shaders),
      std::tuple(this, shaders));
    return &iter.first->second;
  }

  
  void DxvkPipelineManager::registerShader(
    const Rc<DxvkShader>&         shader) {
    if (m_stateCache != nullptr)
      m_stateCache->registerShader(shader);
  }


  DxvkPipelineCount DxvkPipelineManager::getPipelineCount() const {
    DxvkPipelineCount result;
    result.numComputePipelines  = m_numComputePipelines.load();
    result.numGraphicsPipelines = m_numGraphicsPipelines.load();
    return result;
  }


  bool DxvkPipelineManager::isCompilingShaders() const {
    return m_stateCache != nullptr
        && m_stateCache->isCompilingShaders();
  }


  void DxvkPipelineManager::stopWorkerThreads() const {
    if (m_stateCache != nullptr)
      m_stateCache->stopWorkerThreads();

    { std::lock_guard<dxvk::mutex> lock(m_asyncMutex);
      m_asyncStop = true;
    }
    m_asyncCond.notify_all();

    for (auto& thread : m_asyncThreads)
      thread.join();
    m_asyncThreads.clear();
  }
  
}
