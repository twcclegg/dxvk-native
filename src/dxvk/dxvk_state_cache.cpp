#include <algorithm>
#include <cstdio>
#include <cstring>

#include "dxvk_device.h"
#include "dxvk_pipemanager.h"
#include "dxvk_state_cache.h"

namespace dxvk {

  static const Sha1Hash       g_nullHash      = Sha1Hash::compute(nullptr, 0);
  static const DxvkShaderKey  g_nullShaderKey = DxvkShaderKey();


  uint32_t dxvkPipeLogLevel() {
    static const uint32_t level = [] {
      std::string e = env::getEnvVar("LB_PIPE_LOG");
      return e.empty() || e[0] < '0' || e[0] > '9' ? 0u : uint32_t(e[0] - '0');
    }();
    return level;
  }


  // FNV-1a, so keys printed by different launches compare equal
  static uint32_t fnv32(const void* data, size_t size, uint32_t h = 2166136261u) {
    auto bytes = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; i++)
      h = (h ^ bytes[i]) * 16777619u;
    return h;
  }

  static uint32_t formatHash32(const DxvkRenderPassFormat& f) {
    uint32_t words[3 + 2 * MaxNumRenderTargets];
    uint32_t n = 0;
    words[n++] = uint32_t(f.sampleCount);
    words[n++] = f.viewMask;
    words[n++] = uint32_t(f.depth.format) ^ (uint32_t(f.depth.layout) << 16);
    for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
      words[n++] = uint32_t(f.color[i].format);
      words[n++] = uint32_t(f.color[i].layout);
    }
    return fnv32(words, sizeof(words));
  }

  // An entry's state as the pipeline compiles it when replayed (DXVK_PIPELINE_NORMALIZE)
  static void effectiveEntry(
          DxvkGraphicsPipelineStateInfo& state,
    const DxvkRenderPassFormat&          format,
          uint32_t                       fsOutMask) {
    if (dxvkPipelineNormalizeEnabled())
      dxvkNormalizePipelineState(state, format, fsOutMask);
  }

  static std::string shortKey(const DxvkShaderKey& key) {
    return key.eq(g_nullShaderKey) ? std::string("-") : key.sha1().toString().substr(0, 8);
  }


  /**
   * \brief Packed entry header
   */
  struct DxvkStateCacheEntryHeader {
    uint32_t stageMask : 8;
    uint32_t entrySize : 24;
  };

  
  /**
   * \brief State cache entry data
   *
   * Stores data for a single cache entry and
   * provides convenience methods to access it.
   */
  class DxvkStateCacheEntryData {
    constexpr static size_t MaxSize = 1024;
  public:

    size_t size() const {
      return m_size;
    }

    const char* data() const {
      return m_data;
    }

    Sha1Hash computeHash() const {
      return Sha1Hash::compute(m_data, m_size);
    }

    template<typename T>
    bool read(T& data, uint32_t version) {
      return read(data);
    }

    bool read(DxvkBindingMask& data, uint32_t version) {
      if (version < 9) {
        DxvkBindingMaskV8 v8;

        if (!read(v8))
          return false;

        data = v8.convert();
        return true;
      }

      return read(data);
    }

    bool read(DxvkIlBinding& data, uint32_t version) {
      if (version < 10) {
        DxvkIlBindingV9 v9;

        if (!read(v9))
          return false;

        data = v9.convert();
        return true;
      }

      return read(data);
    }

    template<typename T>
    bool write(const T& data) {
      if (m_size + sizeof(T) > MaxSize)
        return false;
      
      std::memcpy(&m_data[m_size], &data, sizeof(T));
      m_size += sizeof(T);
      return true;
    }

    bool readFromStream(std::istream& stream, size_t size) {
      if (size > MaxSize)
        return false;

      if (!stream.read(m_data, size))
        return false;

      m_size = size;
      m_read = 0;
      return true;
    }

  private:

    size_t m_size = 0;
    size_t m_read = 0;
    char   m_data[MaxSize];

    template<typename T>
    bool read(T& data) {
      if (m_read + sizeof(T) > m_size)
        return false;

      std::memcpy(&data, &m_data[m_read], sizeof(T));
      m_read += sizeof(T);
      return true;
    }

  };


  template<typename T>
  bool readCacheEntryTyped(std::istream& stream, T& entry) {
    auto data = reinterpret_cast<char*>(&entry);
    auto size = sizeof(entry);

    if (!stream.read(data, size))
      return false;
    
    Sha1Hash expectedHash = std::exchange(entry.hash, g_nullHash);
    Sha1Hash computedHash = Sha1Hash::compute(entry);
    return expectedHash == computedHash;
  }


  bool DxvkStateCacheKey::eq(const DxvkStateCacheKey& key) const {
    return this->vs.eq(key.vs)
        && this->tcs.eq(key.tcs)
        && this->tes.eq(key.tes)
        && this->gs.eq(key.gs)
        && this->fs.eq(key.fs)
        && this->cs.eq(key.cs);
  }


  size_t DxvkStateCacheKey::hash() const {
    DxvkHashState hash;
    hash.add(this->vs.hash());
    hash.add(this->tcs.hash());
    hash.add(this->tes.hash());
    hash.add(this->gs.hash());
    hash.add(this->fs.hash());
    hash.add(this->cs.hash());
    return hash;
  }


  DxvkStateCache::DxvkStateCache(
    const DxvkDevice*           device,
          DxvkPipelineManager*  pipeManager,
          DxvkRenderPassPool*   passManager)
  : m_pipeManager(pipeManager),
    m_passManager(passManager),
    m_multiview  (device->features().coreMultiview.multiview) {
    bool newFile = !readCacheFile();

    if (newFile) {
      Logger::warn("DXVK: Creating new state cache file");

      // Start with an empty file
      std::ofstream file(getCacheFileName().c_str(),
        std::ios_base::binary |
        std::ios_base::trunc);

      if (!file && env::createDirectory(getCacheDir())) {
        file = std::ofstream(getCacheFileName().c_str(),
          std::ios_base::binary |
          std::ios_base::trunc);
      }

      // Write header with the current version number
      DxvkStateCacheHeader header;

      auto data = reinterpret_cast<const char*>(&header);
      auto size = sizeof(header);

      file.write(data, size);

      // Write all valid entries to the cache file in
      // case we're recovering a corrupted cache file
      for (auto& e : m_entries)
        writeCacheEntry(file, e);
    }

    // Use half the available CPU cores for pipeline compilation
    uint32_t numCpuCores = dxvk::thread::hardware_concurrency();
    uint32_t numWorkers  = ((std::max(1u, numCpuCores) - 1) * 5) / 7;

    if (numWorkers <  1) numWorkers =  1;
    if (numWorkers > 32) numWorkers = 32;

    if (device->config().numCompilerThreads > 0)
      numWorkers = device->config().numCompilerThreads;
    
    Logger::info(str::format("DXVK: Using ", numWorkers, " compiler threads"));
    
    // Start the worker threads and the file writer
    m_workerBusy.store(numWorkers);

    for (uint32_t i = 0; i < numWorkers; i++) {
      m_workerThreads.emplace_back([this] () { workerFunc(); });

      // TODO: better solution for this.
#ifndef DXVK_NATIVE
      m_workerThreads[i].set_priority(ThreadPriority::Lowest);
#endif
    }
    
    m_writerThread = dxvk::thread([this] () { writerFunc(); });
  }
  

  DxvkStateCache::~DxvkStateCache() {
    this->stopWorkerThreads();
  }


  bool DxvkStateCache::addGraphicsPipeline(
    const DxvkStateCacheKey&              shaders,
    const DxvkGraphicsPipelineStateInfo&  state,
    const DxvkRenderPassFormat&           format,
          uint32_t                        fsOutMask) {
    if (shaders.vs.eq(g_nullShaderKey))
      return false;
    
    // Do not add an entry that is already in the cache
    auto entries = m_entryMap.equal_range(shaders);

    for (auto e = entries.first; e != entries.second; e++) {
      const DxvkStateCacheEntry& entry = m_entries[e->second];
      if (!entry.format.eq(format))
        continue;

      DxvkGraphicsPipelineStateInfo entryState = entry.gpState;
      effectiveEntry(entryState, entry.format, fsOutMask);

      if (entryState == state)
        return false;
    }

    // Queue a job to write this pipeline to the cache
    std::unique_lock<dxvk::mutex> lock(m_writerLock);

    m_writerQueue.push({ shaders, state,
      DxvkComputePipelineStateInfo(),
      format, g_nullHash });
    m_writerCond.notify_one();
    return true;
  }


  void DxvkStateCache::addComputePipeline(
    const DxvkStateCacheKey&              shaders,
    const DxvkComputePipelineStateInfo&   state) {
    if (shaders.cs.eq(g_nullShaderKey))
      return;

    // Do not add an entry that is already in the cache
    auto entries = m_entryMap.equal_range(shaders);

    for (auto e = entries.first; e != entries.second; e++) {
      if (m_entries[e->second].cpState == state)
        return;
    }

    // Queue a job to write this pipeline to the cache
    std::unique_lock<dxvk::mutex> lock(m_writerLock);

    m_writerQueue.push({ shaders,
      DxvkGraphicsPipelineStateInfo(), state,
      DxvkRenderPassFormat(), g_nullHash });
    m_writerCond.notify_one();
  }


  void DxvkStateCache::registerShader(const Rc<DxvkShader>& shader) {
    DxvkShaderKey key = shader->getShaderKey();

    if (key.eq(g_nullShaderKey))
      return;
    
    // Add the shader so we can look it up by its key
    std::unique_lock<dxvk::mutex> entryLock(m_entryLock);
    m_shaderMap.insert({ key, shader });
    m_shaderTimes.insert({ key, high_resolution_clock::now() });

    // Deferred lock, don't stall workers unless we have to
    std::unique_lock<dxvk::mutex> workerLock;

    auto pipelines = m_pipelineMap.equal_range(key);

    for (auto p = pipelines.first; p != pipelines.second; p++) {
      WorkerItem item;

      if (!getShaderByKey(p->second.vs,  item.gp.vs)
       || !getShaderByKey(p->second.tcs, item.gp.tcs)
       || !getShaderByKey(p->second.tes, item.gp.tes)
       || !getShaderByKey(p->second.gs,  item.gp.gs)
       || !getShaderByKey(p->second.fs,  item.gp.fs)
       || !getShaderByKey(p->second.cs,  item.cp.cs))
        continue;
      
      if (!workerLock)
        workerLock = std::unique_lock<dxvk::mutex>(m_workerLock);
      
      m_workerQueue.push(item);
    }

    if (workerLock)
      m_workerCond.notify_all();
  }


  std::string DxvkStateCache::describeMiss(
    const DxvkStateCacheKey&              shaders,
    const DxvkGraphicsPipelineStateInfo&  state,
    const DxvkRenderPassFormat&           format,
          uint32_t                        fsOutMask) {
    // m_entries and m_entryMap are only written while reading the file in the constructor
    auto entries = m_entryMap.equal_range(shaders);

    size_t      count     = 0;
    size_t      bestDiffs = ~size_t(0);
    std::string bestText;

    auto same = [] (const auto& a, const auto& b) {
      return std::memcmp(&a, &b, sizeof(a)) == 0;
    };

    for (auto e = entries.first; e != entries.second; e++) {
      DxvkStateCacheEntry entry = m_entries[e->second];
      effectiveEntry(entry.gpState, entry.format, fsOutMask);
      const DxvkGraphicsPipelineStateInfo& c = entry.gpState;
      count += 1;

      std::string text;
      size_t      diffs = 0;
      auto note = [&] (bool differs, const std::string& name) {
        if (differs) {
          text += (diffs++ ? "," : "") + name;
        }
      };

      note(entry.format.viewMask != format.viewMask,
        str::format("viewMask ", entry.format.viewMask, "->", format.viewMask));
      note(entry.format.sampleCount != format.sampleCount, "samples");

      bool rtDiffers = entry.format.depth.format != format.depth.format
                    || entry.format.depth.layout != format.depth.layout;
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        rtDiffers |= entry.format.color[i].format != format.color[i].format
                  || entry.format.color[i].layout != format.color[i].layout;
      }
      note(rtDiffers, "targets");

      note(!same(c.bsBindingMask, state.bsBindingMask), "bindingMask");
      note(!same(c.ia, state.ia), "ia");
      note(!same(c.il, state.il), "il");
      note(!same(c.rs, state.rs), "rs");
      note(!same(c.ms, state.ms), "ms");
      note(!same(c.ds, state.ds), "ds");
      note(!same(c.om, state.om), "om");
      note(!same(c.dsFront, state.dsFront) || !same(c.dsBack, state.dsBack), "stencil");
      note(!same(c.omSwizzle, state.omSwizzle), "swizzle");
      note(!same(c.omBlend, state.omBlend), "blend");
      note(!same(c.ilAttributes, state.ilAttributes), "attrs");
      note(!same(c.ilBindings, state.ilBindings), "vbindings");

      for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
        if (c.sc.specConstants[i] != state.sc.specConstants[i]) {
          char buf[48];
          std::snprintf(buf, sizeof(buf), "sc%u %x->%x", i, c.sc.specConstants[i], state.sc.specConstants[i]);
          note(true, buf);
        }
      }

      if (!diffs) {
        bestDiffs = 0;
        bestText.clear();
        break;
      }

      if (diffs < bestDiffs) {
        bestDiffs = diffs;
        bestText  = std::move(text);
      }
    }

    // Registration age of the youngest shader, and the compiler threads' backlog
    std::string age;
    { std::lock_guard<dxvk::mutex> lock(m_entryLock);
      auto now = high_resolution_clock::now();
      double youngest = -1.0;
      bool   missing  = false;

      for (const DxvkShaderKey* k : { &shaders.vs, &shaders.tcs, &shaders.tes, &shaders.gs, &shaders.fs }) {
        if (k->eq(g_nullShaderKey))
          continue;

        auto t = m_shaderTimes.find(*k);
        if (t == m_shaderTimes.end()) {
          missing = true;
          continue;
        }

        double ms = std::chrono::duration<double, std::milli>(now - t->second).count();
        if (youngest < 0.0 || ms < youngest)
          youngest = ms;
      }

      age = missing ? std::string("unregistered") : str::format(uint64_t(youngest), " ms");
    }

    size_t queued;
    { std::lock_guard<dxvk::mutex> lock(m_workerLock);
      queued = m_workerQueue.size();
    }

    std::string where = str::format("shaders ", age, ", queue ", queued, ", busy ", m_workerBusy.load());

    if (!count) {
      size_t vsUses = m_pipelineMap.count(shaders.vs);
      size_t fsUses = m_pipelineMap.count(shaders.fs);
      return str::format("cache: none (vs in ", vsUses, ", fs in ", fsUses, " entries; ", where, ")");
    }

    if (!bestDiffs)
      return str::format("cache: exact of ", count, " (", where, ")");

    return str::format("cache: differs of ", count, " [", bestText, "] (", where, ")");
  }


  std::string DxvkStateCache::pipelineKey(
    const DxvkStateCacheKey&              shaders,
    const DxvkGraphicsPipelineStateInfo&  state,
    const DxvkRenderPassFormat&           format) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s.%s.%08x.%08x",
      shortKey(shaders.vs).c_str(), shortKey(shaders.fs).c_str(),
      fnv32(&state, sizeof(state)), formatHash32(format));
    return buf;
  }


  /* Field-by-field difference between a cache entry (a) and a pipeline (b), for LB_PIPE_LOG=2. A field is
   * marked "dc" when it cannot change the pipeline Vulkan builds: blend factors and ops of an attachment with
   * blending off in both, any blend or swizzle state of an attachment the fragment shader doesn't write or
   * the pass doesn't have, depth compare and write with the depth test off (Vulkan skips depth writes then)
   * or no depth attachment, stencil ops with the stencil test off, and the logic op with logic ops off: the
   * state dxvkNormalizePipelineState clears. */
  namespace {

    struct StateDiff {
      std::string text;
      uint32_t    all  = 0;
      uint32_t    real = 0;

      void add(const char* name, uint32_t a, uint32_t b, bool dc) {
        if (a == b)
          return;

        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s%s %x->%x%s", all ? "," : "", name, a, b, dc ? " dc" : "");
        text += buf;
        all  += 1;
        real += dc ? 0 : 1;
      }

      void add(const std::string& name, uint32_t a, uint32_t b, bool dc) {
        add(name.c_str(), a, b, dc);
      }
    };

    const char* specConstantName(uint32_t i) {
      // D3D9SpecConstantId (d3d9_spec_constants.h); the D3D11 front end uses none
      static const char* names[] = {
        "alphaOp", "samplerType", "fogEnabled", "vsFogMode", "psFogMode", "pointMode", "projection",
        "vsBools", "psBools", "fetch4", "depthMode", "vsScale", "depthFilter" };
      return i < sizeof(names) / sizeof(names[0]) ? names[i] : "sc";
    }

    StateDiff diffState(
      const DxvkGraphicsPipelineStateInfo& a, const DxvkRenderPassFormat& fa,
      const DxvkGraphicsPipelineStateInfo& b, const DxvkRenderPassFormat& fb,
            uint32_t                       fsOutMask) {
      StateDiff d;

      // Render pass
      d.add("samples", fa.sampleCount, fb.sampleCount, false);
      d.add("viewMask", fa.viewMask, fb.viewMask, false);
      d.add("depthFmt", fa.depth.format, fb.depth.format, false);
      d.add("depthLayout", fa.depth.layout, fb.depth.layout, false);

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        d.add(str::format("rt", i, "Fmt"), fa.color[i].format, fb.color[i].format, false);
        d.add(str::format("rt", i, "Layout"), fa.color[i].layout, fb.color[i].layout, false);
      }

      // Resources, input assembly, vertex input
      if (std::memcmp(&a.bsBindingMask, &b.bsBindingMask, sizeof(a.bsBindingMask)))
        d.add("bindingMask", 0, 1, false);

      d.add("topology", a.ia.primitiveTopology(), b.ia.primitiveTopology(), false);
      d.add("restart", a.ia.primitiveRestart(), b.ia.primitiveRestart(), false);
      d.add("patchVerts", a.ia.patchVertexCount(), b.ia.patchVertexCount(), false);
      d.add("attrCount", a.il.attributeCount(), b.il.attributeCount(), false);
      d.add("bindCount", a.il.bindingCount(), b.il.bindingCount(), false);

      for (uint32_t i = 0; i < std::max(a.il.attributeCount(), b.il.attributeCount()); i++) {
        const auto& x = a.ilAttributes[i];
        const auto& y = b.ilAttributes[i];
        d.add(str::format("attr", i, ".loc"), x.location(), y.location(), false);
        d.add(str::format("attr", i, ".bind"), x.binding(), y.binding(), false);
        d.add(str::format("attr", i, ".fmt"), x.format(), y.format(), false);
        d.add(str::format("attr", i, ".offset"), x.offset(), y.offset(), false);
      }

      for (uint32_t i = 0; i < std::max(a.il.bindingCount(), b.il.bindingCount()); i++) {
        const auto& x = a.ilBindings[i];
        const auto& y = b.ilBindings[i];
        d.add(str::format("vb", i, ".bind"), x.binding(), y.binding(), false);
        d.add(str::format("vb", i, ".stride"), x.stride(), y.stride(), false);
        d.add(str::format("vb", i, ".rate"), x.inputRate(), y.inputRate(), false);
        d.add(str::format("vb", i, ".divisor"), x.divisor(), y.divisor(), false);
      }

      // Rasterizer, multisample
      d.add("depthClip", a.rs.depthClipEnable(), b.rs.depthClipEnable(), false);
      d.add("depthBias", a.rs.depthBiasEnable(), b.rs.depthBiasEnable(), false);
      d.add("polygonMode", a.rs.polygonMode(), b.rs.polygonMode(), false);
      d.add("cullMode", a.rs.cullMode(), b.rs.cullMode(), false);
      d.add("frontFace", a.rs.frontFace(), b.rs.frontFace(), false);
      d.add("viewports", a.rs.viewportCount(), b.rs.viewportCount(), false);
      d.add("rsSamples", a.rs.sampleCount(), b.rs.sampleCount(), false);
      d.add("conservative", a.rs.conservativeMode(), b.rs.conservativeMode(), false);
      d.add("msSamples", a.ms.sampleCount(), b.ms.sampleCount(), false);
      d.add("sampleMask", a.ms.sampleMask(), b.ms.sampleMask(), false);
      d.add("alphaToCoverage", a.ms.enableAlphaToCoverage(), b.ms.enableAlphaToCoverage(), false);

      // Depth and stencil
      const bool noDepth   = fa.depth.format == VK_FORMAT_UNDEFINED && fb.depth.format == VK_FORMAT_UNDEFINED;
      const bool depthOff  = noDepth || (!a.ds.enableDepthTest() && !b.ds.enableDepthTest());
      const bool stenOff   = noDepth || (!a.ds.enableStencilTest() && !b.ds.enableStencilTest());
      d.add("depthTest", a.ds.enableDepthTest(), b.ds.enableDepthTest(), noDepth);
      d.add("depthWrite", a.ds.enableDepthWrite(), b.ds.enableDepthWrite(), depthOff);
      d.add("depthOp", a.ds.depthCompareOp(), b.ds.depthCompareOp(), depthOff);
      d.add("depthBounds", a.ds.enableDepthBoundsTest(), b.ds.enableDepthBoundsTest(), noDepth);
      d.add("stencilTest", a.ds.enableStencilTest(), b.ds.enableStencilTest(), noDepth);

      for (uint32_t face = 0; face < 2; face++) {
        VkStencilOpState x = (face ? a.dsBack : a.dsFront).state();
        VkStencilOpState y = (face ? b.dsBack : b.dsFront).state();
        const char* f = face ? "back" : "front";
        d.add(str::format(f, ".fail"), x.failOp, y.failOp, stenOff);
        d.add(str::format(f, ".pass"), x.passOp, y.passOp, stenOff);
        d.add(str::format(f, ".depthFail"), x.depthFailOp, y.depthFailOp, stenOff);
        d.add(str::format(f, ".op"), x.compareOp, y.compareOp, stenOff);
        d.add(str::format(f, ".readMask"), x.compareMask, y.compareMask, stenOff);
        d.add(str::format(f, ".writeMask"), x.writeMask, y.writeMask, stenOff);
      }

      // Output merger
      d.add("logicOpEnable", a.om.enableLogicOp(), b.om.enableLogicOp(), false);
      d.add("logicOp", a.om.logicOp(), b.om.logicOp(),
        !a.om.enableLogicOp() && !b.om.enableLogicOp());

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        const auto& x = a.omBlend[i];
        const auto& y = b.omBlend[i];
        const bool unused  = !(fsOutMask & (1u << i))
          || (fa.color[i].format == VK_FORMAT_UNDEFINED && fb.color[i].format == VK_FORMAT_UNDEFINED);
        const bool noBlend = unused || (!x.blendEnable() && !y.blendEnable());
        d.add(str::format("rt", i, ".blend"), x.blendEnable(), y.blendEnable(), unused);
        d.add(str::format("rt", i, ".srcColor"), x.srcColorBlendFactor(), y.srcColorBlendFactor(), noBlend);
        d.add(str::format("rt", i, ".dstColor"), x.dstColorBlendFactor(), y.dstColorBlendFactor(), noBlend);
        d.add(str::format("rt", i, ".colorOp"), x.colorBlendOp(), y.colorBlendOp(), noBlend);
        d.add(str::format("rt", i, ".srcAlpha"), x.srcAlphaBlendFactor(), y.srcAlphaBlendFactor(), noBlend);
        d.add(str::format("rt", i, ".dstAlpha"), x.dstAlphaBlendFactor(), y.dstAlphaBlendFactor(), noBlend);
        d.add(str::format("rt", i, ".alphaOp"), x.alphaBlendOp(), y.alphaBlendOp(), noBlend);
        d.add(str::format("rt", i, ".writeMask"), x.colorWriteMask(), y.colorWriteMask(), unused);

        const auto& p = a.omSwizzle[i];
        const auto& q = b.omSwizzle[i];
        d.add(str::format("rt", i, ".swizzle"),
          p.rIndex() | p.gIndex() << 4 | p.bIndex() << 8 | p.aIndex() << 12,
          q.rIndex() | q.gIndex() << 4 | q.bIndex() << 8 | q.aIndex() << 12, unused);
      }

      // Specialization constants
      for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
        d.add(str::format("sc", i, ".", specConstantName(i)), a.sc.specConstants[i], b.sc.specConstants[i], false);

      return d;
    }

  }


  std::string DxvkStateCache::describeMissFields(
    const DxvkStateCacheKey&              shaders,
    const DxvkGraphicsPipelineStateInfo&  state,
    const DxvkRenderPassFormat&           format,
          uint32_t                        fsOutMask) {
    std::string key = pipelineKey(shaders, state, format);
    auto entries = m_entryMap.equal_range(shaders);

    DxvkStateCacheEntry best;
    StateDiff           bestDiff;
    bool                hasBest = false;
    uint32_t            count = 0;

    for (auto e = entries.first; e != entries.second; e++) {
      DxvkStateCacheEntry entry = m_entries[e->second];
      effectiveEntry(entry.gpState, entry.format, fsOutMask);
      StateDiff diff = diffState(entry.gpState, entry.format, state, format, fsOutMask);
      count += 1;

      if (!hasBest || diff.real < bestDiff.real || (diff.real == bestDiff.real && diff.all < bestDiff.all)) {
        hasBest  = true;
        best     = entry;
        bestDiff = std::move(diff);
      }
    }

    if (!hasBest) {
      return str::format("pipe miss: key ", key, " near none (vs in ", m_pipelineMap.count(shaders.vs),
        ", fs in ", m_pipelineMap.count(shaders.fs), " entries)");
    }

    const char* kind = bestDiff.all == 0 ? "exact" : bestDiff.real == 0 ? "dc-only" : "real";
    return str::format("pipe miss: key ", key, " near ", pipelineKey(best.shaders, best.gpState, best.format),
      " of ", count, " ", kind, " [", bestDiff.text, "]");
  }


  void DxvkStateCache::stopWorkerThreads() {
    { std::lock_guard<dxvk::mutex> workerLock(m_workerLock);
      std::lock_guard<dxvk::mutex> writerLock(m_writerLock);

      if (m_stopThreads.exchange(true))
        return;

      m_workerCond.notify_all();
      m_writerCond.notify_all();
    }

    for (auto& worker : m_workerThreads)
      worker.join();
    
    m_writerThread.join();
  }


  DxvkShaderKey DxvkStateCache::getShaderKey(const Rc<DxvkShader>& shader) const {
    return shader != nullptr ? shader->getShaderKey() : g_nullShaderKey;
  }


  bool DxvkStateCache::getShaderByKey(
    const DxvkShaderKey&            key,
          Rc<DxvkShader>&           shader) const {
    if (key.eq(g_nullShaderKey))
      return true;
    
    auto entry = m_shaderMap.find(key);
    if (entry == m_shaderMap.end())
      return false;

    shader = entry->second;
    return true;
  }


  void DxvkStateCache::mapPipelineToEntry(
    const DxvkStateCacheKey&        key,
          size_t                    entryId) {
    m_entryMap.insert({ key, entryId });
  }

  
  void DxvkStateCache::mapShaderToPipeline(
    const DxvkShaderKey&            shader,
    const DxvkStateCacheKey&        key) {
    if (!shader.eq(g_nullShaderKey))
      m_pipelineMap.insert({ shader, key });
  }


  void DxvkStateCache::compilePipelines(const WorkerItem& item) {
    DxvkStateCacheKey key;
    key.vs  = getShaderKey(item.gp.vs);
    key.tcs = getShaderKey(item.gp.tcs);
    key.tes = getShaderKey(item.gp.tes);
    key.gs  = getShaderKey(item.gp.gs);
    key.fs  = getShaderKey(item.gp.fs);
    key.cs  = getShaderKey(item.cp.cs);

    if (item.cp.cs == nullptr) {
      auto pipeline = m_pipeManager->createGraphicsPipeline(item.gp);
      auto entries = m_entryMap.equal_range(key);

      for (auto e = entries.first; e != entries.second; e++) {
        const auto& entry = m_entries[e->second];

        // Written by a multiview session (LB_STEREO_MULTIVIEW): such a render pass is invalid
        // without the multiview feature, and no framebuffer here would ever use it.
        if (entry.format.viewMask && !m_multiview)
          continue;

        if (m_passManager->validateRenderPassFormat(entry.format)) {
          auto rp = m_passManager->getRenderPass(entry.format);
          pipeline->compilePipeline(entry.gpState, rp);
        }
      }
    } else {
      auto pipeline = m_pipeManager->createComputePipeline(item.cp);
      auto entries = m_entryMap.equal_range(key);

      for (auto e = entries.first; e != entries.second; e++) {
        const auto& entry = m_entries[e->second];
        pipeline->compilePipeline(entry.cpState);
      }
    }
  }


  bool DxvkStateCache::readCacheFile() {
    // Open state file and just fail if it doesn't exist
    std::ifstream ifile(getCacheFileName().c_str(), std::ios_base::binary);

    if (!ifile) {
      Logger::warn("DXVK: No state cache file found");
      return false;
    }

    // The header stores the state cache version,
    // we need to regenerate it if it's outdated
    DxvkStateCacheHeader newHeader;
    DxvkStateCacheHeader curHeader;

    if (!readCacheHeader(ifile, curHeader)) {
      Logger::warn("DXVK: Failed to read state cache header");
      return false;
    }

    // Struct size hasn't changed between v2 and v4
    size_t expectedSize = newHeader.entrySize;

    if (curHeader.version <= 4)
      expectedSize = sizeof(DxvkStateCacheEntryV4);
    else if (curHeader.version <= 5)
      expectedSize = sizeof(DxvkStateCacheEntryV5);
    else if (curHeader.version <= 6)
      expectedSize = sizeof(DxvkStateCacheEntryV6);
    else if (curHeader.version <= 7)
      expectedSize = sizeof(DxvkStateCacheEntry);

    if (curHeader.entrySize != expectedSize) {
      Logger::warn("DXVK: State cache entry size changed");
      return false;
    }

    // Discard caches of unsupported versions
    if (curHeader.version < 2 || curHeader.version > newHeader.version) {
      Logger::warn("DXVK: State cache version not supported");
      return false;
    }

    // Notify user about format conversion
    if (curHeader.version != newHeader.version)
      Logger::warn(str::format("DXVK: Updating state cache version to v", newHeader.version));

    // Read actual cache entries from the file.
    // If we encounter invalid entries, we should
    // regenerate the entire state cache file.
    uint32_t numInvalidEntries = 0;

    while (ifile) {
      DxvkStateCacheEntry entry;

      if (readCacheEntry(curHeader.version, ifile, entry)) {
        size_t entryId = m_entries.size();
        m_entries.push_back(entry);

        mapPipelineToEntry(entry.shaders, entryId);

        mapShaderToPipeline(entry.shaders.vs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.tcs, entry.shaders);
        mapShaderToPipeline(entry.shaders.tes, entry.shaders);
        mapShaderToPipeline(entry.shaders.gs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.fs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.cs,  entry.shaders);
      } else if (ifile) {
        numInvalidEntries += 1;
      }
    }

    Logger::info(str::format(
      "DXVK: Read ", m_entries.size(),
      " valid state cache entries"));

    if (numInvalidEntries) {
      Logger::warn(str::format(
        "DXVK: Skipped ", numInvalidEntries,
        " invalid state cache entries"));
      return false;
    }
    
    // Rewrite entire state cache if it is outdated
    return curHeader.version == newHeader.version;
  }


  bool DxvkStateCache::readCacheHeader(
          std::istream&             stream,
          DxvkStateCacheHeader&     header) const {
    DxvkStateCacheHeader expected;

    auto data = reinterpret_cast<char*>(&header);
    auto size = sizeof(header);

    if (!stream.read(data, size))
      return false;
    
    for (uint32_t i = 0; i < 4; i++) {
      if (expected.magic[i] != header.magic[i])
        return false;
    }
    
    return true;
  }


  bool DxvkStateCache::readCacheEntryV7(
          uint32_t                  version,
          std::istream&             stream, 
          DxvkStateCacheEntry&      entry) const {
    if (version <= 6) {
      DxvkStateCacheEntryV6 v6;

      if (version <= 4) {
        DxvkStateCacheEntryV4 v4;

        if (!readCacheEntryTyped(stream, v4))
          return false;

        if (version == 2)
          convertEntryV2(v4);

        if (!convertEntryV4(v4, v6))
          return false;
      } else if (version <= 5) {
        DxvkStateCacheEntryV5 v5;

        if (!readCacheEntryTyped(stream, v5))
          return false;

        if (!convertEntryV5(v5, v6))
          return false;
      } else {
        if (!readCacheEntryTyped(stream, v6))
          return false;
      }

      return convertEntryV6(v6, entry);
    } else {
      return readCacheEntryTyped(stream, entry);
    }
  }


  bool DxvkStateCache::readCacheEntry(
          uint32_t                  version,
          std::istream&             stream, 
          DxvkStateCacheEntry&      entry) const {
    if (version < 8)
      return readCacheEntryV7(version, stream, entry);

    // Read entry metadata and actual data
    DxvkStateCacheEntryHeader header;
    DxvkStateCacheEntryData data;
    Sha1Hash hash;
  
    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header))
     || !stream.read(reinterpret_cast<char*>(&hash), sizeof(hash))
     || !data.readFromStream(stream, header.entrySize))
      return false;

    // Validate hash, skip entry if invalid
    if (hash != data.computeHash())
      return false;

    // Read shader hashes
    VkShaderStageFlags stageMask = VkShaderStageFlags(header.stageMask);
    auto keys = &entry.shaders.vs;

    for (uint32_t i = 0; i < 6; i++) {
      if (stageMask & VkShaderStageFlagBits(1 << i))
        data.read(keys[i], version);
      else
        keys[i] = g_nullShaderKey;
    }

    if (stageMask & VK_SHADER_STAGE_COMPUTE_BIT) {
      if (!data.read(entry.cpState.bsBindingMask, version))
        return false;
    } else {
      // Read packed render pass format
      uint8_t sampleCount = 0;
      uint8_t imageFormat = 0;
      uint8_t imageLayout = 0;

      if (!data.read(sampleCount, version)
       || !data.read(imageFormat, version)
       || !data.read(imageLayout, version))
        return false;

      entry.format.sampleCount = VkSampleCountFlagBits(sampleCount);
      entry.format.depth.format = VkFormat(imageFormat);
      entry.format.depth.layout = unpackImageLayout(imageLayout);

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!data.read(imageFormat, version)
         || !data.read(imageLayout, version))
          return false;

        entry.format.color[i].format = VkFormat(imageFormat);
        entry.format.color[i].layout = unpackImageLayout(imageLayout);
      }

      // v11: multiview view mask (Apple port, stereo). Shader keys hash the D3D9 bytecode, not
      // the SPIR-V, so a session without multiview also matches these entries; compilePipelines
      // skips them there (a non-zero view mask needs the multiview feature).
      entry.format.viewMask = 0;

      if (version >= 11) {
        uint8_t viewMask = 0;

        if (!data.read(viewMask, version))
          return false;

        entry.format.viewMask = viewMask;
      }

      if (!validateRenderPassFormat(entry.format))
        return false;

      // Read common pipeline state
      if (!data.read(entry.gpState.bsBindingMask, version)
       || !data.read(entry.gpState.ia, version)
       || !data.read(entry.gpState.il, version)
       || !data.read(entry.gpState.rs, version)
       || !data.read(entry.gpState.ms, version)
       || !data.read(entry.gpState.ds, version)
       || !data.read(entry.gpState.om, version)
       || !data.read(entry.gpState.dsFront, version)
       || !data.read(entry.gpState.dsBack, version))
        return false;

      if (entry.gpState.il.attributeCount() > MaxNumVertexAttributes
       || entry.gpState.il.bindingCount() > MaxNumVertexBindings)
        return false;

      // Read render target swizzles
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!data.read(entry.gpState.omSwizzle[i], version))
          return false;
      }

      // Read render target blend info
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!data.read(entry.gpState.omBlend[i], version))
          return false;
      }

      // Read defined vertex attributes
      for (uint32_t i = 0; i < entry.gpState.il.attributeCount(); i++) {
        if (!data.read(entry.gpState.ilAttributes[i], version))
          return false;
      }

      // Read defined vertex bindings
      for (uint32_t i = 0; i < entry.gpState.il.bindingCount(); i++) {
        if (!data.read(entry.gpState.ilBindings[i], version))
          return false;
      }
    }

    // Read non-zero spec constants
    auto& sc = (stageMask & VK_SHADER_STAGE_COMPUTE_BIT)
      ? entry.cpState.sc
      : entry.gpState.sc;

    uint32_t specConstantMask = 0;

    if (!data.read(specConstantMask, version))
      return false;

    for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
      if (specConstantMask & (1 << i)) {
        if (!data.read(sc.specConstants[i], version))
          return false;
      }
    }

    return true;
  }


  void DxvkStateCache::writeCacheEntry(
          std::ostream&             stream, 
          DxvkStateCacheEntry&      entry) const {
    DxvkStateCacheEntryData data;
    VkShaderStageFlags stageMask = 0;

    // Write shader hashes
    auto keys = &entry.shaders.vs;

    for (uint32_t i = 0; i < 6; i++) {
      if (!keys[i].eq(g_nullShaderKey)) {
        stageMask |= VkShaderStageFlagBits(1 << i);
        data.write(keys[i]);
      }
    }

    if (stageMask & VK_SHADER_STAGE_COMPUTE_BIT) {
      // Nothing else here to write out
      data.write(entry.cpState.bsBindingMask);
    } else {
      // Pack render pass format
      data.write(uint8_t(entry.format.sampleCount));
      data.write(uint8_t(entry.format.depth.format));
      data.write(packImageLayout(entry.format.depth.layout));

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        data.write(uint8_t(entry.format.color[i].format));
        data.write(packImageLayout(entry.format.color[i].layout));
      }

      data.write(uint8_t(entry.format.viewMask));

      // Write out common pipeline state
      data.write(entry.gpState.bsBindingMask);
      data.write(entry.gpState.ia);
      data.write(entry.gpState.il);
      data.write(entry.gpState.rs);
      data.write(entry.gpState.ms);
      data.write(entry.gpState.ds);
      data.write(entry.gpState.om);
      data.write(entry.gpState.dsFront);
      data.write(entry.gpState.dsBack);

      // Write out render target swizzles and blend info
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++)
        data.write(entry.gpState.omSwizzle[i]);

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++)
        data.write(entry.gpState.omBlend[i]);

      // Write out input layout for defined attributes
      for (uint32_t i = 0; i < entry.gpState.il.attributeCount(); i++)
        data.write(entry.gpState.ilAttributes[i]);

      for (uint32_t i = 0; i < entry.gpState.il.bindingCount(); i++)
        data.write(entry.gpState.ilBindings[i]);
    }

    // Write out all non-zero spec constants
    auto& sc = (stageMask & VK_SHADER_STAGE_COMPUTE_BIT)
      ? entry.cpState.sc
      : entry.gpState.sc;

    uint32_t specConstantMask = 0;

    for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
      specConstantMask |= sc.specConstants[i] ? (1 << i) : 0;

    data.write(specConstantMask);

    for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
      if (specConstantMask & (1 << i))
        data.write(sc.specConstants[i]);
    }

    // General layout: header -> hash -> data
    DxvkStateCacheEntryHeader header;
    header.stageMask = uint8_t(stageMask);
    header.entrySize = data.size();

    Sha1Hash hash = data.computeHash();

    stream.write(reinterpret_cast<char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<char*>(&hash), sizeof(hash));
    stream.write(data.data(), data.size());
    stream.flush();
  }


  bool DxvkStateCache::convertEntryV2(
          DxvkStateCacheEntryV4&    entry) const {
    // Semantics changed:
    // v2: rsDepthClampEnable
    // v3: rsDepthClipEnable
    entry.gpState.rsDepthClipEnable = !entry.gpState.rsDepthClipEnable;

    // Frontend changed: Depth bias
    // will typically be disabled
    entry.gpState.rsDepthBiasEnable = VK_FALSE;
    return true;
  }


  bool DxvkStateCache::convertEntryV4(
    const DxvkStateCacheEntryV4&    in,
          DxvkStateCacheEntryV6&    out) const {
    out.shaders = in.shaders;
    out.format  = in.format;
    out.hash    = in.hash;

    out.cpState.bsBindingMask           = in.cpState.bsBindingMask;
    out.gpState.bsBindingMask           = in.gpState.bsBindingMask;
    
    out.gpState.iaPrimitiveTopology     = in.gpState.iaPrimitiveTopology;
    out.gpState.iaPrimitiveRestart      = in.gpState.iaPrimitiveRestart;
    out.gpState.iaPatchVertexCount      = in.gpState.iaPatchVertexCount;
    
    out.gpState.ilAttributeCount        = in.gpState.ilAttributeCount;
    out.gpState.ilBindingCount          = in.gpState.ilBindingCount;

    for (uint32_t i = 0; i < in.gpState.ilAttributeCount; i++)
      out.gpState.ilAttributes[i]       = in.gpState.ilAttributes[i];

    for (uint32_t i = 0; i < in.gpState.ilBindingCount; i++) {
      out.gpState.ilBindings[i]         = in.gpState.ilBindings[i];
      out.gpState.ilDivisors[i]         = in.gpState.ilDivisors[i];
    }
    
    out.gpState.rsDepthClipEnable       = in.gpState.rsDepthClipEnable;
    out.gpState.rsDepthBiasEnable       = in.gpState.rsDepthBiasEnable;
    out.gpState.rsPolygonMode           = in.gpState.rsPolygonMode;
    out.gpState.rsCullMode              = in.gpState.rsCullMode;
    out.gpState.rsFrontFace             = in.gpState.rsFrontFace;
    out.gpState.rsViewportCount         = in.gpState.rsViewportCount;
    out.gpState.rsSampleCount           = in.gpState.rsSampleCount;
    
    out.gpState.msSampleCount           = in.gpState.msSampleCount;
    out.gpState.msSampleMask            = in.gpState.msSampleMask;
    out.gpState.msEnableAlphaToCoverage = in.gpState.msEnableAlphaToCoverage;
    
    out.gpState.dsEnableDepthTest       = in.gpState.dsEnableDepthTest;
    out.gpState.dsEnableDepthWrite      = in.gpState.dsEnableDepthWrite;
    out.gpState.dsEnableStencilTest     = in.gpState.dsEnableStencilTest;
    out.gpState.dsDepthCompareOp        = in.gpState.dsDepthCompareOp;
    out.gpState.dsStencilOpFront        = in.gpState.dsStencilOpFront;
    out.gpState.dsStencilOpBack         = in.gpState.dsStencilOpBack;
    
    out.gpState.omEnableLogicOp         = in.gpState.omEnableLogicOp;
    out.gpState.omLogicOp               = in.gpState.omLogicOp;

    for (uint32_t i = 0; i < 8; i++) {
      out.gpState.omBlendAttachments[i] = in.gpState.omBlendAttachments[i];
      out.gpState.omComponentMapping[i] = in.gpState.omComponentMapping[i];
    }

    return true;
  }


  bool DxvkStateCache::convertEntryV5(
    const DxvkStateCacheEntryV5&    in,
          DxvkStateCacheEntryV6&    out) const {
    out.shaders = in.shaders;
    out.gpState = in.gpState;
    out.format  = in.format;
    out.hash    = in.hash;

    out.cpState.bsBindingMask = in.cpState.bsBindingMask;
    return true;
  }


  bool DxvkStateCache::convertEntryV6(
    const DxvkStateCacheEntryV6&    in,
          DxvkStateCacheEntry&      out) const {
    out.shaders = in.shaders;
    out.format  = in.format;
    out.hash    = in.hash;

    if (in.shaders.cs.eq(g_nullShaderKey)) {
      // Binding mask
      out.gpState.bsBindingMask = in.gpState.bsBindingMask.convert();

      // Graphics state
      out.gpState.ia = DxvkIaInfo(
        in.gpState.iaPrimitiveTopology,
        in.gpState.iaPrimitiveRestart,
        in.gpState.iaPatchVertexCount);
      
      out.gpState.il = DxvkIlInfo(
        in.gpState.ilAttributeCount,
        in.gpState.ilBindingCount);
      
      for (uint32_t i = 0; i < in.gpState.ilAttributeCount; i++) {
        out.gpState.ilAttributes[i] = DxvkIlAttribute(
          in.gpState.ilAttributes[i].location,
          in.gpState.ilAttributes[i].binding,
          in.gpState.ilAttributes[i].format,
          in.gpState.ilAttributes[i].offset);
      }
      
      for (uint32_t i = 0; i < in.gpState.ilBindingCount; i++) {
        out.gpState.ilBindings[i] = DxvkIlBinding(
          in.gpState.ilBindings[i].binding,
          in.gpState.ilBindings[i].stride,
          in.gpState.ilBindings[i].inputRate,
          in.gpState.ilDivisors[i]);
      }
      
      out.gpState.rs = DxvkRsInfo(
        in.gpState.rsDepthClipEnable,
        in.gpState.rsDepthBiasEnable,
        in.gpState.rsPolygonMode,
        in.gpState.rsCullMode,
        in.gpState.rsFrontFace,
        in.gpState.rsViewportCount,
        in.gpState.rsSampleCount,
        VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);

      out.gpState.ms = DxvkMsInfo(
        in.gpState.msSampleCount,
        in.gpState.msSampleMask,
        in.gpState.msEnableAlphaToCoverage);
      
      out.gpState.ds = DxvkDsInfo(
        in.gpState.dsEnableDepthTest,
        in.gpState.dsEnableDepthWrite,
        in.gpState.dsEnableDepthBoundsTest,
        in.gpState.dsEnableStencilTest,
        in.gpState.dsDepthCompareOp);
      
      out.gpState.dsFront = DxvkDsStencilOp(in.gpState.dsStencilOpFront);
      out.gpState.dsBack  = DxvkDsStencilOp(in.gpState.dsStencilOpBack);

      out.gpState.om = DxvkOmInfo(
        in.gpState.omEnableLogicOp,
        in.gpState.omLogicOp);
      
      for (uint32_t i = 0; i < 8 && i < MaxNumRenderTargets; i++) {
        out.gpState.omBlend[i] = DxvkOmAttachmentBlend(
          in.gpState.omBlendAttachments[i].blendEnable,
          in.gpState.omBlendAttachments[i].srcColorBlendFactor,
          in.gpState.omBlendAttachments[i].dstColorBlendFactor,
          in.gpState.omBlendAttachments[i].colorBlendOp,
          in.gpState.omBlendAttachments[i].srcAlphaBlendFactor,
          in.gpState.omBlendAttachments[i].dstAlphaBlendFactor,
          in.gpState.omBlendAttachments[i].alphaBlendOp,
          in.gpState.omBlendAttachments[i].colorWriteMask);
        
        out.gpState.omSwizzle[i] = DxvkOmAttachmentSwizzle(
          in.gpState.omComponentMapping[i]);
      }

      // Specialization constants
      for (uint32_t i = 0; i < 8 && i < MaxNumSpecConstants; i++)
        out.cpState.sc.specConstants[i] = in.cpState.scSpecConstants[i];
    } else {
      // Binding mask
      out.cpState.bsBindingMask = in.cpState.bsBindingMask.convert();

      for (uint32_t i = 0; i < 8 && i < MaxNumSpecConstants; i++)
        out.gpState.sc.specConstants[i] = in.gpState.scSpecConstants[i];
    }

    return true;
  }


  void DxvkStateCache::workerFunc() {
    env::setThreadName("dxvk-shader");

    while (!m_stopThreads.load()) {
      WorkerItem item;

      { std::unique_lock<dxvk::mutex> lock(m_workerLock);

        if (m_workerQueue.empty()) {
          m_workerBusy -= 1;
          m_workerCond.wait(lock, [this] () {
            return m_workerQueue.size()
                || m_stopThreads.load();
          });

          if (!m_workerQueue.empty())
            m_workerBusy += 1;
        }

        if (m_workerQueue.empty())
          break;
        
        item = m_workerQueue.front();
        m_workerQueue.pop();
      }

      compilePipelines(item);

      // LB_PIPE_LOG=2: when the queue runs dry, so the log shows whether the replay of the cache
      // finished before play started (the line lands among the frame lines)
      if (dxvkPipeLogLevel() >= 2) {
        uint32_t items = m_itemsSinceDrain.fetch_add(1) + 1;
        bool drained;

        { std::lock_guard<dxvk::mutex> lock(m_workerLock);
          drained = m_workerQueue.empty();
        }

        if (drained && m_itemsSinceDrain.compare_exchange_strong(items, 0u)) {
          double ms = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - m_startTime).count();
          std::fprintf(stderr, "pipe cache drained: %u shader sets since the last drain, %.0f ms after start\n",
            items, ms);
        }
      }
    }
  }


  void DxvkStateCache::writerFunc() {
    env::setThreadName("dxvk-writer");

    std::ofstream file;

    while (!m_stopThreads.load()) {
      DxvkStateCacheEntry entry;

      { std::unique_lock<dxvk::mutex> lock(m_writerLock);

        m_writerCond.wait(lock, [this] () {
          return m_writerQueue.size()
              || m_stopThreads.load();
        });

        if (m_writerQueue.size() == 0)
          break;

        entry = m_writerQueue.front();
        m_writerQueue.pop();
      }

      if (!file.is_open()) {
        file.open(getCacheFileName().c_str(),
          std::ios_base::binary |
          std::ios_base::app);
      }

      writeCacheEntry(file, entry);

      if (dxvkPipeLogLevel() >= 2 && entry.shaders.cs.eq(g_nullShaderKey)) {
        std::fprintf(stderr, "pipe cache wrote: %s\n",
          pipelineKey(entry.shaders, entry.gpState, entry.format).c_str());
      }
    }
  }

#ifdef _WIN32
  std::wstring DxvkStateCache::getCacheFileName() const {
#else
  std::string DxvkStateCache::getCacheFileName() const {
#endif
    std::string path = getCacheDir();

    if (!path.empty() && *path.rbegin() != '/')
      path += '/';
    
    std::string exeName = env::getExeBaseName();
    path += exeName + ".dxvk-cache";

#ifdef _WIN32
    return str::tows(path.c_str());
#else
    return path;
#endif
  }


  std::string DxvkStateCache::getCacheDir() const {
    return env::getEnvVar("DXVK_STATE_CACHE_PATH");
  }


  uint8_t DxvkStateCache::packImageLayout(
          VkImageLayout             layout) {
    switch (layout) {
      case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL: return 0x80;
      case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL: return 0x81;
      default: return uint8_t(layout);
    }
  }


  VkImageLayout DxvkStateCache::unpackImageLayout(
          uint8_t                   layout) {
    switch (layout) {
      case 0x80: return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
      case 0x81: return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
      default: return VkImageLayout(layout);
    }
  }


  bool DxvkStateCache::validateRenderPassFormat(
    const DxvkRenderPassFormat&     format) {
    // The render pass code supports up to 8 views (a uint8_t in the cache file).
    bool valid = format.viewMask < 0x100;

    if (format.depth.format) {
      valid &= format.depth.layout == VK_IMAGE_LAYOUT_GENERAL
            || format.depth.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || format.depth.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            || format.depth.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
            || format.depth.layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
    }

    for (uint32_t i = 0; i < MaxNumRenderTargets && valid; i++) {
      if (format.color[i].format) {
        valid &= format.color[i].layout == VK_IMAGE_LAYOUT_GENERAL
              || format.color[i].layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      }
    }

    return valid;
  }

}
