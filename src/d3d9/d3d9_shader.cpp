#include "d3d9_shader.h"

#include "d3d9_device.h"
#include "d3d9_util.h"

#include "../dxvk/dxvk_state_cache.h"

#include <cstdio>

namespace dxvk {

#ifndef DXVK_NATIVE
  typedef HRESULT (STDMETHODCALLTYPE *D3DXDisassembleShader) (
    const void*      pShader, 
          BOOL       EnableColorCode, 
          char*      pComments, 
          ID3DBlob** ppDisassembly); // ppDisassembly is actually a D3DXBUFFER, but it has the exact same vtable as a ID3DBlob at the start.

  D3DXDisassembleShader g_pfnDisassembleShader = nullptr;

  HRESULT DisassembleShader(
    const void*      pShader, 
          BOOL       EnableColorCode, 
          char*      pComments, 
          ID3DBlob** ppDisassembly) {
    if (g_pfnDisassembleShader == nullptr) {
      HMODULE d3d9x = LoadLibraryA("d3dx9.dll");

      if (d3d9x == nullptr)
        d3d9x = LoadLibraryA("d3dx9_43.dll");

      g_pfnDisassembleShader = 
        reinterpret_cast<D3DXDisassembleShader>(GetProcAddress(d3d9x, "D3DXDisassembleShader"));
    }

    if (g_pfnDisassembleShader == nullptr)
      return D3DERR_INVALIDCALL;

    return g_pfnDisassembleShader(
      pShader,
      EnableColorCode,
      pComments,
      ppDisassembly);
  }
#endif


  D3D9CommonShader::D3D9CommonShader() {}

  D3D9CommonShader::D3D9CommonShader(
            D3D9DeviceEx*         pDevice,
            VkShaderStageFlagBits ShaderStage,
      const DxvkShaderKey&        Key,
      const DxsoModuleInfo*       pDxsoModuleInfo,
      const void*                 pShaderBytecode,
      const DxsoAnalysisInfo&     AnalysisInfo,
            DxsoModule*           pModule) {
    const uint32_t bytecodeLength = AnalysisInfo.bytecodeByteLength;
    m_bytecode.resize(bytecodeLength);
    std::memcpy(m_bytecode.data(), pShaderBytecode, bytecodeLength);

    const std::string name = Key.toString();
    Logger::debug(str::format("Compiling shader ", name));
    
    const std::string dumpPath = env::getEnvVar("DXVK_SHADER_DUMP_PATH");
#ifndef DXVK_NATIVE
    // If requested by the user, dump both the raw DXBC
    // shader and the compiled SPIR-V module to a file.
    if (dumpPath.size() != 0) {
      DxsoReader reader(
        reinterpret_cast<const char*>(pShaderBytecode));

      reader.store(std::ofstream(str::tows(str::format(dumpPath, "/", name, ".dxso").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc), bytecodeLength);

      char comment[2048];
      Com<ID3DBlob> blob;
      HRESULT hr = DisassembleShader(
        pShaderBytecode,
        TRUE,
        comment, 
        &blob);
      
      if (SUCCEEDED(hr)) {
        std::ofstream disassembledOut(str::tows(str::format(dumpPath, "/", name, ".dxso.dis").c_str()).c_str(), std::ios_base::binary | std::ios_base::trunc);
        disassembledOut.write(
          reinterpret_cast<const char*>(blob->GetBufferPointer()),
          blob->GetBufferSize());
      }
    }
#endif
    
    // Decide whether we need to create a pass-through
    // geometry shader for vertex shader stream output

    const D3D9ConstantLayout& constantLayout = ShaderStage == VK_SHADER_STAGE_VERTEX_BIT
      ? pDevice->GetVertexConstantLayout()
      : pDevice->GetPixelConstantLayout();
    m_shaders      = pModule->compile(*pDxsoModuleInfo, name, AnalysisInfo, constantLayout);
    m_isgn         = pModule->isgn();
    m_usedSamplers = pModule->usedSamplers();

    // Shift up these sampler bits so we can just
    // do an or per-draw in the device.
    // We shift by 17 because 16 ps samplers + 1 dmap (tess)
    if (ShaderStage == VK_SHADER_STAGE_VERTEX_BIT)
      m_usedSamplers <<= 17;

    m_usedRTs      = pModule->usedRTs();

    m_info      = pModule->info();
    m_meta      = pModule->meta();
    m_constants = pModule->constants();

    m_shaders[0]->setShaderKey(Key);

    if (m_shaders[1] != nullptr) {
      // Lets lie about the shader key type for the state cache.
      m_shaders[1]->setShaderKey({ VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, Key.sha1() });
    }

    if (dumpPath.size() != 0) {
#ifdef _WIN32
      std::ofstream dumpStream(
        str::tows(str::format(dumpPath, "/", name, ".spv").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc);
#else
      std::ofstream dumpStream(
        str::format(dumpPath, "/", name, ".spv").c_str(),
        std::ios_base::binary | std::ios_base::trunc);
#endif
      
      m_shaders[0]->dump(dumpStream);
    }

    pDevice->GetDXVKDevice()->registerShader(m_shaders[0]);

    if (m_shaders[1] != nullptr)
      pDevice->GetDXVKDevice()->registerShader(m_shaders[1]);
  }


  void D3D9ShaderModuleSet::GetShaderModule(
            D3D9DeviceEx*         pDevice,
            D3D9CommonShader*     pShaderModule,
            VkShaderStageFlagBits ShaderStage,
      const DxsoModuleInfo*       pDxbcModuleInfo,
      const void*                 pShaderBytecode) {
    DxsoReader reader(
      reinterpret_cast<const char*>(pShaderBytecode));

    DxsoModule module(reader);

    if (module.info().majorVersion() > pDxbcModuleInfo->options.shaderModel)
      throw DxvkError("GetShaderModule: Out of range of supported shader model");

    if (module.info().shaderStage() != ShaderStage)
      throw DxvkError("GetShaderModule: Bytecode does not match shader stage");

    DxsoAnalysisInfo info = module.analyze();

    DxvkShaderKey lookupKey = DxvkShaderKey(
      ShaderStage,
      Sha1Hash::compute(pShaderBytecode, info.bytecodeByteLength));

    // Use the shader's unique key for the lookup
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto entry = m_modules.find(lookupKey);
      if (entry != m_modules.end()) {
        *pShaderModule = entry->second;
        return;
      }
    }
    
    // This shader has not been compiled yet, so we have to create a
    // new module. This takes a while, so we won't lock the structure.
    *pShaderModule = D3D9CommonShader(
      pDevice, ShaderStage, lookupKey,
      pDxbcModuleInfo, pShaderBytecode,
      info, &module);
    
    // Insert the new module into the lookup table. If another thread
    // has compiled the same shader in the meantime, we should return
    // that object instead and discard the newly created module.
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto status = m_modules.insert({ lookupKey, *pShaderModule });
      if (!status.second) {
        *pShaderModule = status.first->second;
        return;
      }
    }

    AppendBytecode(ShaderStage, lookupKey, pShaderBytecode, info.bytecodeByteLength);
  }


  /* Bytecode cache file (Apple port): a header, then per shader its stage, byte length,
   * the SHA-1 of the bytecode (also its DxvkShaderKey hash) and the bytecode itself. */
  struct D3D9BytecodeCacheHeader {
    char     magic[4] = { 'D', 'X', 'S', 'O' };
    uint32_t version  = 1;
  };

  struct D3D9BytecodeCacheEntryHeader {
    uint32_t stage;
    uint32_t byteLength;
    Sha1Hash hash;
  };

  // Larger than any shader model 3 shader; a corrupt length must not allocate gigabytes
  constexpr uint32_t MaxCachedBytecodeLength = 1u << 20;


  D3D9ShaderModuleSet::~D3D9ShaderModuleSet() {
    StopBytecodeCache();
  }


  void D3D9ShaderModuleSet::StartBytecodeCache(
          D3D9DeviceEx*         pDevice,
    const DxsoModuleInfo&       ModuleInfo) {
    if (env::getEnvVar("DXVK_STATE_CACHE") == "0"
     || env::getEnvVar("DXVK_SHADER_BYTECODE_CACHE") == "0"
     || !pDevice->GetDXVKDevice()->config().enableStateCache)
      return;

    std::string dir = env::getEnvVar("DXVK_STATE_CACHE_PATH");
    if (!dir.empty() && dir.back() != '/')
      dir += '/';
    m_cachePath = dir + env::getExeBaseName() + ".dxso-cache";

    // Read what earlier runs created; stop at the first entry that doesn't check out
    std::vector<std::pair<VkShaderStageFlagBits, std::vector<char>>> shaders;
    bool rewrite = true;

    { std::ifstream file(m_cachePath, std::ios_base::binary);
      D3D9BytecodeCacheHeader expected, header;

      if (file && file.read(reinterpret_cast<char*>(&header), sizeof(header))
       && !std::memcmp(header.magic, expected.magic, sizeof(header.magic))
       && header.version == expected.version) {
        rewrite = false;
        D3D9BytecodeCacheEntryHeader entry;

        while (file.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
          bool valid = (entry.stage == VK_SHADER_STAGE_VERTEX_BIT || entry.stage == VK_SHADER_STAGE_FRAGMENT_BIT)
                    && entry.byteLength && entry.byteLength <= MaxCachedBytecodeLength;
          std::vector<char> code(valid ? entry.byteLength : 0);

          if (!valid || !file.read(code.data(), code.size())
           || Sha1Hash::compute(code.data(), code.size()) != entry.hash) {
            Logger::warn("D3D9: Bytecode cache: invalid entry, rewriting the file");
            rewrite = true;
            break;
          }

          DxvkShaderKey key(VkShaderStageFlagBits(entry.stage), entry.hash);
          if (m_cachedKeys.insert(key).second)
            shaders.emplace_back(VkShaderStageFlagBits(entry.stage), std::move(code));
        }
      }
    }

    if (rewrite) {
      // New, outdated or damaged: start over, keeping the shaders read before the damage
      m_cacheFile.open(m_cachePath, std::ios_base::binary | std::ios_base::trunc);

      if (!m_cacheFile && env::createDirectory(dir))
        m_cacheFile.open(m_cachePath, std::ios_base::binary | std::ios_base::trunc);

      D3D9BytecodeCacheHeader header;
      m_cacheFile.write(reinterpret_cast<const char*>(&header), sizeof(header));

      for (const auto& s : shaders) {
        D3D9BytecodeCacheEntryHeader entry;
        entry.stage      = s.first;
        entry.byteLength = uint32_t(s.second.size());
        entry.hash       = Sha1Hash::compute(s.second.data(), s.second.size());
        m_cacheFile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        m_cacheFile.write(s.second.data(), s.second.size());
      }

      m_cacheFile.flush();
    } else {
      m_cacheFile.open(m_cachePath, std::ios_base::binary | std::ios_base::app);
    }

    if (!m_cacheFile) {
      Logger::warn(str::format("D3D9: Bytecode cache: cannot write ", m_cachePath));
      m_cacheFile = std::ofstream();
    }

    m_cacheEnabled = true;
    Logger::info(str::format("D3D9: Bytecode cache: creating ", shaders.size(), " shaders of earlier runs"));

    if (!shaders.empty()) {
      m_cacheThread = dxvk::thread([this, pDevice, ModuleInfo, list = std::move(shaders)] () mutable {
        CreateCachedShaders(pDevice, ModuleInfo, std::move(list));
      });
    }
  }


  void D3D9ShaderModuleSet::StopBytecodeCache() {
    m_stopCaching.store(true);

    if (m_cacheThread.joinable())
      m_cacheThread.join();
  }


  void D3D9ShaderModuleSet::CreateCachedShaders(
          D3D9DeviceEx*         pDevice,
          DxsoModuleInfo        ModuleInfo,
          std::vector<std::pair<VkShaderStageFlagBits, std::vector<char>>> Shaders) {
    env::setThreadName("dxvk-dxso-cache");

    auto t0 = dxvk::high_resolution_clock::now();
    uint32_t created = 0;

    for (const auto& s : Shaders) {
      if (m_stopCaching.load())
        return;

      try {
        D3D9CommonShader module;
        GetShaderModule(pDevice, &module, s.first, &ModuleInfo, s.second.data());
        created += 1;
      } catch (const DxvkError& e) {
        Logger::warn(str::format("D3D9: Bytecode cache: ", e.message()));
      }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dxvk::high_resolution_clock::now() - t0);
    Logger::info(str::format("D3D9: Bytecode cache: created ", created, " shaders in ", ms.count(), " ms"));
  }


  void D3D9ShaderModuleSet::AppendBytecode(
          VkShaderStageFlagBits ShaderStage,
    const DxvkShaderKey&        Key,
    const void*                 pShaderBytecode,
          uint32_t              ByteLength) {
    if (!m_cacheEnabled || ByteLength == 0 || ByteLength > MaxCachedBytecodeLength)
      return;

    std::lock_guard<dxvk::mutex> lock(m_cacheMutex);

    if (!m_cachedKeys.insert(Key).second)
      return;

    // LB_PIPE_LOG=2: a shader no earlier run created, so the state cache can't have compiled its pipelines
    if (dxvkPipeLogLevel() >= 2) {
      std::fprintf(stderr, "dxso shader new this run: %s %s, %u bytes\n",
        ShaderStage == VK_SHADER_STAGE_VERTEX_BIT ? "vs" : "ps",
        Key.sha1().toString().substr(0, 8).c_str(), ByteLength);
    }

    if (!m_cacheFile.is_open())
      return;

    D3D9BytecodeCacheEntryHeader entry;
    entry.stage      = ShaderStage;
    entry.byteLength = ByteLength;
    entry.hash       = Key.sha1();

    m_cacheFile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    m_cacheFile.write(reinterpret_cast<const char*>(pShaderBytecode), ByteLength);
    m_cacheFile.flush();
  }

}
