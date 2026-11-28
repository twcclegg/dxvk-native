#pragma once

#include "d3d9_resource.h"
#include "../dxso/dxso_module.h"
#include "d3d9_shader_permutations.h"
#include "d3d9_util.h"

#include <array>
#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dxvk {


  /**
   * \brief Common shader object
   * 
   * Stores the compiled SPIR-V shader and the SHA-1
   * hash of the original DXBC shader, which can be
   * used to identify the shader.
   */
  class D3D9CommonShader {

  public:

    D3D9CommonShader();

    D3D9CommonShader(
            D3D9DeviceEx*         pDevice,
            VkShaderStageFlagBits ShaderStage,
      const DxvkShaderKey&        Key,
      const DxsoModuleInfo*       pDxbcModuleInfo,
      const void*                 pShaderBytecode,
      const DxsoAnalysisInfo&     AnalysisInfo,
            DxsoModule*           pModule);

    /**
     * \brief Registers the shaders with the state cache
     *
     * Done once per module, by the shader module set,
     * for the module it keeps.
     */
    void RegisterShaders(D3D9DeviceEx* pDevice) const;


    Rc<DxvkShader> GetShader(D3D9ShaderPermutation Permutation) const {
      return m_shaders[Permutation];
    }

    std::string GetName() const {
      return m_shaders[D3D9ShaderPermutations::None]->debugName();
    }

    const std::vector<uint8_t>& GetBytecode() const {
      return m_bytecode;
    }

    const DxsoIsgn& GetIsgn() const {
      return m_isgn;
    }

    const DxsoShaderMetaInfo& GetMeta() const { return m_meta; }
    const DxsoDefinedConstants& GetConstants() const { return m_constants; }

    D3D9ShaderMasks GetShaderMask() const { return D3D9ShaderMasks{ m_usedSamplers, m_usedRTs }; }

    const DxsoProgramInfo& GetInfo() const { return m_info; }

  private:

    DxsoIsgn              m_isgn;
    uint32_t              m_usedSamplers;
    uint32_t              m_usedRTs;

    DxsoProgramInfo       m_info;
    DxsoShaderMetaInfo    m_meta;
    DxsoDefinedConstants  m_constants;

    DxsoPermutations      m_shaders;

    std::vector<uint8_t>  m_bytecode;

  };

  /**
   * \brief Common shader interface
   * 
   * Implements methods for all D3D11*Shader
   * interfaces and stores the actual shader
   * module object.
   */
  template <typename Base>
  class D3D9Shader : public D3D9DeviceChild<Base> {

  public:

    D3D9Shader(
            D3D9DeviceEx*      pDevice,
      const D3D9CommonShader&  CommonShader)
      : D3D9DeviceChild<Base>( pDevice )
      , m_shader             ( CommonShader ) { }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) {
      if (ppvObject == nullptr)
        return E_POINTER;

      *ppvObject = nullptr;

      if (riid == __uuidof(IUnknown)
       || riid == __uuidof(Base)) {
        *ppvObject = ref(this);
        return S_OK;
      }

      Logger::warn("D3D9Shader::QueryInterface: Unknown interface query");
      Logger::warn(str::format(riid));
      return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE GetFunction(void* pOut, UINT* pSizeOfData) {
      if (pSizeOfData == nullptr)
        return D3DERR_INVALIDCALL;

      const auto& bytecode = m_shader.GetBytecode();

      if (pOut == nullptr) {
        *pSizeOfData = bytecode.size();
        return D3D_OK;
      }

      size_t copyAmount = std::min(size_t(*pSizeOfData), bytecode.size());
      std::memcpy(pOut, bytecode.data(), copyAmount);

      return D3D_OK;
    }

    const D3D9CommonShader* GetCommonShader() const {
      return &m_shader;
    }

  private:

    D3D9CommonShader m_shader;

  };

  // Needs their own classes and not usings for forward decl.

  class D3D9VertexShader final : public D3D9Shader<IDirect3DVertexShader9> {

  public:

    D3D9VertexShader(
            D3D9DeviceEx*      pDevice,
      const D3D9CommonShader&  CommonShader)
      : D3D9Shader<IDirect3DVertexShader9>( pDevice, CommonShader ) { }

  };

  class D3D9PixelShader final : public D3D9Shader<IDirect3DPixelShader9> {

  public:

    D3D9PixelShader(
            D3D9DeviceEx*      pDevice,
      const D3D9CommonShader&  CommonShader)
      : D3D9Shader<IDirect3DPixelShader9>( pDevice, CommonShader ) { }

  };

  /**
   * \brief Shader module set
   * 
   * Some applications may compile the same shader multiple
   * times, so we should cache the resulting shader modules
   * and reuse them rather than creating new ones. This
   * class is thread-safe.
   */
  class D3D9ShaderModuleSet : public RcObject {
    
  public:

    ~D3D9ShaderModuleSet();
    
    void GetShaderModule(
            D3D9DeviceEx*         pDevice,
            D3D9CommonShader*     pShaderModule,
            VkShaderStageFlagBits ShaderStage,
      const DxsoModuleInfo*       pDxbcModuleInfo,
      const void*                 pShaderBytecode);

    /**
     * \brief Starts creating the shaders of earlier runs
     *
     * Apple port. The DXVK state cache compiles a cached pipeline only once
     * all of its shaders exist, and a game that creates its shaders just
     * before it first draws with them (Skyrim does) leaves the cache's
     * compiler threads hundreds of pipelines behind, so they get compiled
     * at draw time anyway. This keeps the D3D9 bytecode of every shader the
     * game creates in <exe>.dxso-cache next to the state cache file, and on
     * the next launch creates those shaders on a background thread right
     * after the device, which lets the state cache compile their pipelines
     * while the game is still in its menus and loading screens. The game's
     * own CreateVertexShader/CreatePixelShader later find the same module.
     * Off when the state cache is off (DXVK_STATE_CACHE=0 or
     * dxvk.enableStateCache = False) or with DXVK_SHADER_BYTECODE_CACHE=0.
     *
     * Trade-off: every shader any earlier run created is created, so the
     * compiler threads compile every pipeline in the state cache file on
     * every launch, including areas, settings and mono/stereo variants this
     * session never uses, and keep them for the life of the device. Neither
     * file is ever pruned, so that work and memory grow with play history.
     * The log says how long it took ("compiler threads idle ... ms after
     * shader creation started"). Whether it needs a bound (for example
     * skipping entries no recent launch used) depends on its time and
     * memory cost on the target device.
     */
    void StartBytecodeCache(
            D3D9DeviceEx*         pDevice,
      const DxsoModuleInfo&       ModuleInfo);

    /**
     * \brief Stops the background shader creation
     *
     * Called by the device before it goes away.
     */
    void StopBytecodeCache();
    
  private:

    void AppendBytecode(
            VkShaderStageFlagBits ShaderStage,
      const DxvkShaderKey&        Key,
      const void*                 pShaderBytecode,
            uint32_t              ByteLength);

    void CreateCachedShaders(
            D3D9DeviceEx*         pDevice,
            DxsoModuleInfo        ModuleInfo,
            std::vector<std::pair<VkShaderStageFlagBits, std::vector<char>>> Shaders);
    
    dxvk::mutex m_mutex;

    // Keys of modules being created, so a second thread waits instead of compiling
    // the same shader again (Apple port; guarded by m_mutex)
    dxvk::condition_variable  m_moduleCond;
    std::unordered_set<DxvkShaderKey, DxvkHash, DxvkEq> m_modulesInFlight;

    // Bytecode cache (Apple port)
    bool                      m_cacheEnabled = false;
    std::string               m_cachePath;
    dxvk::mutex               m_cacheMutex;
    std::ofstream             m_cacheFile;
    std::unordered_set<DxvkShaderKey, DxvkHash, DxvkEq> m_cachedKeys;
    std::atomic<bool>         m_stopCaching = { false };
    dxvk::thread              m_cacheThread;
    
    std::unordered_map<
      DxvkShaderKey,
      D3D9CommonShader,
      DxvkHash, DxvkEq> m_modules;
    
  };

  template<typename T>
  const D3D9CommonShader* GetCommonShader(const T& pShader) {
    return pShader != nullptr ? pShader->GetCommonShader() : nullptr;
  }

}