#include "./core/definition/CoreExports.h"
#include "./engine/tileraster/TileRasterRenderer.h"
#include "./engine/export/TileRasterRendererExport.h"
#include "./engine/bufferman/BufferManager.h"

#define IFRIT_TRNS  Ifrit::Engine::TileRaster
#define IFRIT_BASENS  Ifrit::Engine
#define IFRIT_TRTP Ifrit::Engine::LibraryExport::TileRasterRendererWrapper

using namespace Ifrit::Engine;
using namespace Ifrit::Engine::LibraryExport;
using namespace Ifrit::Engine::TileRaster;

namespace Ifrit::Engine::LibraryExport {
	struct TileRasterRendererWrapper {
		std::shared_ptr<IFRIT_TRNS::TileRasterRenderer> renderer;
		std::vector<std::unique_ptr<IFRIT_BASENS::ShaderBase>> allocatedFuncWrappers;
		std::unique_ptr<std::vector<int>> allocatedIndexBuffer;
	};
	class VertexShaderFunctionalWrapper : virtual public VertexShader {
	public:
		VertexShaderFunctionalPtr func = nullptr;
		virtual void execute(const void* const* input, ifloat4* outPos, ifloat4* const* outVaryings) override {
			if(func)func(input, outPos, outVaryings);
		}
	};
	class FragmentShaderFunctionalWrapper : virtual public FragmentShader {
	public:
		FragmentShaderFunctionalPtr func = nullptr;
		virtual void execute(const void* varyings, void* colorOutput, float* fragmentDepth) override {
			if(func)func(varyings, colorOutput, fragmentDepth);
		}
	};
}
IFRIT_APIDECL_COMPAT IFRIT_TRTP* IFRIT_APICALL  iftrCreateInstance() IFRIT_EXPORT_COMPAT_NOTHROW {
	auto hInst = new TileRasterRendererWrapper();
	hInst->renderer = std::make_shared<IFRIT_TRNS::TileRasterRenderer>();
	return hInst;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyInstance(IFRIT_TRTP* hInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
	delete hInstance;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFrameBuffer(IFRIT_TRTP* hInstance, IFRIT_BASENS::FrameBuffer* frameBuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->bindFrameBuffer(*frameBuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexBuffer(IFRIT_TRTP* hInstance, const IFRIT_BASENS::VertexBuffer* vertexBuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->bindVertexBuffer(*vertexBuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindIndexBuffer(IFRIT_TRTP* hInstance, void* indexBuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
	auto p = reinterpret_cast<BufferManager::IfritBuffer*>(indexBuffer);
	hInstance->renderer->bindIndexBuffer(*p);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShaderFunc(IFRIT_TRTP* hInstance, IFRIT_BASENS::VertexShaderFunctionalPtr func,
	IFRIT_BASENS::VaryingDescriptor* vsOutDescriptors) IFRIT_EXPORT_COMPAT_NOTHROW {
	auto vsInst = std::make_unique<VertexShaderFunctionalWrapper>();
	vsInst->func = func;
	hInstance->renderer->bindVertexShaderLegacy(*vsInst, *vsOutDescriptors);
	hInstance->allocatedFuncWrappers.push_back(std::move(vsInst));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShaderFunc(IFRIT_TRTP* hInstance, IFRIT_BASENS::FragmentShaderFunctionalPtr func) IFRIT_EXPORT_COMPAT_NOTHROW {
	auto fsInst = std::make_unique<FragmentShaderFunctionalWrapper>();
	fsInst->func = func;
	hInstance->renderer->bindFragmentShader(*fsInst);
	hInstance->allocatedFuncWrappers.push_back(std::move(fsInst));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetBlendFunc(IFRIT_TRTP* hInstance, IFRIT_BASENS::IfritColorAttachmentBlendState* state) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->setBlendFunc(*state);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetDepthFunc(IFRIT_TRTP* hInstance, IFRIT_BASENS::IfritCompareOp state) IFRIT_EXPORT_COMPAT_NOTHROW{
	hInstance->renderer->setDepthFunc(state);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetForceDeterministic(IFRIT_TRTP* hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->optsetForceDeterministic(opt);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetDepthTestEnable(IFRIT_TRTP* hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->optsetDepthTestEnable(opt);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDrawLegacy(IFRIT_TRTP* hInstance, int numVertices, int clearFramebuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->drawElements(numVertices,clearFramebuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrClear(IFRIT_TRTP* hInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->clear();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrInit(IFRIT_TRTP* hInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->init();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrTest(void (*p)(int)) IFRIT_EXPORT_COMPAT_NOTHROW {
	p(114514);
}

//Update v1
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShader(IFRIT_TRTP* hInstance, void* pVertexShader) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->bindVertexShader(*(VertexShader*)pVertexShader);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShader(IFRIT_TRTP* hInstance, void* pFragmentShader) IFRIT_EXPORT_COMPAT_NOTHROW {
	hInstance->renderer->bindFragmentShader(*(FragmentShader*)pFragmentShader);
}

#undef IFRIT_TRTP
#undef IFRIT_BASENS
#undef IFRIT_TRNS