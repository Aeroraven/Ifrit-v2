#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
	#define IFRIT_DLLEXPORT __declspec(dllexport)
	#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__MINGW64__)
	#define IFRIT_DLLEXPORT __declspec(dllexport)
	#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__clang__)
	#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
	#define IFRIT_DLLIMPORT
#elif defined(__GNUC__)
	#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
	#define IFRIT_DLLIMPORT
#else
	static_assert(false, "Unsupported compiler")
#endif


#ifdef IFRIT_COMPONENT_LLVMEXEC_EXPORT
	#define IFRIT_COM_LE_API extern "C" IFRIT_DLLEXPORT
#else 
	#define IFRIT_COM_LE_API extern "C" IFRIT_DLLIMPORT
#endif

#ifdef _WIN32
#define IFRIT_COM_LE_API_CALLCONV __stdcall
#else
#define IFRIT_COM_LE_API_CALLCONV
#endif
	struct IfritCompLLVMExecutionSession;
	IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Init();
	IFRIT_COM_LE_API IfritCompLLVMExecutionSession* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Create(const char* ir,const char* identifier);
	IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Destroy(IfritCompLLVMExecutionSession* session);
	IFRIT_COM_LE_API void* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Lookup(IfritCompLLVMExecutionSession* session, const char* symbol);
	
#ifdef __cplusplus
}
#endif