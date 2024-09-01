#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#ifdef IFRIT_COMPONENT_LLVMEXEC_EXPORT
	#define IFRIT_COM_LE_API extern "C" __declspec(dllexport)
#else 
	#define IFRIT_COM_LE_API extern "C" __declspec(dllimport)
#endif

#define IFRIT_COM_LE_API_CALLCONV __stdcall
	struct IfritCompLLVMExecutionSession;
	IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Init();
	IFRIT_COM_LE_API IfritCompLLVMExecutionSession* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Create(const char* ir,const char* identifier);
	IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Destroy(IfritCompLLVMExecutionSession* session);
	IFRIT_COM_LE_API void* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Lookup(IfritCompLLVMExecutionSession* session, const char* symbol);
	
#ifdef __cplusplus
}
#endif