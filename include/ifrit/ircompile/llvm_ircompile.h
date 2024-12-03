
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


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