
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
    enum SpvVMCtxEndian
    {
        IFSP_UNDEFINED     = 0,
        IFSP_LITTLE_ENDIAN = 1,
        IFSP_BIG_ENDIAN    = 2,
        IFSP_MAX_ENUM      = 3
    };
    enum SpvVMIntermediateReprAttribute
    {
        IFSP_IR_SOURCE_TYPE
    };
    enum SpvVMMatrixLayout
    {
        IFSP_MATL_UNDEF,
        IFSP_MATL_COLMAJOR,
        IFSP_MATL_ROWMAJOR
    };
    enum SpvVMIntermediateReprExpTargetType
    {
        IFSP_IRTARGET_UNSPECIFIED,
        IFSP_IRTARGET_INTERMEDIATE_UNDEF,
        IFSP_IRTARGET_STRING,
        IFSP_IRTARGET_TYPE_CONSTANT_COMPOSITE,
        IFSP_IRTARGET_TYPE_CONSTANT
    };
    enum SpvVMIntermediateReprExpTargetDeclType
    {
        IFSP_IRTARGET_DECL_UNSPECIFIED,
        IFSP_IRTARGET_DECL_VOID,
        IFSP_IRTARGET_DECL_BOOL,
        IFSP_IRTARGET_DECL_INT,
        IFSP_IRTARGET_DECL_FLOAT,
        IFSP_IRTARGET_DECL_VECTOR,
        IFSP_IRTARGET_DECL_MATRIX,
        IFSP_IRTARGET_DECL_IMAGE,
        IFSP_IRTARGET_DECL_SAMPLER,
        IFSP_IRTARGET_DECL_SAMPLED_IMAGE,
        IFSP_IRTARGET_DECL_ARRAY,
        IFSP_IRTARGET_DECL_RUNTIME_ARRAY,
        IFSP_IRTARGET_DECL_STRUCT,
        IFSP_IRTARGET_DECL_OPAQUE,
        IFSP_IRTARGET_DECL_POINTER,
        IFSP_IRTARGET_DECL_FUNCTION,

        IFSP_IRTARGET_DECL_ACCELERATION_STRUCTURE
    };
    struct SpvVMCtxInstruction
    {
        u32              opCode;
        u32              opWordCounts;
        std::vector<u32> opParams;
    };

    struct SpvVMIntermediateReprBlock
    {
        std::vector<std::string>                                literals;
        std::unordered_map<SpvVMIntermediateReprAttribute, int> attributes;
        std::vector<SpvVMIntermediateReprBlock*>                children;
    };

    struct SpvShaderExternalMappings
    {
        std::vector<std::string>         inputVarSymbols;
        std::vector<std::string>         outputVarSymbols;
        std::vector<std::string>         uniformVarSymbols;
        std::vector<std::pair<int, int>> uniformVarLoc;
        std::vector<int>                 inputSize;
        std::vector<int>                 outputSize;
        std::vector<int>                 uniformSize;
        std::string                      mainFuncSymbol;
        std::string                      builtinPositionSymbol;
        std::string                      builtinLaunchIdKHR;
        std::string                      builtinLaunchSizeKHR;
        std::string                      incomingRayPayloadKHR;
        int                              incomingRayPayloadKHRSize = 0;

        bool                             requiresInterQuadInfo = false;
    };
    struct SpvDecorationBlock
    {
        int               location = -1;
        int               binding = -1, descSet = -1;
        bool              isBuiltinPos            = false;
        bool              isBuiltinLaunchIdKHR    = false;
        bool              isBuiltinLaunchSizeKHR  = false;
        bool              isIncomingRayPayloadKHR = false;
        SpvVMMatrixLayout matrixLayout            = IFSP_MATL_COLMAJOR;
    };
    struct SpvVMIntermediateReprExpTarget
    {
        int id = -1;

        union DataRegion
        {
            bool  boolValue;
            int   intValue;
            float floatValue;
        };

        bool                                   activated     = false;
        bool                                   isFunction    = false;
        bool                                   isConstant    = false;
        bool                                   isUndef       = false;
        bool                                   isVariable    = false;
        bool                                   isLabel       = false;
        bool                                   isInstance    = false;
        bool                                   isGlobal      = false;
        bool                                   isAccessChain = false;
        bool                                   named         = false;

        bool                                   isUniform     = false;
        bool                                   isMergedBlock = false;

        SpvVMIntermediateReprExpTargetType     exprType = IFSP_IRTARGET_UNSPECIFIED;
        SpvVMIntermediateReprExpTargetDeclType declType = IFSP_IRTARGET_DECL_UNSPECIFIED;

        std::string                            name;
        std::string                            debugString;
        SpvDecorationBlock                     decoration;

        std::vector<std::string>               memberName;
        std::vector<SpvDecorationBlock>        memberDecoration;
        std::vector<int>                       memberOffset;

        int                                    intWidth;
        int                                    intSignedness;
        int                                    floatWidth;
        int                                    componentCount;

        int                                    componentTypeRef;
        std::vector<int>                       memberTypeRef;
        int                                    resultTypeRef;

        DataRegion                             data;
        int                                    storageClass;
        int                                    functionControl;

        std::vector<int>                       compositeDataRef;
        std::vector<int>                       functionParamRef;

        int                                    accessChainRef;
        std::vector<int>                       accessChain;
        int                                    offset = 0;
    };
    struct SpvVMIntermediateRepresentation
    {
        SpvVMIntermediateReprBlock*                             root = nullptr;
        std::unordered_map<u32, SpvVMIntermediateReprExpTarget> targets;
        std::unordered_map<SpvVMIntermediateReprAttribute, int> attributes;
        int                                                     addressingModel;
        int                                                     memoryModel;

        int                                                     entryPointExecutionModel;
        int                                                     entryPointExecutionMode;
        int                                                     entryPointId;
        std::string                                             entryPointName;
        std::vector<int>                                        entryPointInterfaces;

        int                                                     capability;

        /* Rewrite IR */
        int                                                     activatedFunctionRef = -1;
        int                                                     recordedFuncParams   = 0;
        int                                                     currentPass          = 0;
        int                                                     currentInst          = 0;
        std::stringstream                                       generatedIR;
        std::stringstream                                       functionInstIR;

        /* Shader Link */
        SpvShaderExternalMappings                               shaderMaps;
    };

    struct SpvVMContext
    {
        u32                              headerMagic;
        u32                              headerVersion;
        u32                              headerGenerator;
        u32                              headerBound;
        u32                              headerSchema;
        SpvVMCtxEndian                   endianBytecode;
        SpvVMCtxEndian                   endianParserNative;
        std::vector<SpvVMCtxInstruction> instructions;
    };

} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv