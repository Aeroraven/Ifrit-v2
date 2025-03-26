
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMContext.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMExtInstRegistry.h"
#include "ifrit/softgraphics/engine/shadervm/spirvvec/SpvMdBase.h"

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::SpirvVec
{
    enum SpVcQuadGroupedIRStage
    {
        SPVC_QGIR_BLOCK_GENERATION,
        SPVC_QGIR_DEFINITION,
        SPVC_QGIR_DATAFLOW_DEPENDENCY,
        SPVC_QGIR_MASKGEN,
        SPVC_QGIR_CONVERSION
    };

    class SpVcQuadGroupedIRGenerator;

    typedef void (*SpVcDefinitionPassHandler)(int pc, std::vector<u32> params, SpVcVMGeneratorContext* ctx,
        SpVcQuadGroupedIRGenerator* irg);
    typedef void (*SpVcConversionPassHandler)(int pc, std::vector<u32> params, SpVcVMGeneratorContext* ctx,
        SpVcQuadGroupedIRGenerator* irg);

    enum SpVcDataflowDependencySpecial
    {
        SPVC_DATADEPS_ORDINARY = 0x0,
        SPVC_DATADEPS_SSAPHI   = 0x1,
    };

    struct SpVcDataflowDependency
    {
        int                           retArg    = -1;
        std::vector<int>              depArgs   = {};
        int                           depVaArgs = -1;
        SpVcDataflowDependencySpecial special   = SPVC_DATADEPS_ORDINARY;
    };

    class SpVcQuadGroupedIRGenerator
    {
    private:
        using SpVcSpirBytecode = Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv::SpvVMContext;
        SpVcSpirBytecode*                                                mRaw;
        SpVcVMGeneratorContext*                                          mCtx;
        Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv::SpvVMExtRegistry mExtInstGen;

        std::unordered_map<int, SpVcDefinitionPassHandler>               mDefinitionPassHandlers;
        std::unordered_map<int, SpVcConversionPassHandler>               mConvPassHandlers;
        std::unordered_map<int, SpVcDataflowDependency>                  mArgumentDependency;
        SpVcDefinitionPassHandler                                        mUniversalDefinitionPassHandler;
        int                                                              curPc = 0;
        SpVcQuadGroupedIRStage                                           curStage;
        int                                                              curVar     = 0;
        int                                                              curVarMask = 0;

    public:
        void                                                                     bindBytecode(SpVcSpirBytecode* bytecode, SpVcVMGeneratorContext* context);

        void                                                                     performBlockPass();
        void                                                                     Init();
        void                                                                     verbose();

        // Defintion pass defines types and annotations for variables
        void                                                                     performDefinitionPass();
        void                                                                     performDefinitionPassRegister();

        // Data flow dependency resolution
        void                                                                     performDataflowResolutionPass();
        void                                                                     performDataflowResolutionPassRegister();

        // Mask injection pass
        void                                                                     performMaskInjectionPass(int quadSize);

        // Type generation
        void                                                                     performTypeGenerationPass();

        // Conversion pass converts raw bytecode to a intermediate representation,
        // with consideration of execution masks
        void                                                                     performConversionPass();
        void                                                                     performConversionPassRegister();

        // Output generates final LLVM IR code
        // void performOutputPass();
        void                                                                     performSymbolExport();

        void                                                                     parse();
        std::string                                                              generateIR();

        // Utilities
        SpVcVMGenBlock*                                                          getActiveBlock();
        void                                                                     pushActiveBlock(SpVcVMGenBlock* block);
        // void popActiveBlock();

        SpVcVMGenStack*                                                          getActiveStack();
        void                                                                     pushNewStack();
        void                                                                     popNewStack();

        SpVcVMGenVariable*                                                       createExecutionMaskVar();
        SpVcVMGenVariable*                                                       getVariableSafe(u32 id);

        std::string                                                              getParsingProgress();
        void                                                                     setCurrentProgCounter(int pc);
        void                                                                     setCurrentPass(SpVcQuadGroupedIRStage stage);
        std::string                                                              allocateLlvmVarName();
        inline Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv::SpvVMExtRegistry* getExtRegistry()
        {
            return &mExtInstGen;
        }

        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIr(std::unique_ptr<T>&& ir)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            return v;
        }
        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrG(std::unique_ptr<T>&& ir)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            mCtx->globalDefs.push_back(v);
            return v;
        }
        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrB(std::unique_ptr<T>&& ir, SpVcVMGenBlock* b)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            b->ir.push_back(v);
            return v;
        }
        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrBPre(std::unique_ptr<T>&& ir, SpVcVMGenBlock* b)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            b->irPre.push_back(v);
            return v;
        }
        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrBExist(T* ir, SpVcVMGenBlock* b)
        {
            auto v = ir;
            b->ir.push_back(v);
            return v;
        }
        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrF(std::unique_ptr<T>&& ir, SpVcVMGenBlock* b)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            b->funcBelong->ir.push_back(v);
            return v;
        }

        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrFx(std::unique_ptr<T>&& ir, SpVcVMGenFunction* f)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            f->ir.push_back(v);
            return v;
        }

        template <class T>
            requires std::is_base_of_v<LLVM::SpVcLLVMExpr, T>
        T* addIrFxTail(std::unique_ptr<T>&& ir, SpVcVMGenFunction* f)
        {
            auto v = ir.get();
            mCtx->irExprs.push_back(std::move(ir));
            f->irPost.push_back(v);
            return v;
        }

        inline int getQuads() { return 4; }
    };
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::SpirvVec