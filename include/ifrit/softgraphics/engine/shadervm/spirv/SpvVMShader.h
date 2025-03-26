
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/raytracer/RtShaders.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMInterpreter.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
    struct SpvRuntimeSymbolTables
    {
        std::vector<void*> inputs;
        std::vector<int>   inputBytes;
        std::vector<void*> outputs;
        std::vector<int>   outputBytes;
        std::unordered_map<std::pair<int, int>, std::pair<void*, int>,
            Ifrit::Graphics::SoftGraphics::Core::Utility::PairHash>
              uniform;
        void* entry             = nullptr;
        void* builtinPosition   = nullptr;
        void* builtinLaunchId   = nullptr;
        void* builtinLaunchSize = nullptr;

        void* builtinContext      = nullptr;
        void* incomingPayload     = nullptr;
        int   incomingPayloadSize = 0;
    };
    class SpvRuntimeBackend
    {
    protected:
        static int                             CreateTime;
        SpvVMReader                            reader;
        SpvVMInterpreter                       interpreter;
        SpvVMContext                           spctx;
        SpvVMIntermediateRepresentation        spvir;
        const SpvVMIntermediateRepresentation* spvirRef;
        ShaderRuntime*                         runtime;
        SpvRuntimeSymbolTables                 symbolTables;
        std::unique_ptr<ShaderRuntime>         copiedRuntime = nullptr;
        std::string                            irCode;

        std::unique_ptr<ShaderRuntime>         owningRuntime = nullptr;

        // MinGW does not directly store the size of vector
        // it calculates the size of vector by subtracting the address of the first
        // element from the address of the last elements
        int                                    cSISize = 0;
        int                                    cSOSize = 0;
        void (*cEntry)()                               = nullptr;

    public:
        SpvRuntimeBackend(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        SpvRuntimeBackend(const SpvRuntimeBackend& other);

    protected:
        void updateSymbolTable(bool isCopy);
    };

    class SpvVertexShader final : public VertexShader, public SpvRuntimeBackend
    {
    public:
        SpvVertexShader(const SpvVertexShader& p);

    public:
        SpvVertexShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        ~SpvVertexShader() = default;
        IFRIT_DUAL virtual void                             execute(const void* const* input, Vector4f* outPos, Vector4f* const* outVaryings) override;
        IFRIT_HOST virtual VertexShader*                    GetCudaClone() override;
        IFRIT_HOST virtual std::unique_ptr<VertexShader>    getThreadLocalCopy() override;
        IFRIT_HOST virtual void                             updateUniformData(int binding, int set, const void* pData) override;
        IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
        IFRIT_HOST virtual VaryingDescriptor                getVaryingDescriptor() override;
    };

    class SpvFragmentShader final : public FragmentShader, public SpvRuntimeBackend
    {
    public:
        SpvFragmentShader(const SpvFragmentShader& p);

    public:
        SpvFragmentShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        ~SpvFragmentShader() = default;
        IFRIT_DUAL virtual void                             execute(const void* varyings, void* colorOutput, float* fragmentDepth) override;
        IFRIT_HOST virtual FragmentShader*                  GetCudaClone() override;
        IFRIT_HOST virtual std::unique_ptr<FragmentShader>  getThreadLocalCopy() override;
        IFRIT_HOST virtual void                             updateUniformData(int binding, int set, const void* pData) override;
        IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
    };

    // V2
    class SpvRaygenShader final : public Raytracer::RayGenShader, public SpvRuntimeBackend
    {
    public:
        SpvRaygenShader(const SpvRaygenShader& p);

    public:
        SpvRaygenShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        ~SpvRaygenShader() = default;
        IFRIT_DUAL virtual void                                     execute(const Vector3i& inputInvocation, const Vector3i& dimension, void* context) override;
        IFRIT_HOST virtual Raytracer::RayGenShader*                 GetCudaClone() override;
        IFRIT_HOST virtual std::unique_ptr<Raytracer::RayGenShader> getThreadLocalCopy() override;
        IFRIT_HOST virtual void                                     updateUniformData(int binding, int set, const void* pData) override;
        IFRIT_HOST virtual std::vector<std::pair<int, int>>         getUniformList() override;
    };

    class SpvMissShader final : public Raytracer::MissShader, public SpvRuntimeBackend
    {
    public:
        SpvMissShader(const SpvMissShader& p);

    private:
        IFRIT_HOST void updateStack();

    public:
        SpvMissShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        ~SpvMissShader() = default;
        IFRIT_DUAL virtual void                                   execute(void* context) override;
        IFRIT_HOST virtual Raytracer::MissShader*                 GetCudaClone() override;
        IFRIT_HOST virtual std::unique_ptr<Raytracer::MissShader> getThreadLocalCopy() override;
        IFRIT_HOST virtual void                                   updateUniformData(int binding, int set, const void* pData) override;
        IFRIT_HOST virtual std::vector<std::pair<int, int>>       getUniformList() override;
        IFRIT_HOST virtual void                                   onStackPushComplete() override;
        IFRIT_HOST virtual void                                   onStackPopComplete() override;
    };

    class SpvClosestHitShader final : public Raytracer::CloseHitShader, public SpvRuntimeBackend
    {
    public:
        SpvClosestHitShader(const SpvClosestHitShader& p);

    private:
        IFRIT_HOST void updateStack();

    public:
        SpvClosestHitShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
        ~SpvClosestHitShader() = default;
        IFRIT_DUAL virtual void                                       execute(const RayHit& hitAttribute, const RayInternal& ray, void* context) override;
        IFRIT_HOST virtual Raytracer::CloseHitShader*                 GetCudaClone() override;
        IFRIT_HOST virtual std::unique_ptr<Raytracer::CloseHitShader> getThreadLocalCopy() override;
        IFRIT_HOST virtual void                                       updateUniformData(int binding, int set, const void* pData) override;
        IFRIT_HOST virtual std::vector<std::pair<int, int>>           getUniformList() override;
        IFRIT_HOST virtual void                                       onStackPushComplete() override;
        IFRIT_HOST virtual void                                       onStackPopComplete() override;
    };
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv