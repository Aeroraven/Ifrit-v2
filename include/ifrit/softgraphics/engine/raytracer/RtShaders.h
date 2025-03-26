
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
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include <stack>

namespace Ifrit::Graphics::SoftGraphics::Raytracer
{
    // v2
    struct RaytracerShaderStackElement
    {
        RayInternal ray;
        RayHit      rayHit;
        void*       payloadPtr;
    };

    class IFRIT_APIDECL RaytracerShaderExecutionStack
    {
    protected:
        std::vector<RaytracerShaderStackElement> execStack;

    public:
        IFRIT_HOST void         pushStack(const RayInternal& ray, const RayHit& rayHit, void* pPayload);
        IFRIT_HOST void         popStack();
        IFRIT_HOST virtual void onStackPushComplete() = 0;
        IFRIT_HOST virtual void onStackPopComplete()  = 0;
    };

    class IFRIT_APIDECL RayGenShader : public ShaderBase
    {
    public:
        IFRIT_DUAL virtual void execute(const Vector3i& inputInvocation, const Vector3i& dimension, void* context) = 0;
        IFRIT_DUAL virtual ~RayGenShader()                                                                         = default;
        IFRIT_HOST virtual RayGenShader*                    GetCudaClone() { return nullptr; };
        IFRIT_HOST virtual std::unique_ptr<RayGenShader>    getThreadLocalCopy() = 0;
        IFRIT_HOST virtual void                             updateUniformData(int binding, int set, const void* pData) {}
        IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return {}; }
    };

    class IFRIT_APIDECL MissShader : public ShaderBase, public RaytracerShaderExecutionStack
    {
    public:
        IFRIT_DUAL virtual void execute(void* context) = 0;
        IFRIT_DUAL virtual ~MissShader()               = default;
        IFRIT_HOST virtual MissShader*                      GetCudaClone() { return nullptr; };
        IFRIT_HOST virtual std::unique_ptr<MissShader>      getThreadLocalCopy() = 0;
        IFRIT_HOST virtual void                             updateUniformData(int binding, int set, const void* pData) {}
        IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return {}; }
    };

    class IFRIT_APIDECL CloseHitShader : public ShaderBase, public RaytracerShaderExecutionStack
    {
    public:
        IFRIT_DUAL virtual void execute(const RayHit& hitAttribute, const RayInternal& ray, void* context) = 0;
        IFRIT_DUAL virtual ~CloseHitShader()                                                               = default;
        IFRIT_HOST virtual CloseHitShader*                  GetCudaClone() { return nullptr; };
        IFRIT_HOST virtual std::unique_ptr<CloseHitShader>  getThreadLocalCopy() = 0;
        IFRIT_HOST virtual void                             updateUniformData(int binding, int set, const void* pData){};
        IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return {}; }
    };

    class IFRIT_APIDECL CallableShader : public ShaderBase, public RaytracerShaderExecutionStack
    {
    public:
        IFRIT_DUAL virtual void            execute(void* outPayload, void* inPayload, void* context) = 0;
        IFRIT_HOST virtual CallableShader* GetCudaClone() { return nullptr; };
    };

} // namespace Ifrit::Graphics::SoftGraphics::Raytracer