
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
#include "ifrit/runtime/application/ProjectProperty.h"

#include "ifrit/runtime/forwarding/FwdApp.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include "ifrit/rhi/common/RhiForwardingTypes.h"

#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/application/Subsystem.h"

namespace Ifrit::Runtime
{
    class IApplication
    {

    protected:
        virtual void* GetSubsystemInternal(u64 typeId) = 0;

    public:
        virtual void                                    OnStart()  = 0;
        virtual void                                    OnUpdate() = 0;
        virtual void                                    OnEnd()    = 0;

        virtual RHI::RhiBackend*                        GetRhi()                   = 0;
        virtual Ifrit::Display::Window::WindowProvider* GetDisplay()               = 0;
        virtual String                                  GetCacheDir() const        = 0;
        virtual TimingRecorder*                         GetTimeRecorder()          = 0;
        virtual const ProjectProperty&                  GetProjectProperty() const = 0;
        virtual ShaderRegistry*                         GetShaderRegistry()        = 0;
        virtual SharedRenderResource*                   GetSharedRenderResource()  = 0;
        virtual RendererWrapper*                        GetRendererWrapper()       = 0;

        virtual RHI::RhiTexture*                        GetDefaultColorImage() const = 0;
        virtual ApplicationState*                       GetApplicationState()        = 0;
        virtual AssetManager*                           GetAssetRegistry()           = 0;

        template <typename T>
            requires(std::is_base_of<ISubsystem, T>::value)
        T* GetSubsystem()
        {
            return ForcedCheckedCast<T>(GetSubsystemInternal(typeid(T).hash_code()));
        }
    };

    IFRIT_RUNTIME_API IApplication* GetActiveApplication();
    IFRIT_RUNTIME_API void          SetActiveApplication(IApplication* app);
} // namespace Ifrit::Runtime