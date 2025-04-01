
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
#include "ifrit/runtime/util/TimingRecorder.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Runtime
{
    class IApplication
    {
    public:
        virtual void                             OnStart()  = 0;
        virtual void                             OnUpdate() = 0;
        virtual void                             OnEnd()    = 0;

        virtual Graphics::Rhi::RhiBackend*       GetRhi()                   = 0;
        virtual Display::Window::WindowProvider* GetDisplay()               = 0;
        virtual String                           GetCacheDir() const        = 0;
        virtual TimingRecorder*                  GetTimeRecorder()          = 0;
        virtual const ProjectProperty&           GetProjectProperty() const = 0;
    };
} // namespace Ifrit::Runtime