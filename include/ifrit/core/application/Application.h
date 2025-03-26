
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
#include "ifrit/core/application/ProjectProperty.h"
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/input/InputSystem.h"
#include "ifrit/core/scene/SceneAssetManager.h"
#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/core/util/TimingRecorder.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <string>

namespace Ifrit::Core
{

    class IFRIT_APIDECL Application : public IApplication
    {
        using RhiBackend     = Graphics::Rhi::RhiBackend;
        using WindowProvider = Display::Window::WindowProvider;

    protected:
        Uref<RhiBackend>       m_rhiLayer; // should be destroyed last

        Ref<AssetManager>      m_assetManager;
        Ref<SceneManager>      m_sceneManager;
        Ref<SceneAssetManager> m_sceneAssetManager;
        Ref<InputSystem>       m_inputSystem;
        Ref<TimingRecorder>    m_timingRecorder;
        Uref<WindowProvider>   m_windowProvider;
        ProjectProperty        m_info;

    private:
        void        Start();
        void        Update();
        void        End();
        inline bool ApplicationShouldClose() { return true; }

    public:
        virtual void                   OnStart() override {}
        virtual void                   OnUpdate() override {}
        virtual void                   OnEnd() override {}
        void                           Run(const ProjectProperty& info);

        inline virtual RhiBackend*     GetRhi() override { return m_rhiLayer.get(); }
        inline virtual WindowProvider* GetDisplay() override { return m_windowProvider.get(); }
        inline String                  GetCacheDir() const override { return m_info.m_cachePath; }
        inline TimingRecorder*         GetTimeRecorder() override { return m_timingRecorder.get(); }
        inline const ProjectProperty&  GetProjectProperty() const override { return m_info; }
    };
} // namespace Ifrit::Core