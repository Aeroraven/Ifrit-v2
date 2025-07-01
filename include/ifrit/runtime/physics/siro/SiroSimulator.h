/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "ifrit/runtime/physics/siro/SiroIntegrator.h"
#include "ifrit/runtime/base/ApplicationInterface.h"

namespace Ifrit::Runtime::Siro
{

    struct SiroSimulatorPrivateData;

    class IFRIT_RUNTIME_API SiroSimulator
    {
    private:
        IApplication*             m_App;
        SiroSimulatorPrivateData* m_Data = nullptr;

    public:
        SiroSimulator(IApplication* app);
        virtual ~SiroSimulator();

        Owner<Graphics::Rhi::RhiTaskSubmission> Update(f32 deltaTime, Vec<Graphics::Rhi::RhiTaskSubmission*> waitFor);

        void                                    RegisterSolver(ISiroSolver* solver);
    };
} // namespace Ifrit::Runtime::Siro