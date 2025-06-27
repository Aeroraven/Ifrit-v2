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
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime::Siro
{
    struct ParticleCollectionPrivateData;

    class IFRIT_RUNTIME_API ParticleCollection : public Mesh
    {
    private:
        ParticleCollectionPrivateData* m_Data;

    public:
        ParticleCollection(f32 numParticles);
        virtual ~ParticleCollection() = default;

    public:
        void                  Initialize();

        virtual Ref<MeshData> LoadMesh() override;
        virtual MeshData*     LoadMeshUnsafe() override;
        virtual u32           GetNumIndices() override;
        virtual u32           GetNumVertices() override;
        virtual Vec<u32>      GetIndexBufferHost() override;
        virtual Vec<Vector3f> GetVertexBufferHost() override;

    private:
        void BuildTopology();
    };
} // namespace Ifrit::Runtime::Siro