#include "ifrit/runtime/physics/artemis/mpm/visualization/ProceduralMPMMesh.h"
#include "ifrit/runtime/physics/artemis/ArtemisController.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"

namespace Ifrit::Runtime::Artemis
{
    IFRIT_APIDECL void ProceduralMPMMesh::UpdateMesh(FrameGraphBuilder& builder)
    {
        auto artemisController = GetActiveApplication()->GetSubsystem<Artemis::ArtemisController>();
        auto mpmSimulator =
            reinterpret_cast<Artemis::MPMSimulator*>(artemisController->GetPresetSolver(EPresetArtemisSimulator::MPM));
        auto particleBuffer = mpmSimulator->GetParticlePositionBuffer();
        auto particleCount  = mpmSimulator->GetParticleCounterBuffer();
        Super::SetParticleData(particleBuffer, particleCount);
        Super::UpdateMesh(builder);
    }

} // namespace Ifrit::Runtime::Artemis