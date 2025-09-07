#include "ifrit/editor/util/SceneSerialization.h"
#include <fstream>
namespace Ifrit::Editor::Util
{
    IFRIT_EDITOR_API void ExportGameObjectAsPrefab(Runtime::GameObject* gameObject, const String& filePath)
    {
        if (!gameObject)
        {
            IF_LOG_ERROR("Editor", "Cannot export null GameObject as Prefab.");
            return;
        }

        // Serialize the GameObject to a prefab format
        auto          p = gameObject->CreatePrefab();

        // Serialize to JSON or any other format as needed
        String        serializedData = p->mSerializedData;

        // Write to file
        std::ofstream outFile(filePath);
        if (outFile.is_open())
        {
            outFile << serializedData;
            outFile.close();
            IF_LOG_INFO("Editor", "Exported GameObject {} as Prefab to {}", gameObject->GetName(), filePath);
        }
        else
        {
            IF_LOG_ERROR("Editor", "Failed to open file {} for writing.", filePath);
        }
    }
} // namespace Ifrit::Editor::Util