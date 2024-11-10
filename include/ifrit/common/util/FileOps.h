#include <fstream>
#include <string>

namespace Ifrit::Common::Utility {
std::string readTextFile(const std::string &path) {
  std::ifstream file(path);
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }
  return content;
}
} // namespace Ifrit::Common::Utility