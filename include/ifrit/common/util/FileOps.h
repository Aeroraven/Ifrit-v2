
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
#include <fstream>
#include <string>

namespace Ifrit::Common::Utility {
inline std::string readTextFile(const std::string &path) {
  std::ifstream file(path);
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }
  return content;
}

inline void writeBinaryFile(const std::string &path,
                            const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  file.write(content.c_str(), content.size());
}

inline std::string readBinaryFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  std::string content;
  file.seekg(0, std::ios::end);
  content.resize(file.tellg());
  file.seekg(0, std::ios::beg);
  file.read(content.data(), content.size());
  return content;
}
} // namespace Ifrit::Common::Utility