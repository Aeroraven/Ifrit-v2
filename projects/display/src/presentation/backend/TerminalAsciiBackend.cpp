
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

#include "ifrit/display/presentation/backend/TerminalAsciiBackend.h"
#include <iomanip>
#include <iostream>

namespace Ifrit::Display::Backend {
IFRIT_APIDECL TerminalAsciiBackend::TerminalAsciiBackend(int cWid, int cHeight) {
  this->consoleWidth = cWid;
  this->consoleHeight = cHeight;
}
IFRIT_APIDECL void TerminalAsciiBackend::updateTexture(const float *image, int channels, int width, int height) {
  std::string res = "";
  for (int i = consoleHeight - 1; i >= 0; i--) {
    for (int j = 0; j < consoleWidth; j++) {
      int samplePtX = (int)(j * (width / (float)consoleWidth));
      int samplePtY = (int)(i * (height / (float)consoleHeight));
      auto colR = image[samplePtX * width * channels + samplePtY * channels + 0];
      auto colG = image[samplePtX * width * channels + samplePtY * channels + 1];
      auto colB = image[samplePtX * width * channels + samplePtY * channels + 2];
      auto luminance = 1 - (0.2126 * colR + 0.7152 * colG + 0.0722 * colB);
      luminance = (luminance * 71 + 0.5);
      auto luminInt = (int)luminance;

      res += ramp[luminInt];
    }
    res += "\n";
  }
  resultBuffer = res;
}
IFRIT_APIDECL void TerminalAsciiBackend::draw() {
  this->setCursor(0, 0, resultBuffer);
  std::cout << resultBuffer;
}

} // namespace Ifrit::Display::Backend