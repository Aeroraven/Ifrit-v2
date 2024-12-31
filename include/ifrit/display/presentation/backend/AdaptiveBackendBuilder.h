
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
#include "ifrit/display/presentation/backend/BackendProvider.h"
#include <memory>

namespace Ifrit::Display::Backend {
class IFRIT_APIDECL AdaptiveBackendBuilder {
public:
  std::unique_ptr<BackendProvider> buildUniqueBackend();
};
} // namespace Ifrit::Display::Backend