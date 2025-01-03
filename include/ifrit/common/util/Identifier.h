
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
#define UUID_SYSTEM_GENERATOR
#include "stduuid/stduuid.h"

namespace Ifrit::Common::Utility {

inline void generateUuid(std::string &id) {
  uuids::uuid idx = uuids::uuid_system_generator{}();
  id = uuids::to_string(idx);
}
} // namespace Ifrit::Common::Utility