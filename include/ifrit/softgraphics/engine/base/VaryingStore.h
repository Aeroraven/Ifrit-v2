
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
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

namespace Ifrit::Graphics::SoftGraphics
{
	union IFRIT_APIDECL VaryingStore
	{
		float	 vf;
		int		 vi;
		u32		 vui;
		Vector2f vf2;
		Vector3f vf3;
		Vector4f vf4;
		Vector2i vi2;
		Vector3i vi3;
		Vector4i vi4;
		Vector2u vui2;
	};
} // namespace Ifrit::Graphics::SoftGraphics
