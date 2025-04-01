
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/imaging/BufferedImage.h"

namespace Ifrit::Graphics::SoftGraphics::Imaging
{
    class LayeredBufferedImage
    {
    private:
        std::vector<std::shared_ptr<BufferedImage>> layers;

    public:
        LayeredBufferedImage()  = default;
        ~LayeredBufferedImage() = default;

        void                  addLayer(std::shared_ptr<BufferedImage> layer);
        inline BufferedImage& getLayer(int index) const { return *layers[index]; }
        inline int            getLayerCount() const
        {
            using namespace Ifrit;
            return SizeCast<int>(layers.size());
        }
    };
} // namespace Ifrit::Graphics::SoftGraphics::Imaging