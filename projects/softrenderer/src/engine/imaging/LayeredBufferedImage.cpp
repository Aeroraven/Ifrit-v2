#include "engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::Engine::Imaging {
	void LayeredBufferedImage::addLayer(std::shared_ptr<BufferedImage> layer){
		layers.push_back(layer);
	}
}