#define IFRIT_DLL
#include <memory>
#include <softrenderer/include/engine/tileraster/TileRasterRenderer.h>
#include <softrenderer/include/engine/base/Shaders.h>
#include <softrenderer/include/engine/bufferman/BufferManager.h>
#include <softrenderer/include/core/data/Image.h>

class DemoVS: public Ifrit::Engine::SoftRenderer::VertexShader {
public:
    virtual void execute(const void *const *input, ifloat4 *outPos,ifloat4 *const *outVaryings){
        const float *inPos = (const float *)input[0];
        outPos->x = inPos[0];
        outPos->y = inPos[1];
        outPos->z = inPos[2];
        outPos->w = 1.0f;
    }
};

class DemoFS: public Ifrit::Engine::SoftRenderer::FragmentShader {
public:
    virtual void execute(const void *varyings, void *colorOutput,float *fragmentDepth){
        //fill red color
        float *outColor = (float *)colorOutput;
        outColor[0] = 1.0f;
        outColor[1] = 0.0f;
        outColor[2] = 0.0f;
        outColor[3] = 1.0f;
    }
};

int main(){
    using namespace Ifrit::Engine::SoftRenderer::TileRaster;
    using namespace Ifrit::Engine::SoftRenderer::Core::Data;
    using namespace Ifrit::Engine::SoftRenderer::BufferManager;

    std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
    renderer->init();
    std::unique_ptr<ImageF32> color = std::make_unique<ImageF32>(800, 600, 4);
    std::unique_ptr<ImageF32> depth = std::make_unique<ImageF32>(800, 600, 1);
    FrameBuffer frameBuffer;
    frameBuffer.setColorAttachments({color.get()}); 
    frameBuffer.setDepthAttachment(*depth.get());
    renderer->bindFrameBuffer(frameBuffer);
    VertexBuffer vertexBuffer;
    vertexBuffer.setLayout({TypeDescriptors.FLOAT4});
    vertexBuffer.setVertexCount(3);
    vertexBuffer.allocateBuffer(3);


    vertexBuffer.setValue<ifloat4>(0,0,{0.0f, 0.5f, 0.2f, 1.0f});
    vertexBuffer.setValue<ifloat4>(1,0,{0.5f, -0.5f, 0.2f, 1.0f});
    vertexBuffer.setValue<ifloat4>(2,0,{-0.5f, -0.5f, 0.2f, 1.0f});
    renderer->bindVertexBuffer(vertexBuffer);

    DemoVS vs;
    renderer->bindVertexShader(vs);
    DemoFS fs;
    renderer->bindFragmentShader(fs);
    
    std::vector<int> indexBuffer = {2,1,0};
    std::shared_ptr<TrivialBufferManager> bufferman = std::make_shared<TrivialBufferManager>();
    bufferman->init();
    auto indexBuffer1 = bufferman->createBuffer({sizeof(indexBuffer[0]) * indexBuffer.size()});
    bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0,sizeof(indexBuffer[0]) * indexBuffer.size());
    renderer->bindIndexBuffer(indexBuffer1);

    renderer->drawElements(3, true);

    // save to ppm
    std::ofstream ofs("output.ppm", std::ios::out );
    // check if the file is opened
    if (!ofs.is_open()) {
        std::cerr << "Could not open file for writing\n";
        return 1;
    }
    ofs << "P6\n"
        << 800 << " " << 600 << "\n"
        << 255 << "\n";
    for (int i = 0; i < 800 * 600; i++) {
        float* p = color->getPixelRGBAUnsafe(i%800, i/800);
        uint8_t r = p[0] * 255;
        uint8_t g = p[1] * 255;
        uint8_t b = p[2] * 255;
        ofs << r << g << b;  
    }
    ofs.close();
    return 0;
}