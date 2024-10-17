#define IFRIT_DLL
#include <memory>
#include <softrenderer/include/engine/tileraster/TileRasterRenderer.h>
#include <softrenderer/include/engine/base/Shaders.h>
#include <softrenderer/include/engine/bufferman/BufferManager.h>
#include <softrenderer/include/core/data/Image.h>
#include <display/include/presentation/backend/OpenGLBackend.h>
#include <display/include/presentation/window/GLFWWindowProvider.h>

#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>

#ifdef IFRIT_API_EXPORT
static_assert(false, "IFRIT_API_DECL is already defined");
#endif

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
        outColor[0] = 0.2f;
        outColor[1] = 0.0f;
        outColor[2] = 0.0f;
        outColor[3] = 1.0f;
    }
};

int main2(){
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

    using namespace Ifrit::Presentation::Backend;
    using namespace Ifrit::Presentation::Window;
    GLFWWindowProvider windowProvider;
    windowProvider.setup(800, 600);
    windowProvider.setTitle("Ifrit");

    OpenGLBackend backend;
    backend.setViewport(0, 0, 800, 600);

    windowProvider.loop([&](int *repCore) {
        renderer->drawElements(3, true);
        backend.updateTexture(color->getData(), 4, 800, 600);
        backend.draw();
    });
    return 0;
}

int main3(){
    using namespace Ifrit::Engine::VkRenderer;
    InitializeArguments args;
    EngineContext context(args);
    return 0;
}

int main(){
    return main2();
}