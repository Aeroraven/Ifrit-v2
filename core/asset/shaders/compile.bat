glslc  -fshader-stage=vertex demo.frag.hlsl -o demo.frag.hlsl.spv
glslc -fshader-stage=vertex demo.vert.hlsl  -o demo.vert.hlsl.spv 
glslc -fshader-stage=fragment diffuse.frag.hlsl  -o diffuse.frag.hlsl.spv 
glslc -fshader-stage=vertex diffuse.vert.hlsl    -o diffuse.vert.hlsl.spv