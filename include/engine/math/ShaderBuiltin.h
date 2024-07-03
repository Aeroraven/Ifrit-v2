#pragma once

#define isbSampleTex(id,x,y) ((float4*)(atTexture[(id)]))[int((y)*atTextureHei[(id)])*atTextureWid[(id)]+int((x)*atTextureWid[(id)])]
#define isbReadFloat4(x) (*reinterpret_cast<const ifloat4*>((x)))

#define isbcuReadPsVarying(x,y)  ((const ifloat4s256*)(x))[(y)]
#define isbcuReadPsColorOut(x,y)  ((ifloat4s256*)(x))[(y)]