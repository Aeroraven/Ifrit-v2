#pragma once

#define isbSampleTexSimple(id,x,y) ((float4*)(atTexture[(id)]))[int((y)*atTextureHei[(id)])*atTextureWid[(id)]+int((x)*atTextureWid[(id)])]
#define isbReadFloat4(x) (*reinterpret_cast<const ifloat4*>((x)))

#define isbReadGsVarying(x,y) (inVaryings[(x)][(y)])
#define isbStoreGsVarying(x,y,d,v) (outVaryings[(x)*(d)+(y)]=(v))