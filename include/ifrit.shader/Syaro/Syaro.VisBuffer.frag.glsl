#version 450

layout(location = 0) in flat uint ids; // clusterid, triangleid
layout(location = 0) out uint outColor;

void main() {
    uint triangleId = uint(gl_PrimitiveID);
    uint lowPart = triangleId & 0x0000007Fu;
    uint highPart = ((ids+1)<<7) & 0xFFFFFF80u; //1 to tell it's a valid cluster
    outColor = highPart | lowPart;
}