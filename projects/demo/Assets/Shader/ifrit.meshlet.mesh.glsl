#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : enable

struct Meshlet {
  uint vertex_offset;
  uint triangle_offset;
  uint vertex_count;
  uint triangle_count;
};

struct MeshletCull{
  vec4 selfSphere;
  vec4 parentSphere;
  float selfError;
  float parentError;
  uint lod;
  uint dummy;
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

layout(set = 0 , binding = 1) buffer Meshlets { Meshlet meshlets[]; } gMeshlets[];
layout(set = 0 , binding = 1) buffer MeshletVertices { uint meshlet_vertices[]; } gMeshletVert[];
layout(set = 0 , binding = 1) buffer MeshletTriangles { uint meshlet_triangles[]; } gMeshletTri[];
layout(set = 0 , binding = 1) buffer Vertices { vec4 vertices[]; } gVertices[];
layout(set = 0 , binding = 1) buffer FilteredMeshlets { uint data[]; } gCulledMeshlets[];

layout (set = 0,binding = 0) uniform UniformBufferObject {
  mat4 mvp;
  mat4 mv;
  vec4 cameraPos;
  uint meshlet_count;
  float fov;
} ubo[];

layout(set = 1, binding = 0) uniform BindlessMapping {
  uint meshletId;
  uint meshletVertId;
  uint meshletTriId;
  uint vxbufId;
  uint targetMeshId;
  uint uniformId;
  uint dummy1;
  uint dummy2;
  uint dummy3;
} bindlessMapping;

layout(location = 0) out vec3 fragColor[];

// color maps, 24 colors
vec4 colorMap[8] = vec4[8](
  vec4(1.0, 0.0, 0.0, 1.0),
  vec4(0.0, 1.0, 0.0, 1.0),
  vec4(0.0, 0.0, 1.0, 1.0),
  vec4(1.0, 1.0, 0.0, 1.0),
  vec4(1.0, 0.0, 1.0, 1.0),
  vec4(0.0, 1.0, 1.0, 1.0),
  vec4(1.0, 1.0, 1.0, 1.0),
  vec4(0.5, 0.0, 0.0, 1.0)
);


uint readTriangleIndex(uint meshletId, uint offset){
  uint offsetInUint8Local = offset;
  uint meshletTriOffset = gMeshlets[bindlessMapping.meshletId].meshlets[meshletId].triangle_offset;
  uint meshletVertOffset = gMeshlets[bindlessMapping.meshletId].meshlets[meshletId].vertex_offset;
  uint totalUint8Offset = meshletTriOffset + offsetInUint8Local;
  
  uint indexDataU32 = gMeshletTri[bindlessMapping.meshletTriId].meshlet_triangles[totalUint8Offset];
  return indexDataU32;
}

uint readVertIndex(uint meshletId, uint offset){
  uint offsetInUint8Local = offset;
  uint meshletVertOffset = gMeshlets[bindlessMapping.meshletId].meshlets[meshletId].vertex_offset;
  uint totalUint8Offset = meshletVertOffset + offsetInUint8Local;
  return gMeshletVert[bindlessMapping.meshletVertId].meshlet_vertices[totalUint8Offset];
}

float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  return 1.0 / tan(fov) * r / sqrt(d * d - r * r); 
}

void main() {
  uint mio = gl_WorkGroupID.x;
  uint mi = gCulledMeshlets[bindlessMapping.targetMeshId].data[mio]; 

  // Meshlet display
  uint totalTris = gMeshlets[bindlessMapping.meshletId].meshlets[mi].triangle_count;
  uint totalVerts = gMeshlets[bindlessMapping.meshletId].meshlets[mi].vertex_count;
  SetMeshOutputsEXT(totalVerts, totalTris);
  for(uint i=0;i<totalVerts;i++){
    uint vi = readVertIndex(mi, i);
    vec3 v0 = gVertices[bindlessMapping.vxbufId].vertices[vi].xyz;
    gl_MeshVerticesEXT[i].gl_Position = ubo[bindlessMapping.uniformId].mvp * vec4(v0.xyz, 1.0);
    //fragColor[i] = vec3(spr,spr,spr);
    fragColor[i] = colorMap[mi % 8].rgb;
  }

  for(uint i = 0; i < totalTris; i++){
    uint triIndexA = readTriangleIndex(mi, i*3 + 0);
    uint triIndexB = readTriangleIndex(mi, i*3 + 1);
    uint triIndexC = readTriangleIndex(mi, i*3 + 2);
    gl_PrimitiveTriangleIndicesEXT[i] = uvec3(triIndexA, triIndexB, triIndexC);
  }

}