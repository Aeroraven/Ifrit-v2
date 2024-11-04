#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 1) buffer AtomicCounterBuffer { uvec3 counter; } counter[];
layout(set = 1, binding = 0) uniform BindlessBuffer { 
    uint atomicId;
    uint dummy1;
    uint dummy2;
    uint dummy3;
} bindless;


void main() {
    counter[bindless.atomicId].counter = uvec3(0);
}