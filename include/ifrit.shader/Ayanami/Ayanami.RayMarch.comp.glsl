
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

#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.Shared.glsl"


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

RegisterUniform(bPerframe,{
    PerFramePerViewData data;
});

RegisterStorage(bMeshDFDesc,{
    MeshDFDesc data[];
});

RegisterStorage(bMeshDFMeta,{
    MeshDFMeta data;
});

RegisterUniform(bLocalTransform,{
    mat4 m_localToWorld;
    mat4 m_worldToLocal;
    vec4 m_maxScale;
});

layout(push_constant) uniform PushConstant{
    uint perframeId;
    uint totalInsts;
    uint descId;
    uint outTex;
    uint rtH;
    uint rtW;
} pc;

bool rayboxIntersection(vec3 o,vec3 d,vec3 lb,vec3 rt, out float t){
    // from: https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    vec3 dirfrac;
    dirfrac.x = 1.0f / d.x;
    dirfrac.y = 1.0f / d.y;
    dirfrac.z = 1.0f / d.z;
    float t1 = (lb.x - o.x)*dirfrac.x;
    float t2 = (rt.x - o.x)*dirfrac.x;
    float t3 = (lb.y - o.y)*dirfrac.y;
    float t4 = (rt.y - o.y)*dirfrac.y;
    float t5 = (lb.z - o.z)*dirfrac.z;
    float t6 = (rt.z - o.z)*dirfrac.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    if (tmax < 0){
        t = tmax;
        return false;
    }
    if (tmin > tmax){
        t = tmax;
        return false;
    }
    t = tmin;
    return true;
}

void main(){
    float fov = GetResource(bPerframe, pc.perframeId).data.m_cameraFovX;
    float aspect = GetResource(bPerframe, pc.perframeId).data.m_cameraAspect;
    vec3 camPos = GetResource(bPerframe, pc.perframeId).data.m_cameraPosition.xyz;
    int tX = int(gl_GlobalInvocationID.x);
    int tY = int(gl_GlobalInvocationID.y);
    if(tX >= pc.rtW || tY >= pc.rtH) return;

    float ndcX = (2.0 * (float(tX)+0.5) / float(pc.rtW) - 1.0) * aspect;
    float ndcY = 1.0 - 2.0 * (float(tY)+0.5) / float(pc.rtH);
    float tanFov = tan(fov * 0.5);
    vec3 rayDir = normalize(vec3(ndcX * tanFov, ndcY * tanFov, 1.0));

    vec3 normal = vec3(0.0, 0.0, 0.0);
    float bestT = 1000000.0;

    for(int T=0;T<pc.totalInsts;T++){
        MeshDFDesc desc0 = GetResource(bMeshDFDesc, pc.descId).data[T];
        mat4 worldToLocal = GetResource(bLocalTransform, desc0.m_TransformId).m_worldToLocal;
        uint mdfMetaId = desc0.m_MdfMetaId;
        MeshDFMeta meta = GetResource(bMeshDFMeta, mdfMetaId).data;

        vec3 lb = meta.bboxMin.xyz;
        vec3 rt = meta.bboxMax.xyz;

        vec3 d = rayDir;
        vec3 o = camPos;
        float dLen = length(d);

        
        vec4 pO = vec4(o, 1.0);
        pO = worldToLocal * pO;
        pO = pO / pO.w;

        vec4 pO2 = vec4(o + d, 1.0);
        pO2 = worldToLocal * pO2;
        pO2 = pO2 / pO2.w;

        d = pO2.xyz - pO.xyz;
        o = pO.xyz;

        vec3 nD = normalize(d);

        float t;
        bool hit = rayboxIntersection(o,d,lb,rt,t);
        vec3 hitp = o + d*t;
        bool finalHit = false;
        bool outp = false;
        vec3 normalEps = vec3(0.02, 0.02, 0.02);

        // Begin sdf tracing
        if(hit){
            for(int i=0;i<400;i++){
                // get sdf value
                vec3 uvw= (hitp - lb) / (rt - lb);
                uvw = clamp(uvw, 0.0, 1.0);
                float sdf = texture(GetSampler3D(meta.sdfId), uvw).x-1.0;
                if(abs(sdf) < 1.0){
                    finalHit = true;

                    float tval = length(hitp - o) / length(d);
                    if(tval < bestT){
                        float dx1 = texture(GetSampler3D(meta.sdfId), uvw + vec3(normalEps.x, 0.0, 0.0)).x;
                        float dx2 = texture(GetSampler3D(meta.sdfId), uvw - vec3(normalEps.x, 0.0, 0.0)).x;
                        float dy1 = texture(GetSampler3D(meta.sdfId), uvw + vec3(0.0, normalEps.y, 0.0)).x;
                        float dy2 = texture(GetSampler3D(meta.sdfId), uvw - vec3(0.0, normalEps.y, 0.0)).x;
                        float dz1 = texture(GetSampler3D(meta.sdfId), uvw + vec3(0.0, 0.0, normalEps.z)).x;
                        float dz2 = texture(GetSampler3D(meta.sdfId), uvw - vec3(0.0, 0.0, normalEps.z)).x;
                        normal.x = dx1 - dx2;
                        normal.y = dy1 - dy2;
                        normal.z = dz1 - dz2;
                        normal = normalize(normal);
                        bestT = tval;
                    }
                   
                    break;
                }
                hitp += sdf * nD ;
            }
        }
    }
    imageStore(GetUAVImage2DR32F(pc.outTex), ivec2(tX, tY), vec4(abs(normal),1.0));
}