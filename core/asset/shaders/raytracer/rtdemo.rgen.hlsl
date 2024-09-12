RaytracingAccelerationStructure accStruct : register(t0,space0);
RWTexture2D<float4> outImage : register(u0,space1);

struct Payload
{
    [[vk::location(0)]] float4 hitv;
};

[shader("raygeneration")]
void main()
{
    uint3 lId = DispatchRaysIndex();
    uint3 lSize = DispatchRaysDimensions();

    float2 r = float2(lId.x, lId.y) / float2(lSize.x, lSize.y) * 0.225 - 0.125;
    r.y += 0.1; 

    RayDesc ray;
    ray.Origin = float3(r.x, r.y, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    Payload payload;
    TraceRay(accStruct, 0x01, 0xff, 0, 0, 0, ray, payload);
    outImage[int2(lId.xy)] = payload.hitv;
}