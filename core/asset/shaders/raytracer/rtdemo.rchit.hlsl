struct Attributes{
    float2 bary;
};
struct Payload{
    [[vk::location(0)]] float4 hitv;
};

[shader("closesthit")]
void main(inout Payload p, in Attributes at){    
    // Unpack Buffers
    p.hitv = float4(0.0,1.0,0.0,1.0);
}