struct Payload
{
    [[vk::location(0)]] float4 hitv;
};

[shader("miss")]
void main(inout Payload p)
{
    p.hitv = float4(0.0,1.0,0.0,1.0);
}