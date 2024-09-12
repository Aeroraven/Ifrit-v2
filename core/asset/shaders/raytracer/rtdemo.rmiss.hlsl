struct Payload
{
    [[vk::location(0)]] float4 hitv;
};

[shader("miss")]
void main(inout Payload p)
{
    p.hitv = float4(0.2,0.2,0.2,1.0);
}