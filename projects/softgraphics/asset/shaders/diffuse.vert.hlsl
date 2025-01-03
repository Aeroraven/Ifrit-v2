struct VSInput
{
    [[vk::location(0)]] float4 pos : POSITION0;
    [[vk::location(1)]] float4 color : COLOR0;
};

struct VSOutput
{
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
};

struct Uniform2 {
	float4x4 tm;
};
ConstantBuffer<Uniform2> d: register(b1, space0);


VSOutput main(VSInput vsIn){
	VSOutput vsOut;
	vsOut.pos = mul(d.tm, vsIn.pos);
	vsOut.color = vsIn.color;
	return vsOut;
}