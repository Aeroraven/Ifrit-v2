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

struct Uniform {
	float4 t1;
	float4 t2;
};
ConstantBuffer<Uniform> c : register(b0, space0);

VSOutput main(VSInput vsIn){
	VSOutput vsOut;
	vsOut.pos = vsIn.pos + c.t2;
	vsOut.color = vsIn.color + c.t1;
	return vsOut;
}