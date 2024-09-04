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


VSOutput main(VSInput vsIn){
	VSOutput vsOut;
	vsOut.pos = vsIn.pos;
	vsOut.color = vsIn.color;
	return vsOut;
}