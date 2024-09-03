struct Hello {
	float4 t1;
	float4 t2;
};
ConstantBuffer<Hello> cam : register(b0, space0);

float4 main(float4 color : COLOR0) : SV_TARGET{
	float4 base = float4(1.0f,1.0f,1.0f,1.0f);
	float4 val = color * 0.5f + (base + cam.t1 + cam.t2) * 0.125f;
	return sin(val*8.0f);
}