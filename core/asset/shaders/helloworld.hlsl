float4 main(float4 color : COLOR0) : SV_TARGET{
	float4 base = float4(1.0f,1.0f,1.0f,1.0f);
	return color * 0.5f + base * 0.5f;
}