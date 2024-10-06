float4 main(float4 color : COLOR0) : SV_TARGET{
	float4 p = color*0.5+0.5;
	return ddx(p)*30.0;
}