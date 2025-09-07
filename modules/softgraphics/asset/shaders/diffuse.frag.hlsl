struct Uniform {
	float4 lightDir;
	float4 t2;
};
ConstantBuffer<Uniform> u: register(b0, space0);

float4 main(float4 norm : COLOR0) : SV_TARGET{
	float3 n = normalize(norm);
	float3 l = normalize(u.lightDir.xyz);
	float d = dot(n, l) * 0.5 + 0.5;
	return float4(d, d, d, 1);
}