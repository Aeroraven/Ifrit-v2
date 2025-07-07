
#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"

namespace IfritShader {
namespace Math {


    void QRDecomposition(float3x3 A, out float3x3 Q, out float3x3 R)
    {
        float3 a1 = (float3(A._11, A._21, A._31));
        float3 a2 = (float3(A._12, A._22, A._32));
        float3 a3 = (float3(A._13, A._23, A._33));

        float3 b1 = a1;
        float k21 = dot(a2,b1) / dot(b1, b1);

        float3 b2 = a2 - k21 * b1;
        float k31 = dot(a3,b1) / dot(b1, b1);
        float k32 = dot(a3,b2) / dot(b2, b2);
        
        float3 b3 = a3 - k31 * b1 - k32 * b2;

        float3 q1 = normalize(b1);
        float3 q2 = normalize(b2);
        float3 q3 = normalize(b3);

        Q = float3x3(q1, q2, q3);
        R = float3x3(
            length(b1), k21 * length(b1), k31 * length(b1),
            0.0f, length(b2), k32 * length(b2),
            0.0f, 0.0f, length(b3)
        );
    }

    void QRDecomposition(float2x2 A, out float2x2 Q, out float2x2 R)
    {
        float2 a1 = (float2(A._11, A._21));
        float2 a2 = (float2(A._12, A._22));

        float2 b1 = a1;
        float k21 = dot(a2,b1) / dot(b1, b1);

        float2 b2 = a2 - k21 * b1;

        float2 q1 = normalize(b1);
        float2 q2 = normalize(b2);

        Q = float2x2(q1, q2);
        R = float2x2(
            length(b1), k21 * length(b1),
            0.0f, length(b2)
        );
    }

    IFSHADER_TEMPLATE<uint P, uint Q>
    void JacobiEigenvalueStep(inout float3x3 A, inout float3x3 V)
    {
        float Apq = A[P][Q];
        float App = A[P][P];
        float Aqq = A[Q][Q];
        float theta = (App == Aqq) ? kPI * 0.25f : 0.5f * atan2(2.0f * Apq, App - Aqq);

        float C = cos(theta);
        float S = sin(theta);
        float3x3 R = 0;
        R[P][P] = C;
        R[Q][Q] = C;
        R[P][Q] = -S;
        R[Q][P] = S;
        int M = 3 - (P + Q);
        R[M][M] = 1.0f;
        A = mul(Math::Transpose(R),mul(A, R));
        V = mul(V, R);
    }

    void JacobiEigenvalueStep(inout float2x2 A, inout float2x2 V)
    {
        float Apq = A[0][1];
        float App = A[0][0];
        float Aqq = A[1][1];
        float theta = (App == Aqq) ? kPI * 0.25f : 0.5f * atan2(2.0f * Apq, App - Aqq);

        float C = cos(theta);
        float S = sin(theta);
        float2x2 R = Identity2();
        R[0][0] = C;
        R[1][1] = C;
        R[0][1] = -S;
        R[1][0] = S;
        A = mul(Math::Transpose(R), mul(A, R));
        V = mul(V, R);
    }

    void JacobiEigenvalueAnalysis(float3x3 A, out float3 E, out float3x3 V)
    {
        float3x3 R = Identity3();
        float3x3 B = A;
        for(int i = 0; i < 4; ++i)
        {
            JacobiEigenvalueStep<0,1>(B, R);
            JacobiEigenvalueStep<0,2>(B, R);
            JacobiEigenvalueStep<1,2>(B, R);
        }
        E = abs(float3(B[0][0], B[1][1], B[2][2]));
        V = R;
    }

    void JacobiEigenvalueAnalysis(float2x2 A, out float2 E, out float2x2 V)
    {
        float2x2 R = Identity2();
        float2x2 B = A;
        JacobiEigenvalueStep(B, R);
        E = abs(float2(B[0][0], B[1][1]));
        V = R;
    }

    IFSHADER_TEMPLATE<uint P,uint Q>
    void SwapColumn(inout float3x3 A)
    {
        float T1 = A[0][P];
        float T2 = A[1][P];
        float T3 = A[2][P];
        A[0][P] = A[0][Q];
        A[1][P] = A[1][Q];
        A[2][P] = A[2][Q];
        A[0][Q] = T1;
        A[1][Q] = T2;
        A[2][Q] = T3;
    }

    void SwapColumn(inout float2x2 A)
    {
        float T1 = A[0][0];
        float T2 = A[1][0];
        A[0][0] = A[0][1];
        A[1][0] = A[1][1];
        A[0][1] = T1;
        A[1][1] = T2;
    }

    void SortEigenvaluesAndVectors(inout float3 E, inout float3x3 V)
    {
        if(E[0] < E[1])
        {
            SwapColumn<0,1>(V);
            float T = E[0];
            E[0] = E[1];
            E[1] = T;
        }
        if(E[0] < E[2])
        {
            SwapColumn<0,2>(V);
            float T = E[0];
            E[0] = E[2];
            E[2] = T;
        }
        if(E[1] < E[2])
        {
            SwapColumn<1,2>(V);
            float T = E[1];
            E[1] = E[2];
            E[2] = T;
        }
    }

    void SortEigenvaluesAndVectors(inout float2 E, inout float2x2 V)
    {
        if(E[0] < E[1])
        {
            SwapColumn(V);
            float T = E[0];
            E[0] = E[1];
            E[1] = T;
        }
    }

    void SVD(float3x3 A, out float3x3 U, out float3x3 S, out float3x3 V)
    {
        float3 E;
        float3x3 ATA = mul(Math::Transpose(A), A);
        JacobiEigenvalueAnalysis(ATA, E, V);
        SortEigenvaluesAndVectors(E, V);
        S = 0;
        S[0][0] = sqrt(abs(E[0]));
        S[1][1] = sqrt(abs(E[1]));
        S[2][2] = sqrt(abs(E[2]));

        float3x3 SI = S;
        SI[0][0] = rcp(SI[0][0]);
        SI[1][1] = rcp(SI[1][1]);
        SI[2][2] = rcp(SI[2][2]);
        U = mul(mul(A, V), SI);
    }

    void SVD(float2x2 A, out float2x2 U, out float2x2 S, out float2x2 V)
    {
        float2 E;
        float2x2 ATA = mul(Math::Transpose(A), A);
        JacobiEigenvalueAnalysis(ATA, E, V);
        SortEigenvaluesAndVectors(E, V);
        S = 0;
        S[0][0] = sqrt(abs(E[0]));
        S[1][1] = sqrt(abs(E[1]));

        float2x2 SI = S;
        SI[0][0] = rcp(SI[0][0]);
        SI[1][1] = rcp(SI[1][1]);
        U = mul(mul(A, V), SI);
    }

    void PolarDecomposition(float3x3 A, out float3x3 R, out float3x3 S)
    {
        float3x3 U;
        float3x3 V;
        float3x3 Sigma;
        SVD(A, U, Sigma, V);
        
        float3x3 Vt = Math::Transpose(V);
        R = mul(U, Vt);
        S = mul(mul(V, Sigma), Vt);
    }

    void PolarDecomposition(float2x2 A, out float2x2 R, out float2x2 S)
    {
        float2x2 U;
        float2x2 V;
        float2x2 Sigma;
        SVD(A, U, Sigma, V);
        
        float2x2 Vt = Math::Transpose(V);
        R = mul(U, Vt);
        S = mul(mul(V, Sigma), Vt);
    }

}}