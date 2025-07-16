#pragma once
//#define IFSHADER_MPM_3D
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"


namespace IfritShader{
namespace Artemis{
namespace MPM{

    IFSHADER_DEFINE_CONST_INT32(kMpmSearchRangeWidthPerDim, 2*kMpmGridSearchRange+1);

#ifdef IFSHADER_MPM_3D
    IFSHADER_DEFINE_CONST_INT32(kMpmSearchRangeVolume, kMpmSearchRangeWidthPerDim * kMpmSearchRangeWidthPerDim * kMpmSearchRangeWidthPerDim);
    IFSHADER_TYPEALIAS(FCpicLinearBasis, float4);
    IFSHADER_TYPEALIAS(FCpicLinearBasisMat, float4x4);
    IFSHADER_DEFINE_CONST_INT32(kCpicLinearBasisMatDim, 4);
    
#else
    IFSHADER_DEFINE_CONST_INT32(kMpmSearchRangeVolume, kMpmSearchRangeWidthPerDim * kMpmSearchRangeWidthPerDim);
    IFSHADER_TYPEALIAS(FCpicLinearBasis, float3);
    IFSHADER_TYPEALIAS(FCpicLinearBasisMat, float3x3);
    IFSHADER_DEFINE_CONST_INT32(kCpicLinearBasisMatDim, 3);
#endif

    struct FCpicLinearBasisQ
    {
        FCpicLinearBasis m_Rows[kMpmSearchRangeVolume];
    };

    struct FCpicLinearBasisKsi
    {
        FScalar m_Diag[kMpmSearchRangeVolume];
    };

    struct FCpicNeighbourValues
    {
        FScalar m_Values[kMpmSearchRangeVolume];
    };


    FCpicLinearBasisMat GetCpicLinearMInv(FCpicLinearBasisQ BasisQ, FCpicLinearBasisKsi BasisKsi)
    {
        // M <- Q.T @ Diag(Ksi) @ Q
        // M <- M^(-1)
        FCpicLinearBasisQ KsiQ;
        FCpicLinearBasisMat M;
        for(int i = 0; i < kMpmSearchRangeVolume; ++i)
        {
            KsiQ.m_Rows[i] = BasisQ.m_Rows[i] * BasisKsi.m_Diag[i];
        }
        for(int i = 0; i<kCpicLinearBasisMatDim; ++i)
        {
            for(int j = 0; j<kCpicLinearBasisMatDim; ++j)
            {
                M[i][j] = 0.0f;
                for(int k = 0; k < kMpmSearchRangeVolume; ++k)
                {
                    // Q.T @ KsiQ
                    M[i][j] += BasisQ.m_Rows[k][i] * KsiQ.m_Rows[k][j];
                }
            }
        }
        M = Math::Inverse(M);
    }

    FCpicLinearBasis GetCpicReconValueAndGradient_LinearBasis(
        FCpicLinearBasisMat MInv, FCpicLinearBasisQ BasisQ, FCpicLinearBasisKsi BasisKsi, FCpicNeighbourValues Values)
    {
        // Pos <- MInv @ Q.T @ Diag(Ksi) @ Values
        FCpicNeighbourValues KsiValues;
        for(int i = 0; i < kMpmSearchRangeVolume; ++i)
        {
            KsiValues.m_Values[i] = Values.m_Values[i] * BasisKsi.m_Diag[i];
        }
        FCpicLinearBasis QtKV;
        for(int i = 0; i < kCpicLinearBasisMatDim; ++i)
        {
            QtKV[i] = 0.0f;
            for(int j = 0; j < kMpmSearchRangeVolume; ++j)
            {
                // Q.T @ KsiValues
                QtKV[i] += BasisQ.m_Rows[j][i] * KsiValues.m_Values[j];
            }
        }
        return mul(MInv, QtKV);
    }

    int EncodeRangeSearchOffset(int dx,int dy,int dz)
    {
        int dxA = dx + kMpmGridSearchRange;
        int dyA = dy + kMpmGridSearchRange;
        int dzA = dz + kMpmGridSearchRange;

        int Offset = 0;
#ifdef IFSHADER_MPM_3D
        Offset += dzA * kMpmSearchRangeWidthPerDim * kMpmSearchRangeWidthPerDim;
        Offset += dyA * kMpmSearchRangeWidthPerDim;
        Offset += dxA;
#else
        Offset += dyA * kMpmSearchRangeWidthPerDim;
        Offset += dxA;
#endif
        return Offset;
    }

    FCpicLinearBasis EncodeLinearBasis(FSpatialVector Position)
    {
        // linear basis 2d (1,x,y) or 3d (1,x,y,z)
        FCpicLinearBasis Basis;
#ifdef IFSHADER_MPM_3D
        Basis[0] = 1.0f;
        Basis[1] = Position.x;
        Basis[2] = Position.y;
        Basis[3] = Position.z;
#else
        Basis[0] = 1.0f;
        Basis[1] = Position.x;
        Basis[2] = Position.y;
#endif
        return Basis;
    }


    void GetCpicPDistAndPNormal(FCpicLinearBasis DistAndNormal, out FScalar Dist, out FSpatialVector Normal)
    {
#ifdef IFSHADER_MPM_3D
        Dist = DistAndNormal[0];
        Normal.x = DistAndNormal[1];
        Normal.y = DistAndNormal[2];
        Normal.z = DistAndNormal[3];
#else
        Dist = DistAndNormal[0];
        Normal.x = DistAndNormal[1];
        Normal.y = DistAndNormal[2];
#endif
    }

}}}
