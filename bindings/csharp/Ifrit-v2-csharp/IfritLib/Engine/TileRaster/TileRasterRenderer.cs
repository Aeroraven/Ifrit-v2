using IfritLib.Core;
using IfritLib.Engine.Base;
using IfritLib.Native;
using System.Runtime.InteropServices;

namespace IfritLib.Engine.TileRaster
{
    public sealed class TileRasterRenderer : NativeObjectWrapper
    {
        public TileRasterRenderer()
        {
            _internalObject = NativeMethods.IftrCreateInstance();
        }
        public void BindFrameBuffer(FrameBuffer frameBuffer)
        {
            NativeMethods.IftrBindFrameBuffer(_internalObject, frameBuffer.InternalObject);
        }
        public void BindVertexBuffer(VertexBuffer vertexBuffer)
        {
            NativeMethods.IftrBindVertexBuffer(_internalObject, vertexBuffer.InternalObject);
        }
        public void BindIndexBuffer(int[] indexBuffer)
        {
            IntPtr rawPtr = Marshal.AllocHGlobal(Marshal.SizeOf<int>() * indexBuffer.Length);
            Marshal.Copy(indexBuffer, 0, rawPtr, indexBuffer.Length);
            NativeMethods.IftrBindIndexBuffer(_internalObject, rawPtr, (uint)indexBuffer.Length);
            Marshal.FreeHGlobal(rawPtr);
        }
        public void BindVertexShaderLegacy1(ShaderDelegates.VertexShaderFunctionPtrRaw vertexShader, VaryingDescriptors layout)
        {
            IntPtr funcPtr = Marshal.GetFunctionPointerForDelegate(vertexShader);
            NativeMethods.IftrBindVertexShaderFunc(_internalObject, funcPtr, layout.InternalObject);
        }
        public void BindFragmentShaderLegacy1(ShaderDelegates.FragmentShaderFunctionPtrRaw fragmentShader)
        {
            IntPtr funcPtr = Marshal.GetFunctionPointerForDelegate(fragmentShader);
            NativeMethods.IftrBindFragmentShaderFunc(_internalObject, funcPtr);
        }
        public void BindVertexShader(VertexShader vertexShader)
        {
            NativeMethods.IftrBindVertexShader(_internalObject, vertexShader.InternalObject);
        }
        public void BindFragmentShader(FragmentShader fragmentShader)
        {
            NativeMethods.IftrBindFragmentShader(_internalObject, fragmentShader.InternalObject);
        }
        public void SetDepthFunc(int state)
        {
            NativeMethods.IftrSetDepthFunc(_internalObject, state);
        }
        public void SetForceDeterministic(int opt)
        {
            NativeMethods.IftrOptsetForceDeterministic(_internalObject, opt);
        }
        public void SetDepthTestEnable(int opt)
        {
            NativeMethods.IftrOptsetDepthTestEnable(_internalObject, opt);
        }
        public void DrawLegacy(int clearFramebuffer)
        {
            NativeMethods.IftrDrawLegacy(_internalObject, clearFramebuffer);
        }
        public void Clear()
        {
            NativeMethods.IftrClear(_internalObject);
        }
        public void Init()
        {
            NativeMethods.IftrInit(_internalObject);
        }
        ~TileRasterRenderer()
        {
            if (_internalObject != 0)
            {
                NativeMethods.IftrDestroyInstance(_internalObject);
            }
        }
    }
}
