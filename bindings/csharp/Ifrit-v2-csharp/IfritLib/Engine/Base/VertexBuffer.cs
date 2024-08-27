using IfritLib.Core;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.Base
{
    public sealed class VertexBuffer : NativeObjectWrapper
    {
        public VertexBuffer()
        {
            _internalObject = NativeMethods.IftrCreateVertexBuffer();
        }
        public void SetLayout(int[] layouts)
        {
            IntPtr rawPtr = Marshal.AllocHGlobal(Marshal.SizeOf<int>() * layouts.Length);
            Marshal.Copy(layouts, 0, rawPtr, layouts.Length);
            NativeMethods.IftrSetVertexBufferLayout(_internalObject, rawPtr, (uint)layouts.Length);
            Marshal.FreeHGlobal(rawPtr);
        }
        public void SetVertexSize(uint size)
        {
            NativeMethods.IftrSetVertexBufferSize(_internalObject, size);
        }
        public void AllocateBuffer()
        {
            NativeMethods.IftrAllocateVertexBuffer(_internalObject);
        }
        public void SetEntry(int index, int attribute, float[] value)
        {
            IntPtr rawPtr = Marshal.AllocHGlobal(Marshal.SizeOf<float>() * value.Length);
            Marshal.Copy(value,0, rawPtr, value.Length);
            NativeMethods.IftrSetVertexBufferValueFloat4(_internalObject, index, attribute, rawPtr);
            Marshal.FreeHGlobal(rawPtr);
        }
        ~VertexBuffer()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IftrDestroyVertexBuffer(_internalObject);
            }
        }
    }
}
