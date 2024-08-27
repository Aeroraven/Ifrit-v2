using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.Base
{
    public static class ShaderDelegates
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void VertexShaderFunctionPtrRaw(IntPtr input,IntPtr outPos, IntPtr outVaryings);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void FragmentShaderFunctionPtrRaw(IntPtr inVaryings, IntPtr outColor, IntPtr fragmentDepth);
    }
}
