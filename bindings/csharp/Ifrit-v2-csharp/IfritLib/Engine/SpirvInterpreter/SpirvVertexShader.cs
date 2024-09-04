using IfritLib.Engine.Base;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.SpirvInterpreter
{
    public class SpirvVertexShader : VertexShader
    {
        public SpirvVertexShader(SpirvShaderCompilerBackend compiler, string path)
        {
            var rawPtr = Marshal.StringToHGlobalAnsi(path);
            _internalObject = NativeMethods.IfspvmCreateVertexShaderFromFile(compiler.InternalObject, rawPtr);
        }
        ~SpirvVertexShader()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IfspvmDestroyVertexShaderFromFile(_internalObject);
            }
        }
    }
}
