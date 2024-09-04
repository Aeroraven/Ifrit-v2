using IfritLib.Engine.Base;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.SpirvInterpreter
{
    public class SpirvFragmentShader : FragmentShader
    {
        public SpirvFragmentShader(SpirvShaderCompilerBackend compiler, string path) {
            var rawPtr = Marshal.StringToHGlobalAnsi(path);
            _internalObject = NativeMethods.IfspvmCreateFragmentShaderFromFile(compiler.InternalObject, rawPtr);
        }
        ~SpirvFragmentShader()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IfspvmDestroyFragmentShaderFromFile(_internalObject);
            }
        }
    }
}
