using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.SpirvInterpreter
{
    public class SpirvLLVMBackend : SpirvShaderCompilerBackend
    {
        public SpirvLLVMBackend()
        {
            _internalObject = NativeMethods.IfvmCreateLLVMRuntimeBuilder();
        }
        ~SpirvLLVMBackend()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IfvmDestroyLLVMRuntimeBuilder(_internalObject);
            }
        }
    }
}
