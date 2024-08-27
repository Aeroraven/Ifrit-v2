using IfritLib.Core;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.Base
{
    public sealed class BufferLayout : NativeObjectWrapper
    {
        public BufferLayout()
        {
            _internalObject = NativeMethods.IftrCreateBufferLayout();
        }
        ~BufferLayout()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IftrDestroyBufferLayout(_internalObject);
            }
        }
    }
}
