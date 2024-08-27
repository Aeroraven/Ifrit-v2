using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Core
{
    public class NativeObjectWrapper
    {
        protected IntPtr _internalObject = IntPtr.Zero;
        public IntPtr InternalObject { get => _internalObject; }
    }
}
