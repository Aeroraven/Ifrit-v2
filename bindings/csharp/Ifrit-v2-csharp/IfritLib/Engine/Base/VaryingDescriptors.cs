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
    public sealed class VaryingDescriptors : NativeObjectWrapper
    {
        public VaryingDescriptors()
        {
            _internalObject = NativeMethods.IftrCreateVaryingDescriptor();
        }
        public void WriteDescriptor(int[] descriptors)
        {
            IntPtr rawPtr = Marshal.AllocHGlobal(Marshal.SizeOf<int>() * descriptors.Length);
            Marshal.Copy(descriptors, 0, rawPtr, descriptors.Length);
            NativeMethods.IftrWriteVaryingDescriptor(_internalObject, rawPtr, (uint)descriptors.Length);
            Marshal.FreeHGlobal(rawPtr);
        }
        ~VaryingDescriptors()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IftrDestroyVaryingDescriptor(_internalObject);
            }
        }

    }
}
