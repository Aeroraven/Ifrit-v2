using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Native
{
    internal static partial class NativeMethods
    {
        /* Image */
        [LibraryImport(DllName, EntryPoint = "ifcrCreateImageFP32")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfcrCreateImageFP32(nuint width, nuint height, nuint channel);

        [LibraryImport(DllName, EntryPoint = "ifcrDestroyImageFP32")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfcrDestroyImageFP32(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "ifcrGetImageRawDataFP32")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfcrGetImageRawDataFP32(IntPtr pInstance);
    }
}
