using IfritLib.Core;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.BufferManagement
{
    public class BufferObject : NativeObjectWrapper {
        internal BufferObject(IntPtr internalObject)
        {
            this._internalObject = internalObject;
        }
        ~BufferObject()
        {
            if (this._internalObject != IntPtr.Zero)
            {
                NativeMethods.IfbufDestroyBuffer(this._internalObject);
            }
        }
        
        public void BufferDataFromList<T>(IList<T> pArrayData,int offset) where T: IList<T>
        {
            //To Unmanaged
            IntPtr pArrayDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<T>() * pArrayData.Count);
            for (int i = 0; i < pArrayData.Count; i++)
            {
                Marshal.StructureToPtr(pArrayData[i], pArrayDataPtr + i * Marshal.SizeOf<T>(), false);
            }
            nuint offsetN = (nuint)offset;
            nuint sizeN = (nuint)Marshal.SizeOf<T>() * (nuint)pArrayData.Count;
            NativeMethods.IfbufBufferData(_internalObject,pArrayDataPtr, offsetN, sizeN);
            Marshal.FreeHGlobal(pArrayDataPtr);
        }
    }
}
