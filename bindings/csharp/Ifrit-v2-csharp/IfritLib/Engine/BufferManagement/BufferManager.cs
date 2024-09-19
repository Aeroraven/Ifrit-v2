using IfritLib.Core;
using IfritLib.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IfritLib.Engine.BufferManagement
{
    public class BufferManager :NativeObjectWrapper {
        public BufferManager()
        {
            this._internalObject = NativeMethods.IfbufCreateBufferManager();
        }
        ~BufferManager()
        {
            if (this._internalObject != IntPtr.Zero)
            {
                NativeMethods.IfbufDestroyBufferManager(this._internalObject);
            }
        }
        public BufferObject CreateBuffer(int size)
        {
            IntPtr buffer = NativeMethods.IfbufCreateBuffer(this._internalObject, (nuint)size);
            return new BufferObject(buffer);
        }
    }
}
