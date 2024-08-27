using IfritLib.Core;
using IfritLib.Native;
using System.Runtime.InteropServices;

namespace IfritLib.Engine.Base
{
    public sealed class FrameBuffer : NativeObjectWrapper
    {
        public FrameBuffer()
        {
            _internalObject = NativeMethods.IftrCreateFrameBuffer();
        }
        public void SetColorAttachmentsFP32(ImageContainer[] images)
        {
            IntPtr[] rawArray = new IntPtr[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                rawArray[i] = images[i].InternalObject;
            }
            IntPtr pRawArray = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * rawArray.Length);
            Marshal.Copy(rawArray, 0, pRawArray, rawArray.Length);
            NativeMethods.IftrSetFrameBufferColorAttachmentFP32(_internalObject, pRawArray, (uint)images.Length);
            Marshal.FreeHGlobal(pRawArray);
        }
        public void SetDepthAttachmentFP32(ImageContainer image)
        {
            NativeMethods.IftrSetFrameBufferDepthAttachmentFP32(_internalObject, image.InternalObject);
        }
        ~FrameBuffer()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IftrDestroyFrameBuffer(_internalObject);
            }
        }
    }
}
