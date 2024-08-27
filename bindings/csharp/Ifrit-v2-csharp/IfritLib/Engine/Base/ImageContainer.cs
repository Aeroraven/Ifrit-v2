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
    public sealed class ImageContainer : NativeObjectWrapper
    {
        private uint _width;
        private uint _height;
        private uint _channel;

        public uint Width { get=>_width; }
        public uint Height { get=>_height; }
        public uint Channel { get=>_channel; }

        public ImageContainer(uint width, uint height, uint channel)
        {
            _width = width;
            _height = height;
            _channel = channel;
            _internalObject = NativeMethods.IfcrCreateImageFP32(width, height, channel);
        }
        public void CopyToArray(float[] data)
        {
            IntPtr rawPtr = NativeMethods.IfcrGetImageRawDataFP32(_internalObject);
            Marshal.Copy(rawPtr, data, 0, (int)(Width*Height*Channel));
        }
        ~ImageContainer()
        {
            if (_internalObject != IntPtr.Zero)
            {
                NativeMethods.IfcrDestroyImageFP32(_internalObject);
            }
        }

    }
}
