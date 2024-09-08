using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace IfritLib.Native
{
    internal static partial class NativeMethods
    {
        /* Engine */
        [LibraryImport(DllName, EntryPoint = "iftrCreateFrameBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IftrCreateFrameBuffer();

        [LibraryImport(DllName, EntryPoint = "iftrDestroyFrameBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDestroyFrameBuffer(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "iftrSetFrameBufferColorAttachmentFP32")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetFrameBufferColorAttachmentFP32(IntPtr pInstance, IntPtr pImage, nuint nums);

        [LibraryImport(DllName, EntryPoint = "iftrSetFrameBufferDepthAttachmentFP32")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetFrameBufferDepthAttachmentFP32(IntPtr pInstance, IntPtr pImage);

        [LibraryImport(DllName, EntryPoint = "iftrCreateVaryingDescriptor")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IftrCreateVaryingDescriptor();

        [LibraryImport(DllName, EntryPoint = "iftrDestroyVaryingDescriptor")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDestroyVaryingDescriptor(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "iftrWriteVaryingDescriptor")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrWriteVaryingDescriptor(IntPtr pInstance, IntPtr pDesc, nuint nums);

        [LibraryImport(DllName, EntryPoint = "iftrCreateVertexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IftrCreateVertexBuffer();

        [LibraryImport(DllName, EntryPoint = "iftrDestroyVertexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDestroyVertexBuffer(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "iftrSetVertexBufferLayout")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetVertexBufferLayout(IntPtr pInstance, IntPtr pDesc, nuint nums);

        [LibraryImport(DllName, EntryPoint = "iftrSetVertexBufferSize")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetVertexBufferSize(IntPtr pInstance, nuint nums);

        [LibraryImport(DllName, EntryPoint = "iftrAllocateVertexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrAllocateVertexBuffer(IntPtr pInstances);

        [LibraryImport(DllName, EntryPoint = "iftrSetVertexBufferValueFloat4")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetVertexBufferValueFloat4(IntPtr pInstances, int index, int attr, IntPtr value);


        /* TileRaster */
        [LibraryImport(DllName,EntryPoint = "iftrCreateInstance")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IftrCreateInstance();

        [LibraryImport(DllName, EntryPoint = "iftrDestroyInstance")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDestroyInstance(IntPtr hInstance);

        [LibraryImport(DllName, EntryPoint = "iftrBindFrameBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindFrameBuffer(IntPtr hInstance, IntPtr frameBuffer);

        [LibraryImport(DllName, EntryPoint = "iftrBindVertexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindVertexBuffer(IntPtr hInstance, IntPtr vertexBuffer);

        /*[LibraryImport(DllName, EntryPoint = "iftrBindIndexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindIndexBuffer(IntPtr hInstance, IntPtr indexBuffer);*/

        [LibraryImport(DllName, EntryPoint = "iftrBindVertexShaderFunc")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindVertexShaderFunc(IntPtr hInstance, IntPtr func, IntPtr vsOutDescriptors);

        [LibraryImport(DllName, EntryPoint = "iftrBindFragmentShaderFunc")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindFragmentShaderFunc(IntPtr hInstance, IntPtr func);

        [LibraryImport(DllName, EntryPoint = "iftrSetBlendFunc")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetBlendFunc(IntPtr hInstance, IntPtr state);

        [LibraryImport(DllName, EntryPoint = "iftrSetDepthFunc")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrSetDepthFunc(IntPtr hInstance, int state);

        [LibraryImport(DllName, EntryPoint = "iftrOptsetForceDeterministic")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrOptsetForceDeterministic(IntPtr hInstance, int opt);

        [LibraryImport(DllName, EntryPoint = "iftrOptsetDepthTestEnable")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrOptsetDepthTestEnable(IntPtr hInstance, int opt);

        /*[LibraryImport(DllName, EntryPoint = "iftrDrawLegacy")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDrawLegacy(IntPtr hInstance, int clearFramebuffer);*/

        [LibraryImport(DllName, EntryPoint = "iftrClear")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrClear(IntPtr hInstance);

        [LibraryImport(DllName, EntryPoint = "iftrInit")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrInit(IntPtr hInstance);

        /* Debug */
        [LibraryImport(DllName, EntryPoint = "iftrTest")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrTest(IntPtr func);

        /* ========= Update V1 ======== */
        [LibraryImport(DllName, EntryPoint = "iftrBindVertexShader")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindVertexShader(IntPtr hInstance, IntPtr func);

        [LibraryImport(DllName, EntryPoint = "iftrBindFragmentShader")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindFragmentShader(IntPtr hInstance, IntPtr func);

        [LibraryImport(DllName, EntryPoint = "ifvmCreateLLVMRuntimeBuilder")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfvmCreateLLVMRuntimeBuilder();

        [LibraryImport(DllName, EntryPoint = "ifvmDestroyLLVMRuntimeBuilder")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfvmDestroyLLVMRuntimeBuilder(IntPtr pInstance);

        /* SPIRVM */
        [LibraryImport(DllName, EntryPoint = "ifspvmCreateVertexShaderFromFile")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfspvmCreateVertexShaderFromFile(IntPtr pRuntimeBuilder, IntPtr path);

        [LibraryImport(DllName, EntryPoint = "ifspvmCreateFragmentShaderFromFile")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfspvmCreateFragmentShaderFromFile(IntPtr pRuntimeBuilder, IntPtr path);

        [LibraryImport(DllName, EntryPoint = "ifspvmDestroyVertexShaderFromFile")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfspvmDestroyVertexShaderFromFile(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "ifspvmDestroyFragmentShaderFromFile")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfspvmDestroyFragmentShaderFromFile(IntPtr pInstance);

        /* ========= Update V2 ======== */

        // Renderer
        [LibraryImport(DllName, EntryPoint = "iftrBindIndexBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrBindIndexBuffer(IntPtr hInstance, IntPtr indexBuffer);

        [LibraryImport(DllName, EntryPoint = "iftrDrawLegacy")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IftrDrawLegacy(IntPtr hInstance, int numVertices, int clearFramebuffer);

        // Buffer Management
        [LibraryImport(DllName, EntryPoint = "ifbufCreateBufferManager")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfbufCreateBufferManager();

        [LibraryImport(DllName, EntryPoint = "ifbufDestroyBufferManager")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfbufDestroyBufferManager(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "ifbufCreateBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial IntPtr IfbufCreateBuffer(IntPtr pManager, nuint size);

        [LibraryImport(DllName, EntryPoint = "ifbufDestroyBuffer")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfbufDestroyBuffer(IntPtr pInstance);

        [LibraryImport(DllName, EntryPoint = "ifbufBufferData")]
        [UnmanagedCallConv(CallConvs = [typeof(CallConvStdcall)])]
        public static partial void IfbufBufferData(IntPtr pBuffer, IntPtr pSrc, nuint offset,nuint size);

    }
}
