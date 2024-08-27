using IfritLib.Engine.Base;
using IfritLib.Engine.TileRaster;
using System.Runtime.InteropServices;
using System.Text;
using System.Drawing;

namespace IfritDemo
{
    internal class Program
    {
        public static void SavePPMImage(string path,int height,int width, int channels, byte[] data)
        {
            if (File.Exists(path)) {
                File.Delete(path);
            }
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.WriteLine("P3");
                sw.WriteLine($"{width} {height}");
                sw.WriteLine("255");
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < channels; k++)
                        {
                            if (k >= 3)
                            {
                                continue;
                            }
                            sw.Write(data[i * width * channels + j * channels + k]);
                            sw.Write(" ");
                        }
                    }
                    sw.WriteLine();
                }
            }
        }
        public static void MarshalUnmananagedArray2Struct<T>(IntPtr unmanagedArray, int length, out T[] mangagedArray)
        {
            //https://stackoverflow.com/questions/6747112/getting-array-of-struct-from-intptr
            var size = Marshal.SizeOf(typeof(T));
            mangagedArray = new T[length];

            for (int i = 0; i < length; i++)
            {
                IntPtr ins = new IntPtr(unmanagedArray.ToInt64() + i * size);
                mangagedArray[i] = Marshal.PtrToStructure<T>(ins);
            }
        }
        static void DemoVS(IntPtr input,IntPtr pos,IntPtr varyings)
        {
            IntPtr[] inputPtr = new IntPtr[2];
            MarshalUnmananagedArray2Struct<IntPtr>(input,2,out inputPtr);
            float[] posData = new float[4];
            MarshalUnmananagedArray2Struct<float>(inputPtr[0],4,out posData);
            float[] colorData = new float[4];
            MarshalUnmananagedArray2Struct<float>(inputPtr[1],4,out colorData);
            Marshal.Copy(posData, 0, pos, 4);

            IntPtr[] varyingPtr = new nint[1];
            MarshalUnmananagedArray2Struct<IntPtr>(varyings,1,out varyingPtr);
            Marshal.Copy(colorData, 0, varyingPtr[0], 4);

        }
        static void DemoFS(IntPtr varyings,IntPtr color,IntPtr depth)
        {
            float[] colorData = new float[4];
            Marshal.Copy(varyings, colorData, 0, 4);
            Marshal.Copy(colorData, 0, color, 4);
        }
        static void Main(string[] args)
        {
            const int DEMO_RESOLUTION = 512;
            var image = new ImageContainer(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
            var depth = new ImageContainer(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
            var renderer = new TileRasterRenderer();
            renderer.Init();
            
            var frameBuffer = new FrameBuffer();
            var vertexBuffer = new VertexBuffer();
            int[] indexBuffer = [2, 1, 0];
            int[] vsInLayout = [4, 4];

            vertexBuffer.SetLayout(vsInLayout);
            vertexBuffer.SetVertexSize(3);
            vertexBuffer.AllocateBuffer();
            float[] vA = [0.0f, 0.5f, 0.2f, 1.0f];
            float[] vB = [0.5f, -0.5f, 0.2f, 1.0f];
            float[] vC = [-0.5f, -0.5f, 0.2f, 1.0f];

            float[] cA = [255.0f, 0.0f, 0.0f, 255.0f];
            float[] cB = [0.0f, 255.0f, 0.0f, 255.0f];
            float[] cC = [0.0f, 0.0f, 255.0f, 255.0f];
            vertexBuffer.SetEntry(0, 0, vA);
            vertexBuffer.SetEntry(1, 0, vB);
            vertexBuffer.SetEntry(2, 0, vC);
            vertexBuffer.SetEntry(0, 1, cA);
            vertexBuffer.SetEntry(1, 1, cB);
            vertexBuffer.SetEntry(2, 1, cC);

            frameBuffer.SetColorAttachmentsFP32([image]);
            frameBuffer.SetDepthAttachmentFP32(depth);

            renderer.BindFrameBuffer(frameBuffer);
            renderer.BindVertexBuffer(vertexBuffer);
            renderer.BindIndexBuffer(indexBuffer);
            renderer.SetForceDeterministic(1);

            int[] vsOutLayout = [4];
            var varyingDescriptors = new VaryingDescriptors();
            varyingDescriptors.WriteDescriptor(vsOutLayout);

            renderer.BindVertexShader(DemoVS, varyingDescriptors);
            renderer.BindFragmentShader(DemoFS);

            renderer.DrawLegacy(1);
            renderer.DebugPrint();

            float[] imageData = new float[4 * DEMO_RESOLUTION * DEMO_RESOLUTION];
            image.CopyToArray(imageData);
            byte[] imageDataByte = new byte[4 * DEMO_RESOLUTION * DEMO_RESOLUTION];
            for(int i=0;i<imageData.Length; i++)
            {
                imageDataByte[i] = (byte)imageData[i];
            }
            SavePPMImage("output.ppm", DEMO_RESOLUTION, DEMO_RESOLUTION, 4, imageDataByte);
        }
    }
}
