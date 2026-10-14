using System;
using UnityEngine;

namespace PPOCR
{
    /// <summary>
    /// 8 位 RGB 图像容器（HWC、行主序、左上角为原点）。
    /// 与参考实现中 OpenCV 的 BGR/RGB numpy 数组布局一致：
    /// Python 管线在入口处把 cv2.imread 的 BGR 转成 RGB，之后全程 RGB，
    /// 因此本容器统一使用 RGB 通道顺序。
    /// </summary>
    public class ImageBuffer
    {
        public readonly int Width;
        public readonly int Height;

        /// <summary>像素数据，长度 = Width * Height * 3，布局为 R,G,B 交错。</summary>
        public readonly byte[] Data;

        public ImageBuffer(int width, int height)
        {
            if (width <= 0 || height <= 0)
            {
                throw new ArgumentException("ImageBuffer size must be positive: " + width + "x" + height);
            }

            Width = width;
            Height = height;
            Data = new byte[width * height * 3];
        }

        public ImageBuffer(int width, int height, byte[] data)
        {
            if (data == null || data.Length != width * height * 3)
            {
                throw new ArgumentException("ImageBuffer data length mismatch");
            }

            Width = width;
            Height = height;
            Data = data;
        }

        public void SetRgb(int x, int y, byte r, byte g, byte b)
        {
            int idx = (y * Width + x) * 3;
            Data[idx] = r;
            Data[idx + 1] = g;
            Data[idx + 2] = b;
        }

        public void GetRgb(int x, int y, out byte r, out byte g, out byte b)
        {
            int idx = (y * Width + x) * 3;
            r = Data[idx];
            g = Data[idx + 1];
            b = Data[idx + 2];
        }

        /// <summary>
        /// 从 Unity Texture2D 转换。Unity 纹理的第 0 行在底部，
        /// 这里做一次垂直翻转，使图像左上角为原点。
        /// 注意：纹理必须开启 Read/Write（可读），否则会抛异常。
        /// </summary>
        public static ImageBuffer FromTexture(Texture2D texture)
        {
            if (texture == null)
            {
                throw new ArgumentNullException("texture");
            }

            int w = texture.width;
            int h = texture.height;
            Color32[] pixels = texture.GetPixels32();
            if (pixels == null || pixels.Length != w * h)
            {
                throw new InvalidOperationException(
                    "Failed to read pixels from texture. 请先在导入设置中开启 Read/Write。");
            }

            var image = new ImageBuffer(w, h);
            for (int y = 0; y < h; y++)
            {
                // Unity 行序自底向上，翻转为自顶向下
                int srcRow = (h - 1 - y) * w;
                int dstRow = y * w;
                for (int x = 0; x < w; x++)
                {
                    Color32 c = pixels[srcRow + x];
                    int idx = (dstRow + x) * 3;
                    image.Data[idx] = c.r;
                    image.Data[idx + 1] = c.g;
                    image.Data[idx + 2] = c.b;
                }
            }

            return image;
        }

        /// <summary>
        /// 转回 Unity Texture2D（RGB24）。从左上角原点翻回 Unity 的底部原点行序。
        /// </summary>
        public Texture2D ToTexture()
        {
            var texture = new Texture2D(Width, Height, TextureFormat.RGB24, false);
            byte[] raw = new byte[Data.Length];
            for (int y = 0; y < Height; y++)
            {
                int srcRow = y * Width * 3;
                // 翻回自底向上的行序
                int dstRow = (Height - 1 - y) * Width * 3;
                Array.Copy(Data, srcRow, raw, dstRow, Width * 3);
            }

            texture.LoadRawTextureData(raw);
            texture.Apply();
            return texture;
        }

        /// <summary>深拷贝。</summary>
        public ImageBuffer Clone()
        {
            var copy = new ImageBuffer(Width, Height);
            Array.Copy(Data, copy.Data, Data.Length);
            return copy;
        }
    }
}
