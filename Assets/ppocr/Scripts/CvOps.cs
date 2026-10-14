using System;
using System.Collections.Generic;

namespace PPOCR
{
    /// <summary>
    /// OpenCV 图像算子的 C# 移植，仅实现 PP-OCR 管线用到的部分：
    /// 双线性 resize、双三次 warpPerspective（BORDER_REPLICATE）、
    /// rot90、扫描线多边形填充、mask 均值、Bresenham 画线（Demo 用）。
    /// </summary>
    public static class CvOps
    {
        /// <summary>
        /// 双线性缩放，对应 cv2.resize(img, (w, h))（INTER_LINEAR）。
        /// 逐位复刻 OpenCV hal::resize 的定点路径（HResizeLinearVec_8u32s +
        /// VResizeLinearVec_32s8u），与 cv2 4.13 输出完全一致：
        /// 1) 系数按 float32 计算：fx=(float)((d+0.5)*scale-0.5)，a=round(f*2048)
        ///    （各自独立四舍五入，半舍入到偶数）；横向在边缘强制 fx=0，纵向不强制；
        /// 2) 横向趟存精确 int32 二抽头和（scale 2^11，不舍入）；
        /// 3) 纵向趟：t=S&gt;&gt;4；m=(t*b)&gt;&gt;16（截断）；out=((m0+m1+2)&gt;&gt;2)；
        ///    纵向 tap 钳制到 [0, srcH-1]（sy=-1 时两 tap 均取第 0 行）。
        /// </summary>
        public static ImageBuffer ResizeBilinear(ImageBuffer src, int dstW, int dstH)
        {
            var dst = new ImageBuffer(dstW, dstH);
            int sw = src.Width, sh = src.Height;
            if (dstW == sw && dstH == sh)
            {
                Array.Copy(src.Data, dst.Data, src.Data.Length);
                return dst;
            }

            // ---- 系数 setup（与 hal::resize 逐位一致）----
            double invScaleX = (double)dstW / (double)sw;
            double invScaleY = (double)dstH / (double)sh;
            double scaleX = 1.0 / invScaleX;
            double scaleY = 1.0 / invScaleY;

            int[] hx0 = new int[dstW], hx1 = new int[dstW];
            int[] ha0 = new int[dstW], ha1 = new int[dstW];
            for (int x = 0; x < dstW; x++)
            {
                float fx = (float)((x + 0.5) * scaleX - 0.5);
                int sx = (int)Math.Floor((double)fx);
                fx = (float)(fx - sx);
                if (sx < 0)
                {
                    sx = 0;
                    fx = 0f;
                }
                else if (sx >= sw - 1)
                {
                    sx = sw - 1;
                    fx = 0f;
                }
                float c0 = 1f - fx, c1 = fx;
                ha0[x] = CvRound2048(c0);
                ha1[x] = CvRound2048(c1);
                hx0[x] = sx;
                hx1[x] = Math.Min(sx + 1, sw - 1);
            }

            int[] vy0 = new int[dstH], vy1 = new int[dstH];
            int[] vb0 = new int[dstH], vb1 = new int[dstH];
            for (int y = 0; y < dstH; y++)
            {
                float fy = (float)((y + 0.5) * scaleY - 0.5);
                int sy = (int)Math.Floor((double)fy);
                fy = (float)(fy - sy);
                // 纵向不做边缘强制（与 OpenCV beta setup 一致）
                float c0 = 1f - fy, c1 = fy;
                vb0[y] = CvRound2048(c0);
                vb1[y] = CvRound2048(c1);
                vy0[y] = Math.Min(Math.Max(sy, 0), sh - 1);
                vy1[y] = Math.Min(Math.Max(sy + 1, 0), sh - 1);
            }

            // ---- 横向趟：精确 int32 二抽头和（scale 2^11，不舍入）----
            int[] tmp = new int[sh * dstW * 3];
            byte[] sd = src.Data;
            for (int r = 0; r < sh; r++)
            {
                int srcRow = r * sw * 3;
                int dstRow = r * dstW * 3;
                for (int x = 0; x < dstW; x++)
                {
                    int i0 = srcRow + hx0[x] * 3;
                    int i1 = srcRow + hx1[x] * 3;
                    int a0 = ha0[x], a1 = ha1[x];
                    int o = dstRow + x * 3;
                    tmp[o] = sd[i0] * a0 + sd[i1] * a1;
                    tmp[o + 1] = sd[i0 + 1] * a0 + sd[i1 + 1] * a1;
                    tmp[o + 2] = sd[i0 + 2] * a0 + sd[i1 + 2] * a1;
                }
            }

            // ---- 纵向趟：t=S>>4；m=(t*b)>>16；out=((m0+m1+2)>>2) ----
            byte[] dd = dst.Data;
            int rowLen = dstW * 3;
            for (int y = 0; y < dstH; y++)
            {
                int r0 = vy0[y] * rowLen;
                int r1 = vy1[y] * rowLen;
                int b0 = vb0[y], b1 = vb1[y];
                int dRow = y * rowLen;
                for (int i = 0; i < rowLen; i++)
                {
                    int t0 = tmp[r0 + i] >> 4;
                    int t1 = tmp[r1 + i] >> 4;
                    int m0 = (t0 * b0) >> 16;
                    int m1 = (t1 * b1) >> 16;
                    dd[dRow + i] = (byte)((m0 + m1 + 2) >> 2);
                }
            }

            return dst;
        }

        /// <summary>saturate_cast&lt;short&gt;(f * 2048f)：float32 乘法 + cvRound（半舍入到偶数）。</summary>
        private static int CvRound2048(float f)
        {
            float p = f * 2048f;
            int iv = (int)Math.Round((double)p, MidpointRounding.ToEven);
            if (iv < 0) return 0;
            if (iv > 2048) return 2048;
            return iv;
        }

        /// <summary>
        /// 透视变换采样，对应 cv2.warpPerspective(img, M, (w, h),
        /// borderMode=BORDER_REPLICATE, flags=INTER_CUBIC)。
        /// 逐位复刻 OpenCV 的定点路径（WarpPerspectiveInvoker + remap INTER_CUBIC）：
        /// 1) 按 BLOCK_SZ=32 分块（bh0=min(16,h)，bw0=min(1024/bh0,w)，bh0=min(1024/bw0,h)）；
        /// 2) 坐标：X0=M[0]*x+M[1]*(y+y1)+M[2]（双精度分块累加），
        ///    W=(W0+M[6]*x1)，fX=(X0+M[0]*x1)*(32/W)，cvRound（半舍入到偶数）；
        /// 3) tap 基址 sx=(X&gt;&gt;5)-1，分数 alpha=(Y&amp;31)*32+(X&amp;31)；
        /// 4) 权重表 BicubicTab_i[fy][fx]（15 位定点，isum 修正到 32768）；
        /// 5) 4x4 邻域 × 16 权重精确整数和，BORDER_REPLICATE 钳位；
        /// 6) 输出 saturate_u8((sum+16384)&gt;&gt;15)。
        /// 约定：传入的 m 为 dst→src 的映射（即 cv2.getPerspectiveTransform(pts_std, points)
        /// 或 getPerspectiveTransform(points, pts_std) 再经 3x3 求逆的结果）。
        /// </summary>
        public static ImageBuffer WarpPerspective(ImageBuffer src, double[] m, int dstW, int dstH)
        {
            var dst = new ImageBuffer(dstW, dstH);
            int sw = src.Width, sh = src.Height;
            byte[] sd = src.Data;
            byte[] dd = dst.Data;
            short[] tab = BicubicTab;

            // ---- OpenCV WarpPerspectiveInvoker 分块（BLOCK_SZ = 32）----
            int bh0 = Math.Min(16, dstH);
            int bw0 = Math.Min(1024 / bh0, dstW);
            bh0 = Math.Min(1024 / bw0, dstH);

            for (int y = 0; y < dstH; y += bh0)
            {
                for (int x = 0; x < dstW; x += bw0)
                {
                    int bw = Math.Min(bw0, dstW - x);
                    int bh = Math.Min(bh0, dstH - y);
                    for (int y1 = 0; y1 < bh; y1++)
                    {
                        int yabs = y + y1;
                        double X0 = m[0] * x + m[1] * yabs + m[2];
                        double Y0 = m[3] * x + m[4] * yabs + m[5];
                        double W0 = m[6] * x + m[7] * yabs + m[8];
                        int dRow = yabs * dstW;
                        for (int x1 = 0; x1 < bw; x1++)
                        {
                            double W = W0 + m[6] * x1;
                            double Wq = W != 0.0 ? 32.0 / W : 0.0;
                            double fX = (X0 + m[0] * x1) * Wq;
                            double fY = (Y0 + m[3] * x1) * Wq;
                            if (fX < -2147483648.0) fX = -2147483648.0;
                            else if (fX > 2147483647.0) fX = 2147483647.0;
                            if (fY < -2147483648.0) fY = -2147483648.0;
                            else if (fY > 2147483647.0) fY = 2147483647.0;
                            int X = (int)Math.Round(fX, MidpointRounding.ToEven);
                            int Y = (int)Math.Round(fY, MidpointRounding.ToEven);

                            // xy 以 short 存储（saturate_cast<short>(X >> INTER_BITS)）
                            int xi = X >> 5;
                            int yi = Y >> 5;
                            if (xi < short.MinValue) xi = short.MinValue;
                            else if (xi > short.MaxValue) xi = short.MaxValue;
                            if (yi < short.MinValue) yi = short.MinValue;
                            else if (yi > short.MaxValue) yi = short.MaxValue;
                            int sx = xi - 1;
                            int sy = yi - 1;

                            int aIdx = ((Y & 31) << 5) + (X & 31);
                            int d = (dRow + x + x1) * 3;
                            for (int c = 0; c < 3; c++)
                            {
                                int sum = 0;
                                for (int k1 = 0; k1 < 4; k1++)
                                {
                                    int yy = sy + k1;
                                    if (yy < 0) yy = 0;
                                    else if (yy > sh - 1) yy = sh - 1;
                                    int rowBase = yy * sw * 3 + c;
                                    int wi = (aIdx << 4) + (k1 << 2);
                                    for (int k2 = 0; k2 < 4; k2++)
                                    {
                                        int xx = sx + k2;
                                        if (xx < 0) xx = 0;
                                        else if (xx > sw - 1) xx = sw - 1;
                                        sum += sd[rowBase + xx * 3] * tab[wi + k2];
                                    }
                                }

                                int v = (sum + 16384) >> 15;
                                dd[d + c] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
                            }
                        }
                    }
                }
            }

            return dst;
        }

        /// <summary>
        /// BicubicTab_i[32][32][4][4]（int16，1024*16 平铺），
        /// 复刻 imgwarp.cpp initInterTab2D：float 三次核 → 2D 外积 →
        /// saturate_cast&lt;short&gt;(v*32768)（cvRound 半舍入到偶数）→ isum 修正到 32768。
        /// </summary>
        private static readonly short[] BicubicTab = BuildBicubicTab();

        private static short[] BuildBicubicTab()
        {
            // initInterTab1D：interpolateCubic(i/32)，A = -0.75
            var tab1d = new float[32 * 4];
            for (int i = 0; i < 32; i++)
            {
                float scale = 1f / 32f;
                float xx = i * scale;
                float x1 = xx + 1f;
                const float A = -0.75f;
                tab1d[i * 4 + 0] = ((A * x1 - 5f * A) * x1 + 8f * A) * x1 - 4f * A;
                tab1d[i * 4 + 1] = ((A + 2f) * xx - (A + 3f)) * xx * xx + 1f;
                float om = 1f - xx;
                tab1d[i * 4 + 2] = ((A + 2f) * om - (A + 3f)) * om * om + 1f;
                tab1d[i * 4 + 3] = 1f - tab1d[i * 4 + 0] - tab1d[i * 4 + 1] - tab1d[i * 4 + 2];
            }

            var itab = new short[32 * 32 * 16];
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    int blk = (i * 32 + j) * 16;
                    int isum = 0;
                    for (int k1 = 0; k1 < 4; k1++)
                    {
                        float vy = tab1d[i * 4 + k1];
                        for (int k2 = 0; k2 < 4; k2++)
                        {
                            float v = vy * tab1d[j * 4 + k2];
                            float vs = v * 32768f;
                            int q = (int)Math.Round((double)vs, MidpointRounding.ToEven);
                            if (q < short.MinValue) q = short.MinValue;
                            else if (q > short.MaxValue) q = short.MaxValue;
                            itab[blk + k1 * 4 + k2] = (short)q;
                            isum += q;
                        }
                    }

                    if (isum != 32768)
                    {
                        int diff = isum - 32768;
                        int mk1 = 2, mk2 = 2, Mk1 = 2, Mk2 = 2;
                        for (int k1 = 2; k1 < 4; k1++)
                        {
                            for (int k2 = 2; k2 < 4; k2++)
                            {
                                if (itab[blk + k1 * 4 + k2] < itab[blk + mk1 * 4 + mk2])
                                {
                                    mk1 = k1;
                                    mk2 = k2;
                                }
                                else if (itab[blk + k1 * 4 + k2] > itab[blk + Mk1 * 4 + Mk2])
                                {
                                    Mk1 = k1;
                                    Mk2 = k2;
                                }
                            }
                        }

                        if (diff < 0)
                        {
                            itab[blk + Mk1 * 4 + Mk2] = (short)(itab[blk + Mk1 * 4 + Mk2] - diff);
                        }
                        else
                        {
                            itab[blk + mk1 * 4 + mk2] = (short)(itab[blk + mk1 * 4 + mk2] - diff);
                        }
                    }
                }
            }

            return itab;
        }

        /// <summary>逆时针旋转 90°，对应 np.rot90(img)。</summary>
        public static ImageBuffer Rotate90CCW(ImageBuffer src)
        {
            int w = src.Width;
            int h = src.Height;
            var dst = new ImageBuffer(h, w);
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    // 目标坐标：(dstX, dstY) = (y, w - 1 - x)
                    int srcIdx = (y * w + x) * 3;
                    int dstIdx = ((w - 1 - x) * h + y) * 3;
                    dst.Data[dstIdx] = src.Data[srcIdx];
                    dst.Data[dstIdx + 1] = src.Data[srcIdx + 1];
                    dst.Data[dstIdx + 2] = src.Data[srcIdx + 2];
                }
            }

            return dst;
        }

        /// <summary>
        /// 扫描线偶数规则多边形填充，对应 cv2.fillPoly(mask, pts, 1)。
        /// mask 为 0/1 字节数组，尺寸 w*h；顶点向零截断为整数
        /// （对应 Python 端 box_score_fast 的 box.reshape(1,-1,2).astype('int32')）。
        /// </summary>
        public static void FillPolygon(byte[] mask, int w, int h, List<Pt2> poly)
        {
            if (poly == null || poly.Count < 3)
            {
                return;
            }

            int n = poly.Count;
            int yMin = int.MaxValue, yMax = int.MinValue;
            var ipts = new int[n * 2];
            for (int i = 0; i < n; i++)
            {
                // 向零截断，对应 Python box_score_fast 的
                // `box.reshape(1,-1,2).astype(np.int32)` 后再 cv2.fillPoly。
                int ix = (int)poly[i].X;
                int iy = (int)poly[i].Y;
                ipts[i * 2] = ix;
                ipts[i * 2 + 1] = iy;
                if (iy < yMin) yMin = iy;
                if (iy > yMax) yMax = iy;
            }

            if (yMin < 0) yMin = 0;
            if (yMax > h - 1) yMax = h - 1;

            var xs = new List<double>(16);
            for (int y = yMin; y <= yMax; y++)
            {
                xs.Clear();
                for (int i = 0; i < n; i++)
                {
                    int x1 = ipts[i * 2];
                    int y1 = ipts[i * 2 + 1];
                    int j = (i + 1) % n;
                    int x2 = ipts[j * 2];
                    int y2 = ipts[j * 2 + 1];
                    if (y1 == y2)
                    {
                        continue;
                    }

                    // 半开区间规则，避免顶点重复计数
                    int lo = Math.Min(y1, y2);
                    int hi = Math.Max(y1, y2);
                    if (y < lo || y >= hi)
                    {
                        continue;
                    }

                    double t = (double)(y - y1) / (y2 - y1);
                    xs.Add(x1 + t * (x2 - x1));
                }

                if (xs.Count < 2)
                {
                    continue;
                }

                xs.Sort();
                for (int k = 0; k + 1 < xs.Count; k += 2)
                {
                    int xStart = (int)Math.Ceiling(xs[k] - 0.5);
                    int xEnd = (int)Math.Floor(xs[k + 1] - 0.5);
                    if (xStart < 0) xStart = 0;
                    if (xEnd > w - 1) xEnd = w - 1;
                    for (int x = xStart; x <= xEnd; x++)
                    {
                        mask[y * w + x] = 1;
                    }
                }
            }
        }

        /// <summary>
        /// mask 区域均值，对应 cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)。
        /// mask 尺寸为 (ymax-ymin+1) x (xmax-xmin+1)。
        /// </summary>
        public static float MaskedMean(float[] bitmap, int bitmapW, byte[] mask,
            int xMin, int yMin, int xMax, int yMax)
        {
            double sum = 0;
            long count = 0;
            int maskW = xMax - xMin + 1;
            for (int y = yMin; y <= yMax; y++)
            {
                for (int x = xMin; x <= xMax; x++)
                {
                    if (mask[(y - yMin) * maskW + (x - xMin)] != 0)
                    {
                        sum += bitmap[y * bitmapW + x];
                        count++;
                    }
                }
            }

            return count > 0 ? (float)(sum / count) : 0f;
        }

        /// <summary>Bresenham 直线（Demo 绘制结果框用）。</summary>
        public static void DrawLine(byte[] rgb, int w, int h,
            int x0, int y0, int x1, int y1, byte r, byte g, byte b)
        {
            int dx = Math.Abs(x1 - x0);
            int dy = Math.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;
            while (true)
            {
                if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
                {
                    int idx = (y0 * w + x0) * 3;
                    rgb[idx] = r;
                    rgb[idx + 1] = g;
                    rgb[idx + 2] = b;
                }

                if (x0 == x1 && y0 == y1)
                {
                    break;
                }

                int e2 = 2 * err;
                if (e2 > -dy)
                {
                    err -= dy;
                    x0 += sx;
                }

                if (e2 < dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        private static int ClampInt(int v, int lo, int hi)
        {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }
    }
}
