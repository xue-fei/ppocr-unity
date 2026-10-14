using System.Collections.Generic;

namespace PPOCR
{
    /// <summary>
    /// 二值图轮廓查找，对应 cv2.findContours(bitmap, RETR_LIST, CHAIN_APPROX_SIMPLE)
    /// 在 DB 后处理场景下的等价实现。
    ///
    /// 实现说明：
    /// 参考实现的下游只对轮廓点集求最小面积外接矩形和面积，
    /// 而 CHAIN_APPROX_SIMPLE 只剔除共线中间点、不改变点集的凸包与极值点，
    /// 因此这里直接输出"边界像素集合"即可得到完全一致的下游结果，
    /// 无需逐点追踪轮廓路径（Suzuki-Abe / Moore 追踪）。
    ///
    /// 连通性约定（避免 Jordan 悖论）：
    /// 前景按 8 连通标记，背景按 4 连通标记；
    /// 外轮廓 = 前景中拥有背景 8 邻域的像素；
    /// 孔洞轮廓 = 被前景完全包围的背景区域周围的前景像素。
    /// </summary>
    public static class ContourFinder
    {
        /// <param name="bitmap">0/1 二值图，长度 = w*h</param>
        public static List<List<Pt2>> FindContours(byte[] bitmap, int w, int h)
        {
            var contours = new List<List<Pt2>>();
            int total = w * h;

            // 1. 前景 8 连通标记，按光栅顺序编号
            int[] comp = new int[total];
            int compCount = 0;
            var stack = new Stack<int>();
            for (int i = 0; i < total; i++)
            {
                if (bitmap[i] == 0 || comp[i] != 0)
                {
                    continue;
                }

                compCount++;
                comp[i] = compCount;
                stack.Push(i);
                while (stack.Count > 0)
                {
                    int p = stack.Pop();
                    int px = p % w;
                    int py = p / w;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int ny = py + dy;
                        if (ny < 0 || ny >= h)
                        {
                            continue;
                        }

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0)
                            {
                                continue;
                            }

                            int nx = px + dx;
                            if (nx < 0 || nx >= w)
                            {
                                continue;
                            }

                            int q = ny * w + nx;
                            if (bitmap[q] != 0 && comp[q] == 0)
                            {
                                comp[q] = compCount;
                                stack.Push(q);
                            }
                        }
                    }
                }
            }

            // 2. 每个前景连通域的外边界像素（有背景 8 邻域或越界）
            var outerBoundaries = new List<Pt2>[compCount + 1];
            for (int i = 0; i < total; i++)
            {
                if (bitmap[i] == 0)
                {
                    continue;
                }

                int px = i % w;
                int py = i / w;
                if (!HasBackgroundNeighbor8(bitmap, w, h, px, py))
                {
                    continue;
                }

                int c = comp[i];
                if (outerBoundaries[c] == null)
                {
                    outerBoundaries[c] = new List<Pt2>();
                }

                outerBoundaries[c].Add(new Pt2(px, py));
            }

            for (int c = 1; c <= compCount; c++)
            {
                if (outerBoundaries[c] != null && outerBoundaries[c].Count > 0)
                {
                    contours.Add(outerBoundaries[c]);
                }
            }

            // 3. 背景 4 连通；不触图像边界的背景区域即孔洞
            bool[] bgVisited = new bool[total];
            var bgStack = new Stack<int>();
            var bgPixels = new List<int>(64);
            for (int i = 0; i < total; i++)
            {
                if (bitmap[i] != 0 || bgVisited[i])
                {
                    continue;
                }

                bgPixels.Clear();
                bgVisited[i] = true;
                bgStack.Push(i);
                bool touchesBorder = false;
                while (bgStack.Count > 0)
                {
                    int p = bgStack.Pop();
                    bgPixels.Add(p);
                    int px = p % w;
                    int py = p / w;
                    if (px == 0 || py == 0 || px == w - 1 || py == h - 1)
                    {
                        touchesBorder = true;
                    }

                    // 4 邻域
                    if (px > 0)
                    {
                        int q = p - 1;
                        if (bitmap[q] == 0 && !bgVisited[q])
                        {
                            bgVisited[q] = true;
                            bgStack.Push(q);
                        }
                    }

                    if (px < w - 1)
                    {
                        int q = p + 1;
                        if (bitmap[q] == 0 && !bgVisited[q])
                        {
                            bgVisited[q] = true;
                            bgStack.Push(q);
                        }
                    }

                    if (py > 0)
                    {
                        int q = p - w;
                        if (bitmap[q] == 0 && !bgVisited[q])
                        {
                            bgVisited[q] = true;
                            bgStack.Push(q);
                        }
                    }

                    if (py < h - 1)
                    {
                        int q = p + w;
                        if (bitmap[q] == 0 && !bgVisited[q])
                        {
                            bgVisited[q] = true;
                            bgStack.Push(q);
                        }
                    }
                }

                if (touchesBorder)
                {
                    continue;
                }

                // 孔洞轮廓 = 与孔洞 4 邻接的前景像素（去重）
                var holeBoundary = new List<Pt2>();
                var emitted = new HashSet<int>();
                foreach (int p in bgPixels)
                {
                    int px = p % w;
                    int py = p / w;
                    if (px > 0)
                    {
                        TryAddHolePixel(bitmap, w, p - 1, holeBoundary, emitted);
                    }

                    if (px < w - 1)
                    {
                        TryAddHolePixel(bitmap, w, p + 1, holeBoundary, emitted);
                    }

                    if (py > 0)
                    {
                        TryAddHolePixel(bitmap, w, p - w, holeBoundary, emitted);
                    }

                    if (py < h - 1)
                    {
                        TryAddHolePixel(bitmap, w, p + w, holeBoundary, emitted);
                    }
                }

                if (holeBoundary.Count > 0)
                {
                    contours.Add(holeBoundary);
                }
            }

            return contours;
        }

        private static void TryAddHolePixel(byte[] bitmap, int w, int index, List<Pt2> output, HashSet<int> emitted)
        {
            if (bitmap[index] == 0 || emitted.Contains(index))
            {
                return;
            }

            emitted.Add(index);
            output.Add(new Pt2(index % w, index / w));
        }

        private static bool HasBackgroundNeighbor8(byte[] bitmap, int w, int h, int px, int py)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                int ny = py + dy;
                if (ny < 0 || ny >= h)
                {
                    return true; // 越界视为背景
                }

                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0)
                    {
                        continue;
                    }

                    int nx = px + dx;
                    if (nx < 0 || nx >= w)
                    {
                        return true;
                    }

                    if (bitmap[ny * w + nx] == 0)
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }
}
