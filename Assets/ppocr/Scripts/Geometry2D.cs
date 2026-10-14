using System;
using System.Collections.Generic;
using System.Linq;

namespace PPOCR
{
    /// <summary>二维浮点坐标。对应 Python 端 numpy 的 (x, y) 点。</summary>
    public struct Pt2
    {
        public float X;
        public float Y;

        public Pt2(float x, float y)
        {
            X = x;
            Y = y;
        }

        public static float Distance(Pt2 a, Pt2 b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public override string ToString()
        {
            return "(" + X.ToString("F1") + "," + Y.ToString("F1") + ")";
        }
    }

    /// <summary>
    /// OpenCV 几何算法的 C# 移植，均为 DB 后处理 / 裁剪所需的最小集合：
    /// 凸包、最小面积外接矩形（旋转卡壳）、四边形角点排序、
    /// 多边形面积/周长、四点透视变换（getPerspectiveTransform）。
    /// </summary>
    public static class Geom
    {
        /// <summary>
        /// Andrew 单调链凸包，返回逆时针顺序的点列（不含共线中间点）。
        /// </summary>
        public static List<Pt2> ConvexHull(List<Pt2> points)
        {
            if (points == null || points.Count < 3)
            {
                return points != null ? new List<Pt2>(points) : new List<Pt2>();
            }

            var pts = new List<Pt2>(points);
            // 按 x 优先、y 次优先排序（保证重复点相邻，便于去重）
            pts.Sort((a, b) =>
            {
                int c = a.X.CompareTo(b.X);
                return c != 0 ? c : a.Y.CompareTo(b.Y);
            });

            var unique = new List<Pt2>();
            foreach (var p in pts)
            {
                if (unique.Count == 0 || unique[unique.Count - 1].X != p.X || unique[unique.Count - 1].Y != p.Y)
                {
                    unique.Add(p);
                }
            }

            if (unique.Count < 3)
            {
                return unique;
            }

            var lower = new List<Pt2>();
            foreach (var p in unique)
            {
                while (lower.Count >= 2 && Cross(lower[lower.Count - 2], lower[lower.Count - 1], p) <= 0)
                {
                    lower.RemoveAt(lower.Count - 1);
                }

                lower.Add(p);
            }

            var upper = new List<Pt2>();
            for (int i = unique.Count - 1; i >= 0; i--)
            {
                var p = unique[i];
                while (upper.Count >= 2 && Cross(upper[upper.Count - 2], upper[upper.Count - 1], p) <= 0)
                {
                    upper.RemoveAt(upper.Count - 1);
                }

                upper.Add(p);
            }

            lower.RemoveAt(lower.Count - 1);
            upper.RemoveAt(upper.Count - 1);
            lower.AddRange(upper);
            return lower; // 逆时针
        }

        private static float Cross(Pt2 o, Pt2 a, Pt2 b)
        {
            return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
        }

        /// <summary>
        /// 最小面积外接矩形（旋转卡壳）。
        /// 经典结论：最小面积外接矩形至少有一条边与凸包的某条边共线，
        /// 因此枚举凸包的每条边作为基准边即可。
        /// 返回 4 个角点（无序集合，顺序不保证）。
        /// </summary>
        public static List<Pt2> MinAreaRect(List<Pt2> points)
        {
            if (points == null || points.Count == 0)
            {
                return new List<Pt2>();
            }

            var hull = ConvexHull(points);
            if (hull.Count < 3)
            {
                return AxisAlignedBBox(points);
            }

            double bestArea = double.MaxValue;
            double bestUMin = 0, bestUMax = 0, bestNMin = 0, bestNMax = 0;
            double bestUx = 1, bestUy = 0, bestNx = 0, bestNy = 1;

            for (int i = 0; i < hull.Count; i++)
            {
                Pt2 a = hull[i];
                Pt2 b = hull[(i + 1) % hull.Count];
                double dx = b.X - a.X;
                double dy = b.Y - a.Y;
                double len = Math.Sqrt(dx * dx + dy * dy);
                if (len < 1e-12)
                {
                    continue;
                }

                double ux = dx / len;
                double uy = dy / len;
                double nx = -uy;
                double ny = ux;

                double uMin = double.MaxValue, uMax = double.MinValue;
                double nMin = double.MaxValue, nMax = double.MinValue;
                foreach (var p in hull)
                {
                    double u = p.X * ux + p.Y * uy;
                    double n = p.X * nx + p.Y * ny;
                    if (u < uMin) uMin = u;
                    if (u > uMax) uMax = u;
                    if (n < nMin) nMin = n;
                    if (n > nMax) nMax = n;
                }

                double area = (uMax - uMin) * (nMax - nMin);
                if (area < bestArea)
                {
                    bestArea = area;
                    bestUMin = uMin;
                    bestUMax = uMax;
                    bestNMin = nMin;
                    bestNMax = nMax;
                    bestUx = ux;
                    bestUy = uy;
                    bestNx = nx;
                    bestNy = ny;
                }
            }

            if (bestArea == double.MaxValue)
            {
                return AxisAlignedBBox(points);
            }

            // (u,n) 坐标系下的 4 个角，转回 xy
            var corners = new List<Pt2>(4);
            corners.Add(ToXY(bestUMin, bestNMin, bestUx, bestUy, bestNx, bestNy));
            corners.Add(ToXY(bestUMax, bestNMin, bestUx, bestUy, bestNx, bestNy));
            corners.Add(ToXY(bestUMax, bestNMax, bestUx, bestUy, bestNx, bestNy));
            corners.Add(ToXY(bestUMin, bestNMax, bestUx, bestUy, bestNx, bestNy));
            return corners;
        }

        private static Pt2 ToXY(double u, double n, double ux, double uy, double nx, double ny)
        {
            return new Pt2((float)(u * ux + n * nx), (float)(u * uy + n * ny));
        }

        private static List<Pt2> AxisAlignedBBox(List<Pt2> points)
        {
            float xMin = float.MaxValue, yMin = float.MaxValue;
            float xMax = float.MinValue, yMax = float.MinValue;
            foreach (var p in points)
            {
                if (p.X < xMin) xMin = p.X;
                if (p.Y < yMin) yMin = p.Y;
                if (p.X > xMax) xMax = p.X;
                if (p.Y > yMax) yMax = p.Y;
            }

            return new List<Pt2>
            {
                new Pt2(xMin, yMin),
                new Pt2(xMax, yMin),
                new Pt2(xMax, yMax),
                new Pt2(xMin, yMax)
            };
        }

        /// <summary>
        /// 与 Python get_mini_boxes / crop_poly 的角点重排逻辑完全一致：
        /// 先按 x 稳定排序，再按 y 的上下关系整理为
        /// [左上, 右上, 右下, 左下]。
        /// 输入为最小面积矩形的 4 个角点（顺序任意）。
        /// </summary>
        public static List<Pt2> OrderQuad(List<Pt2> corners)
        {
            if (corners == null || corners.Count != 4)
            {
                throw new ArgumentException("OrderQuad 需要 4 个角点");
            }

            // LINQ OrderBy 是稳定排序，对应 Python sorted(key=x)
            var points = corners.OrderBy(p => p.X).ToList();

            int i1, i4, i2, i3;
            if (points[1].Y > points[0].Y)
            {
                i1 = 0;
                i4 = 1;
            }
            else
            {
                i1 = 1;
                i4 = 0;
            }

            if (points[3].Y > points[2].Y)
            {
                i2 = 2;
                i3 = 3;
            }
            else
            {
                i2 = 3;
                i3 = 2;
            }

            return new List<Pt2>
            {
                points[i1], // 左上
                points[i2], // 右上
                points[i3], // 右下
                points[i4]  // 左下
            };
        }

        /// <summary>有序四边形的短边长度（对应 get_mini_boxes 的 sside）。</summary>
        public static float QuadMinSide(List<Pt2> quad)
        {
            float s0 = Pt2.Distance(quad[0], quad[1]);
            float s1 = Pt2.Distance(quad[1], quad[2]);
            return Math.Min(s0, s1);
        }

        /// <summary>鞋带公式面积（绝对值）。对应 cv2.contourArea。</summary>
        public static double ShoelaceArea(List<Pt2> poly)
        {
            double area = 0;
            int n = poly.Count;
            for (int i = 0; i < n; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % n];
                area += a.X * b.Y - b.X * a.Y;
            }

            return Math.Abs(area) * 0.5;
        }

        /// <summary>多边形周长。对应 cv2.arcLength(closed=True)。</summary>
        public static double Perimeter(List<Pt2> poly, bool closed)
        {
            double length = 0;
            int n = poly.Count;
            if (n < 2)
            {
                return 0;
            }

            int last = closed ? n : n - 1;
            for (int i = 0; i < last; i++)
            {
                length += Pt2.Distance(poly[i], poly[(i + 1) % n]);
            }

            return length;
        }

        /// <summary>
        /// 四点透视变换，逐位复刻 cv2.getPerspectiveTransform（imgwarp.cpp）：
        /// 8x8 线性系统的右侧乘积项 -src.x*dst.x 等按 **float32** 计算后再转 double
        /// （C++ 中 src/dst 为 Point2f），随后经 cv::solve(DECOMP_LU) →
        /// hal::LU64f（LUImpl：部分选主元 + 原地消元 + 回代）求解。
        /// 返回 3x3 行主序矩阵，使得 dst ≈ M * src（齐次坐标，M[2][2] = 1），
        /// 与 cv2.getPerspectiveTransform(src, dst) 输出逐位一致。
        /// </summary>
        public static double[] GetPerspectiveTransform(List<Pt2> src, List<Pt2> dst)
        {
            if (src.Count != 4 || dst.Count != 4)
            {
                throw new ArgumentException("getPerspectiveTransform 需要 4 对点");
            }

            // 与 cv::getPerspectiveTransform 相同的 8x8 增广矩阵（A 与 b 分离存储）
            double[,] a = new double[8, 8];
            double[] b = new double[8];
            for (int i = 0; i < 4; i++)
            {
                float sx = src[i].X;
                float sy = src[i].Y;
                float dx = dst[i].X;
                float dy = dst[i].Y;

                a[i, 0] = sx;
                a[i, 1] = sy;
                a[i, 2] = 1;
                a[i, 3] = 0;
                a[i, 4] = 0;
                a[i, 5] = 0;
                a[i, 6] = -(float)(sx * dx); // float32 乘积（对应 C++ Point2f 乘法）
                a[i, 7] = -(float)(sy * dx);

                a[i + 4, 0] = 0;
                a[i + 4, 1] = 0;
                a[i + 4, 2] = 0;
                a[i + 4, 3] = sx;
                a[i + 4, 4] = sy;
                a[i + 4, 5] = 1;
                a[i + 4, 6] = -(float)(sx * dy);
                a[i + 4, 7] = -(float)(sy * dy);

                b[i] = dx;
                b[i + 4] = dy;
            }

            if (!LuSolve64(a, b, 8))
            {
                throw new InvalidOperationException("getPerspectiveTransform: 矩阵奇异，无法求解");
            }

            var m = new double[9];
            for (int i = 0; i < 8; i++)
            {
                m[i] = b[i];
            }

            m[8] = 1.0;
            return m;
        }

        /// <summary>
        /// 3x3 矩阵求逆，逐位复刻 cv::invert(DECOMP_LU) 的 n&lt;=3 分支
        /// （lapack.cpp：余子式公式，det3 的运算顺序原样保留）。
        /// warpPerspective 内部即用此公式把 src→dst 矩阵反转为 dst→src。
        /// </summary>
        public static double[] Invert3x3(double[] m)
        {
            double m00 = m[0], m01 = m[1], m02 = m[2];
            double m10 = m[3], m11 = m[4], m12 = m[5];
            double m20 = m[6], m21 = m[7], m22 = m[8];

            double d = m00 * (m11 * m22 - m12 * m21)
                     - m01 * (m10 * m22 - m12 * m20)
                     + m02 * (m10 * m21 - m11 * m20);
            d = 1.0 / d;

            var r = new double[9];
            r[0] = (m11 * m22 - m12 * m21) * d;
            r[1] = (m02 * m21 - m01 * m22) * d;
            r[2] = (m01 * m12 - m02 * m11) * d;
            r[3] = (m12 * m20 - m10 * m22) * d;
            r[4] = (m00 * m22 - m02 * m20) * d;
            r[5] = (m02 * m10 - m00 * m12) * d;
            r[6] = (m10 * m21 - m11 * m20) * d;
            r[7] = (m01 * m20 - m00 * m21) * d;
            r[8] = (m00 * m11 - m01 * m10) * d;
            return r;
        }

        /// <summary>
        /// 复刻 OpenCV hal::LU64f / LUImpl（matrix_decomp.cpp）：
        /// 部分选主元（首个最大者，严格大于才换）、eps = DBL_EPSILON*100 判奇异、
        /// alpha = A[j][i] * (-1/A[i][i]) 原地消元（含 b 的单列更新）、最后回代。
        /// A 为 8x8 行主序，b 为右端项，解写回 b。
        /// </summary>
        private static bool LuSolve64(double[,] a, double[] b, int m)
        {
            const double eps = 2.220446049250313e-14; // DBL_EPSILON * 100
            for (int i = 0; i < m; i++)
            {
                int k = i;
                for (int j = i + 1; j < m; j++)
                {
                    if (Math.Abs(a[j, i]) > Math.Abs(a[k, i]))
                    {
                        k = j;
                    }
                }

                if (Math.Abs(a[k, i]) < eps)
                {
                    return false;
                }

                if (k != i)
                {
                    for (int j = i; j < m; j++)
                    {
                        double t = a[i, j];
                        a[i, j] = a[k, j];
                        a[k, j] = t;
                    }

                    double tb = b[i];
                    b[i] = b[k];
                    b[k] = tb;
                }

                double d = -1.0 / a[i, i];
                for (int j = i + 1; j < m; j++)
                {
                    double alpha = a[j, i] * d;
                    for (int k2 = i + 1; k2 < m; k2++)
                    {
                        a[j, k2] += alpha * a[i, k2];
                    }

                    b[j] += alpha * b[i];
                }
            }

            for (int i = m - 1; i >= 0; i--)
            {
                double s = b[i];
                for (int k2 = i + 1; k2 < m; k2++)
                {
                    s -= a[i, k2] * b[k2];
                }

                b[i] = s / a[i, i];
            }

            return true;
        }
    }
}
