using System;
using System.Collections.Generic;

namespace PPOCR
{
    /// <summary>
    /// 凸多边形的圆角外扩，对应 Python 端 DB 后处理中的：
    ///     offset = pyclipper.PyclipperOffset()
    ///     offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    ///     expanded = offset.Execute(distance)
    ///
    /// 数学上，凸多边形外扩 distance 等价于与半径为 distance 的圆盘
    /// 做 Minkowski 和：每条边向外平移 distance，顶点处用圆弧连接。
    /// 本实现即按此构造（圆弧按固定角步长离散）。
    ///
    /// 注意：参考实现中 unclip 的输入始终是 get_mini_boxes 输出的
    /// 最小面积矩形（凸四边形），因此凸假设成立；
    /// 下游仅对外扩结果求最小面积外接矩形，圆弧离散化的结果与
    /// pyclipper 在该场景下完全一致。
    /// </summary>
    public static class PolygonExpander
    {
        /// <summary>
        /// 外扩凸多边形。roundJoins 为 true 时顶点处生成圆弧离散点
        /// （更贴近 pyclipper 输出，可用于绘制）；为 false 时仅输出
        /// 平移后的边（求最小面积外接矩形的结果相同，速度更快）。
        /// </summary>
        public static List<Pt2> OffsetConvexRound(List<Pt2> poly, double distance, bool roundJoins = true)
        {
            if (poly == null || poly.Count < 3 || distance <= 0)
            {
                return poly != null ? new List<Pt2>(poly) : new List<Pt2>();
            }

            var pts = new List<Pt2>(poly);
            // 统一为逆时针（外法线 = 边方向的右侧法线）
            if (IsClockwise(pts))
            {
                pts.Reverse();
            }

            int n = pts.Count;
            var result = new List<Pt2>(n * 7);

            for (int i = 0; i < n; i++)
            {
                Pt2 a = pts[i];
                Pt2 b = pts[(i + 1) % n];
                double dx = b.X - a.X;
                double dy = b.Y - a.Y;
                double len = Math.Sqrt(dx * dx + dy * dy);
                if (len < 1e-12)
                {
                    continue;
                }

                // 逆时针多边形的外法线：(dy, -dx) / len
                double nx = dy / len;
                double ny = -dx / len;
                result.Add(new Pt2((float)(a.X + distance * nx), (float)(a.Y + distance * ny)));

                if (roundJoins)
                {
                    // 顶点 b 处的圆弧：从当前边外法线转向下一条边外法线
                    Pt2 c = pts[(i + 2) % n];
                    double dx2 = c.X - b.X;
                    double dy2 = c.Y - b.Y;
                    double len2 = Math.Sqrt(dx2 * dx2 + dy2 * dy2);
                    if (len2 < 1e-12)
                    {
                        continue;
                    }

                    double nx2 = dy2 / len2;
                    double ny2 = -dx2 / len2;

                    double a1 = Math.Atan2(ny, nx);
                    double a2 = Math.Atan2(ny2, nx2);
                    double delta = a2 - a1;
                    while (delta <= -Math.PI) delta += 2 * Math.PI;
                    while (delta > Math.PI) delta -= 2 * Math.PI;
                    if (delta < 0) delta += 2 * Math.PI; // 凸多边形外角 ∈ (0, π)

                    int steps = Math.Max(1, (int)Math.Ceiling(delta / (Math.PI / 9.0)));
                    for (int s = 1; s <= steps; s++)
                    {
                        double t = a1 + delta * s / steps;
                        result.Add(new Pt2(
                            (float)(b.X + distance * Math.Cos(t)),
                            (float)(b.Y + distance * Math.Sin(t))));
                    }
                }
            }

            return result;
        }

        private static bool IsClockwise(List<Pt2> pts)
        {
            double area = 0;
            int n = pts.Count;
            for (int i = 0; i < n; i++)
            {
                var a = pts[i];
                var b = pts[(i + 1) % n];
                area += a.X * b.Y - b.X * a.Y;
            }

            return area < 0; // 鞋带公式为负表示顺时针
        }
    }
}
