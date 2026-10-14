using System.Collections.Generic;
using UnityEngine;

namespace PPOCR
{
    /// <summary>
    /// 单个文本区域的识别结果。
    /// 对应 Python 参考实现 ppocrv5_onnx.schema.OCRResult。
    /// </summary>
    public class OcrResult
    {
        /// <summary>识别出的文本。</summary>
        public string Text;

        /// <summary>该区域的平均字符置信度（0~1）。</summary>
        public float Score;

        /// <summary>
        /// 文本框的四个角点，顺序固定为 [左上, 右上, 右下, 左下]，
        /// 坐标为原图像素坐标（左上角为原点，单位像素）。
        /// </summary>
        public List<Vector2> Points;

        public OcrResult()
        {
        }

        public OcrResult(string text, float score, List<Vector2> points)
        {
            Text = text;
            Score = score;
            Points = points;
        }

        /// <summary>
        /// 四边形角点的轴对齐外接矩形 (xMin, yMin, xMax, yMax)。
        /// 对应 Python 端 OCRResult.xyxy()。
        /// </summary>
        public Rect GetBoundingRect()
        {
            if (Points == null || Points.Count == 0)
            {
                return new Rect(0f, 0f, 0f, 0f);
            }

            float xMin = float.MaxValue;
            float yMin = float.MaxValue;
            float xMax = float.MinValue;
            float yMax = float.MinValue;
            foreach (var p in Points)
            {
                if (p.x < xMin) xMin = p.x;
                if (p.y < yMin) yMin = p.y;
                if (p.x > xMax) xMax = p.x;
                if (p.y > yMax) yMax = p.y;
            }

            return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
        }

        public override string ToString()
        {
            var r = GetBoundingRect();
            return string.Format(
                "[{0:F3}] \"{1}\" @ ({2:F0},{3:F0})-({4:F0},{5:F0})",
                Score, Text, r.xMin, r.yMin, r.xMax, r.yMax);
        }
    }
}
