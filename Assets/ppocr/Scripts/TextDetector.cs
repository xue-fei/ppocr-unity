using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PPOCR
{
    /// <summary>单个检测框（原图像素坐标的四点 + 置信度）。</summary>
    public class DetBox
    {
        /// <summary>[左上, 右上, 右下, 左下]。</summary>
        public List<Pt2> Points = new List<Pt2>();

        public float Score;
    }

    /// <summary>
    /// PP-OCR 文本检测器（DB 算法）。
    /// 对应 Python 参考实现 ppocrv5_onnx/text_detector/ 下的
    /// Detector + preprocess.DetResize/NormalizeImage + postprocess.DBPostProcess。
    ///
    /// 流程：resize_long 缩放 → 标准化 → ONNX 推理 → 二值化 →
    /// 轮廓查找 → 最小面积矩形 → box_score 过滤 → unclip 外扩 →
    /// 最小面积矩形 → 缩放回原图坐标。
    /// </summary>
    public class TextDetector : IDisposable
    {
        private const int MaxSideLimit = 4000; // 与参考实现 preprocess.max_side_limit 一致
        private const int MinBoxSize = 3;      // DBPostProcess.min_size

        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly string _inputName = "x";
        private readonly string _outputName = "fetch_name_0";

        private readonly int _resizeLong;
        private readonly float _thresh;
        private readonly float _boxThresh;
        private readonly float _unclipRatio;
        private readonly int _maxCandidates;

        /// <summary>调试转储钩子（tag, 原始 float 数据）。默认 null，仅供排查用。</summary>
        public static Action<string, float[]> DebugDump;

        /// <summary>最近一次 Detect 的轮廓数量（调试用，对比 cv2.findContours）。</summary>
        public static int LastContourCount;

        /// <summary>最近一次 Detect 经 min_size 过滤后的候选数（调试用）。</summary>
        public static int LastCandidateCount;

        /// <summary>调试：候选框信息（原图缩放前坐标、打分、是否保留），最多 256 条。</summary>
        public class CandInfo
        {
            public List<Pt2> Quad;
            public float Score;
            public bool Kept;
        }

        public static List<CandInfo> LastCandidates = new List<CandInfo>();

        // 标准化参数（与参考实现 NormalizeImage 默认值一致）
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        public TextDetector(byte[] modelBytes, DetectorConfig config)
        {
            if (modelBytes == null || modelBytes.Length == 0)
            {
                throw new ArgumentException("detector onnx model bytes is empty");
            }

            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            _resizeLong = config.ResizeLong;
            _thresh = config.Thresh;
            _boxThresh = config.BoxThresh;
            _unclipRatio = config.UnclipRatio;
            _maxCandidates = config.MaxCandidates;

            try
            {
                var options = new SessionOptions();
                _session = new InferenceSession(modelBytes, options);
                _runOptions = new RunOptions();

                // 从模型元数据读取真实的输入输出名（各版本 PP-OCR 均为 x / fetch_name_0）
                if (_session.InputNames.Count > 0)
                {
                    _inputName = _session.InputMetadata.ContainsKey("x")
                        ? "x"
                        : _session.InputNames[0];
                }

                if (_session.OutputNames.Count > 0)
                {
                    _outputName = _session.OutputNames[0];
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    "创建检测推理会话失败: " + ex.Message, ex);
            }
        }

        /// <summary>
        /// 对 RGB 图像执行文本检测，返回文本框列表（顺序近似光栅扫描序，
        /// 与参考实现的 findContours 输出序一致）。
        /// </summary>
        public List<DetBox> Detect(ImageBuffer rgb)
        {
            if (rgb == null)
            {
                throw new ArgumentNullException("rgb");
            }

            // ---------- 1. 预处理（DetResize + NormalizeImage） ----------
            ImageBuffer img = rgb;
            int srcH = img.Height;
            int srcW = img.Width;
            if (srcH + srcW < 64)
            {
                img = PadToMin32(img);
                srcH = img.Height;
                srcW = img.Width;
            }

            // resize_long：长边缩放到 resize_long，短边等比
            double ratio = (double)_resizeLong / Math.Max(srcH, srcW);
            int resizeH = (int)(srcH * ratio);
            int resizeW = (int)(srcW * ratio);

            // 超过 max_side_limit 时整体缩小
            if (Math.Max(resizeH, resizeW) > MaxSideLimit)
            {
                ratio = (double)MaxSideLimit / Math.Max(resizeH, resizeW);
                resizeH = (int)(resizeH * ratio);
                resizeW = (int)(resizeW * ratio);
            }

            // 网络要求边长为 32 的倍数（最小 32）
            resizeH = Math.Max((int)Math.Round(resizeH / 32.0) * 32, 32);
            resizeW = Math.Max((int)Math.Round(resizeW / 32.0) * 32, 32);

            ImageBuffer resized = (resizeH == srcH && resizeW == srcW)
                ? img
                : CvOps.ResizeBilinear(img, resizeW, resizeH);

            // (pixel/255 - mean) / std，HWC -> CHW
            float[] tensor = new float[3 * resizeH * resizeW];
            int planeSize = resizeH * resizeW;
            for (int y = 0; y < resizeH; y++)
            {
                for (int x = 0; x < resizeW; x++)
                {
                    int srcIdx = (y * resizeW + x) * 3;
                    for (int c = 0; c < 3; c++)
                    {
                        float v = resized.Data[srcIdx + c] / 255f;
                        tensor[c * planeSize + y * resizeW + x] = (v - Mean[c]) / Std[c];
                    }
                }
            }

            // ---------- 2. 推理 ----------
            if (DebugDump != null)
            {
                DebugDump("det_in_" + resizeW + "x" + resizeH, tensor);
            }

            long[] outShape;
            var pred = new float[planeSize];
            RunDet(tensor, 1, 3, resizeH, resizeW, pred, out outShape);

            int outH = (int)outShape[outShape.Length - 2];
            int outW = (int)outShape[outShape.Length - 1];

            if (DebugDump != null)
            {
                DebugDump("det_out_" + outW + "x" + outH, pred);
            }

            if (outH != resizeH || outW != resizeW)
            {
                // 个别模型输出分辨率与输入不一致时，取左上角区域（正常不会发生）
                PpocrLog.Warn(string.Format(
                    "det 输出尺寸 {0}x{1} 与输入 {2}x{3} 不一致", outW, outH, resizeW, resizeH));
            }

            // ---------- 3. DB 后处理 ----------
            // 3.1 二值化
            var bitmap = new byte[planeSize];
            for (int i = 0; i < planeSize; i++)
            {
                bitmap[i] = pred[i] > _thresh ? (byte)1 : (byte)0;
            }

            // 3.2 轮廓查找
            List<List<Pt2>> contours = ContourFinder.FindContours(bitmap, outW, outH);
            LastContourCount = contours.Count;

            // 3.3 逐轮廓：最小面积矩形 -> 打分 -> unclip -> 最小面积矩形 -> 缩放回原图
            double widthScale = (double)srcW / outW;
            double heightScale = (double)srcH / outH;

            var boxes = new List<DetBox>();
            int numContours = Math.Min(contours.Count, _maxCandidates);
            int cand = 0;
            LastCandidates.Clear();
            for (int ci = 0; ci < numContours; ci++)
            {
                List<Pt2> contour = contours[ci];

                var quad = Geom.OrderQuad(Geom.MinAreaRect(contour));
                if (Geom.QuadMinSide(quad) < MinBoxSize)
                {
                    continue;
                }

                cand++;

                float score = BoxScoreFast(pred, outW, outH, quad);
                bool kept = score >= _boxThresh;
                if (LastCandidates.Count < 256)
                {
                    // 记录检测图坐标（outW x outH 空间）的 quad，便于与 cv2 对比
                    var q = new List<Pt2>(4);
                    foreach (var p in quad)
                    {
                        q.Add(new Pt2(p.X, p.Y));
                    }

                    LastCandidates.Add(new CandInfo { Quad = q, Score = score, Kept = kept });
                }

                if (!kept)
                {
                    continue;
                }

                double area = Geom.ShoelaceArea(quad);
                double length = Geom.Perimeter(quad, true);
                if (length < 1e-6)
                {
                    continue;
                }

                double distance = area * _unclipRatio / length;
                List<Pt2> expanded = PolygonExpander.OffsetConvexRound(quad, distance);
                var quad2 = Geom.OrderQuad(Geom.MinAreaRect(expanded));
                if (Geom.QuadMinSide(quad2) < MinBoxSize + 2)
                {
                    continue;
                }

                // 缩放回原图坐标并裁剪到图像范围
                var scaled = new List<Pt2>(4);
                foreach (var p in quad2)
                {
                    float bx = ClampF((float)Math.Round(p.X * widthScale), 0f, srcW);
                    float by = ClampF((float)Math.Round(p.Y * heightScale), 0f, srcH);
                    scaled.Add(new Pt2(bx, by));
                }

                boxes.Add(new DetBox { Points = scaled, Score = score });
            }

            LastCandidateCount = cand;
            return boxes;
        }

        /// <summary>
        /// box_score_fast：四边形区域内概率图的均值。
        /// 对应 DBPostProcess.box_score_fast。
        /// </summary>
        private static float BoxScoreFast(float[] pred, int w, int h, List<Pt2> quad)
        {
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            foreach (var p in quad)
            {
                if (p.X < minX) minX = p.X;
                if (p.X > maxX) maxX = p.X;
                if (p.Y < minY) minY = p.Y;
                if (p.Y > maxY) maxY = p.Y;
            }

            int xMin = Math.Max(0, Math.Min((int)Math.Floor(minX), w - 1));
            int xMax = Math.Max(0, Math.Min((int)Math.Ceiling(maxX), w - 1));
            int yMin = Math.Max(0, Math.Min((int)Math.Floor(minY), h - 1));
            int yMax = Math.Max(0, Math.Min((int)Math.Ceiling(maxY), h - 1));
            if (xMax < xMin || yMax < yMin)
            {
                return 0f;
            }

            int maskW = xMax - xMin + 1;
            int maskH = yMax - yMin + 1;
            var mask = new byte[maskW * maskH];

            // 与参考实现一致：四边形平移后取整填充
            var shifted = new List<Pt2>(4);
            foreach (var p in quad)
            {
                shifted.Add(new Pt2(p.X - xMin, p.Y - yMin));
            }

            CvOps.FillPolygon(mask, maskW, maskH, shifted);
            return CvOps.MaskedMean(pred, w, mask, xMin, yMin, xMax, yMax);
        }

        /// <summary>图像补齐到最小 32x32（对应 preprocess.image_padding）。</summary>
        private static float ClampF(float v, float lo, float hi)
        {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }

        private static ImageBuffer PadToMin32(ImageBuffer src)
        {
            int w = Math.Max(32, src.Width);
            int h = Math.Max(32, src.Height);
            if (w == src.Width && h == src.Height)
            {
                return src;
            }

            var padded = new ImageBuffer(w, h); // 零填充
            for (int y = 0; y < src.Height; y++)
            {
                Array.Copy(src.Data, y * src.Width * 3, padded.Data, y * w * 3, src.Width * 3);
            }

            return padded;
        }

        /// <summary>
        /// 执行检测推理，把输出概率图（[1,1,H,W] 的第一张图）拷贝进 predBuffer。
        /// 输出 span 只在 OrtValue 存活期间有效，因此必须在 using 内完成拷贝。
        /// </summary>
        /// <summary>调试用：直接对给定输入张量执行 det 推理（跳过预处理）。</summary>
        public void RunDetDebug(float[] tensor, int n, int c, int h, int w,
            float[] pred, out long[] shape)
        {
            RunDet(tensor, n, c, h, w, pred, out shape);
        }

        private void RunDet(float[] tensor, int n, int c, int h, int w,
            float[] predBuffer, out long[] shape)
        {
            long[] dims = { n, c, h, w };
            using (var inputValue = OrtValue.CreateTensorValueFromMemory(tensor, dims))
            {
                var inputs = new Dictionary<string, OrtValue> { { _inputName, inputValue } };
                var outputNames = new List<string> { _outputName };
                using (var results = _session.Run(_runOptions, inputs, outputNames))
                {
                    var outValue = results[0];
                    shape = outValue.GetTensorTypeAndShape().Shape;
                    ReadOnlySpan<float> span = outValue.GetTensorDataAsSpan<float>();
                    int copyCount = Math.Min(predBuffer.Length, span.Length);
                    for (int i = 0; i < copyCount; i++)
                    {
                        predBuffer[i] = span[i];
                    }
                }
            }
        }

        public void Dispose()
        {
            if (_runOptions != null)
            {
                _runOptions.Dispose();
            }

            if (_session != null)
            {
                _session.Dispose();
            }
        }
    }
}
