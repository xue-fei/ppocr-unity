using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine; // 仅使用 Vector2 值类型，无引擎调用
using Stopwatch = System.Diagnostics.Stopwatch;

namespace PPOCR
{
    /// <summary>
    /// PP-OCR 端到端流水线：检测 → 裁剪 → 识别。
    /// 对应 Python 参考实现 ppocrv5_onnx/pipeline.py 的 OCRPipeline。
    ///
    /// 典型用法：
    ///     var pipeline = OcrPipeline.Create(ModelSize.Small);   // 桌面/编辑器，同步
    ///     List&lt;OcrResult&gt; results = pipeline.Run(image, recBatchSize: 6);
    /// 或异步（各平台通用，Android 必需）：
    ///     OcrPipeline.CreateAsync(config, p =&gt; { ... }, err =&gt; Debug.LogError(err));
    /// </summary>
    public class OcrPipeline : IDisposable
    {
        private readonly TextDetector _detector;
        private readonly TextRecognizer _recognizer;

        public DetectorConfig DetectorConfig { get; private set; }
        public RecognizerConfig RecognizerConfig { get; private set; }

        /// <summary>调试转储钩子（crop 序号, 裁剪图）。默认 null，仅供排查用。</summary>
        public static Action<int, ImageBuffer> DebugCropDump;

        private OcrPipeline(TextDetector detector, TextRecognizer recognizer,
            DetectorConfig detConfig, RecognizerConfig recConfig)
        {
            _detector = detector;
            _recognizer = recognizer;
            DetectorConfig = detConfig;
            RecognizerConfig = recConfig;
        }

        /// <summary>
        /// 同步创建流水线（Editor / Windows / macOS / Linux）。
        /// Android 请使用 CreateAsync（StreamingAssets 在 APK 内，无法直接读文件）。
        /// streamingRoot 为空时使用 Unity 的 Application.streamingAssetsPath，
        /// 也可传入自定义根目录（如 persistentDataPath，或测试用的绝对路径）。
        /// </summary>
        public static OcrPipeline Create(ModelSize size = ModelSize.Small, string streamingRoot = null)
        {
            return Create(DetectorConfig.FromSize(size), RecognizerConfig.FromSize(size), streamingRoot);
        }

        public static OcrPipeline Create(DetectorConfig detConfig, RecognizerConfig recConfig,
            string streamingRoot = null)
        {
            if (detConfig == null || recConfig == null)
            {
                throw new ArgumentNullException("detConfig / recConfig");
            }

            var sw = Stopwatch.StartNew();

            // 读取模型自带 inference.yml，用其参数覆盖默认值
            string detYmlPath = PpocrAssets.BuildPath(detConfig.ModelDir, detConfig.YmlFileName, streamingRoot);
            InferenceYml detYml = InferenceYml.Parse(PpocrAssets.ReadAllText(detYmlPath, streamingRoot));
            detConfig.ResizeLong = detYml.ResizeLong;
            detConfig.Thresh = detYml.Thresh;
            detConfig.BoxThresh = detYml.BoxThresh;
            detConfig.UnclipRatio = detYml.UnclipRatio;
            detConfig.MaxCandidates = detYml.MaxCandidates;

            string recYmlPath = PpocrAssets.BuildPath(recConfig.ModelDir, recConfig.YmlFileName, streamingRoot);
            InferenceYml recYml = InferenceYml.Parse(PpocrAssets.ReadAllText(recYmlPath, streamingRoot));
            recConfig.Channels = recYml.RecChannels;
            recConfig.Height = recYml.RecHeight;
            recConfig.Width = recYml.RecWidth;

            if (recYml.CharacterDict.Count == 0)
            {
                throw new InvalidOperationException(
                    "inference.yml 中未找到 character_dict: " + recYmlPath);
            }

            byte[] detBytes = PpocrAssets.ReadAllBytes(
                PpocrAssets.BuildPath(detConfig.ModelDir, detConfig.OnnxFileName, streamingRoot),
                streamingRoot);
            byte[] recBytes = PpocrAssets.ReadAllBytes(
                PpocrAssets.BuildPath(recConfig.ModelDir, recConfig.OnnxFileName, streamingRoot),
                streamingRoot);

            var detector = new TextDetector(detBytes, detConfig);
            var recognizer = new TextRecognizer(recBytes, recConfig, recYml.CharacterDict);

            sw.Stop();
            PpocrLog.Info(string.Format(
                "[PPOCR] 模型加载完成 ({0} ms) det={1} rec={2} dict={3} 项",
                sw.ElapsedMilliseconds, detConfig.ModelDir, recConfig.ModelDir,
                recYml.CharacterDict.Count));

            return new OcrPipeline(detector, recognizer, detConfig, recConfig);
        }

        /// <summary>
        /// 异步创建流水线（协程式回调，全平台可用；Android 下 StreamingAssets
        /// 位于 APK 内，必须走 UnityWebRequest 通道）。
        /// </summary>
        public static void CreateAsync(DetectorConfig detConfig, RecognizerConfig recConfig,
            Action<OcrPipeline> onDone, Action<Exception> onError)
        {
            if (detConfig == null || recConfig == null)
            {
                onError(new ArgumentNullException("detConfig / recConfig"));
                return;
            }

            PpocrCoroutine.Run(CreateCoroutine(detConfig, recConfig, onDone, onError));
        }

        public static void CreateAsync(ModelSize size,
            Action<OcrPipeline> onDone, Action<Exception> onError)
        {
            CreateAsync(DetectorConfig.FromSize(size), RecognizerConfig.FromSize(size), onDone, onError);
        }

        /// <summary>
        /// 协程版异步创建：可在协程中 yield 等待完成。
        /// </summary>
        public static IEnumerator CreateAsyncCoroutine(DetectorConfig detConfig,
            RecognizerConfig recConfig, Action<OcrPipeline> onDone, Action<Exception> onError)
        {
            return CreateCoroutine(detConfig, recConfig, onDone, onError);
        }

        public static IEnumerator CreateAsyncCoroutine(ModelSize size,
            Action<OcrPipeline> onDone, Action<Exception> onError)
        {
            return CreateCoroutine(DetectorConfig.FromSize(size),
                RecognizerConfig.FromSize(size), onDone, onError);
        }

        private static IEnumerator<object> CreateCoroutine(DetectorConfig detConfig,
            RecognizerConfig recConfig, Action<OcrPipeline> onDone, Action<Exception> onError)
        {
            string detYmlPath = PpocrAssets.BuildPath(detConfig.ModelDir, detConfig.YmlFileName);
            string recYmlPath = PpocrAssets.BuildPath(recConfig.ModelDir, recConfig.YmlFileName);
            string detOnnxPath = PpocrAssets.BuildPath(detConfig.ModelDir, detConfig.OnnxFileName);
            string recOnnxPath = PpocrAssets.BuildPath(recConfig.ModelDir, recConfig.OnnxFileName);

            byte[] detYmlBytes = null;
            byte[] recYmlBytes = null;
            byte[] detBytes = null;
            byte[] recBytes = null;
            Exception error = null;

            yield return PpocrAssets.ReadAllBytesAsync(detYmlPath,
                b => detYmlBytes = b, e => error = e);
            if (error != null) { onError(error); yield break; }

            yield return PpocrAssets.ReadAllBytesAsync(recYmlPath,
                b => recYmlBytes = b, e => error = e);
            if (error != null) { onError(error); yield break; }

            yield return PpocrAssets.ReadAllBytesAsync(detOnnxPath,
                b => detBytes = b, e => error = e);
            if (error != null) { onError(error); yield break; }

            yield return PpocrAssets.ReadAllBytesAsync(recOnnxPath,
                b => recBytes = b, e => error = e);
            if (error != null) { onError(error); yield break; }

            try
            {
                var sw = Stopwatch.StartNew();

                // 读取模型自带 inference.yml，用其参数覆盖默认值
                InferenceYml detYml = InferenceYml.Parse(Encoding.UTF8.GetString(detYmlBytes));
                detConfig.ResizeLong = detYml.ResizeLong;
                detConfig.Thresh = detYml.Thresh;
                detConfig.BoxThresh = detYml.BoxThresh;
                detConfig.UnclipRatio = detYml.UnclipRatio;
                detConfig.MaxCandidates = detYml.MaxCandidates;

                InferenceYml recYml = InferenceYml.Parse(Encoding.UTF8.GetString(recYmlBytes));
                recConfig.Channels = recYml.RecChannels;
                recConfig.Height = recYml.RecHeight;
                recConfig.Width = recYml.RecWidth;

                if (recYml.CharacterDict.Count == 0)
                {
                    throw new InvalidOperationException(
                        "inference.yml 中未找到 character_dict: " + recYmlPath);
                }

                var detector = new TextDetector(detBytes, detConfig);
                var recognizer = new TextRecognizer(recBytes, recConfig, recYml.CharacterDict);

                sw.Stop();
                PpocrLog.Info(string.Format(
                    "[PPOCR] 模型加载完成 ({0} ms) det={1} rec={2} dict={3} 项",
                    sw.ElapsedMilliseconds, detConfig.ModelDir, recConfig.ModelDir,
                    recYml.CharacterDict.Count));

                onDone(new OcrPipeline(detector, recognizer, detConfig, recConfig));
            }
            catch (Exception ex)
            {
                onError(ex);
            }
        }

        /// <summary>
        /// 执行 OCR。
        /// </summary>
        /// <param name="image">RGB 输入图像</param>
        /// <param name="recBatchSize">识别批次大小（对应参考实现的 rec_batch_size，默认 8）</param>
        public List<OcrResult> Run(ImageBuffer image, int recBatchSize = 8)
        {
            if (image == null)
            {
                throw new ArgumentNullException("image");
            }

            if (recBatchSize < 1)
            {
                recBatchSize = 1;
            }

            var swDet = Stopwatch.StartNew();
            List<DetBox> boxes = _detector.Detect(image);
            swDet.Stop();

            if (boxes.Count == 0)
            {
                return new List<OcrResult>();
            }

            // 裁剪（对应 detector.crop_poly.get_minarea_rect_crop）
            var crops = new List<ImageBuffer>(boxes.Count);
            foreach (var box in boxes)
            {
                var crop = CropPoly(image, box.Points);
                if (DebugCropDump != null)
                {
                    DebugCropDump(crops.Count, crop);
                }

                crops.Add(crop);
            }

            // 分批识别
            var swRec = Stopwatch.StartNew();
            var recResults = new List<RecResult>(boxes.Count);
            for (int i = 0; i < crops.Count; i += recBatchSize)
            {
                int count = Math.Min(recBatchSize, crops.Count - i);
                var batch = crops.GetRange(i, count);
                recResults.AddRange(_recognizer.Recognize(batch));
            }

            swRec.Stop();

            var results = new List<OcrResult>(boxes.Count);
            for (int i = 0; i < boxes.Count; i++)
            {
                var points = new List<Vector2>(4);
                foreach (var p in boxes[i].Points)
                {
                    points.Add(new Vector2(p.X, p.Y));
                }

                results.Add(new OcrResult(recResults[i].Text, recResults[i].Score, points));
            }

            PpocrLog.Info(string.Format("[PPOCR] 检测 {0} 框 ({1} ms)，识别 ({2} ms)",
                boxes.Count, swDet.ElapsedMilliseconds, swRec.ElapsedMilliseconds));

            return results;
        }

        /// <summary>
        /// 最小面积矩形裁剪 + 透视校正。
        /// 对应 crop_poly.get_minarea_rect_crop / get_rotate_crop_image：
        /// INTER_CUBIC + BORDER_REPLICATE，且当裁剪图高宽比 ≥ 1.5 时旋转 90°。
        /// </summary>
        private static ImageBuffer CropPoly(ImageBuffer img, List<Pt2> points)
        {
            List<Pt2> box = Geom.OrderQuad(Geom.MinAreaRect(points));

            int cropW = (int)Math.Max(
                Pt2.Distance(box[0], box[1]),
                Pt2.Distance(box[2], box[3]));
            int cropH = (int)Math.Max(
                Pt2.Distance(box[0], box[3]),
                Pt2.Distance(box[1], box[2]));

            if (cropW < 1) cropW = 1;
            if (cropH < 1) cropH = 1;

            var std = new List<Pt2>
            {
                new Pt2(0, 0),
                new Pt2(cropW, 0),
                new Pt2(cropW, cropH),
                new Pt2(0, cropH)
            };

            // 与参考实现一致：getPerspectiveTransform(points, pts_std) 得到 src->dst，
            // warpPerspective 内部再用 3x3 余子式公式求逆得到 dst->src
            double[] m = Geom.GetPerspectiveTransform(box, std);
            double[] minv = Geom.Invert3x3(m);
            ImageBuffer dst = CvOps.WarpPerspective(img, minv, cropW, cropH);

            if (dst.Height * 1.0 / dst.Width >= 1.5)
            {
                dst = CvOps.Rotate90CCW(dst);
            }

            return dst;
        }

        /// <summary>把识别结果整理为可读文本（每行一条，带分数）。</summary>
        public static string FormatResults(List<OcrResult> results)
        {
            if (results == null || results.Count == 0)
            {
                return "(无文本)";
            }

            var sb = new StringBuilder();
            for (int i = 0; i < results.Count; i++)
            {
                sb.Append(i + 1).Append(". ").Append(results[i].ToString()).Append('\n');
            }

            return sb.ToString();
        }

        public void Dispose()
        {
            if (_detector != null)
            {
                _detector.Dispose();
            }

            if (_recognizer != null)
            {
                _recognizer.Dispose();
            }
        }
    }
}
