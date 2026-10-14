using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace PPOCR.EditorTools
{
    /// <summary>
    /// 批处理验证工具：在真实 Unity 运行时中执行一次 OCR，
    /// 并把结果写成 JSON，便于与 Python 参考实现（ppocrv5-onnx）做数值比对。
    ///
    /// 命令行用法：
    ///   Unity.exe -batchmode -quit -projectPath &lt;工程&gt; ^
    ///     -executeMethod PPOCR.EditorTools.PpocrValidation.RunValidation
    /// </summary>
    public static class PpocrValidation
    {
        private const string OutputPath = "PpocrUnityResult.json";
        private const string OutputPathMediumTest = "PpocrUnityResult_medium_test.json";

        [MenuItem("Tools/PPOCR/Run Validation")]
        public static void RunValidation()
        {
            try
            {
                Run("demo.png", ModelSize.Small, OutputPath);
            }
            catch (Exception ex)
            {
                Debug.LogError("[PPOCR] 验证失败: " + ex);
                EditorApplication.Exit(1);
                return;
            }

            EditorApplication.Exit(0);
        }

        [MenuItem("Tools/PPOCR/Run Validation (medium + test.jpg)")]
        public static void RunValidationMediumTest()
        {
            try
            {
                Run("test.jpg", ModelSize.Medium, OutputPathMediumTest);
            }
            catch (Exception ex)
            {
                Debug.LogError("[PPOCR] 验证失败: " + ex);
                EditorApplication.Exit(1);
                return;
            }

            EditorApplication.Exit(0);
        }

        /// <summary>
        /// CvOps.ResizeBilinear 的 bit 级验证：读取 Python 端（已与 cv2 4.13
        /// 100% 对齐）生成的测试向量，逐字节比对 C# 实现输出。
        /// 向量格式：[int32 count] + 每例 [int32 sw, sh, dstW, dstH] [src bytes] [expected bytes]
        /// </summary>
        [MenuItem("Tools/PPOCR/Run Resize Bit-Exact Validation")]
        public static void RunResizeValidation()
        {
            string vectorPath = Path.Combine(
                Path.GetTempPath(), "ppocr_probe", "resize_vectors.bin");
            if (!File.Exists(vectorPath))
            {
                Debug.LogError("[PPOCR] 测试向量不存在: " + vectorPath);
                EditorApplication.Exit(1);
                return;
            }

            byte[] all = File.ReadAllBytes(vectorPath);
            int pos = 0;
            int count = BitConverter.ToInt32(all, pos); pos += 4;
            int pass = 0, fail = 0;
            var sb = new StringBuilder();

            for (int t = 0; t < count; t++)
            {
                int sw = BitConverter.ToInt32(all, pos);
                int sh = BitConverter.ToInt32(all, pos + 4);
                int dw = BitConverter.ToInt32(all, pos + 8);
                int dh = BitConverter.ToInt32(all, pos + 12);
                pos += 16;
                int srcLen = sw * sh * 3;
                int dstLen = dw * dh * 3;

                var srcBuf = new ImageBuffer(sw, sh);
                Buffer.BlockCopy(all, pos, srcBuf.Data, 0, srcLen);
                pos += srcLen;
                var expected = new byte[dstLen];
                Buffer.BlockCopy(all, pos, expected, 0, dstLen);
                pos += dstLen;

                ImageBuffer got = CvOps.ResizeBilinear(srcBuf, dw, dh);
                int nDiff = 0;
                for (int i = 0; i < dstLen; i++)
                {
                    if (got.Data[i] != expected[i])
                    {
                        nDiff++;
                    }
                }

                if (nDiff == 0)
                {
                    pass++;
                }
                else
                {
                    fail++;
                    sb.AppendLine(string.Format("FAIL {0}x{1}->{2}x{3}: {4}/{5} bytes differ",
                        sw, sh, dw, dh, nDiff, dstLen));
                }
            }

            string report = string.Format("[PPOCR] Resize bit-exact: {0}/{1} pass, {2} fail",
                pass, count, fail);
            Debug.Log(report);
            if (sb.Length > 0)
            {
                Debug.LogError(sb.ToString());
            }

            string outPath = Path.Combine(
                Directory.GetParent(Application.dataPath).FullName, "PpocrResizeValidation.txt");
            File.WriteAllText(outPath, report + "\n" + sb.ToString());
            EditorApplication.Exit(fail == 0 ? 0 : 1);
        }

        private static void Run(string imageFileName, ModelSize size, string outputFileName)
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string outputPath = Path.Combine(projectRoot, outputFileName);

            // 1. 读取测试图片（Assets 根目录或 StreamingAssets）
            string imagePath = Path.Combine(Application.dataPath, imageFileName);
            if (!File.Exists(imagePath))
            {
                imagePath = Path.Combine(Application.streamingAssetsPath, imageFileName);
            }

            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException("测试图片不存在: " + imagePath);
            }

            byte[] imageBytes = File.ReadAllBytes(imagePath);
            var tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
            if (!tex.LoadImage(imageBytes))
            {
                throw new InvalidOperationException(imageFileName + " 解码失败");
            }

            ImageBuffer image = ImageBuffer.FromTexture(tex);
            Debug.Log(string.Format("[PPOCR] 输入图像 {0}x{1}，模型档位 {2}",
                image.Width, image.Height, size));

            // 2. 执行 OCR
            using (var pipeline = OcrPipeline.Create(size))
            {
                List<OcrResult> results = pipeline.Run(image, 6);

                // 3. 输出 JSON
                var sb = new StringBuilder();
                sb.Append("[\n");
                for (int i = 0; i < results.Count; i++)
                {
                    OcrResult r = results[i];
                    sb.Append("  {\"text\": \"").Append(JsonEscape(r.Text))
                      .Append("\", \"score\": ").Append(r.Score.ToString("F6", CultureInfo.InvariantCulture))
                      .Append(", \"points\": [");
                    for (int p = 0; p < r.Points.Count; p++)
                    {
                        if (p > 0)
                        {
                            sb.Append(", ");
                        }

                        sb.Append('[')
                          .Append(r.Points[p].x.ToString("F1", CultureInfo.InvariantCulture)).Append(", ")
                          .Append(r.Points[p].y.ToString("F1", CultureInfo.InvariantCulture)).Append(']');
                    }

                    sb.Append("]}");
                    if (i < results.Count - 1)
                    {
                        sb.Append(',');
                    }

                    sb.Append('\n');
                }

                sb.Append("]\n");

                File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
                Debug.Log(string.Format("[PPOCR] 验证完成，{0} 个结果已写入 {1}",
                    results.Count, outputPath));
            }
        }

        private static string JsonEscape(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }

            var sb = new StringBuilder(s.Length + 16);
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20)
                        {
                            sb.Append("\\u").Append(((int)c).ToString("x4"));
                        }
                        else
                        {
                            sb.Append(c);
                        }

                        break;
                }
            }

            return sb.ToString();
        }
    }
}
