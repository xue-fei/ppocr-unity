using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace PPOCR
{
    /// <summary>
    /// PP-OCR 演示组件：加载模型 → 对输入纹理执行 OCR → 绘制检测框 → 输出结果。
    ///
    /// 用法：
    ///   1. 用菜单 Tools/PPOCR/Create Demo Scene 生成演示场景（自动绑定 demo.png）；
    ///      或任意场景中建空物体挂上本组件，并在 Inspector 指定 Input Image。
    ///   2. 运行后自动执行一次；也可在 Inspector 组件右键菜单选择 "Run Again"。
    /// 输入纹理需开启 Read/Write；未指定 Input Image 时自动加载
    /// StreamingAssets/demo.png。
    /// </summary>
    public class PpocrDemo : MonoBehaviour
    {
        [Tooltip("输入图像（需可读）。为空时自动加载 StreamingAssets/demo.png")]
        public Texture2D inputImage;

        [Tooltip("模型档位：Tiny / Small / Medium")]
        public ModelSize modelSize = ModelSize.Small;

        [Tooltip("识别批次大小（对应参考实现的 rec_batch_size）")]
        public int recBatchSize = 6;

        [Tooltip("是否在结果图上绘制检测框")]
        public bool drawBoxes = true;

        [Tooltip("识别结果文本（只读，Inspector 中查看）")]
        [Multiline(20)]
        public string resultSummary = string.Empty;

        private OcrPipeline _pipeline;
        private Texture2D _resultTexture;
        private List<OcrResult> _results;
        private string _status = "初始化…";
        private bool _busy;

        private void Awake()
        {
            // 把库内日志接到 Unity 控制台
            PpocrLog.Handler = Debug.Log;
        }

        private IEnumerator Start()
        {
            yield return RunRoutine();
        }

        /// <summary>在 Inspector 组件右键菜单中手动重新执行。</summary>
        [ContextMenu("Run Again")]
        private void RunAgain()
        {
            if (_busy)
            {
                return;
            }

            StartCoroutine(RunRoutine());
        }

        private IEnumerator RunRoutine()
        {
            _busy = true;
            _status = "加载模型…";

            // 1. 准备输入图像
            ImageBuffer image = null;
            if (inputImage != null)
            {
                image = ImageBuffer.FromTexture(inputImage);
            }
            else
            {
                byte[] pngBytes = null;
                Exception error = null;
                yield return PpocrAssets.ReadAllBytesAsync("demo.png",
                    b => pngBytes = b, e => error = e);
                if (error != null)
                {
                    _status = "加载 demo.png 失败: " + error.Message;
                    Debug.LogError("[PPOCR] " + _status);
                    _busy = false;
                    yield break;
                }

                var tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
                if (!tex.LoadImage(pngBytes))
                {
                    _status = "demo.png 解码失败";
                    Debug.LogError("[PPOCR] " + _status);
                    _busy = false;
                    yield break;
                }

                image = ImageBuffer.FromTexture(tex);
                Destroy(tex);
            }

            _status = "推理中…";
            yield return null;

            // 2. 创建/复用流水线
            bool created = false;
            if (_pipeline == null)
            {
                Exception createError = null;
                yield return OcrPipeline.CreateAsyncCoroutine(modelSize, p =>
                {
                    _pipeline = p;
                    created = true;
                }, e => createError = e);

                if (createError != null || !created)
                {
                    _status = "模型加载失败: " + (createError != null ? createError.Message : "unknown");
                    Debug.LogError("[PPOCR] " + _status);
                    _busy = false;
                    yield break;
                }
            }

            // 3. 执行 OCR
            var sw = Stopwatch.StartNew();
            List<OcrResult> results = _pipeline.Run(image, recBatchSize);
            sw.Stop();
            _results = results;

            // 4. 绘制结果图
            if (drawBoxes)
            {
                _resultTexture = DrawResults(image, results);
            }

            // 5. 汇总输出
            var sb = new StringBuilder();
            sb.Append("耗时: ").Append(sw.ElapsedMilliseconds).Append(" ms，识别 ")
              .Append(results.Count).Append(" 个文本区域\n");
            sb.Append(OcrPipeline.FormatResults(results));
            resultSummary = sb.ToString();
            Debug.Log("[PPOCR] OCR 完成 (" + sw.ElapsedMilliseconds + " ms)\n" + resultSummary);

            _status = "完成";
            _busy = false;
        }

        /// <summary>把检测框画到结果图上（红色描边）。</summary>
        private static Texture2D DrawResults(ImageBuffer image, List<OcrResult> results)
        {
            var outImage = image.Clone();
            const byte cr = 255, cg = 0, cb = 0;
            foreach (var r in results)
            {
                if (r.Points == null || r.Points.Count < 4)
                {
                    continue;
                }

                for (int i = 0; i < 4; i++)
                {
                    var a = r.Points[i];
                    var b = r.Points[(i + 1) % 4];
                    CvOps.DrawLine(outImage.Data, outImage.Width, outImage.Height,
                        Mathf.RoundToInt(a.x), Mathf.RoundToInt(a.y),
                        Mathf.RoundToInt(b.x), Mathf.RoundToInt(b.y), cr, cg, cb);
                }
            }

            return outImage.ToTexture();
        }

        private void OnGUI()
        {
            const int w = 420;
            GUILayout.BeginArea(new Rect(10, 10, w, Screen.height - 20));

            GUILayout.Label("PP-OCRv6 Unity Demo (" + modelSize + ")  " + _status,
                GUILayout.Height(24));

            if (GUILayout.Button("重新识别", GUILayout.Width(120), GUILayout.Height(28)))
            {
                RunAgain();
            }

            GUILayout.Space(8);

            // 显示结果图（无结果图时显示原图）
            Texture2D show = _resultTexture != null ? _resultTexture : inputImage;
            if (show != null)
            {
                float scale = Mathf.Min((w - 8) / (float)show.width, 420f / show.height);
                int dw = Mathf.RoundToInt(show.width * scale);
                int dh = Mathf.RoundToInt(show.height * scale);
                GUILayout.Box(show, GUILayout.Width(dw), GUILayout.Height(dh));
            }

            GUILayout.Space(8);

            if (_results != null)
            {
                GUILayout.Label("识别到 " + _results.Count + " 个文本区域：");
                foreach (var r in _results)
                {
                    GUILayout.Label(string.Format("[{0:F2}] {1}", r.Score, r.Text));
                }
            }

            GUILayout.EndArea();
        }

        private void OnDestroy()
        {
            if (_pipeline != null)
            {
                _pipeline.Dispose();
                _pipeline = null;
            }

            if (_resultTexture != null)
            {
                Destroy(_resultTexture);
                _resultTexture = null;
            }
        }
    }
}
