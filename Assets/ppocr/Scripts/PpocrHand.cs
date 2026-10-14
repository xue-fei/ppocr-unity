using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PPOCR
{
    /// <summary>
    /// PP-OCR 手写识别组件：在左侧画板用鼠标写字，点“识别”后用识别模型
    /// 做 CTC 前缀集束搜索，返回按置信度降序排列的 top-K 候选字。
    ///
    /// 用法：
    ///   1. 菜单 Tools/PPOCR/Create Handwriting Scene 一键生成场景；
    ///      或任意场景建空物体挂上本组件即可。
    ///   2. 运行后用鼠标在白色画板上书写，松开后点“识别”。
    ///   3. 右侧列出候选字与置信度，首选字高亮。
    ///
    /// 说明：底层识别器即 PP-OCRv6 文本识别模型（印刷体训练），
    /// 对规范手写单字识别效果较好；自由连笔手写建议写端正一些。
    /// </summary>
    public class PpocrHand : MonoBehaviour
    {
        [Tooltip("识别模型档位：Tiny / Small / Medium")]
        public ModelSize modelSize = ModelSize.Small;

        [Tooltip("返回的候选字数量（按置信度降序）")]
        public int candidateCount = 5;

        [Tooltip("手写画板像素尺寸（正方形）")]
        public int canvasSize = 360;

        [Tooltip("画笔半径（像素）")]
        public int brushRadius = 5;

        private TextRecognizer _recognizer;
        private ImageBuffer _ink;
        private Texture2D _displayTex;
        private Texture2D _cropTex;
        private bool _drawing;
        private int _lastX = -1, _lastY = -1;
        private bool _dirty;
        private bool _hasInk;
        private bool _busy;
        private string _status = "初始化…";
        private List<RecCandidate> _candidates;
        private Rect _canvasRect;

        private void Awake()
        {
            PpocrLog.Handler = Debug.Log;

            _ink = new ImageBuffer(canvasSize, canvasSize);
            ClearInk();
            _displayTex = new Texture2D(canvasSize, canvasSize, TextureFormat.RGB24, false);
            _cropTex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        }

        private IEnumerator Start()
        {
            _status = "加载识别模型…";

            bool ok = false;
            Exception err = null;
            OcrPipeline.CreateRecognizerOnlyAsync(modelSize,
                r => { _recognizer = r; ok = true; },
                e => { err = e; });

            while (!ok && err == null)
            {
                yield return null;
            }

            if (err != null)
            {
                _status = "模型加载失败: " + err.Message;
                Debug.LogError("[PPOCR-Hand] " + _status);
                yield break;
            }

            _status = "就绪：在左侧画板写字后点“识别”";
        }

        private void OnGUI()
        {
            int x = 10;
            int y = 10;

            GUI.Label(new Rect(x, y, 480, 24),
                "PP-OCR 手写识别 (" + modelSize + ")  " + _status);
            y += 28;

            // ---- 画板 ----
            _canvasRect = new Rect(x, y, canvasSize, canvasSize);
            GUI.Box(_canvasRect, GUIContent.none);
            if (_displayTex != null)
            {
                GUI.DrawTexture(_canvasRect, _displayTex);
            }

            // ---- 鼠标落墨 ----
            Event e = Event.current;
            if (_displayTex != null && e != null)
            {
                if (e.type == EventType.MouseDown && _canvasRect.Contains(e.mousePosition))
                {
                    _drawing = true;
                    int lx = (int)(e.mousePosition.x - _canvasRect.x);
                    int ly = (int)(e.mousePosition.y - _canvasRect.y);
                    Stamp(lx, ly);
                    _lastX = lx;
                    _lastY = ly;
                    e.Use();
                }
                else if (e.type == EventType.MouseDrag && _drawing)
                {
                    int lx = (int)(e.mousePosition.x - _canvasRect.x);
                    int ly = (int)(e.mousePosition.y - _canvasRect.y);
                    DrawLineSeg(_lastX, _lastY, lx, ly);
                    _lastX = lx;
                    _lastY = ly;
                    e.Use();
                }
                else if (e.type == EventType.MouseUp)
                {
                    _drawing = false;
                    _lastX = -1;
                    _lastY = -1;
                }
            }

            if (_dirty)
            {
                _ink.UploadToTexture(_displayTex);
                _dirty = false;
            }

            y += canvasSize + 10;

            // ---- 按钮 ----
            GUI.enabled = _recognizer != null && !_busy && _hasInk;
            if (GUI.Button(new Rect(x, y, 120, 30), "识别"))
            {
                StartCoroutine(RecognizeRoutine());
            }

            GUI.enabled = _recognizer != null && !_busy;
            if (GUI.Button(new Rect(x + 130, y, 120, 30), "清除"))
            {
                ClearInk();
            }

            GUI.enabled = true;
            y += 40;

            // ---- 裁剪预览 ----
            if (_cropTex != null && _cropTex.width > 2)
            {
                GUI.Label(new Rect(x, y, 200, 20), "识别用裁剪图：");
                y += 22;
                float cw = Mathf.Min(140f, _cropTex.width);
                float ch = cw * _cropTex.height / _cropTex.width;
                GUI.DrawTexture(new Rect(x, y, cw, ch), _cropTex);
                y += (int)ch + 10;
            }

            // ---- 候选列表 ----
            GUI.Label(new Rect(x, y, 200, 20), "候选字（按置信度）：");
            y += 22;
            if (_candidates != null && _candidates.Count > 0)
            {
                for (int i = 0; i < _candidates.Count; i++)
                {
                    var c = _candidates[i];
                    // 每个候选只显示首字符（单字），避免多字串误显
                    string ch = (c.Text != null && c.Text.Length > 0) ? c.Text[0].ToString() : "（空）";
                    string label = (i + 1) + ".  " + ch
                        + "   置信度 " + (c.Score * 100f).ToString("F1") + "%";
                    GUI.Label(new Rect(x, y, 460, 20), label);
                    y += 22;
                }
            }
            else if (_hasInk)
            {
                GUI.Label(new Rect(x, y, 300, 20), "（点击“识别”获取候选）");
            }
        }

        private IEnumerator RecognizeRoutine()
        {
            _busy = true;
            _status = "识别中…";
            yield return null; // 让出一帧，避免界面卡顿

            // 1. 找到墨迹外接框并裁剪（留边距）
            int minX, minY, maxX, maxY;
            if (!FindInkBounds(out minX, out minY, out maxX, out maxY))
            {
                _status = "画板为空";
                _busy = false;
                yield break;
            }

            int pad = Math.Max(8, (int)(0.1 * Math.Max(maxX - minX, maxY - minY)));
            int cx = Math.Max(0, minX - pad);
            int cy = Math.Max(0, minY - pad);
            int cw = Math.Min(_ink.Width - cx, (maxX - minX) + 2 * pad);
            int ch = Math.Min(_ink.Height - cy, (maxY - minY) + 2 * pad);
            ImageBuffer crop = _ink.Crop(cx, cy, cw, ch);

            // 2. 更新裁剪预览
            if (_cropTex != null) Destroy(_cropTex);
            _cropTex = new Texture2D(crop.Width, crop.Height, TextureFormat.RGB24, false);
            crop.UploadToTexture(_cropTex);

            // 3. 集束搜索解码 top-K
            _candidates = _recognizer.RecognizeCandidates(crop, candidateCount);

            _status = _candidates.Count > 0
                ? "识别完成：首选 “" + (_candidates[0].Text.Length > 0 ? _candidates[0].Text[0].ToString() : "") + "”"
                : "未识别出字符";

            _busy = false;
        }

        // ---------- 画板操作 ----------

        private void ClearInk()
        {
            var d = _ink.Data;
            for (int i = 0; i < d.Length; i++) d[i] = 255; // 白底
            _dirty = true;
            _hasInk = false;
            _candidates = null;
        }

        private void Stamp(int x, int y)
        {
            int r = brushRadius;
            int W = _ink.Width, H = _ink.Height;
            for (int dy = -r; dy <= r; dy++)
            {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -r; dx <= r; dx++)
                {
                    if (dx * dx + dy * dy > r * r) continue;
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    int idx = (yy * W + xx) * 3;
                    _ink.Data[idx] = 0;
                    _ink.Data[idx + 1] = 0;
                    _ink.Data[idx + 2] = 0;
                }
            }

            _dirty = true;
            _hasInk = true;
        }

        private void DrawLineSeg(int x0, int y0, int x1, int y1)
        {
            if (x0 < 0 || y0 < 0)
            {
                Stamp(x1, y1);
                return;
            }

            int dx = x1 - x0;
            int dy = y1 - y0;
            int steps = Math.Max(Math.Abs(dx), Math.Abs(dy));
            if (steps == 0)
            {
                Stamp(x1, y1);
                return;
            }

            for (int i = 0; i <= steps; i++)
            {
                int x = x0 + dx * i / steps;
                int y = y0 + dy * i / steps;
                Stamp(x, y);
            }
        }

        /// <summary>扫描墨迹（黑笔，R&lt;128）外接框；无墨迹返回 false。</summary>
        private bool FindInkBounds(out int minX, out int minY, out int maxX, out int maxY)
        {
            minX = int.MaxValue; minY = int.MaxValue;
            maxX = int.MinValue; maxY = int.MinValue;

            int W = _ink.Width, H = _ink.Height;
            var d = _ink.Data;
            for (int y = 0; y < H; y++)
            {
                int row = y * W * 3;
                for (int x = 0; x < W; x++)
                {
                    if (d[row + x * 3] < 128)
                    {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            return minX != int.MaxValue;
        }

        private void OnDestroy()
        {
            if (_recognizer != null)
            {
                _recognizer.Dispose();
                _recognizer = null;
            }

            if (_displayTex != null)
            {
                Destroy(_displayTex);
                _displayTex = null;
            }

            if (_cropTex != null)
            {
                Destroy(_cropTex);
                _cropTex = null;
            }
        }
    }
}
