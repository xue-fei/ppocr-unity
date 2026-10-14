using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.OnnxRuntime;

namespace PPOCR
{
    /// <summary>单个文本区域的识别结果。</summary>
    public class RecResult
    {
        public string Text = string.Empty;
        public float Score;
    }

    /// <summary>
    /// PP-OCR 文本识别器（CTC）。
    /// 对应 Python 参考实现 ppocrv5_onnx/text_recognizer/ 下的
    /// Recognizer + preprocess.OCRReisizeNormImg + postprocess.CTCLabelDecode。
    ///
    /// 流程：按批次把裁剪图缩放到固定高度（默认 48）→ 宽度取批次最大值
    /// （上限 3200）并右侧补零 → 标准化 → ONNX 推理 →
    /// CTC 解码（去重 + 去 blank）。
    /// </summary>
    public class TextRecognizer : IDisposable
    {
        private const int MaxImageWidth = 3200; // 与参考实现 OCRReisizeNormImg.max_imgW 一致

        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly string _inputName = "x";
        private readonly string _outputName = "fetch_name_0";

        private readonly int _imgC;
        private readonly int _imgH;
        private readonly int _imgW;

        /// <summary>
        /// 字符表：index 0 为 CTC blank，其后依次是 inference.yml 中
        /// character_dict 的条目，最后追加一个空格（对应参考实现的
        /// use_space_char=True）。
        /// </summary>
        private readonly List<string> _chars = new List<string>();

        /// <summary>调试转储钩子（tag, 原始 float 数据）。默认 null，仅供排查批处理问题使用。</summary>
        public static Action<string, float[]> DebugDump;

        public TextRecognizer(byte[] modelBytes, RecognizerConfig config, List<string> characterDict)
        {
            if (modelBytes == null || modelBytes.Length == 0)
            {
                throw new ArgumentException("recognizer onnx model bytes is empty");
            }

            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            if (characterDict == null || characterDict.Count == 0)
            {
                throw new ArgumentException("character dictionary is empty");
            }

            _imgC = config.Channels;
            _imgH = config.Height;
            _imgW = config.Width;

            // ["blank"] + dict + [" "]
            _chars.Add("blank");
            _chars.AddRange(characterDict);
            _chars.Add(" ");

            try
            {
                var options = new SessionOptions();
                _session = new InferenceSession(modelBytes, options);
                _runOptions = new RunOptions();

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
                    "创建识别推理会话失败: " + ex.Message, ex);
            }
        }

        /// <summary>
        /// 识别一批裁剪图（作为一个批次推理）。
        /// 裁剪图顺序由调用方保证与检测框一一对应。
        /// </summary>
        public List<RecResult> Recognize(List<ImageBuffer> crops)
        {
            if (crops == null || crops.Count == 0)
            {
                return new List<RecResult>();
            }

            // ---------- 1. 预处理：缩放 + 标准化 + 右侧补零 ----------
            // 批次宽度 = max(int(imgH * max(imgW/imgH, ratio))) ，下限 3200
            int maxBatchW = 0;
            foreach (var crop in crops)
            {
                double ratio = (double)crop.Width / crop.Height;
                double maxWhRatio = Math.Max((double)_imgW / _imgH, ratio);
                int dynW = (int)(_imgH * maxWhRatio);
                if (dynW > maxBatchW)
                {
                    maxBatchW = dynW;
                }
            }

            if (maxBatchW > MaxImageWidth)
            {
                maxBatchW = MaxImageWidth;
            }

            if (maxBatchW < 1)
            {
                maxBatchW = 1;
            }

            int batch = crops.Count;
            float[] tensor = new float[batch * _imgC * _imgH * maxBatchW]; // 零填充

            for (int b = 0; b < batch; b++)
            {
                ImageBuffer crop = crops[b];
                double ratio = (double)crop.Width / crop.Height;
                int resizedW = (int)Math.Ceiling(_imgH * ratio);
                if (resizedW > maxBatchW)
                {
                    resizedW = maxBatchW;
                }

                if (resizedW < 1)
                {
                    resizedW = 1;
                }

                ImageBuffer resized = (resizedW == crop.Width && _imgH == crop.Height)
                    ? crop
                    : CvOps.ResizeBilinear(crop, resizedW, _imgH);

                // (pixel/255 - 0.5) / 0.5，HWC -> CHW
                // 注意：源图行宽为 resizedW，目标平面行宽为 maxBatchW，
                // 必须按 (y, x) 显式映射，不能用源图线性索引直接写平面
                // （resizedW != maxBatchW 时会导致内容对角线散射）。
                int batchBase = b * _imgC * _imgH * maxBatchW;
                for (int c = 0; c < _imgC; c++)
                {
                    int planeOffset = batchBase + c * _imgH * maxBatchW;
                    for (int y = 0; y < _imgH; y++)
                    {
                        int srcRow = y * resizedW;
                        int dstRow = planeOffset + y * maxBatchW;
                        for (int x = 0; x < resizedW; x++)
                        {
                            float v = resized.Data[(srcRow + x) * 3 + c] / 255f;
                            tensor[dstRow + x] = (v - 0.5f) / 0.5f;
                        }
                    }
                }
                // 右侧 (maxBatchW - resizedW) 保持 0，即参考实现的零填充
                if (DebugDump != null)
                {
                    var slice = new float[_imgC * _imgH * maxBatchW];
                    Array.Copy(tensor, batchBase, slice, 0, slice.Length);
                    DebugDump("rec_in_b" + b + "_w" + maxBatchW, slice);
                }
            }

            // ---------- 2. 推理 ----------
            long[] shape;
            float[] output;
            RunRec(tensor, batch, _imgC, _imgH, maxBatchW, out output, out shape);

            // 输出形状 [B, T, C]
            if (shape.Length != 3)
            {
                throw new InvalidOperationException(
                    "识别模型输出维度异常: [" + string.Join(",", shape) + "]");
            }

            int outB = (int)shape[0];
            int outT = (int)shape[1];
            int outC = (int)shape[2];
            if (outB != batch)
            {
                throw new InvalidOperationException(
                    "识别模型输出批次 " + outB + " 与输入 " + batch + " 不一致");
            }

            // ---------- 3. CTC 解码 ----------
            if (DebugDump != null)
            {
                for (int b = 0; b < outB; b++)
                {
                    var slice = new float[outT * outC];
                    Array.Copy(output, b * outT * outC, slice, 0, slice.Length);
                    DebugDump("rec_out_b" + b + "_t" + outT + "_c" + outC, slice);
                }
            }

            // 判断输出是概率还是 logits：采样若干时间步，看每类和是否约为 1
            bool isProbs = LooksLikeProbabilities(output, outB, outT, outC);

            var results = new List<RecResult>(outB);
            for (int b = 0; b < outB; b++)
            {
                var sb = new StringBuilder();
                var confs = new List<float>();
                int prevIdx = -1;

                for (int t = 0; t < outT; t++)
                {
                    int baseIdx = (b * outT + t) * outC;

                    // argmax + max
                    int best = 0;
                    float bestVal = output[baseIdx];
                    for (int c = 1; c < outC; c++)
                    {
                        float v = output[baseIdx + c];
                        if (v > bestVal)
                        {
                            bestVal = v;
                            best = c;
                        }
                    }

                    // 对应位置的字符概率
                    float prob;
                    if (isProbs)
                    {
                        prob = bestVal;
                    }
                    else
                    {
                        // logits：仅在 argmax 位置做一次 softmax
                        double sum = 0;
                        for (int c = 0; c < outC; c++)
                        {
                            sum += Math.Exp(output[baseIdx + c] - bestVal);
                        }

                        prob = (float)(1.0 / sum);
                    }

                    // CTC：相邻去重 + 去 blank
                    if (best == prevIdx)
                    {
                        continue;
                    }

                    prevIdx = best;
                    if (best == 0)
                    {
                        continue;
                    }

                    if (best < _chars.Count)
                    {
                        sb.Append(_chars[best]);
                        confs.Add(prob);
                    }
                }

                float score = 0f;
                if (confs.Count > 0)
                {
                    double sum = 0;
                    foreach (var f in confs)
                    {
                        sum += f;
                    }

                    score = (float)(sum / confs.Count);
                }

                results.Add(new RecResult { Text = sb.ToString(), Score = score });
            }

            return results;
        }

        /// <summary>
        /// 抽样检测输出是否已经是概率（每类和 ≈ 1）。
        /// 对应 Python 端 np.allclose(np.sum(preds, axis=2), 1.0) 的快速版本。
        /// </summary>
        private static bool LooksLikeProbabilities(float[] output, int b, int t, int c)
        {
            if (b <= 0 || t <= 0 || c <= 0)
            {
                return false;
            }

            // 采样每个批次项的首、中、末三个时间步
            int[] sampleT = { 0, t / 2, t - 1 };
            const float tolerance = 1e-3f;
            foreach (int bt in sampleT)
            {
                if (bt < 0 || bt >= t)
                {
                    continue;
                }

                for (int bi = 0; bi < b; bi++)
                {
                    int baseIdx = (bi * t + bt) * c;
                    double sum = 0;
                    for (int ci = 0; ci < c; ci++)
                    {
                        sum += output[baseIdx + ci];
                    }

                    if (Math.Abs(sum - 1.0) > tolerance)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        private void RunRec(float[] tensor, int n, int c, int h, int w,
            out float[] output, out long[] shape)
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
                    output = new float[span.Length];
                    for (int i = 0; i < span.Length; i++)
                    {
                        output[i] = span[i];
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
