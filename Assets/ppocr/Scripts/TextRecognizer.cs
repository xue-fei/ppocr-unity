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
    /// 识别候选（含置信度）。用于手写单字 Top-K 候选展示。
    /// Score 为经 CTC 前缀集束搜索归一化后的条件概率（0~1）。
    /// </summary>
    public class RecCandidate
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
        /// 识别单张裁剪图，返回置信度最高的 topK 个候选字符串
        /// （含置信度，已按降序排序）。对应手写单字识别场景：
        /// 跳过检测，直接把裁剪图送识别，再用 CTC 前缀集束搜索
        /// 解码出最有可能的若干结果。
        /// </summary>
        public List<RecCandidate> RecognizeCandidates(ImageBuffer crop, int topK)
        {
            if (crop == null || crop.Width < 1 || crop.Height < 1)
            {
                return new List<RecCandidate>();
            }

            if (topK < 1) topK = 1;

            int w;
            float[] tensor = PreprocessSingle(crop, out w);

            float[] output;
            long[] shape;
            RunRec(tensor, 1, _imgC, _imgH, w, out output, out shape);
            if (shape.Length != 3)
            {
                throw new InvalidOperationException(
                    "识别模型输出维度异常: [" + string.Join(",", shape) + "]");
            }

            int outT = (int)shape[1];
            int outC = (int)shape[2];
            return CtcBeamSearch(output, outT, outC, topK);
        }

        /// <summary>
        /// 单图预处理（批次维度固定为 1）：缩放 + 标准化 + 右侧零填充。
        /// 与 Recognize 的单图分支逻辑一致，仅省去多图最大宽度对齐。
        /// </summary>
        private float[] PreprocessSingle(ImageBuffer crop, out int outW)
        {
            double ratio = (double)crop.Width / crop.Height;
            int resizedW = (int)Math.Ceiling(_imgH * ratio);
            if (resizedW > MaxImageWidth) resizedW = MaxImageWidth;
            if (resizedW < 1) resizedW = 1;
            outW = resizedW;

            ImageBuffer resized = (resizedW == crop.Width && _imgH == crop.Height)
                ? crop
                : CvOps.ResizeBilinear(crop, resizedW, _imgH);

            float[] tensor = new float[_imgC * _imgH * resizedW]; // 零填充
            for (int c = 0; c < _imgC; c++)
            {
                int planeOffset = c * _imgH * resizedW;
                for (int y = 0; y < _imgH; y++)
                {
                    int srcRow = y * resizedW;
                    int dstRow = planeOffset + y * resizedW;
                    for (int x = 0; x < resizedW; x++)
                    {
                        float v = resized.Data[(srcRow + x) * 3 + c] / 255f;
                        tensor[dstRow + x] = (v - 0.5f) / 0.5f;
                    }
                }
            }

            return tensor;
        }

        /// <summary>
        /// CTC 前缀集束搜索（log 空间），返回 topK 个最有可能的标签及其
        /// 归一化概率。每个时间步只对置信度最高的若干类展开（类剪枝），
        /// 在巨大字符表（中文 1.8 万类）下仍然高效。
        ///
        /// 递推（Graves 前缀搜索，l 为前缀，c 为类别）：
        ///   - c == blank：pb[l] += (pb[l] + pnb[l]) · p(c)
        ///   - c 非末字符：pnb[l+c] += (pb[l] + pnb[l]) · p(c)
        ///   - c 为末字符：pnb[l+c] += pb[l] · p(c)（补空格视为新字）
        ///                  pnb[l]   += pnb[l] · p(c)（重复同字，CTC 折叠）
        /// 末尾对所有前缀做 logSumExp 归一化得到置信度。
        /// </summary>
        private List<RecCandidate> CtcBeamSearch(float[] output, int T, int C, int topK)
        {
            double negInf = double.NegativeInfinity;

            // ---- 1. 逐时间步 log 概率（log_softmax；概率输入则取 log）----
            bool isProbs = LooksLikeProbabilities(output, 1, T, C);
            var lp = new double[T * C];
            for (int t = 0; t < T; t++)
            {
                int baseT = t * C;
                if (isProbs)
                {
                    for (int c = 0; c < C; c++)
                    {
                        float p = output[baseT + c];
                        if (p < 1e-12f) p = 1e-12f;
                        lp[baseT + c] = Math.Log(p);
                    }
                }
                else
                {
                    double maxv = output[baseT];
                    for (int c = 1; c < C; c++)
                    {
                        if (output[baseT + c] > maxv) maxv = output[baseT + c];
                    }

                    double sum = 0;
                    for (int c = 0; c < C; c++)
                    {
                        sum += Math.Exp(output[baseT + c] - maxv);
                    }

                    double lse = maxv + Math.Log(sum);
                    for (int c = 0; c < C; c++)
                    {
                        lp[baseT + c] = output[baseT + c] - lse;
                    }
                }
            }

            const int blank = 0;
            int beamWidth = Math.Max(topK + 2, 12);
            const int expandTopK = 30;

            // 前缀以字符串表示；pb = 以 blank 结尾的概率，pnb = 非 blank 结尾
            var pb = new Dictionary<string, double> { { "", 0.0 } };
            var pnb = new Dictionary<string, double> { { "", negInf } };

            for (int t = 0; t < T; t++)
            {
                int baseT = t * C;

                // 本时间步概率最高的若干类（类剪枝）
                int[] topClasses = TopKClasses(lp, baseT, C, expandTopK);

                // 当前前缀按总分剪枝到 beamWidth（pb / pnb 并集，避免遗漏只含 pnb 的前缀）
                var prefixScores = new List<KeyValuePair<string, double>>();
                var keys = new HashSet<string>(pb.Keys);
                foreach (var k in pnb.Keys) keys.Add(k);
                foreach (var k in keys)
                {
                    prefixScores.Add(new KeyValuePair<string, double>(
                        k, LogAdd(GetLog(pb, k), GetLog(pnb, k))));
                }

                prefixScores.Sort((a, b) => b.Value.CompareTo(a.Value));
                if (prefixScores.Count > beamWidth)
                {
                    prefixScores.RemoveRange(beamWidth, prefixScores.Count - beamWidth);
                }

                var pbNext = new Dictionary<string, double>();
                var pnbNext = new Dictionary<string, double>();

                foreach (var ps in prefixScores)
                {
                    string l = ps.Key;
                    double pbl = GetLog(pb, l);
                    double pnbl = GetLog(pnb, l);
                    double totalL = LogAdd(pbl, pnbl);
                    string lastChar = l.Length > 0 ? l[l.Length - 1].ToString() : string.Empty;

                    foreach (int c in topClasses)
                    {
                        if (c >= _chars.Count) continue;
                        double p = lp[baseT + c];

                        if (c == blank)
                        {
                            pbNext[l] = LogAdd(GetLog(pbNext, l), totalL + p);
                        }
                        else
                        {
                            string ch = _chars[c];
                            if (l.Length > 0 && lastChar == ch)
                            {
                                // 经 blank 补空格后接同字 -> 视为新字（l+ch）
                                pnbNext[l + ch] = LogAdd(GetLog(pnbNext, l + ch), pbl + p);
                                // 紧接同字 -> CTC 折叠，保持 l
                                pnbNext[l] = LogAdd(GetLog(pnbNext, l), pnbl + p);
                            }
                            else
                            {
                                pnbNext[l + ch] = LogAdd(GetLog(pnbNext, l + ch), totalL + p);
                            }
                        }
                    }
                }

                pb = pbNext;
                pnb = pnbNext;
            }

            // ---- 2. 归一化并取 topK（pb / pnb 并集）----
            var finals = new List<RecCandidate>();
            var fkeys = new HashSet<string>(pb.Keys);
            foreach (var k in pnb.Keys) fkeys.Add(k);
            foreach (var k in fkeys)
            {
                if (k.Length == 0) continue; // 跳过纯 blank 路径
                double total = LogAdd(GetLog(pb, k), GetLog(pnb, k));
                finals.Add(new RecCandidate { Text = k, Score = (float)total });
            }

            if (finals.Count == 0) return new List<RecCandidate>();

            double norm = negInf;
            foreach (var f in finals) norm = LogAdd(norm, f.Score);
            foreach (var f in finals) f.Score = (float)Math.Exp(f.Score - norm);

            finals.Sort((a, b) => b.Score.CompareTo(a.Score));
            if (finals.Count > topK) finals.RemoveRange(topK, finals.Count - topK);

            return finals;
        }

        /// <summary>log 空间加法：log(exp(a) + exp(b))，含 -∞ 处理。</summary>
        private static double LogAdd(double a, double b)
        {
            if (a == double.NegativeInfinity) return b;
            if (b == double.NegativeInfinity) return a;
            double m = a > b ? a : b;
            return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
        }

        private static double GetLog(Dictionary<string, double> d, string key)
        {
            double v;
            return d.TryGetValue(key, out v) ? v : double.NegativeInfinity;
        }

        /// <summary>返回 lp[baseT..baseT+C) 中取值最大的 k 个索引（类剪枝）。</summary>
        private static int[] TopKClasses(double[] lp, int baseT, int C, int k)
        {
            if (k >= C)
            {
                var all = new int[C];
                for (int i = 0; i < C; i++) all[i] = i;
                return all;
            }

            var idx = new int[k];
            var val = new double[k];
            for (int i = 0; i < k; i++)
            {
                idx[i] = i;
                val[i] = lp[baseT + i];
            }

            for (int i = k; i < C; i++)
            {
                double v = lp[baseT + i];
                int minPos = 0;
                double minV = val[0];
                for (int j = 1; j < k; j++)
                {
                    if (val[j] < minV)
                    {
                        minV = val[j];
                        minPos = j;
                    }
                }

                if (v > minV)
                {
                    idx[minPos] = i;
                    val[minPos] = v;
                }
            }

            return idx;
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
