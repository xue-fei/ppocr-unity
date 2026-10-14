using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace PPOCR
{
    /// <summary>
    /// PaddleOCR inference.yml 的最小解析器。
    /// 对应 Python 参考实现 ppocrv5_onnx/inference_config.py 的读取逻辑：
    ///   - PreProcess.transform_ops 中的 DetResizeForTest.resize_long（缺省 960）
    ///   - PreProcess.transform_ops 中的 RecResizeImg.image_shape（缺省 [3,32,320]）
    ///   - PostProcess 中的 thresh / box_thresh / unclip_ratio / max_candidates
    ///   - PostProcess.character_dict 字符字典列表
    ///
    /// 只支持该文件实际用到的 YAML 子集：
    /// 顶层 "Key:" 分节、节内 "  key: value"、列表项 "  - value"、
    /// transform_ops 的 "  - OpName:" 及其子键、行内数组 "[a, b, c]"。
    /// </summary>
    public class InferenceYml
    {
        public const int DefaultResizeLong = 960;
        public const float DefaultThresh = 0.3f;
        public const float DefaultBoxThresh = 0.6f;
        public const float DefaultUnclipRatio = 1.5f;
        public const int DefaultMaxCandidates = 1000;

        public string ModelName = string.Empty;

        // 检测参数（来自 PostProcess / DetResizeForTest）
        public int ResizeLong = DefaultResizeLong;
        public float Thresh = DefaultThresh;
        public float BoxThresh = DefaultBoxThresh;
        public float UnclipRatio = DefaultUnclipRatio;
        public int MaxCandidates = DefaultMaxCandidates;

        // 识别输入尺寸 [C, H, W]（来自 RecResizeImg.image_shape）
        public int RecChannels = 3;
        public int RecHeight = 32;
        public int RecWidth = 320;

        /// <summary>PostProcess.character_dict，未加 "blank" 前缀和空格后缀。</summary>
        public List<string> CharacterDict = new List<string>();

        public static InferenceYml Parse(string text)
        {
            var yml = new InferenceYml();
            if (string.IsNullOrEmpty(text))
            {
                return yml;
            }

            string section = null;   // 顶层节名：Global / Hpi / PreProcess / PostProcess
            string op = null;        // transform_ops 中的算子名
            bool inCharacterDict = false;
            bool inRecImageShape = false;
            var recShapeItems = new List<int>();

            string[] lines = text.Split('\n');
            foreach (string rawLine in lines)
            {
                // 只去掉行尾换行符；绝不能 Trim 空白，
                // 字典条目可能包含 U+3000 表意空格等"空白"字符。
                string line = rawLine.TrimEnd('\r', '\n');

                Match m;
                if ((m = Regex.Match(line, @"^([A-Za-z_]\w*):(.*)$")).Success)
                {
                    // 顶层节
                    section = m.Groups[1].Value;
                    op = null;
                    inCharacterDict = false;
                    inRecImageShape = false;
                    if (section == "Global")
                    {
                        string v = m.Groups[2].Value.Trim();
                        // model_name 可能在后续缩进行里，这里仅占位
                        yml.ModelName = v;
                    }

                    continue;
                }

                // 列表项："- item"。
                // 注意：item 允许是"空白"字符（如 U+3000 表意空格，
                // .NET 的 \s/\S 会把 \p{Z} 类字符视为空白），
                // 因此不能用 \S 限定，否则字典中的 U+3000 条目会被丢弃，
                // 导致其后所有字符索引整体前移、识别结果全乱。
                if ((m = Regex.Match(line, @"^\s+- (.*)$")).Success)
                {
                    string item = m.Groups[1].Value;
                    // transform_ops 列表项："- OpName:" 或 "- OpName: null"
                    Match opm = Regex.Match(item, @"^(\w+):(.*)$");
                    if (opm.Success)
                    {
                        op = opm.Groups[1].Value;
                        inCharacterDict = false;
                        inRecImageShape = false;
                        continue;
                    }

                    // 普通列表项
                    if (inCharacterDict)
                    {
                        yml.CharacterDict.Add(UnquoteScalar(item));
                    }
                    else if (inRecImageShape)
                    {
                        int v;
                        if (int.TryParse(item, NumberStyles.Integer, CultureInfo.InvariantCulture, out v))
                        {
                            recShapeItems.Add(v);
                        }
                    }

                    continue;
                }

                if ((m = Regex.Match(line, @"^\s+(\w+):(.*)$")).Success)
                {
                    string key = m.Groups[1].Value;
                    string value = m.Groups[2].Value.Trim();
                    inCharacterDict = false;
                    inRecImageShape = false;

                    if (section == "PostProcess")
                    {
                        switch (key)
                        {
                            case "thresh":
                                yml.Thresh = ParseFloat(value, DefaultThresh);
                                break;
                            case "box_thresh":
                                yml.BoxThresh = ParseFloat(value, DefaultBoxThresh);
                                break;
                            case "unclip_ratio":
                                yml.UnclipRatio = ParseFloat(value, DefaultUnclipRatio);
                                break;
                            case "max_candidates":
                                yml.MaxCandidates = ParseInt(value, DefaultMaxCandidates);
                                break;
                            case "character_dict":
                                if (string.IsNullOrEmpty(value))
                                {
                                    inCharacterDict = true; // 后续 "  - x" 行是字典条目
                                }

                                break;
                        }

                        continue;
                    }

                    if (section == "Global")
                    {
                        if (key == "model_name")
                        {
                            yml.ModelName = value;
                        }

                        continue;
                    }

                    if (section == "PreProcess")
                    {
                        if (op == "DetResizeForTest" && key == "resize_long")
                        {
                            yml.ResizeLong = ParseInt(value, DefaultResizeLong);
                        }
                        else if (op == "RecResizeImg" && key == "image_shape")
                        {
                            // 兼容行内数组 [3, 48, 320] 与块状列表两种写法
                            int[] shape = ParseInlineIntArray(value);
                            if (shape != null && shape.Length == 3)
                            {
                                yml.RecChannels = shape[0];
                                yml.RecHeight = shape[1];
                                yml.RecWidth = shape[2];
                            }
                            else if (string.IsNullOrEmpty(value))
                            {
                                inRecImageShape = true; // 后续 "  - 3" 行是维度条目
                            }
                        }

                        continue;
                    }

                    continue;
                }
            }

            // image_shape 块状列表在遍历结束后统一生效
            if (recShapeItems.Count == 3)
            {
                yml.RecChannels = recShapeItems[0];
                yml.RecHeight = recShapeItems[1];
                yml.RecWidth = recShapeItems[2];
            }

            return yml;
        }

        /// <summary>
        /// 解码 YAML 单引号标量：'x' → x，'''' → '。
        /// 双引号标量做最基础的 \" 反转义。
        /// 未加引号的标量原样返回（注意不可 Trim，U+3000 等是合法字符）。
        /// </summary>
        private static string UnquoteScalar(string s)
        {
            if (s.Length >= 2 && s[0] == '\'' && s[s.Length - 1] == '\'')
            {
                return s.Substring(1, s.Length - 2).Replace("''", "'");
            }

            if (s.Length >= 2 && s[0] == '"' && s[s.Length - 1] == '"')
            {
                return s.Substring(1, s.Length - 2).Replace("\\\"", "\"");
            }

            return s;
        }

        private static float ParseFloat(string s, float fallback)
        {
            float v;
            if (float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v))
            {
                return v;
            }

            return fallback;
        }

        private static int ParseInt(string s, int fallback)
        {
            int v;
            if (int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out v))
            {
                return v;
            }

            return fallback;
        }

        /// <summary>解析 "[3, 48, 320]" 形式的内联整数数组。</summary>
        private static int[] ParseInlineIntArray(string s)
        {
            s = s.Trim();
            if (!s.StartsWith("[") || !s.EndsWith("]"))
            {
                return null;
            }

            string inner = s.Substring(1, s.Length - 2);
            if (string.IsNullOrEmpty(inner.Trim()))
            {
                return new int[0];
            }

            string[] parts = inner.Split(',');
            var result = new List<int>(parts.Length);
            foreach (string part in parts)
            {
                int v;
                if (!int.TryParse(part.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out v))
                {
                    return null;
                }

                result.Add(v);
            }

            return result.ToArray();
        }
    }
}
