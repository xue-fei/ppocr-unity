using System;

namespace PPOCR
{
    /// <summary>PP-OCRv6 模型档位，对应 StreamingAssets 下的目录名。</summary>
    public enum ModelSize
    {
        Tiny,
        Small,
        Medium
    }

    /// <summary>
    /// 检测器配置。参数默认值来自模型自带的 inference.yml
    /// （PP-OCRv6 为 thresh=0.2 / box_thresh=0.45 / unclip_ratio=1.4 /
    ///  max_candidates=3000 / resize_long=960），与 Python 参考实现一致。
    /// </summary>
    [Serializable]
    public class DetectorConfig
    {
        /// <summary>StreamingAssets 下的模型目录名。</summary>
        public string ModelDir = "PP-OCRv6_small_det_onnx";

        public int ResizeLong = 960;
        public float Thresh = 0.2f;
        public float BoxThresh = 0.45f;
        public float UnclipRatio = 1.4f;
        public int MaxCandidates = 3000;

        public string OnnxFileName
        {
            get { return "inference.onnx"; }
        }

        public string YmlFileName
        {
            get { return "inference.yml"; }
        }

        public static DetectorConfig FromSize(ModelSize size)
        {
            return new DetectorConfig
            {
                ModelDir = "PP-OCRv6_" + size.ToString().ToLowerInvariant() + "_det_onnx"
            };
        }
    }

    /// <summary>
    /// 识别器配置。input_shape 来自 inference.yml 的 RecResizeImg
    /// （PP-OCRv6 为 [3, 48, 320]）。
    /// </summary>
    [Serializable]
    public class RecognizerConfig
    {
        /// <summary>StreamingAssets 下的模型目录名。</summary>
        public string ModelDir = "PP-OCRv6_small_rec_onnx";

        public int Channels = 3;
        public int Height = 48;
        public int Width = 320;

        public string OnnxFileName
        {
            get { return "inference.onnx"; }
        }

        public string YmlFileName
        {
            get { return "inference.yml"; }
        }

        public static RecognizerConfig FromSize(ModelSize size)
        {
            return new RecognizerConfig
            {
                ModelDir = "PP-OCRv6_" + size.ToString().ToLowerInvariant() + "_rec_onnx"
            };
        }
    }
}
