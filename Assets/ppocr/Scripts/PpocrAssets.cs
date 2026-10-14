using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace PPOCR
{
    /// <summary>
    /// StreamingAssets 读取工具。
    /// 桌面平台（Editor / Windows / macOS / Linux）可直接读文件；
    /// Android 的 StreamingAssets 打包在 APK 内，必须走 UnityWebRequest。
    /// </summary>
    public static class PpocrAssets
    {
        /// <summary>
        /// 拼接 StreamingAssets 下的完整路径。
        /// root 为空时使用 Unity 的 Application.streamingAssetsPath；
        /// 也可传入自定义根目录（如 persistentDataPath 或测试用绝对路径）。
        /// </summary>
        public static string BuildPath(string dir, string fileName, string root = null)
        {
            if (string.IsNullOrEmpty(root))
            {
                root = DefaultStreamingRoot();
            }

            return root + "/" + dir + "/" + fileName;
        }

        /// <summary>
        /// Unity 引擎调用单独隔离（且禁止内联）：
        /// 非 Unity 宿主（如控制台验证 harness）即使不走到该分支，
        /// 也不会因 JIT 绑定引擎 internal call 而抛 SecurityException。
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(
            System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static string DefaultStreamingRoot()
        {
            return Application.streamingAssetsPath;
        }

        /// <summary>同步读取（桌面/编辑器可用；Android 会抛异常）。</summary>
        public static byte[] ReadAllBytes(string streamingRelativePath, string root = null)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            throw new NotSupportedException(
                "Android 平台 StreamingAssets 位于 APK 内，请使用 ReadAllBytesAsync 或 OcrPipeline.CreateAsync");
#else
            return File.ReadAllBytes(streamingRelativePath);
#endif
        }

        /// <summary>
        /// 同步读取文本（UTF-8）。path 应为 BuildPath 生成的完整路径；
        /// root 参数仅为与 ReadAllBytes 签名保持一致，可传 null。
        /// </summary>
        public static string ReadAllText(string path, string root = null)
        {
            byte[] bytes = ReadAllBytes(path, root);
            // 去掉 UTF-8 BOM，避免 YAML 首行解析异常
            if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            {
                return System.Text.Encoding.UTF8.GetString(bytes, 3, bytes.Length - 3);
            }

            return System.Text.Encoding.UTF8.GetString(bytes);
        }

        /// <summary>
        /// 异步读取文件（协程），全平台可用。
        /// 返回 Coroutine，可被其他协程 yield 等待。
        /// </summary>
        public static Coroutine ReadAllBytesAsync(string streamingRelativePath,
            Action<byte[]> onDone, Action<Exception> onError)
        {
            return PpocrCoroutine.Run(ReadFileCoroutine(streamingRelativePath, onDone, onError));
        }

        private static IEnumerator ReadFileCoroutine(string path, Action<byte[]> onDone, Action<Exception> onError)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            using (var www = UnityWebRequest.Get(path))
            {
                www.downloadHandler = new DownloadHandlerBuffer();
                yield return www.SendWebRequest();

                while (!www.isDone)
                {
                    yield return null;
                }

                if (www.isNetworkError || www.isHttpError)
                {
                    onError(new IOException("读取 StreamingAssets 失败: " + path + " - " + www.error));
                }
                else
                {
                    onDone(www.downloadHandler.data);
                }
            }
#else
            byte[] data = null;
            Exception error = null;
            try
            {
                data = File.ReadAllBytes(path);
            }
            catch (Exception ex)
            {
                error = ex;
            }

            yield return null; // 让出一帧，保持协程语义一致

            if (error != null)
            {
                onError(error);
            }
            else
            {
                onDone(data);
            }
#endif
        }
    }

    /// <summary>
    /// 隐藏的协程宿主：让纯静态类也能启动 Unity 协程。
    /// </summary>
    public class PpocrCoroutine : MonoBehaviour
    {
        private static PpocrCoroutine _instance;

        public static PpocrCoroutine Instance
        {
            get
            {
                if (_instance == null)
                {
                    var go = new GameObject("[PpocrCoroutine]");
                    _instance = go.AddComponent<PpocrCoroutine>();
                    DontDestroyOnLoad(go);
                    go.hideFlags = HideFlags.HideInHierarchy;
                }

                return _instance;
            }
        }

        public static Coroutine Run(IEnumerator routine)
        {
            return Instance.StartCoroutine(routine);
        }
    }
}
