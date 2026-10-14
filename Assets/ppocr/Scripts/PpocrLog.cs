namespace PPOCR
{
    /// <summary>
    /// 轻量日志出口。默认静默；在 Unity 中把 Handler 指向 Debug.Log 即可：
    ///     PpocrLog.Handler = Debug.Log;
    /// 核心库不直接依赖 UnityEngine，便于在非 Unity 环境中测试。
    /// </summary>
    public static class PpocrLog
    {
        public static System.Action<string> Handler;

        public static void Info(string message)
        {
            if (Handler != null)
            {
                Handler(message);
            }
        }

        public static void Warn(string message)
        {
            if (Handler != null)
            {
                Handler("[PPOCR][Warn] " + message);
            }
        }

        public static void Error(string message)
        {
            if (Handler != null)
            {
                Handler("[PPOCR][Error] " + message);
            }
        }
    }
}
