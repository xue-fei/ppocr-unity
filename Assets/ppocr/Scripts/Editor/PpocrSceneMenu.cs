using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PPOCR.EditorTools
{
    /// <summary>
    /// 一键生成 PP-OCR 演示场景：
    /// 主相机 + 挂载 PpocrDemo 的空物体，并自动绑定
    /// StreamingAssets/demo.png 作为输入图像。
    /// </summary>
    public static class PpocrSceneMenu
    {
        private const string ScenePath = "Assets/ppocr/Scenes/PpocrDemo.unity";
        private const string DemoImagePath = "Assets/StreamingAssets/demo.png";

        [MenuItem("Tools/PPOCR/Create Demo Scene", false, 10)]
        public static void CreateDemoScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 主相机
            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.18f, 0.18f, 0.18f);
            camGo.transform.position = new Vector3(0, 0, -10);

            // 演示组件
            var demoGo = new GameObject("PPOCR");
            var demo = demoGo.AddComponent<PpocrDemo>();

            // 自动绑定演示图片（若存在）
            var demoImage = AssetDatabase.LoadAssetAtPath<Texture2D>(DemoImagePath);
            if (demoImage != null)
            {
                demo.inputImage = demoImage;
            }
            else
            {
                Debug.LogWarning("[PPOCR] 未找到 " + DemoImagePath +
                    "，场景创建后请在 Inspector 手动指定 Input Image。");
            }

            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.Refresh();
            Debug.Log("[PPOCR] 演示场景已创建: " + ScenePath);
        }

        [MenuItem("Tools/PPOCR/Create Demo Scene", true)]
        private static bool ValidateCreateDemoScene()
        {
            return !EditorApplication.isPlaying;
        }
    }
}
