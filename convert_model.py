import os
import pickle
import xgboost as xgb

def convert_pkl_to_json(pkl_path, json_path=None):
    """
    自动检测并将 sklearn 封装的 XGBoost 模型 (.pkl) 转换为 .json 格式。
    """
    if not os.path.exists(pkl_path):
        print(f"❌ 找不到模型文件: {pkl_path}")
        return

    if json_path is None:
        json_path = os.path.splitext(pkl_path)[0] + ".json"

    try:
        print(f"🔍 正在加载 {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        # 尝试获取底层 booster
        booster = None
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif hasattr(model, "save_model"):
            booster = model
        else:
            print("⚠️ 未找到 XGBoost 模型结构，跳过转换。")
            return

        booster.save_model(json_path)
        print(f"✅ 模型已成功转换为 {json_path}")

    except Exception as e:
        print(f"❌ 转换失败: {e}")


if __name__ == "__main__":
    convert_pkl_to_json("E:\\shear\\single_model.pkl")
    convert_pkl_to_json("E:\\shear\\group_model.pkl")
