# AI_CUP
本專案用於從穿戴式裝置感測數據中萃取特徵，並預測使用者的多個屬性，包括：
性別（gender）、持拍手（hold racket handed）、球齡（play years）、程度（level）
## 📁 專案結構
```bash
.
├── train_data/ # 訓練用感測資料 (.txt 檔，每個檔案對應一個 unique_id)
├── test_data/ # 測試用感測資料
├── train_info.csv # 包含每個 unique_id 的標籤欄位
├── test_info.csv # 測試資料的 ID 清單
├── main.py # 主程式
└── sample_submission.csv # 模型輸出的預測結果
```
## ⚙️ 安裝需求

使用此專案前，請確保你已安裝以下 Python 套件：
```bash
pip install numpy pandas scikit-learn
```

## 🚀 執行方式
在指令列輸入以下指令以執行整個流程：

```bash
python main.py
```
此程式會執行：  
1.從 train_data/ 與 test_data/ 中讀取感測資料。  
2.對每筆資料執行特徵萃取（FFT + 時域統計量）。  
3.標準化特徵。  
4.使用 RandomForestClassifier 分別訓練四個屬性模型。  
5.預測測試資料中的各屬性機率。  
6.輸出結果至 sample_submission.csv。  
## 特徵萃取方式
每筆感測資料包含 6 個感測器軸向資料（Ax, Ay, Az, Gx, Gy, Gz），本程式針對以下特徵進行萃取：
Ax 軸的前 64 點進行 FFT，取其 mean、std、max、min。
每一軸向的 mean 與 std。
共計 16 維特徵。
## 預測輸出格式
輸出為 sample_submission.csv，包含以下欄位：
unique_id
gender, hold racket handed：二分類機率（label=1 的機率）
play years_0, play years_1, play years_2
level_2, level_3, level_4, level_5

所有值皆為機率，並四捨五入至小數點第四位。

