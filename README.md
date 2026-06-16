# PART 01 · 機器學習基礎 (Machine Learning Basics)

> 學習筆記｜AI Learning Roadmap
> 日期：2026-06-16

---

## 一、ML 演算法的三個元素

任何機器學習方法，都可以拆成這三個要素：

| **1. Hypothesis class（假設空間）** | 要從「哪一群函數」裡面挑？線性模型？決策樹？神經網路？ |
| **2. Loss function（損失函數）** | `L(f, data)`，f 有多差，Loss越小越好 |
| **3. Optimization（最佳化）** | 怎麼在 Hypothesis class 中找到 loss 最小的 f？ |

核心目標：找一個函數 `f(x) ≈ y`。

---

## 二、Loss Function（損失函數）

「預測」跟「真實答案」差多少。

### MSE — Mean Square Error（迴歸用）

$$ L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2 $$

- 預測點到真實點的「距離平方」平均
- 最小化 MSE ≈ 最大化相似度

### Cross-Entropy（分類用）

$$ L = -\frac{1}{n}\sum_i \sum_c y_{ic}\,\log(p_{ic}) $$

- Cross-Entropy 算的是「罰款」。 模型訓練時想辦法把這個罰款壓到最小,逼自己「該有信心的時候有信心,而且要答對」。

---
為什麼錯得越有信心,罰得越重?

關鍵在那個 log(對數):

給的預測是正確答案的機率    log 罰款
            0.9       →   很小（押對了，輕罰）
            0.5       →   中等
            0.1       →   很大
            0.01      →   超級大
            趨近 0     →   罰款趨近無限大

意思是:模型越是「斬釘截鐵地說正確答案不可能」,罰款就飆高。 這逼模型學會謙虛——不確定的時候別亂押 100%,不然一旦錯了會死得很慘。

### Hinge Loss（SVM 用）

$$ L = \max(0,\; 1 - y\hat{y}) $$

- 這裡的 `y` 是 **+1 / −1**（不是 0/1），公式才成立

---

## 三、Gradient Descent（梯度下降）：怎麼找 f

$$ \theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta) $$

> 每一步往 loss 下降最快的方向走 η 的距離（η = 學習率 learning rate）

| 種類 | 做法 | 特性 |
|------|------|------|
| **Batch GD** | 每步用「全部」資料算 gradient | 資料量大記憶體會爆 |
| **SGD**（Stochastic GD） | 每步只用「一筆」資料 | noise 較大，實務上效率不足 |
| **Mini-batch SGD** | 每步用 32 / 64 / 256 筆資料 | 所有現代 Deep Learning 都用這個 |

---

## 四、Bias & Variance（影響 ML 的兩大誤差來源）

給一個資料集 D，訓練模型 f_D。期望誤差可拆成三部份：

$$ E\big[(y - f_D(x))^2\big] = \text{Bias}^2 + \text{Variance} + \underbrace{\text{noise}}_{\text{不可避免的干擾}} $$

| 來源 | 意思 | 對應狀況 |
|------|------|----------|
| **Bias（偏差）** | 模型太簡單，怎麼學都學不會 | **underfit** |
| **Variance（變異）** | 模型太靈活，太貼合 D 的細節而過擬合 | **overfit** |

### 怎麼診斷？（看 train / val 表現）

| 觀察 | 診斷 |
|------|------|
| train acc 高，val acc 低 | → **High Variance**（overfit） |
| train acc 低，val acc 低 | → **High Bias**（underfit） |

### 怎麼解決？

| 問題 | 解法 |
|------|------|
| **High Variance** | 更多資料、regularization、early stopping |
| **High Bias** | 換更大模型、更長 training、更好的 feature |

---

## 五、Regularization（防 overfit，對付 High Variance）

| 方法 | 做法 | 備註 |
|------|------|------|
| **L2 / weight decay** | 在 loss 後面加 `λ‖θ‖²`，逼 weight 不要太大 | 最常用 |
| **L1 / Lasso** | 加 `λ‖θ‖`，會把不重要的 weight 壓到 0 | Deep Learning 少見；feature selection / 線性模型常見 |
| **Dropout** | 訓練時隨機把 p% 的神經元關掉 | 小資料 + CV 還是好用 |
| **Early stopping** | 看 val loss，連續 n 個 epoch 不下降就停止訓練 | 配合 best checkpoint 保存 |

> 💡 L1 的靈魂 = 稀疏性（把不重要的權重歸零），等於自動幫你做特徵選擇。

---

## 六、Train / Validation / Test 切分

| 集合 | 模型有沒有看到？ | 用途 |
|------|------------------|------|
| **Train** | 看到 | 用來算 gradient（訓練） |
| **Validation** | 沒「直接」看到 | 用來調 hyperparameter（learning rate、batch size、early stop） |
| **Test** | 完全沒碰 | 做最終驗收 / demo 才會遇到 |

> 💡 Validation 沒被拿去算 gradient，但你用它「選模型、調參數」，所以你的選擇**間接**被它影響了。這就是為什麼還要留一個完全沒碰過的 **Test** 做最終公正驗收。

---

## 七、經典演算法 · 七大家族

### ① Linear / Logistic Regression（Baseline 首選）

$$ f(x) = w^\top x + b $$

- 線性迴歸輸出是「一個任意數字」
- 外面包一個 **sigmoid**，把任意數字壓到 0~1 → 變成機率輸出 → **Logistic Regression**
- **用途**：先當 baseline model，未來要驗證新模型效能時，可拿來對比

### ② SVM — Margin 最大化（常配 RBF Kernel）

- 在能分開兩類的所有 hyperplane 中，找 **margin（間隔）最大**的那一個
- **margin 越大 → 泛化越好**（選離兩邊都最遠的那條線）
- **Support Vector**：只有壓在邊界線上的那幾個點，決定了線要畫哪
- **Kernel Trick**：資料在平面上無法用直線分開時，把資料升維，用更高維座標來切；常見 kernel 是 **RBF**
- **適用**：樣本少、維度高時好用

### ③ k-Nearest Neighbors（k-NN）

- 要預測一個新點是貓 or 狗？
  1. 找最相鄰的 k 個鄰居（例如 k=5）
  2. 看這 5 個大多是什麼 → 投票決定
- **沒有訓練過程**（沒有要學的參數）
- **Lazy learner**：訓練時什麼都不做，但每次預測都要跟所有資料算距離 → 資料量大就需要加速索引
- **RAG 的底層就是 k-NN**：把問題變成一個座標，去資料庫找最近的幾筆資料
- **缺點**：維度太高，距離會變得沒意義　→ 通常先把資料壓縮成 embedding

### ④ Decision Tree（決策樹）

- 一連串 yes / no 的問題（例：屋齡 < 10 年？坪數夠大？市中心？）
- 從最上面開始，每個節點問一個「某特徵 > 某數值」，一路往下走直到葉（leaf）給出預測答案
- **怎麼決定問什麼問題？** 用 **Gini impurity** 或 **entropy**：問完之後最能把兩類分乾淨的問題先問
- **缺點**：極度容易 overfit；對資料很敏感（改一點資料，整棵樹可能長得完全不一樣）
- **優點**：不需要正規化（資料不用縮放）、天生能處理分類型特徵、**可解釋（能畫整棵樹給人看）**

### ⑤ Random Forest（隨機森林 · bagging）

- 單棵 tree 易 overfit，那就種很多棵樹，每棵都不太一樣，然後讓它們投票
- **讓每棵樹不一樣的關鍵：**
  - 每棵樹只看隨機一部份資料（bootstrap 抽樣）
  - 每次分裂又只用隨機的一部份特徵
- 原本單棵樹變異很大，用很多棵樹的「變異平均」會較穩定
- **修正版**：理論上樹「完全獨立」時 variance 才會降到原本的 1/n；但實際上樹彼此正相關，所以下降幅度**達不到 1/n**。→ **這正是要「隨機選特徵」的原因：讓樹之間盡量不相關，variance 才壓得越低。**
- **優點**：加更多樹只會更穩定、較難 overfit、不太需要調參（預設就很好）、能給 **feature importance**（特徵重要性排行榜）

### ⑥ Gradient Boosting Trees（GBT｜XGBoost / LightGBM / CatBoost）

跟 Random Forest 不同的點：樹的關係是**一棵接一棵**，專門在修正前面留下的錯誤。

1. 先種第一棵樹做粗略預測，一定有誤差（**residual = 真實值 − 預測值**）
2. 第二棵樹的任務是學習第一棵的殘差
3. 把兩棵樹相加，預測更準確
4. 第三棵樹用來補第二棵的錯……（一直接力下去）

$$ F_{t+1}(x) = F_t(x) + \eta \cdot h_t(x), \qquad h_t(x) = -\nabla L(F_t(x)) $$

> ⭐ 概念就是 **Gradient Descent**——只是這次是在「函數空間」做梯度下降，每棵新樹 = 一個梯度步伐。

### ⑦ Naive Bayes

- 用機率算「這東西可能是哪一類」
- 看到這些特徵 x，計算它屬於各類別 y 的機率
- **Naive（天真）的意思**：假設所有特徵彼此獨立、互不相關
  - 例：判斷垃圾郵件，假設「免費」「中獎」兩詞不相關；但實際上這兩個詞常一起出現在垃圾郵件 → 這個假設很天真，但實務上夠用
- **優點**：在文本分類上很好用、速度快、幾乎不用調參

---

## 八、無監督學習（沒有標準答案）

### Clustering（分群）：把相似的東西聚成一堆

| 方法 | 做法 |
|------|------|
| **k-means** | 先決定要分成幾群（k），反覆把點分到最近的中心 → 更新中心 → 重複 |
| **DBSCAN** | 基於密度分群（不用先指定 k，能找出任意形狀、自動排除雜訊） |
| **GMM** | 軟性分群：一個點「70% 像 A、30% 像 B」 |

### 降維：把高維資料壓扁

| 方法 | 做法 |
|------|------|
| **PCA** | 用線性變換找出「資料變化最大的方向」，高維壓成低維 |
| **UMAP / t-SNE** | 把超高維資料壓成 2 維畫成圖，自動把相似的聚在一起（視覺化用） |

---

## 九、拿到問題該選哪個演算法

1. **先用 Linear / Logistic Regression** 當基準線，未來驗證新模型效能時可拿來對比
2. **看樣本數（表格資料）**：
   - 樣本數 < 1000 → 用 **SVM**
   - 反之（資料量大）→ 用 **Boosting Tree（XGBoost）**
   - （樣本少 + 高維 → SVM）
3. **是否 streaming（資料持續進來）？**
   - 需要 streaming → 用 **SGD** 或 **Naive Bayes**（有資料進來就更新一次，不用整批重訓）
4. **是否需要展示 / 解釋性？**
   - 需要解釋 → 用 **決策樹**，可清楚看到 feature importance
5. **圖片 / 文字 / 語音輸入？**
   - 直接用 **深度學習**（這類資料的相鄰單位有關聯）
   - 圖片 → **CNN**（利用空間局部性）
   - 文字 → **Transformer**（利用序列前後依賴）

---

## 十、Mini Project 實戰心得（Kaggle House Prices）

> 用真實資料把上面所有概念跑過一遍。目標是親手實作「模型一步步變強、為什麼變強」。
> 檔案：`kaggle-house-prices/house_prices_part01.ipynb`

### 完整進步歷程

| 階段 | 做了什麼 | Val RMSE | 比上一步進步 |
|------|----------|----------|------------|
| ① Linear baseline | 只用數值欄位、缺值補中位數 | 0.1519 | — |
| ② 特徵工程 + RandomForest | 造 TotalSF/HouseAge…、one-hot、種樹 | 0.1451 | 0.0068 |
| ③ XGBoost（原始參數） | 表格之王 + 5-fold CV | 0.1221 | 0.0230 |
| ④ 調參對付 overfit | 加 regularization 參數 | 0.1211 | 0.0010（小） |
| ⑤ 處理 outlier | 刪 2 棟「超大坪數卻超便宜」的怪屋 | **0.1142** | **0.0069** |

### 學到的關鍵重點

**1. 目標是降低 val，不是縮小 train/val 差距**
- 差距大只是「症狀」，不是「病」。判斷成功看 **val RMSE 有沒有降**。
- 親手做過「調過頭」的經驗：把 `subsample`、`colsample` 壓到 0.2、正規化加太重 → 差距從 0.095 縮到 0.033（很漂亮），但 val 反而升到 0.1260。
- 原因：train RMSE 從 0.0268 暴衝到 0.0933，代表模型連訓練資料都學不好了 → **從 overfit 用力過頭衝向 underfit**。這就是 Bias-Variance trade-off 。

**2. 正規化的「甜蜜點」在中間**
```
正規化太少 → overfit（train 超低、val 高、差距大）
正規化太多 → underfit（train 也變高、val 沒救、差距小但沒意義）
        ↓
   目標 = 中間讓 val 最低的甜蜜點 🎯
```

**3. ⭐ 改善資料 > 拼命調參（今天最大的領悟）**
- 拼命調參只擠出 0.0010；刪掉 **2 棵怪屋**（占 0.14%）直接降 0.0069，效果是調參的 **7 倍**。
- 為什麼 2 筆資料威力這麼大？因為 **MSE 的平方項**會放大離群值的破壞力（差 5 倍 → 罰 25 倍），模型為了討好怪屋會扭曲自己。
- 當模型接近天花板時：**先把資料弄乾淨，遠比無止境調參有效**。

**4. 刪 outlier 的紀律**
- 只刪「有真實世界合理解釋」的（這裡是建商特殊交易、非市場價）。
- 用雙條件 `GrLivArea > 4000 且 SalePrice < 300000` 精準鎖定，不誤刪正常豪宅。
- 不能為了分數好看亂刪正常資料 → 那是自欺欺人。


- **`early_stopping_rounds` 不能配 `cross_val_score`**：early stopping 需要固定的 `eval_set`，但 CV 是自動輪流切、沒有固定驗證集 → 會產生報錯。兩者要分開用。

---

