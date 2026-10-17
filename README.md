# 臺大校總區周遭 YouBike 2.0 車輛數最適化動態調度模型與演算法

> **國立臺灣大學 經濟學系《資料科學與社會探究》（Econ 5166）第一組專題報告**  
> **專案儲存庫**：[https://github.com/peterko2005/DSSI2026-group-1-project-latex](https://github.com/peterko2005/DSSI2026-group-1-project-latex)

---

## 👥 團隊成員與專案管理規範

* **專案經理與主幹管理（Maintainer / PM）**：柯典辰（B13303082）
* **資料工程與分析團隊（DE / DA）**：蔡睿燊（B13303057）、李芸榛（B13703038）、馮芳英（B13303096）

### ⚠️ 協作審核機制（Pull Request Review Protocol）
本專案為確保 LaTeX 排版語法嚴謹度、防止編譯衝突（Merge Conflicts）並維持 0 Overfull `\hbox`，**主幹分支由 PM 獨立審核管理**：
1. **嚴禁直接 Push 至 `main` 或 `backup` 分支**。
2. 所有組員若需修改文字、新增章節或更新圖表，**必須先建立獨立分支（Branch）進行修改，完成後透過 GitHub 發起「推送請求（Pull Request, PR）」**。
3. 由 PM 在本地編譯確認 `XeLaTeX` 通過且版面無溢出後，點擊允許合併（Merge）至主幹。

---

## 🚀 組員新手指南：如何下載、修改與提交 PR

### 第一步：整包抓取專案（Clone）
在終端機（Terminal）執行以下指令，將整包專案克隆至本地：

```bash
git clone https://github.com/peterko2005/DSSI2026-group-1-project-latex.git
cd DSSI2026-group-1-project-latex
```

> **💡 提示**：若需要完整的中繼編譯快取，可拉取 `backup` 分支：
> ```bash
> git clone -b backup https://github.com/peterko2005/DSSI2026-group-1-project-latex.git
> ```

---

### 第二步：本地編譯檢驗（Compile）
本專案基於臺大碩博士學位論文母版（`ntuthesis.cls`），排版依賴 `xeCJK` 與 `fontspec`，**必須且只能使用 XeLaTeX 引擎編譯**（pdfLaTeX 無法運行）。

#### 終端機一鍵編譯：
```bash
/Library/TeX/texbin/xelatex -interaction=nonstopmode -shell-escape main.tex
```
*(Windows 使用者直接執行 `xelatex -interaction=nonstopmode -shell-escape main.tex`)*

編譯成功後，根目錄會生成最新版的 `main.pdf`。

#### 編輯器設定（VS Code / TeXShop / Overleaf）：
* **VS Code**：安裝 `LaTeX Workshop` 套件，將 Recipe 設定為 `xelatex ➞ xelatex`。
* **Overleaf**：如需上傳至 Overleaf，進入 **Menu → Compiler**，強制切換為 **XeLaTeX**。

---

### 第三步：建立功能分支並進行修改（Branch Workflow）
**永遠不要在本地的 `main` 分支直接修改！** 請建立屬於你自己或該任務的分支：

```bash
# 1. 切換並建立新分支（例如修改第二章資料處理）
git checkout -b feature/ch02-data-cleaning

# 2. 編輯對應的 LaTeX 手稿（例如 contents/chapter02.tex）
# 3. 本地編譯驗證通過（確認 main.pdf 正常產出）

# 4. 提交你的修改
git add contents/chapter02.tex
git commit -m "feat(ch02): update exposure-adjusted count regression description"

# 5. 推送你的分支至 GitHub
git push -u origin feature/ch02-data-cleaning
```

---

### 第四步：在 GitHub 上發起推送請求（Pull Request）
1. 打開瀏覽器進入 GitHub 儲存庫：[DSSI2026-group-1-project-latex](https://github.com/peterko2005/DSSI2026-group-1-project-latex)。
2. 頂部會自動彈出黃色提示條，點擊綠色的 **"Compare & pull request"**。
3. 填寫你本次修改的內容摘要（例如：「完成了第二章 3.2 截斷資料處理說明」）。
4. 點擊 **"Create pull request"** 送出。
5. PM 收到通知後會進行代碼審查與編譯測試，確認無誤後點擊允許合併（Merge）！

---

## 雙分支架構說明（Dual-Branch Architecture）

本專案與母庫同步採用嚴格的「雙分支推播體系」：

* **`main`（展示版）**：透過 `.gitignore` 自動排除所有編譯暫存檔（`.aux`, `.log`, `.synctex.gz`, `.toc`, `.lof`, `.lot` 等），保持 GitHub 網頁端乾淨易讀。
* **`backup`（全量備份版）**：100% 完整狀態鏡像，收錄手稿、PDF 以及所有編譯快取，供跨設備極速還原與續寫。

### PM 雙分支一鍵同步指令：
```bash
./sync.sh push "commit message"   # 自動完成 main 乾淨版與 backup 全量版同步推送
./sync.sh pull                    # 自遠端 backup 分支拉取最新全量檔案
```

---

## 📂 專案核心目錄架構

```text
DSSI2026-group-1-project-latex/
├── main.tex                    # 主版面與排版樣式配置入口
├── ntusetup.tex                # 論文個人資訊、指導教授與組員名單
├── ntuthesis.cls               # NTU 官方論文類別定義檔
├── environments.tex            # 特殊數學與定理環境設定
├── sync.sh                     # main + backup 雙分支同步腳本
├── .gitignore                  # LaTeX 編譯暫存檔過濾規範
├── README.md                   # 本文件（小組協作與 PR 提交指南）
├── TEMPLATE.md                 # 原 NTU Thesis 樣板詳細設定說明
├── AGENTS.md                   # AI 助理寫作與程式碼開發規範
├── CLAUDE.md                   # Claude 專用導引說明
├── contents/                   # 論文各章節主體手稿
│   ├── chapter01.tex           # 第一章：緒論（研究背景與動機）
│   ├── chapter02.tex           # 第二章：資料特徵與真實需求還原
│   ├── chapter03.tex           # 第三章：動態重平衡最佳化模型與演算法
│   ├── chapter04.tex           # 第四章：模擬環境設計與候選策略評估
│   └── chapter05.tex           # 第五章：結論與政策意涵
├── front/                      # 前篇（中英文摘要、誌謝、符號表）
│   ├── abstract.tex            # 中文摘要
│   └── denotation.tex          # 符號系統定義表
├── back/                       # 後篇（附錄、參考文獻）
│   └── references.bib          # BibLaTeX 格式參考文獻庫
├── figures/                    # 論文插圖與校徽向量檔
└── fonts/                      # 內建開源中英文字型（TW-Kai、Tinos 等）
```
