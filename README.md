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
