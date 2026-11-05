#!/usr/bin/env bash
# ==============================================================================
# DSSI 2026 Group 1 LaTeX 雙分支極速同步工具 (Dual-Branch Git Sync)
# 遠端儲存庫：https://github.com/peterko2005/DSSI2026-group-1-project-latex.git
# 用途：
#   1. ./sync.sh push "Commit message" : 同步乾淨展示版至 main 分支，全量版至 backup 分支
#   2. ./sync.sh pull                  : 自遠端 backup 分支拉取最新檔案
#   3. ./sync.sh status                : 檢視分支狀態
# ==============================================================================

set -e

PROPOSAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROPOSAL_DIR"

ACTION="${1:-status}"
MSG="${2:-update: sync proposal changes}"
DSSI_URL="https://github.com/peterko2005/DSSI2026-group-1-project-latex.git"

TOP_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
IS_MONOREPO=false
if [ -n "$TOP_DIR" ] && [ "$TOP_DIR" != "$PROPOSAL_DIR" ]; then
  IS_MONOREPO=true
  REL_PATH="${PROPOSAL_DIR#$TOP_DIR/}"
fi

case "$ACTION" in
  push)
    if [ "$IS_MONOREPO" = true ]; then
      echo "🚀 [1/4] 偵測到於 LaTeX 母庫內運行，正在確保本地變更已納入..."
      cd "$TOP_DIR"
      git remote get-url dssi >/dev/null 2>&1 || git remote add dssi "$DSSI_URL"
      
      git add "$REL_PATH"
      if ! git diff-index --quiet HEAD -- "$REL_PATH"; then
        git commit -m "$MSG"
      else
        echo "  母庫 main 分支對應路徑無新變更需要 commit。"
      fi

      echo "🔄 [2/4] 正在拉取 DSSI 遠端分支指標..."
      git fetch dssi main backup >/dev/null 2>&1 || true

      echo "📦 [3/4] 正在生成 DSSI 獨立專案之 main（乾淨版）與 backup（全量版）樹狀物件..."
      # 1. 提取乾淨版（main）樹狀物件
      CLEAN_TREE=$(git rev-parse "HEAD:$REL_PATH")
      DSSI_MAIN_PARENT=$(git rev-parse dssi/main 2>/dev/null || git rev-parse dssi-standalone 2>/dev/null || echo "")
      if [ -n "$DSSI_MAIN_PARENT" ]; then
        MAIN_COMMIT=$(git commit-tree "$CLEAN_TREE" -p "$DSSI_MAIN_PARENT" -m "$MSG")
      else
        MAIN_COMMIT=$(git commit-tree "$CLEAN_TREE" -m "$MSG")
      fi
      git update-ref refs/heads/dssi-standalone "$MAIN_COMMIT"

      # 2. 提取全量版（backup，含編譯中繼暫存檔）樹狀物件
      TEMP_INDEX=$(mktemp "${TMPDIR:-/tmp}/git_dssi_backup_XXXXXX")
      rm -f "$TEMP_INDEX"
      export GIT_INDEX_FILE="$TEMP_INDEX"
      git add -A --force "$REL_PATH"
      git rm --cached -r --ignore-unmatch '*.DS_Store' >/dev/null 2>&1 || true
      FULL_BACKUP_TREE=$(git write-tree)
      BACKUP_SUBTREE=$(git rev-parse "$FULL_BACKUP_TREE:$REL_PATH")
      unset GIT_INDEX_FILE
      rm -f "$TEMP_INDEX"

      DSSI_BACKUP_PARENT=$(git rev-parse dssi/backup 2>/dev/null || git rev-parse dssi-backup-standalone 2>/dev/null || echo "")
      if [ -n "$DSSI_BACKUP_PARENT" ]; then
        BACKUP_COMMIT=$(git commit-tree "$BACKUP_SUBTREE" -p "$DSSI_BACKUP_PARENT" -m "backup: $MSG (full backup with aux files)")
      else
        BACKUP_COMMIT=$(git commit-tree "$BACKUP_SUBTREE" -m "backup: $MSG (full backup with aux files)")
      fi
      git update-ref refs/heads/dssi-backup-standalone "$BACKUP_COMMIT"

      echo "🌐 [4/4] 正在推送至 DSSI 專屬 GitHub 儲存庫 ($DSSI_URL)..."
      git push dssi dssi-standalone:main dssi-backup-standalone:backup
      echo "✅ DSSI 雙分支同步完成！"
      echo "  - 遠端：$DSSI_URL"
      echo "  - main 分支：已更新（首頁乾淨瀏覽，無暫存檔）"
      echo "  - backup 分支：已更新（全量備份，含所有中繼暫存檔）"
    else
      echo "🚀 [1/3] 正在同步 main 分支（乾淨瀏覽版）..."
      CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
      if [ "$CURRENT_BRANCH" != "main" ]; then
        git checkout main
      fi
      git add -A
      if ! git diff-index --quiet HEAD --; then
        git commit -m "$MSG"
      else
        echo "  main 分支無新變更需要 commit。"
      fi

      echo "📦 [2/3] 正在生成 backup 分支（全量備份版，含編譯暫存檔）..."
      TEMP_INDEX=$(mktemp "${TMPDIR:-/tmp}/git_backup_idx_XXXXXX")
      rm -f "$TEMP_INDEX"
      export GIT_INDEX_FILE="$TEMP_INDEX"
      git add -A --force
      git rm --cached -r --ignore-unmatch '*.DS_Store' >/dev/null 2>&1 || true
      BACKUP_GITIGNORE="# 系統檔案\n.DS_Store\nThumbs.db\n"
      HASH_OBJ=$(printf "%b" "$BACKUP_GITIGNORE" | git hash-object -w --stdin)
      git update-index --cacheinfo 100644 "$HASH_OBJ" .gitignore
      TREE_ID=$(git write-tree)
      unset GIT_INDEX_FILE
      rm -f "$TEMP_INDEX"

      BACKUP_PARENT=$(git rev-parse backup 2>/dev/null || echo "")
      if [ -n "$BACKUP_PARENT" ]; then
        BACKUP_COMMIT=$(git commit-tree "$TREE_ID" -p "$BACKUP_PARENT" -m "backup: $MSG (full backup with aux files)")
      else
        BACKUP_COMMIT=$(git commit-tree "$TREE_ID" -m "backup: $MSG (full backup with aux files)")
      fi
      git update-ref refs/heads/backup "$BACKUP_COMMIT"

      echo "🌐 [3/3] 正在推送至 GitHub (main + backup)..."
      git push origin main backup
      echo "✅ 雙分支同步完成！"
    fi
    ;;

  pull)
    if [ "$IS_MONOREPO" = true ]; then
      cd "$TOP_DIR"
      git remote get-url dssi >/dev/null 2>&1 || git remote add dssi "$DSSI_URL"
      echo "📥 正在自 DSSI GitHub 拉取最新分支指針..."
      git fetch dssi main backup
      echo "✅ 已抓取最新 dssi/main 與 dssi/backup 指針。"
    else
      echo "📥 正在自 GitHub backup 分支拉取最新全量檔案..."
      git fetch origin backup main
      CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
      if [ "$CURRENT_BRANCH" = "backup" ]; then
        git pull origin backup
      else
        git branch -f backup origin/backup
        git pull origin main
      fi
      echo "✅ 拉取完成！若在其他電腦需完整全量檔案，請確保使用 backup 分支。"
    fi
    ;;

  status)
    if [ "$IS_MONOREPO" = true ]; then
      echo "📊 DSSI 專屬遠端狀態 ($DSSI_URL)："
      cd "$TOP_DIR"
      git remote get-url dssi >/dev/null 2>&1 || git remote add dssi "$DSSI_URL"
      git fetch dssi main backup >/dev/null 2>&1 || true
      echo "遠端最新提交："
      git log -n 1 --oneline dssi/main || true
      echo ""
      git status -s "$REL_PATH"
    else
      echo "📊 當前分支狀態："
      git branch -vv
      echo ""
      git status -s
    fi
    ;;

  *)
    echo "用法: $0 {push [訊息]|pull|status}"
    exit 1
    ;;
esac
