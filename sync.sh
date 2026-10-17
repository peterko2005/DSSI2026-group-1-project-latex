#!/usr/bin/env bash
# ==============================================================================
# DSSI 2026 Group 1 LaTeX 雙分支極速同步工具 (Dual-Branch Git Sync)
# 用途：
#   1. ./sync.sh push "Commit message" : 同步乾淨展示版至 main 分支，全量版至 backup 分支
#   2. ./sync.sh pull                  : 自遠端 backup 分支拉取最新檔案
#   3. ./sync.sh status                : 檢視分支狀態
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ACTION="${1:-status}"
MSG="${2:-update: sync proposal changes}"

case "$ACTION" in
  push)
    echo "🚀 [1/3] 正在同步 main 分支（乾淨瀏覽版）..."
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
    if [ "$CURRENT_BRANCH" != "main" ]; then
      echo "  切換至 main 分支..."
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
    echo "  - main 分支：已更新（首頁乾淨瀏覽，無暫存檔）"
    echo "  - backup 分支：已更新（全量備份，含所有中繼暫存檔）"
    ;;

  pull)
    echo "📥 正在自 GitHub backup 分支拉取最新全量檔案..."
    git fetch origin backup main
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
    if [ "$CURRENT_BRANCH" = "backup" ]; then
      git pull origin backup
    else
      echo "  更新本地 backup 分支指針..."
      git branch -f backup origin/backup
      echo "  拉取 main 分支更新..."
      git pull origin main
    fi
    echo "✅ 拉取完成！若在其他電腦需完整全量檔案，請確保使用 backup 分支。"
    ;;

  status)
    echo "📊 當前分支狀態："
    git branch -vv
    echo ""
    git status -s
    ;;

  *)
    echo "用法: $0 {push [訊息]|pull|status}"
    exit 1
    ;;
esac
