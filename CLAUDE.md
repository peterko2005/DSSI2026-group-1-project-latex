# CLAUDE.md

When working in this repository, you MUST first read [AGENTS.md](AGENTS.md) for complete project architecture, build instructions, LaTeX typesetting conventions, and workflow guidelines.

## Core Directives for Claude
1. **Always Read `AGENTS.md` First**: Adhere strictly to the project structure, metadata separation (`main.tex` vs `ntusetup.tex`), and writing conventions defined in `AGENTS.md`.
2. **Compilation Engine**: Always build using **XeLaTeX** (required by `xeCJK` and `fontspec`). Recommended command:
   ```bash
   xelatex -interaction=nonstopmode -shell-escape main.tex
   ```
3. **Clean Main Branch Discipline**: Do not commit compilation artifacts (`.aux`, `.log`, `.synctex.gz`, `.toc`, etc.) to the `main` branch.
4. **Dual-Branch Synchronization**: Synchronize repository updates using `./sync.sh push "commit message"` to maintain both `main` (clean view) and `backup` (complete state) branches.
