# Read-only storage audit — 2026-11-03

**Nothing was deleted.** This is an inventory only.

## The headline finding

`df` reported **219 MB** available, but `diskutil info /` reported
**18.5 GB of container free space**. The gap was *purgeable* space — content
macOS will reclaim under pressure but does not report as available. It was
released during this audit, and the volume now shows **~13 GB free** after the
project's own 1.4 GB virtualenv, the 95 MB dataset and the DistilBERT download.

So the original blocker cleared itself, and **no deletion is strictly required**
to finish this project. The inventory below is for durable headroom.

Note also: `/Users` holds 193 GB, of which ~50 GB belongs to a second account
(`seifallahmohamedkishtah`). That is not ours to touch.

## Safe to delete — rebuildable caches, no user data

| Path | Size | What it is | How it comes back |
|---|---|---|---|
| `~/Library/Caches/pip` | 3.36 GB | pip wheel cache | re-downloaded on demand; `pip cache purge` |
| `~/Library/Caches/puccinialin` | 1.53 GB | Rust/maturin build cache | rebuilt on next build |
| `~/Library/Caches/trivy` (incl. `db/trivy.db`, 1.14 GB) | 1.17 GB | vulnerability database | re-downloaded on next scan |
| `~/Library/Caches/ms-playwright` | 1.66 GB | Playwright browser binaries | `playwright install` |
| `~/Library/Caches/Homebrew` | 644 MB | downloaded bottles + `LibreOffice_26.8.1…dmg` (280 MB) | `brew cleanup` |
| `~/.cache/uv` | 421 MB | uv package cache | re-downloaded |
| `~/.cache/puppeteer` | 204 MB | Chromium for Puppeteer | re-downloaded |
| `~/Library/Caches/ollama/updates/…/Ollama-darwin.zip` | 210 MB | stale app updater download | n/a |
| `~/Library/Caches/VisualStudio/17.0/TempDownload/visualstudioformac-17.6.14.413-arm64.dmg` | 780 MB | installer for a discontinued product | n/a |
| **subtotal** | **~10.0 GB** | | |

## Probably safe — confirm first

| Path | Size | Why confirm |
|---|---|---|
| `~/.ollama/models` | **19.7 GB** | 8 local LLMs: `llama3` 4.7 GB, `codellama` 3.8 GB, `gemma3:4b` 3.3 GB, `phi3:mini` 2.2 GB, `qwen2.5:3b` 1.9 GB, `moondream` 1.7 GB, `gemma2:2b` 1.6 GB, `llama3.2:1b` 1.3 GB. All but one last used **6 months ago**. Re-pullable, but each is a multi-GB download. Drop the ones you don't use: `ollama rm codellama llama3 gemma3:4b` ≈ 11.8 GB. |
| `~/Library/Containers/com.docker.docker/…/Docker.raw` | **42 GB** (60 GB sparse) | The Docker Desktop VM disk. Docker is running but `docker system df` would not respond, so I could not see which images and volumes are inside. `docker system prune -a --volumes` typically reclaims most of it — **but it destroys every image, container and named volume**. Only worth doing if you know nothing in Docker matters. |
| `~/Library/Parallels/Downloads/…25h2…A64FRE_en-us.iso` | 6.22 GB | A Windows 11 installer ISO. Safe if the VM is already installed or you can re-download it. |
| `~/.cache/hyperframes/whisper/models/ggml-large-v3.bin` | 2.88 GB | Whisper large-v3 weights. Re-downloadable; delete only if you are not using local transcription. |
| `~/.cache/codex-runtimes` | 1.63 GB | Codex runtime downloads |
| `~/Library/Caches/com.openai.codex` + `~/Library/Caches/Codex` | 2.52 GB | app caches, rebuilt on use |
| `~/.cache/whisper` | 741 MB | second Whisper model cache |
| `~/Library/Caches/com.spotify.client` | 817 MB | offline/stream cache; clears itself, costs re-streaming |
| `~/Library/Caches/Google`, `Comet`, `BraveSoftware`, `com.brave.Browser` | 4.06 GB | **browser caches, not profiles.** Clearing loses no bookmarks, passwords or history, but logs nothing out either. Still: confirm. |
| `~/Desktop/*.dmg` (`ChatGPT` 560 MB, `Codex` 440 MB, `Cursor-darwin-arm64` 240 MB) | 1.24 GB | installers for apps that look already installed |

## Do not delete

| Path | Size | Why |
|---|---|---|
| `~/Library/Application Support/Claude/vm_bundles/claudevm.bundle/rootfs.img` (+ `.zst`) | 11.2 GB | active application data for a running tool |
| `~/Documents`, `~/Desktop` (non-installer), `~/Movies`, `~/Projects` | 23 GB | personal and project files |
| `/Users/seifallahmohamedkishtah` | ~50 GB | a different user's account |
| `~/.kaggle`, `~/.ssh`, `~/.config`, keychains | small | credentials |
| Browser **profiles** (`~/Library/Application Support/{Google,BraveSoftware,…}`) | 28 GB total App Support | logins, bookmarks, history |
| `~/.cache/huggingface/hub` | 1.37 GB | **this project uses it** (DistilBERT base weights) |
| `~/.cache/kagglehub` | 345 MB | dataset cache |
| `~/Library/Caches` as a whole | 20 GB | never delete wholesale; pick entries |

## Recommended combination for ≥15 GB

| Step | Reclaims | Risk |
|---|---|---|
| 1. Everything in **Safe to delete** | ~10.0 GB | none — all rebuildable |
| 2. `ollama rm codellama llama3 gemma3:4b` (the three largest, unused 6 months) | ~11.8 GB | re-pullable |
| **total** | **~21.8 GB** | |

That clears the target without touching Docker, the Parallels ISO, browser
caches, or anything personal. If you want more, the Parallels ISO (6.2 GB) is
the next safest single item.

Commands, for when you approve — one per step so you can stop anywhere:

```bash
rm -rf ~/Library/Caches/pip ~/Library/Caches/puccinialin ~/Library/Caches/trivy ~/Library/Caches/ms-playwright ~/.cache/uv ~/.cache/puppeteer
```

```bash
brew cleanup --prune=all
```

```bash
rm -f ~/Library/Caches/VisualStudio/17.0/TempDownload/visualstudioformac-17.6.14.413-arm64.dmg ~/Library/Caches/ollama/updates/*/Ollama-darwin.zip
```

```bash
ollama rm codellama llama3 gemma3:4b
```
