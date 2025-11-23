Project overview
This repository contains code to process a corpus of speeches, build or query a Chroma vector store, and run evaluation scripts. The primary entry points are `main.py` and `evaluation.py`.

**Prerequisites**
- Python 3.9+ installed and available as `python` on PATH.
- On Windows, PowerShell is the recommended shell for the commands below.
- (Optional) A GPU and compatible CUDA toolchain if you plan to use a CUDA-enabled `torch` build.

**Ollama & Mistral (optional)**
- If you want to run local LLMs via Ollama (for example Mistral models), install Ollama first.
- Download and install Ollama from: https://ollama.com/download (choose the Windows installer) or use a package manager if available.

Example (PowerShell) — install via `winget` if you have it:

```powershell
winget install Ollama.Ollama
```

- After installing Ollama, you can pull Mistral models (model names may vary). Example:

```powershell
# Pull a Mistral model (example model name; confirm the exact model on the Ollama model registry)
ollama pull mistralai/mistral-7b-instruct

# Test-run the model locally
ollama run mistralai/mistral-7b-instruct --prompt "Hello from Ollama"
```

- Notes:
	- Model names and availability change; check Ollama's docs or registry for exact model identifiers.
	- Ollama runs a local daemon — the `ollama` CLI communicates with it. If you have firewall or permission issues, follow the Ollama troubleshooting guide on their site.

**Recommended workflow (Windows / PowerShell)**

- Create and activate a virtual environment at the project root (recommended):

```powershell
python -m venv .venv
# If PowerShell execution policy prevents running the activate script, run this once in the shell:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.
\ .venv\Scripts\Activate.ps1
# Alternatively (works in cmd): .venv\Scripts\activate
```

- Install dependencies from `requirements.txt`:

```powershell
pip install -r .\requirements.txt
```

Notes about `torch` on Windows
- The `requirements.txt` contains a `torch` entry that may require a platform-specific wheel. If `pip install -r requirements.txt` fails for `torch`, follow PyTorch's official install instructions and select the proper command for your CUDA/CPU configuration from https://pytorch.org/get-started/locally/.
Example CPU-only install (Windows):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Quick run**
- Run the main program (project root):

```powershell
python main.py
```

- Run evaluation script:

```powershell
python evaluation.py
```

**Files and directories**
- `main.py`: Primary script to run the application pipeline.
- `evaluation.py`: Script that executes evaluation routines and writes results to `results/`.
- `requirements.txt`: Python dependencies for the project.
- `corpus/`: Text files used as input for building the vector store.
- `chroma_db*/`: Example Chroma DB sqlite files (prebuilt stores).
- `results/`: Output files (evaluation outputs, analysis, and JSON results).

**Troubleshooting & tips**
- If a package fails to install, try upgrading `pip` first:

```powershell
python -m pip install --upgrade pip
```
- If you use a different Python installation, replace `python` with the full path to the desired interpreter (for example `.venv\Scripts\python.exe`).
- If you expect to run with GPU acceleration, ensure CUDA and the correct `torch` wheel are installed.

**Next steps / optional improvements**
- Create an `environment.yml` or `requirements.lock` for reproducible installs.
- Add a short examples section describing expected inputs/outputs for `main.py` and `evaluation.py` once you confirm their runtime options.

If you'd like, I can: update this README with example command-line arguments for `main.py`/`evaluation.py`, create an `environment.yml`, or generate a `requirements.lock` file. Which would you prefer next?

