"""Kaggle GPU kernel (2x T4): runs both solve sets for the four open models.

One Ollama server per GPU, two models per server, so the two workers run in parallel.
The bundle (code, prompts, data sheets) comes in as a private dataset built by
scripts/kaggle/push.sh. Results land in /kaggle/working/eval/results.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = "/kaggle/working/eval"
MODELS_DIR = "/tmp/ollama_models"
WORKERS = [
    {"gpu": 0, "port": 11434, "models": ["qwen3:8b", "gemma3:12b"]},
    {"gpu": 1, "port": 11435, "models": ["llama3.1:8b", "qwen3:14b"]},
]


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


# Kaggle may hand the bundle over as the archive or already extracted
archives = glob.glob("/kaggle/input/**/bundle.tar.gz", recursive=True)
extracted = glob.glob("/kaggle/input/**/evalset/run_solve.py", recursive=True)
if archives:
    os.makedirs(WORK, exist_ok=True)
    with tarfile.open(archives[0]) as tar:
        tar.extractall(WORK)
elif extracted:
    shutil.copytree(os.path.dirname(os.path.dirname(extracted[0])), WORK, dirs_exist_ok=True)
else:
    sh("find /kaggle/input -maxdepth 4 | head -50")
    raise SystemExit("bundle not found")
os.makedirs(f"{WORK}/results/logs", exist_ok=True)

sh(f"{sys.executable} -m pip install -q pandas==2.2.3 numpy==2.1.3 scipy==1.14.1 statsmodels==0.14.4 "
   "scikit-learn==1.5.2 requests==2.32.3")
sh("apt-get install -y -qq zstd pciutils lshw > /dev/null 2>&1 || (apt-get update -qq > /dev/null && apt-get install -y -qq zstd pciutils lshw > /dev/null)")
try:
    sh("curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1")
except subprocess.CalledProcessError:
    print(open("/tmp/ollama_install.log").read()[-3000:], flush=True)
    raise
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

servers = []
for w in WORKERS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w["gpu"]), OLLAMA_HOST=f"127.0.0.1:{w['port']}",
               OLLAMA_MODELS=MODELS_DIR)
    log = open(f"/tmp/ollama_{w['gpu']}.log", "w")
    servers.append(subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log))
time.sleep(10)

for w in WORKERS:
    for m in w["models"]:
        sh(f"OLLAMA_HOST=127.0.0.1:{w['port']} ollama pull {m} > /dev/null 2>&1")

procs = []
for w in WORKERS:
    models = ",".join(f"ollama/{m}" for m in w["models"])
    tag = f"_gpu{w['gpu']}"
    cmd = (f"cd {WORK} && {sys.executable} -m evalset.run_solve --set prompts --models {models} --tag {tag} && "
           f"{sys.executable} -m evalset.run_solve --set qrdata --sample 100 --seed 0 --models {models} --tag {tag}")
    env = dict(os.environ, OLLAMA_URL=f"http://127.0.0.1:{w['port']}")
    log = open(f"{WORK}/results/logs/kaggle_gpu{w['gpu']}.log", "w")
    procs.append(subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT))

start = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(300)
    counts = {os.path.basename(f): sum(1 for _ in open(f)) for f in glob.glob(f"{WORK}/results/solve_*.jsonl")}
    print(f"{(time.time() - start) / 60:.0f} min: {counts}", flush=True)

print("exit codes:", [p.returncode for p in procs], flush=True)
for p in servers:
    p.terminate()
