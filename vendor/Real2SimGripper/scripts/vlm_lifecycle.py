#!/usr/bin/env python3
"""vlm_lifecycle.py — VLM 服务生命周期管理（共享模块）。

**硬性要求：agent 跑完必须把占用的显存还回去。**

- agent 自己起 VLM 服务（子进程 + 独立进程组）
- `finally` + `atexit` 双保险，保证正常结束/异常/被中断都会停
- 停完跑 `nvidia-smi` 验证显存确实释放
- `AGENTIC_KEEP_VLM=1` 可显式保留（调试用）

用法
----
    from vlm_lifecycle import VlmServer, ensure_vlm, release_vlm
    ensure_vlm()                      # 起服务(已在跑则直接用)
    ... 用 VLM ...
    release_vlm()                     # 停 + 验证显存释放
"""
import atexit
import json
import os
import signal
import subprocess
import time
import urllib.request

VLM_SERVER_SCRIPT = os.environ.get(
    "VLM_SERVER_SCRIPT", "/path/to/workspace/qwen_test/openai_server.py")
VLM_PYTHON = os.environ.get(
    "VLM_PYTHON", os.path.expanduser("~/miniconda3/envs/opentouch/bin/python"))
VLM_PORT = int(os.environ.get("VLM_PORT", "18000"))
VLM_LOG = os.environ.get("VLM_LOG", "/tmp/agentic_vlm_server.log")

_VLM = None


def _server_process_alive():
    """是否还有 openai_server.py 进程（比端口探测更可靠）。"""
    try:
        r = subprocess.run(["pgrep", "-f", "openai_server.py"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def gpu_mem_summary():
    """返回每卡显存与合计（MB）；拿不到返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        per = [int(l.split(",")[1].strip()) for l in out.strip().splitlines() if l.strip()]
        return {"per_gpu_mb": per, "total_used_mb": sum(per)}
    except Exception:
        return None


class VlmServer:
    def __init__(self, port=VLM_PORT):
        self.port = port
        self.proc = None
        self.started_by_us = False
        self.baseline_mb = None      # 启动前的显存基线(集群上别人也占卡, 不能拿绝对值比)

    # ---- 探测 ----
    def is_up(self):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                return bool(json.load(r).get("model_loaded", False))
        except Exception:
            return False

    # ---- 启动 ----
    def start(self, wait_s=900):
        if self.is_up():
            print(f"[VLM] :{self.port} 已在运行（非本 agent 启动, 结束时会保留; 只有本 agent 自己起的才会被释放）")
            self.started_by_us = False
            return True
        if not os.path.exists(VLM_SERVER_SCRIPT):
            print(f"[VLM] 服务脚本不存在 {VLM_SERVER_SCRIPT} → 跳过 VLM")
            return False
        if not os.path.exists(VLM_PYTHON):
            print(f"[VLM] python 不存在 {VLM_PYTHON} → 跳过 VLM")
            return False
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)   # device_map=auto 用全部卡分摊, 避免 OOM
        self.baseline_mb = gpu_mem_summary()    # 记下启动前基线
        print(f"[VLM] 启动服务（port {self.port}，模型加载约 1 分钟）…", flush=True)
        logf = open(VLM_LOG, "w")
        self.proc = subprocess.Popen(
            [VLM_PYTHON, VLM_SERVER_SCRIPT, "--port", str(self.port)],
            stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.started_by_us = True
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.is_up():
                print(f"[VLM] 就绪（{int(time.time() - t0)}s）", flush=True)
                return True
            if self.proc.poll() is not None:
                print(f"[VLM] 服务进程已退出 code={self.proc.returncode}，见 {VLM_LOG}")
                return False
            time.sleep(5)
        print(f"[VLM] 等待超时（{wait_s}s）")
        return False

    # ---- 停止 + 验证 ----
    def stop(self):
        """停服务并验证显存释放。返回 (stopped, gpu_ok)。"""
        stopped = False
        if self.proc is not None and self.proc.poll() is None:
            print("[VLM] 停止本 agent 启动的服务 …", flush=True)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=90)
                stopped = True
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    stopped = True
                except Exception:
                    pass
        # 兜底：端口还占着就按脚本名清 —— **但只在本进程起过服务时做**
        # （否则 COACD 的 multiprocessing 子进程退会时会把父进程的服务杀掉）
        if self.started_by_us and self.is_up():
            print("[VLM] 端口仍占用，按脚本名兜底清理 …", flush=True)
            subprocess.run(["pkill", "-f", "openai_server.py"], check=False)
            for _ in range(30):
                if not self.is_up():
                    stopped = True
                    break
                time.sleep(2)
        gpu = gpu_mem_summary()
        # 判据：**看服务进程还在不在**，而不是比绝对显存。
        # （比显存会在集群上误报：别人/自己在这段时间起了别的任务，显存自然涨，
        #   实测被自己的训练任务 +5.7GB 误判成"未释放"。）
        alive = self.is_up() or _server_process_alive()
        if alive:
            ok = False
            detail = "服务进程仍在运行"
        else:
            ok = True
            base = self.baseline_mb["total_used_mb"] if self.baseline_mb else None
            detail = (f"服务已退出，端口释放；8 卡当前合计 "
                      f"{gpu['total_used_mb'] if gpu else '?'} MB"
                      + (f"（启动前 {base} MB，差值含期间其他任务）" if base else ""))
        if ok:
            print(f"[VLM] ✅ 显存已释放：{detail}", flush=True)
        else:
            print(f"[VLM] ⚠️ {detail}；per_gpu={gpu['per_gpu_mb'] if gpu else None}", flush=True)
        return stopped, ok


def get_server():
    global _VLM
    if _VLM is None:
        _VLM = VlmServer()
    return _VLM


def ensure_vlm():
    return get_server().start()


def release_vlm():
    """释放显存。**只有"本进程真的起过服务"才动手。**

    为什么必须这么保守（实测 2026-10-16 踩到的坑）：
    `sapien/wrapper/coacd.py` 用 `multiprocessing`。启用 COACD 后它的 worker 子进程
    以 **spawn** 方式启动 → 会重新 import 本模块 → **继承 atexit 里注册的这个函数**。
    子进程退出时若不加判断，会：新建一个"没起过服务"的实例 → 发现端口被占（其实是
    父进程的服务）→ 走 pkill 兜底 → **把父进程的 VLM 服务杀掉**，导致 agent 闭环失效。
    """
    if os.environ.get("AGENTIC_KEEP_VLM") == "1":
        print("[VLM] AGENTIC_KEEP_VLM=1，保留服务不清显存")
        return
    srv = _VLM
    force = os.environ.get("AGENTIC_VLM_FORCE_STOP") == "1"
    if srv is None or not srv.started_by_us:
        if not force:
            # 本进程没起过服务（通常是 COACD 的 multiprocessing 子进程）→ 不碰
            return
    try:
        get_server().stop()
    except Exception as e:
        print(f"[VLM] 清理异常（忽略）: {e}")


# 注册即生效：任何退出路径都会尝试释放
atexit.register(release_vlm)
