# Setting Up an NVIDIA GPU for CUDA Development on Linux

This guide covers installing everything needed to run the examples in
`examples/06_cuda/` on a Linux machine with an NVIDIA GPU: the proprietary
driver, the CUDA Toolkit, and a Python environment with GPU-enabled
`numba` (and optionally `cupy`).

## 1. Verify you have a supported GPU

```bash
lspci | grep -i nvidia
```

You should see your NVIDIA GPU listed (e.g. `NVIDIA Corporation ... GeForce/RTX/Tesla/A100 ...`).
Check CUDA compute capability compatibility for your card against the
[CUDA GPUs list](https://developer.nvidia.com/cuda-gpus) if unsure.

## 2. Install the NVIDIA driver

### Ubuntu/Debian

```bash
sudo apt update
sudo ubuntu-drivers autoinstall      # Ubuntu: auto-detects the recommended driver
# or, to pick a specific version:
sudo apt install -y nvidia-driver-550
sudo reboot
```

### RHEL/Fedora/CentOS

```bash
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf module install -y nvidia-driver:latest-dkms
sudo reboot
```

### Verify the driver

```bash
nvidia-smi
```

Expected output includes the driver version, CUDA version supported by
the driver, GPU name, memory, and utilization. If this command fails,
the driver did not load correctly -- check `dmesg | grep -i nvidia` for
errors (common causes: Secure Boot blocking the kernel module, or a
mismatched kernel header version).

## 3. Install the CUDA Toolkit

The driver alone is enough to *run* CUDA programs; the Toolkit adds the
compiler (`nvcc`) and development libraries needed to *build* them
(Numba uses its own bundled NVVM/PTX compiler for JIT, but installing the
Toolkit is still recommended for a complete, debuggable environment).

```bash
# Ubuntu example (adjust the distro/version path for yours):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Add CUDA to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload your shell and verify:

```bash
source ~/.bashrc
nvcc --version
```

## 4. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy numba
# Optional: cupy for a NumPy-compatible GPU array API
pip install cupy-cuda12x   # match the suffix to your installed CUDA major version
```

Verify Numba can see the GPU:

```bash
python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available()); print(cuda.gpus)"
```

Expected output: `CUDA available: True` and a list containing your GPU
device(s).

## 5. Run the CUDA examples in this repository

```bash
python3 examples/06_cuda/cuda_vector_and_matrix_kernels.py
```

Expected output: both kernels report `matches NumPy: True`, confirming
the GPU-computed results are numerically identical to the NumPy
reference computation.

## 6. Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` not found / fails | Driver not installed or not loaded | Reinstall driver, check `dmesg`, disable Secure Boot if the kernel module is unsigned |
| `cuda.is_available()` returns `False` despite `nvidia-smi` working | Numba/CUDA driver API version mismatch, or running inside a container without `--gpus all` | Update `numba`, confirm `nvidia-container-toolkit` is installed for Docker, pass `--gpus all` to `docker run` |
| `nvcc: command not found` | Toolkit `bin` directory not on `PATH` | Re-check step 3's `PATH`/`LD_LIBRARY_PATH` exports |
| Kernel launch succeeds but results are wrong | Grid/block dimensions miscomputed, or forgetting a bounds check (`if i < n:`) inside the kernel | Review the bounds-check pattern used in `examples/06_cuda/cuda_vector_and_matrix_kernels.py` |
| Driver installed but GPU not visible in a VM/cloud instance | GPU passthrough not enabled, or wrong instance type | Confirm the VM/instance type includes a GPU (e.g. AWS `p3`/`g4dn`, GCP `a2`/`n1` with GPU attached) and that passthrough/virtualization settings allow it |

## 7. Running on a cloud GPU instead

If no local NVIDIA GPU is available, the same examples run unmodified on
any cloud VM with a CUDA-capable GPU (AWS EC2 `g4dn`/`p3`, GCP Compute
Engine with an attached GPU, Azure `NC`-series, or a free-tier Google
Colab GPU runtime) -- repeat steps 4-5 on that machine; steps 1-3 are
typically already provisioned by the cloud provider's GPU-enabled images.
