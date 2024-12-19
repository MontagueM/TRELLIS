Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = 1
$Env:UV_NO_CACHE = 0
$Env:UV_LINK_MODE = "symlink"
$Env:GIT_LFS_SKIP_SMUDGE = 1

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        Write-Output "Install failed"
        Read-Host | Out-Null ;
        Exit
    }
}

if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    . ./venv/Scripts/activate
}
elseif (Test-Path "../.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    . ../.venv/Scripts/activate
}
else {
    Write-Output "Create .venv"
    ~/.local/bin/uv venv -p 3.10
    . ./.venv/Scripts/activate
}

Write-Output "Installing main requirements"

~/.local/bin/uv pip install --upgrade setuptools wheel

~/.local/bin/uv pip sync requirements-uv.txt --index-strategy unsafe-best-match
Check "Install main requirements failed"


~/.local/bin/uv pip install --no-build-isolation -e extensions/vox2seq/
Check "Install vox2seq failed"


~/.local/bin/uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
Check "Install kaolin failed"

~/.local/bin/uv pip install --no-build-isolation git+https://github.com/JeffreyXiang/diffoctreerast.git
Check "Install diffoctreerast failed"

~/.local/bin/uv pip install git+https://github.com/sdbds/diff-gaussian-rasterization
Check "Install diff-gaussian-rasterization failed"

# triton
~/.local/bin/uv pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl
Check "Install triton failed"

# accelerate
~/.local/bin/uv pip install accelerate
Check "Install accelerate failed"

# flash attn
~/.local/bin/uv pip install https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl
Check "Install flash_attn failed"

# wheel
~/.local/bin/uv pip install wheel
Check "Install wheel failed"


# tensorrt
# ~/.local/bin/uv pip install nvidia_stub
# Check "Install nvidia_stub for tensorrt failed"

# ~/.local/bin/uv pip install torch-tensorrt
# Check "Install tensorrt failed"

Write-Output "Install finished"
