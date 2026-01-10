$env:LLVM_SYS_211_PREFIX = "D:\LLVM"
Remove-Item Env:\LLVM_CONFIG_PATH -ErrorAction SilentlyContinue

Write-Host "Configured LLVM environment variables for D:\LLVM"
Write-Host "Running cargo check..."

cargo check --workspace --exclude satellite-ide
