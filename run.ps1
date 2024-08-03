# 检测操作系统类型的PowerShell脚本

$os = Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object -ExpandProperty Caption

Write-Output "Installing dependency package using pip..."
pip install -r requirements.txt
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

Write-Output "Generating Datasets..."
python new_data.py
Write-Output "Generating negative data..."
python generate_neg.py --dataset=Games
Write-Output "OK!"
python new_main.py