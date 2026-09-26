bash /workspace/setup_pod_6000.sh qwen # 약 5~10분
source /root/stv_env.sh
cd /workspace/stv/home/yeseo-qwen/scene-text-vision-repo/notebooks
mkdir -p logs

$PY train_qwen3vl_32b.py --benchmark-steps 20

nohup $PY -u train_qwen3vl_32b.py > logs/train_qwen3vl_32b.log 2>&1 &
echo $! > logs/train_qwen3vl_32b.pid
tail -f logs/train_qwen3vl_32b.log

# ---

bash /workspace/setup_pod_6000.sh gemma
source /root/stv_env.sh
cd /workspace/stv/home/yeseo-gemma/scene-text-vision-repo/notebooks
mkdir -p logs

$PY train_gemma4_31b.py --benchmark-steps 20

nohup $PY -u train_gemma4_31b.py > logs/train_gemma4_31b.log 2>&1 &
echo $! > logs/train_gemma4_31b.pid
tail -f logs/train_gemma4_31b.log

# ---

bash /workspace/setup_pod_6000.sh hcx # 마지막에 'HCX class True'가 나와야 함
source /root/stv_env.sh
cd /workspace/stv/home/yeseo-hcx/scene-text-vision-repo/notebooks
mkdir -p logs

# $PY train_hcx_think_32b.py --precision 4bit --benchmark-steps 20

# nohup $PY -u train_hcx_think_32b.py --precision 4bit > logs/train_hcx_4bit.log 2>&1 &
# echo $! > logs/train_hcx_4bit.pid
# tail -f logs/train_hcx_4bit.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $PY -u train_hcx_think_32b.py --precision 4bit --augs orig \
  --batch-size 8 --grad-accum 2 \
  > logs/train_hcx_4bit_noaug.log 2>&1 &
echo $! > logs/train_hcx_4bit_noaug.pid
tail -f logs/train_hcx_4bit_noaug.log

# [자동 pod 종료]

which runpodctl && echo "POD_ID=$RUNPOD_POD_ID"

nohup sh -c '
  while kill -0 $(cat logs/train_gemma4_31b.pid) 2>/dev/null; do sleep 60; done
  echo "[process ended] $(date)"
  sleep 60
  runpodctl stop pod $RUNPOD_POD_ID
' > logs/autostop.log 2>&1 &

nohup sh -c '
  while kill -0 $(cat logs/train_qwen3vl_32b.pid) 2>/dev/null; do sleep 60; done
  echo "[process ended] $(date)"
  sleep 60
  runpodctl stop pod $RUNPOD_POD_ID
' > logs/autostop.log 2>&1 &

nohup sh -c '
  while kill -0 $(cat logs/train_hcx_4bit_noaug.pid) 2>/dev/null; do sleep 60; done
  echo "[process ended] $(date)"
  sleep 60
  runpodctl stop pod $RUNPOD_POD_ID
' > logs/autostop.log 2>&1 &

# [checkpoint 별도 저장]
cp keep_ckpts.sh /workspace/stv/keep_ckpts.sh     # 볼륨에 한 번 올려 두기

# Qwen pod
nohup bash /workspace/stv/keep_ckpts.sh /workspace/stv/outputs/final_qwen3vl32b_4bit 800 1200 1600 \
  > /workspace/stv/keep_qwen.log 2>&1 &

# Gemma pod
nohup bash /workspace/stv/keep_ckpts.sh /workspace/stv/outputs/final_gemma4_31b_4bit 800 1200 1600 \
  > /workspace/stv/keep_gemma.log 2>&1 &

# HCX pod
nohup bash /workspace/stv/keep_ckpts.sh /workspace/stv/outputs/final_hcx_think32b_4bit_aug-orig 200 400 \
  > /workspace/stv/keep_hcx.log 2>&1 &

