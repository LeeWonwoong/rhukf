#!/usr/bin/env bash
# =============================================================================
#  RHUKF / DDQN  Ablation Sweep
#  Grid: activation × N_horizon × alpha × q(init=end) × state_form (× seeds)
#
#  주의: rhukf.py의 출력 폴더명(param_str)에는 horizon/activation/q가 안 들어가서
#        그대로 돌리면 런들이 ./results_cartpole/ 안에서 서로 덮어쓴다.
#        => 런마다 별도 작업 디렉토리에서 실행해 결과를 격리한다 (코드 수정 불필요).
#
#  SSH 끊겨도 살아남게: tmux/nohup 안에서 돌리길 권장 (아래 USAGE 참고).
# =============================================================================
set -uo pipefail

# ---- 0. 사용자 설정 (여기만 고치면 됨) --------------------------------------
SCRIPT="${SCRIPT:-./rhukf.py}"                 # rhukf.py 경로 (상대/절대 다 OK)
PYTHON="${PYTHON:-python}"                      # venv면 미리 activate 하거나 venv python 절대경로
OUTROOT="${OUTROOT:-$HOME/rhukf_sweep/$(date +%Y%m%d_%H%M%S)}"
ENVNAME="${ENVNAME:-CartPole-v1}"
EPISODES="${EPISODES:-}"                        # 비우면 코드 기본값(CartPole=120).
                                                #  후반 붕괴 보려면 예: EPISODES=400
MAX_PARALLEL="${MAX_PARALLEL:-1}"               # 동시 실행 수. CPU/단일GPU면 1~2 권장
DRYRUN="${DRYRUN:-0}"                           # 1이면 명령만 출력하고 실행 안 함

# ---- grid (필요하면 배열만 수정) --------------------------------------------
ACTS=(silu mish)
HORIZONS=(6 7)
ALPHAS=(0.01 0.1 0.9)
QS=(0.5 0.1)                                    # q_init = q_end = 같은 값
SFORMS=(error absolute)
SEEDS=(42)                                      # collapse rate 보려면 (42 1 2 3 7) 처럼 늘리기
                                                #  → 런 수가 seed 개수만큼 곱해짐
# -----------------------------------------------------------------------------

# 스크립트 경로 절대화 (런마다 cd 하므로 필수)
SCRIPT="$(readlink -f "$SCRIPT" 2>/dev/null || echo "$SCRIPT")"
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERR] SCRIPT not found: $SCRIPT  (SCRIPT=... 로 경로 지정)"; exit 1
fi

mkdir -p "$OUTROOT"
SUMMARY_TSV="$OUTROOT/_summary.tsv"
printf "run_id\tact\thorizon\talpha\tq\tsform\tseed\tstatus\tpeak_avg20\tfinal_avg20\tcollapsed\tsecs\n" > "$SUMMARY_TSV"

TOTAL=$(( ${#ACTS[@]} * ${#HORIZONS[@]} * ${#ALPHAS[@]} * ${#QS[@]} * ${#SFORMS[@]} * ${#SEEDS[@]} ))
echo "=================================================================="
echo " RHUKF sweep  |  total runs = $TOTAL  |  parallel = $MAX_PARALLEL"
echo " script   : $SCRIPT"
echo " python   : $PYTHON"
echo " outroot  : $OUTROOT"
echo " episodes : ${EPISODES:-<code default>}   env: $ENVNAME"
echo "=================================================================="

# ---- 한 런 실행 (실패해도 sweep 전체는 안 죽음) ------------------------------
run_one() {
  local act="$1" H="$2" alpha="$3" q="$4" sform="$5" seed="$6" idx="$7"
  local run_id="i${idx}_${act}_h${H}_a${alpha}_q${q}_${sform}_s${seed}"
  local rundir="$OUTROOT/$run_id"
  local log="$rundir/run.log"
  mkdir -p "$rundir"

  # 재실행(resume): 이미 끝난 런은 건너뜀
  if [[ -f "$rundir/.done" ]]; then
    echo "[skip] $run_id (already .done)"
    return 0
  fi

  local cmd=( "$PYTHON" "$SCRIPT"
      --env "$ENVNAME"
      --activation_fn "$act"
      --N_horizon "$H"
      --alpha "$alpha"
      --q_init "$q" --q_end "$q"
      --state_form "$sform"
      --network_seed "$seed" --env_seed "$seed" )
  [[ -n "$EPISODES" ]] && cmd+=( --episodes "$EPISODES" )

  if [[ "$DRYRUN" == "1" ]]; then
    echo "[dry] ($idx/$TOTAL) cd $rundir && ${cmd[*]}"
    return 0
  fi

  echo "[run] ($idx/$TOTAL) $run_id  $(date +%H:%M:%S)"
  local t0 t1 secs status="ok"
  t0=$(date +%s)
  # 상대경로 결과(./results_cartpole/...)를 rundir 안에 가두기 위해 cd 후 실행
  ( cd "$rundir" && "${cmd[@]}" ) > "$log" 2>&1 || status="FAIL"
  t1=$(date +%s); secs=$(( t1 - t0 ))

  # ---- 결과 파싱 (현재 로그 포맷 기준: "Ep N | Rwd: .. | Avg20: ..") ----
  local peak final collapsed="?"
  read -r peak final < <(
    grep -E " Ep +[0-9]+ .*Avg20:" "$log" 2>/dev/null \
    | grep -oE "Avg20: *[0-9.]+" | grep -oE "[0-9.]+" \
    | awk 'NR==1{mx=$1} {if($1>mx)mx=$1; last=$1} END{if(NR==0){print "NA NA"}else{printf "%.1f %.1f", mx, last}}'
  )
  peak="${peak:-NA}"; final="${final:-NA}"
  # collapse 판정 (proxy): 한 번 475+ 찍었는데 끝이 peak의 85% 미만이면 collapse
  if [[ "$peak" != "NA" && "$final" != "NA" ]]; then
    collapsed=$(awk -v p="$peak" -v f="$final" 'BEGIN{print (p>=475 && f < 0.85*p) ? "YES":"no"}')
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" "$act" "$H" "$alpha" "$q" "$sform" "$seed" \
    "$status" "$peak" "$final" "$collapsed" "$secs" >> "$SUMMARY_TSV"

  if [[ "$status" == "ok" ]]; then
    touch "$rundir/.done"
    echo "      └ done  peak=$peak final=$final collapsed=$collapsed  ${secs}s"
  else
    echo "      └ FAIL  (see $log)"
  fi
}

# ---- 디스패치 (직렬 / 병렬) -------------------------------------------------
idx=0
for seed in "${SEEDS[@]}"; do
 for sform in "${SFORMS[@]}"; do
  for act in "${ACTS[@]}"; do
   for H in "${HORIZONS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
     for q in "${QS[@]}"; do
       idx=$((idx+1))
       if (( MAX_PARALLEL <= 1 )); then
         run_one "$act" "$H" "$alpha" "$q" "$sform" "$seed" "$idx"
       else
         run_one "$act" "$H" "$alpha" "$q" "$sform" "$seed" "$idx" &
         # 동시 실행 수 제한 (bash 4.3+ 의 wait -n 필요)
         while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n 2>/dev/null || sleep 1; done
       fi
     done
    done
   done
  done
 done
done
wait

# ---- 요약 출력 --------------------------------------------------------------
echo
echo "=================================================================="
echo " SWEEP DONE.  summary: $SUMMARY_TSV"
echo "=================================================================="
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$SUMMARY_TSV"
else
  cat "$SUMMARY_TSV"
fi
echo
echo "[collapse 의심 런]"
awk -F'\t' 'NR==1||$11=="YES"||$8=="FAIL"' "$SUMMARY_TSV" | { command -v column >/dev/null 2>&1 && column -t -s $'\t' || cat; }
