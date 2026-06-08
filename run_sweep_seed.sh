#!/usr/bin/env bash
# =============================================================================
# run_sweep_seed.sh  —  ui4 고정, 두 "성공" 셀을 seed 5개로 (seed 의존성 검정)
#   A: N_horizon=6, alpha=0.01   (이미 seed flip 관측: s0 성공 / s42 붕괴)
#   B: N_horizon=7, alpha=0.3    (s42만 성공 — robust인지 knife-edge인지 확인)
# 목적: solve-rate + "초기 Jacobian 조건수 → 운명" 가설을 [G]/초기-ep로 검정.
#
# ※ update_interval은 CLI 플래그가 아니라 cfg.update_interval(기본 4)이다.
#    → 이 스크립트는 실행 전에 그 값이 4인지 검사하고, 폴더명에 ui4를 박는다.
#
# 사용:
#   chmod +x run_sweep_seed.sh
#   SCRIPT=~/projects/rhukf/rhukf.py PYTHON=python ./run_sweep_seed.sh
#   (백그라운드)  nohup bash run_sweep_seed.sh > sweep.out 2>&1 &
#   끝나면:  python analyze_collapse.py "$OUTROOT"
# 환경변수(선택): SCRIPT PYTHON OUTROOT EPISODES SEEDS DRYRUN
# =============================================================================
set -u

SCRIPT="${SCRIPT:-./rhukf.py}"          # rhukf.py 경로
PYTHON="${PYTHON:-python}"              # venv python (예: python3 / ~/venv/bin/python)
OUTROOT="${OUTROOT:-./results_seed}"   # 결과 루트 (분석기는 여기를 가리킴)
EPISODES="${EPISODES:-150}"            # CartPole 기본 150 (기존 분석과 동일)
SEEDS="${SEEDS:-0 7 21 42 100}"        # 5 seed (기존 관측치 0,42 포함)
DRYRUN="${DRYRUN:-0}"                  # 1이면 명령만 출력

# 두 셀: "H A" 쌍
CONFIGS=( "6 0.01" "7 0.3" )

# ---- 사전 점검 ----------------------------------------------------------------
if [[ ! -f "$SCRIPT" ]]; then echo "[ERR] SCRIPT 없음: $SCRIPT (SCRIPT=경로 지정)"; exit 1; fi
if ! "$PYTHON" -c "import sys" 2>/dev/null; then echo "[ERR] PYTHON 실행 불가: $PYTHON (venv 활성화/경로 확인)"; exit 1; fi

# update_interval==4 확인 (CLI로 못 바꾸므로 소스 기본값이 맞아야 함)
UI_LINE="$(grep -nE 'update_interval *: *int *= *[0-9]+' "$SCRIPT" | head -1)"
UI_VAL="$(echo "$UI_LINE" | grep -oE '= *[0-9]+' | grep -oE '[0-9]+')"
echo "[check] $SCRIPT update_interval = ${UI_VAL:-?}  ($UI_LINE)"
if [[ "${UI_VAL:-}" != "4" ]]; then
  echo "[ERR] update_interval 이 4가 아니다 (현재 ${UI_VAL:-?}). rhukf.py에서 'update_interval: int = 4' 로 고친 뒤 다시 실행." 
  exit 1
fi

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/_summary.tsv"
echo -e "config\tH\talpha\tseed\texit\tfinal_avg20\tstatus\trundir" > "$SUMMARY"

run_one() {
  local H="$1" A="$2" S="$3"
  local name="h${H}_a${A}_tau0.02_ui4_s${S}"     # ★ ui4 를 폴더명에 박음(분석기가 읽음)
  local rundir="$OUTROOT/$name"
  local done="$rundir/.done"
  mkdir -p "$rundir"
  if [[ -f "$done" && "$DRYRUN" != "1" ]]; then echo "[skip] $name (이미 완료)"; return; fi

  # 분석한 성공/붕괴 런과 동일 레짐을 명시적으로 고정
  local args=(
    "$SCRIPT"
    --activation_fn silu
    --state_form error
    --anchor_type target
    --ddqn_argmax online_moving
    --measurement_mode q_target
    --target_update_mode soft --tau 0.02
    --use_n_step --n_step 3
    --N_horizon "$H" --alpha "$A"
    --network_seed "$S" --env_seed "$S" --seed "$S"
    --episodes "$EPISODES"
  )

  echo "[run ] $name"
  if [[ "$DRYRUN" == "1" ]]; then echo "    (cd $rundir && MPLBACKEND=Agg $PYTHON ${args[*]})"; return; fi

  ( cd "$rundir" && MPLBACKEND=Agg "$PYTHON" "${args[@]}" ) > "$rundir/run.log" 2>&1
  local ec=$?
  # 최종 Avg20 + 상태 추출(training_log.txt는 results_cartpole/<param_str>/ 아래)
  local tl; tl="$(find "$rundir" -name training_log.txt | head -1)"
  local fin="-" stat="?"
  if [[ -n "$tl" ]]; then
    fin="$(grep -oE 'Avg20: *[0-9.]+' "$tl" | tail -1 | grep -oE '[0-9.]+')"
    if [[ -n "$fin" ]]; then
      stat="$(awk -v f="$fin" 'BEGIN{ if(f>=480) print "SOLVED"; else if(f<450) print "COLLAPSED/LOW"; else print "PARTIAL" }')"
    fi
  fi
  echo -e "${H}_${A}\t${H}\t${A}\t${S}\t${ec}\t${fin:--}\t${stat}\t${rundir}" >> "$SUMMARY"
  if [[ $ec -eq 0 ]]; then touch "$done"; echo "    -> exit0  final=${fin:--}  $stat"
  else echo "    -> exit$ec (run.log 확인)"; fi
}

echo "=== ui4 seed sweep | configs=${#CONFIGS[@]} x seeds=$(echo $SEEDS|wc -w) | episodes=$EPISODES ==="
for cfg in "${CONFIGS[@]}"; do
  read -r H A <<< "$cfg"
  for S in $SEEDS; do run_one "$H" "$A" "$S"; done
done

echo
echo "=== 완료. 요약: $SUMMARY ==="
[[ "$DRYRUN" != "1" ]] && column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "분석:  $PYTHON analyze_collapse.py \"$OUTROOT\""
