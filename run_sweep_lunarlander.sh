#!/usr/bin/env bash
# LunarLander RHUKF sweep
# 72 runs = horizon{5,7} x alpha{0.1,0.3} x r_std{1.0,1.5,2.0} x p_init{0.03,0.05,0.07} x seed{0,42}
# mode = filter (RHUKF only, no Adam baseline), batch = 256
#
# -e 안 씀: run 하나 죽어도 나머지 계속 돌리려고. 실패는 명시적으로 처리.
set -uo pipefail

# ============================================================
#  CONFIG  --  네 argparse 기준으로 ↓ 이 블록만 확인/수정하면 됨
# ============================================================
PY="python3"
SCRIPT="srrhuif_v9.py"           # <-- 현재 쓰는 파일명으로 (v10+면 그걸로)
ENV="LunarLander-v2"             # <-- gym id 확인 (v2 / v3)
EPISODES=1500                    # <-- LunarLander 목표 episode 수 (timing에 직결)

# 플래그 이름 (네 코드와 다르면 여기만 고쳐) -----------------
F_ENV="--env"
F_EPISODES="--episodes"
F_BATCH="--batch"
F_HORIZON="--N_horizon"
F_ALPHA="--alpha"                # <-- 추정: UT spread 플래그명 확인
F_RSTD="--r_std"                 # <-- 추정: 측정노이즈 std 플래그명 확인
F_PINIT="--p_init"
F_SEED="--seed"
F_MODE="--mode"                  # <-- 추정: filter/compare/adam 선택 플래그명 확인
F_OUTDIR="--outdir"              # <-- 출력 폴더 지정 플래그명 확인

# 모든 run 공통 고정 플래그를 여기 추가 (activation, gamma, measurement_mode,
# state_form, target_update_mode 등 평소 쓰던 것들). 예시는 주석:
EXTRA_FLAGS=(
  # --measurement_mode pure_reward
  # --filter_form covariance
  # --state_form absolute
  # --activation_fn tanh
  # --gamma 0.99
)
# ============================================================

BATCH=256
MODE="filter"

HORIZONS=(5 7)
ALPHAS=(0.1 0.3)
R_STDS=(1.0 1.5 2.0)
P_INITS=(0.03 0.05 0.07)
SEEDS=(0 42)

OUTROOT="sweep_lunarlander_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTROOT"

total=$(( ${#HORIZONS[@]} * ${#ALPHAS[@]} * ${#R_STDS[@]} * ${#P_INITS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " LunarLander RHUKF sweep"
echo " total runs : $total"
echo " episodes   : $EPISODES   batch : $BATCH   mode : $MODE"
echo " out root   : $OUTROOT"
echo "=========================================="

i=0
start_all=$(date +%s)

for n in "${HORIZONS[@]}"; do
 for a in "${ALPHAS[@]}"; do
  for r in "${R_STDS[@]}"; do
   for p in "${P_INITS[@]}"; do
    for s in "${SEEDS[@]}"; do
      i=$((i+1))
      # tag: horizon은 h{}로 표기 (네 기존 naming의 n{nstep}과 충돌 방지)
      tag="lunar_a${a}_r${r}_p${p}_b${BATCH}_h${n}_s${s}"
      outdir="${OUTROOT}/${tag}"
      mkdir -p "$outdir"

      # --- resume: 이미 끝난 run은 건너뜀 (.done 마커) ---
      if [[ -f "${outdir}/.done" ]]; then
        echo "[$i/$total] SKIP (done) $tag"
        continue
      fi

      echo "[$i/$total] RUN  $tag"
      t0=$(date +%s)

      "$PY" "$SCRIPT" \
        "$F_ENV" "$ENV" \
        "$F_EPISODES" "$EPISODES" \
        "$F_BATCH" "$BATCH" \
        "$F_HORIZON" "$n" \
        "$F_ALPHA" "$a" \
        "$F_RSTD" "$r" \
        "$F_PINIT" "$p" \
        "$F_SEED" "$s" \
        "$F_MODE" "$MODE" \
        "$F_OUTDIR" "$outdir" \
        "${EXTRA_FLAGS[@]}" \
        > "${outdir}/train.log" 2>&1
      rc=$?

      t1=$(date +%s)
      if [[ $rc -eq 0 ]]; then
        touch "${outdir}/.done"
        echo "      done in $((t1-t0))s"
      else
        echo "      FAILED (rc=$rc) -> ${outdir}/train.log"
      fi
    done
   done
  done
 done
done

end_all=$(date +%s)
echo "=========================================="
echo " finished $total runs in $(( (end_all-start_all)/60 )) min ($((end_all-start_all))s)"
echo " logs under: $OUTROOT/*/train.log"
echo "=========================================="
