import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import os
import sys
import random
import argparse
import warnings
import math
import functools
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('Agg')

# =========================================================================
# Dual Output Logger (console + file) & File-only Print
# =========================================================================

class DualLogger:
    """콘솔 + 파일 동시 출력"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        # Windows 콘솔(cp949 등)에서 이모지/유니코드가 UnicodeEncodeError를 내지 않도록
        # 콘솔 인코딩을 가능하면 utf-8 + replace로 전환.
        try:
            self.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    def write(self, msg):
        try:
            self.stdout.write(msg)
        except UnicodeEncodeError:
            # 콘솔이 인코딩하지 못하는 문자는 대체 문자로 출력 (파일에는 원본 그대로 기록)
            enc = getattr(self.stdout, 'encoding', None) or 'ascii'
            self.stdout.write(msg.encode(enc, errors='replace').decode(enc, errors='replace'))
        self.file.write(msg)
        self.file.flush()
        
    def write_file_only(self, msg):
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

_dual_logger = None

def setup_file_logging(filepath):
    global _dual_logger
    _dual_logger = DualLogger(filepath)
    sys.stdout = _dual_logger
    print(f"[Logging] 모든 stdout이 저장됨: {filepath}")

def close_file_logging():
    global _dual_logger
    if _dual_logger is not None:
        sys.stdout = _dual_logger.stdout
        _dual_logger.close()
        _dual_logger = None

def file_print(*args, **kwargs):
    """터미널에는 출력하지 않고 텍스트 파일(training_log.txt)에만 기록합니다."""
    global _dual_logger
    if _dual_logger is not None:
        msg = " ".join(map(str, args))
        _dual_logger.write_file_only(msg + "\n")

print("=" * 70)
print(f"SRRHUIF/RHUKF v9.0 (Error/Absolute state | FV/Node/Layer | FiMos 제거) | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    # TF32는 여기서 전역으로 켜지 않는다 — apply_tf32_config()가 cfg를 보고
    # NN forward 전용으로만 활성화하고, 행렬연산(필터 공분산 bmm)은 FP32로 남긴다.
print("=" * 70)

# =========================================================================
# TF32 분리: NN forward만 TF32 matmul, 필터 행렬연산은 FP32
#   - TF32는 Ampere+ (compute capability ≥ 8.0) GPU의 float32 matmul/bmm에만 적용.
#   - torch.linalg 분해(cholesky/qr/solve_triangular)는 이 플래그와 무관하게 FP32
#     → 필터의 분해는 항상 안전. 분리 대상은 matmul/bmm 정밀도뿐.
#   - 전역 기본은 FP32(allow_tf32=False). forward 함수만 데코레이터로 호출 동안 TF32 on.
# =========================================================================
TF32_FORWARD_ENABLED = False  # apply_tf32_config()에서 cfg + 하드웨어 보고 확정

def _tf32_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

def apply_tf32_config(cfg):
    """전역 matmul을 FP32로 고정하고, GPU 지원 + cfg.use_tf32_forward일 때만 forward TF32 활성.
    Returns (enabled: bool, supported: bool)."""
    global TF32_FORWARD_ENABLED
    supported = _tf32_supported()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    TF32_FORWARD_ENABLED = bool(getattr(cfg, 'use_tf32_forward', True) and supported)
    return TF32_FORWARD_ENABLED, supported

def tf32_forward(fn):
    """NN forward 함수 데코레이터: 호출 동안만 TF32 matmul 허용(활성 시), 종료 시 원복.
    비활성/미지원이면 완전 no-op(오버헤드 없음)이라 항상 붙여둬도 안전."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not TF32_FORWARD_ENABLED:
            return fn(*args, **kwargs)
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_mm
            torch.backends.cudnn.allow_tf32 = prev_cudnn
    return wrapper

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.float32)
DTYPE = torch.float32
DTYPE_FWD = torch.float32
JITTER = 1e-7
JITTER_TRIA = 1e-7

# ── Fallback 발동 횟수 카운터 (수치적 안정성 진단용) ──
#   chol_1e5: Cholesky(1e-6 jitter) 실패 → 1e-5 jitter 재시도 횟수
#   tria_qr : tria_operation_batch에서 Cholesky 실패 → QR 폴백 횟수
FALLBACK_COUNTS = {'chol_1e5': 0, 'tria_qr': 0}

def safe_cholesky_fallback(M, eye, base_jitter=JITTER):
    """Cholesky를 base_jitter로 먼저 시도하고, 실패 시 1e-5 jitter로 재시도하며
    그 횟수를 FALLBACK_COUNTS['chol_1e5']에 누적한다."""
    try:
        return torch.linalg.cholesky(M + base_jitter * eye)
    except Exception:
        FALLBACK_COUNTS['chol_1e5'] += 1
        return torch.linalg.cholesky(M + 1e-5 * eye)

# =========================================================================
# 0. Environment Registry
#   환경별 설정을 한 곳에서 관리. 새 환경 추가 시 여기에 항목만 넣으면 됨.
#     obs_scale   : InputNormalizer가 관측값을 나눌 per-dim 스케일 (길이 = obs dim)
#     max_steps   : 에피소드당 최대 스텝 (env truncation)
#     results_dir : 결과 저장 폴더명
# =========================================================================
ENV_CONFIGS: Dict[str, Dict] = {
    "CartPole-v1": {
        "obs_scale": [2.4, 3.0, 0.21, 2.0],
        "max_steps": 500,
        "max_episodes": 150,
        "eps_decay_steps": 2000,
        "buffer_size": 50000,
        "results_dir": "results_cartpole",
        "reward_threshold": 195,       # solved 기준 (avg reward)
        "reward_ylim": [0, 520],       # 보상 그래프 y축 (CartPole: 0~500)
    },
    "LunarLander-v3": {
        # obs: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
        "obs_scale": [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0],
        "max_steps": 1000,
        "max_episodes": 1000,
        "eps_decay_steps": 15000,
        "buffer_size": 300000,
        "results_dir": "results_lunarlander",
        "reward_threshold": 200,       # solved 기준 (avg reward)
        "reward_ylim": [-450, 320],    # 보상 그래프 y축 (LunarLander: 음수 추락보상 포함)
    },
}


def build_env_kwargs(cfg) -> Dict:
    """env별 gym.make 추가 kwargs. LunarLander 계열에만 wind 파라미터 적용.
    (CartPole 등 다른 env에 enable_wind를 넘기면 gym이 에러내므로 env 이름으로 분기)."""
    kwargs: Dict = {}
    if cfg.env_name.startswith("LunarLander"):
        kwargs['enable_wind'] = cfg.enable_wind
        kwargs['wind_power'] = cfg.wind_power
        kwargs['turbulence_power'] = cfg.turbulence_power
    return kwargs

# ========================================================================
# 1. Configuration
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"  #"LunarLander-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # [TF32] NN forward(matmul/bmm)만 TF32 허용, 필터 행렬연산은 FP32 유지.
    #   Ampere+ (compute capability ≥ 8.0) GPU에서만 실제로 효과. CPU/구형 GPU면 무시(FP32).
    use_tf32_forward: bool = True
    max_episodes: int = 200
    max_steps: int = 500

    # [env config] None이면 __post_init__에서 ENV_CONFIGS[env_name] 값으로 자동 채움.
    #   명시적으로 주면(CLI/직접) 그 값이 우선.
    obs_scale: Optional[List[float]] = None
    results_dir: Optional[str] = None
    
    _max_steps_explicit: bool = False  # --max_steps로 직접 지정 시 env 기본값이 덮어쓰지 않도록
    _max_episodes_explicit: bool = False  # --episodes로 직접 지정 시 env 기본값이 덮어쓰지 않도록
    _eps_decay_steps_explicit: bool = False  # --eps_decay_steps로 직접 지정 시 env 기본값이 덮어쓰지 않도록
    _buffer_size_explicit: bool = False  # --buffer로 직접 지정 시 env 기본값이 덮어쓰지 않도록

    # [LunarLander-v3 wind] gym.make에 전달되는 바람 옵션. LunarLander 계열에만 적용됨
    #   (build_env_kwargs 참조). enable_wind=False면 wind_power/turbulence_power는 무시.
    enable_wind: bool = False
    wind_power: float = 15.0        # gym 기본값 15.0 (권장 0.0~20.0)
    turbulence_power: float = 1.5   # gym 기본값 1.5 (권장 0.0~2.0)


    # Decoupling Mode 
    #   'node'  = per-neuron block-diagonal (mean-field, 가장 가벼움)
    #   'layer' = per-layer joint (K-FAC-like, within-layer covariance 보존)
    #   'fv'    = full vector (모든 파라미터를 한 블록으로, 가장 정확하지만 무거움)
    decoupling_mode: str = 'fv'

    filter_form: str = 'covariance' # information or covariance
    state_form: str = 'error' # absolute or error
    measurement_mode: str = 'q_target' # q_target or pure_reward
    
    # [v7: Anchor type] state_form='error'에서 θ_anchor 결정 방식
    #   'target'  = θ_anchor = θ_target (soft anchor, Moving Target 안전)
    #   'current' = θ_anchor = θ_active (직전 step 결과, 선형화 오차 최소)
    #   'init'    = θ_anchor = θ_init (학습 시작 시 frozen된 임의 초기값, FIR/RHE 정신)
    anchor_type: str = 'target'
    
    # [v7: DDQN argmax policy] state_form='error'에서 Y_batch 계산 시 argmax θ
    #   'target'         = θ_target argmax (DQN-with-target, 완전 캐싱)
    #   'online_frozen'  = θ_active (호라이즌 시작 직전 동결) argmax (표준 DDQN, 완전 캐싱) ★권장
    #   'online_moving'  = θ_anchor + Δμ^{h-1} argmax (매 h마다 갱신, 캐싱 불가, ablation용)
    #   'spas'           = Sigma-point ensemble argmax (sigma around θ_anchor, mean Q),  θ 하나의 max-bias를 sigma ensemble 평균으로 완화. 캐싱 가능.
    ddqn_argmax: str = 'online_moving'

    # [v9+: online_moving h=0 초기화] ddqn_argmax='online_moving'에서 호라이즌 첫 스텝(h=0)의
    # argmax 정책. h≥1에서는 항상 θ_anchor + Δμ^{h-1} 사용.
    #   'prev_est'     = 직전 호라이즌 종료 시점 active θ (horizon 직전 theta_active)
    #   'theta_target' = θ_target (보수적, target net 안정성 활용)
    #   'spas'         = sigma-point ensemble mean argmax (FV 전용)
    h0_online_moving_init: str = 'theta_target'
    
    use_twin: bool = False  # ★ Overestimation 구조적 해결: 페널티 c 없이 min으로 안전 (TD3 식)

    use_residual: bool = False
    use_residual_auto_depth: Optional[int] = 3   # 3 hidden 이상이면 auto-on

    node_layer_other_source: str = 'current' #prior or current
    
    # [FIR 철학] h=0의 prior source
    #   'target' = target net (현재 코드 기존 동작)
    #   'init'   = 학습 시작시 frozen된 θ_init (RHE/FIR 정신에 더 가까움)
    h0_prior_source: str = 'target'
    
    shared_layers: List[int] = field(default_factory=lambda: [24,24])   # [] = no hidden shared layers
    value_layers: List[int] = field(default_factory=lambda: [])
    advantage_layers: List[int] = field(default_factory=lambda: [])
    q_layers: List[int] = field(default_factory=lambda: [])        # [] = sing le linear layer (dimS → nA)

    use_dueling: bool = False # False로 두어 순수 DDQN 아키텍처 사용 (Layer 모드 최적화)

    gamma: float = 0.9
    scale_factor: float = 1.0
    
    tau_srrhuif: float = 0.02
    update_interval: int = 4
    
    target_update_mode: str = 'soft'
    target_update_period: int = 200   # hard mode 시 호라이즌 업데이트 카운트 기준
    
    activation_fn: str = 'silu' #tanh
    init_scheme: str = 'he' #xavier

    buffer_size: int = 50000
    batch_size: int = 128
    N_horizon: int = 6
    
    q_init: float = 1e-2
    q_end: float = 1e-2

    r_init: float = 1.5
    r_end: float = 1.5
    huber_c: float = 1000
    
    tikhonov_lambda: float = 1e-8

    p_init: float = 0.05
    p_delta_init: float = 0.05
    
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0
    
    use_n_step: bool = True
    n_step_size: int = 3

    use_per: bool = False
    per_alpha: float = 0.6      # priority 지수 (Schaul 2016 기본값)
    per_eps: float = 1e-6       # 0 priority 방지용 offset
    per_apply_is_weight: bool = False  # IS weight를 R inflation으로 반영 (실험용)

    warmup_step : int = 0

    train_mode: str = 'filter' #filter or adam
    use_adam_warmup: bool = False
    adam_lr: float = 1e-3

    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 15000

    max_layer_step: float = 0.0
    max_k_gain: float = 0.0

    use_spas: bool =  True

    use_input_norm: bool = True
    use_compile: bool = True
    plot_interval: int = 60
    log_interval : int = 1

    # [video] RecordVideo 백그라운드(headless rgb_array) 녹화
    #   record_video=True 면 매 video_interval 에피소드마다 현재 θ로 greedy rollout 1판을
    #   mp4로 저장. 학습 env와 분리된 별도 env에서 돌아가므로 학습/카운트에 영향 없음.
    record_video: bool = True
    video_interval: int = 100           # 매 N 에피소드마다 1개 녹화 (ep % N == 0)
    video_dir: Optional[str] = None    # None이면 {outdir}/videos
    video_async: bool = True           # True면 데몬 스레드에서 녹화(학습 비차단)

    seed: int = 42
    network_seed: Optional[int] = 42
    env_seed: Optional[int] = 42
    
    use_full_eigvalsh: bool = True
    diag_ref_states: bool = True
    diag_argmax_flip: bool = True
    diag_eff_rank: bool = True
    diag_horizon_cond: bool = True
    diag_buffer: bool = True
    
    # [v9+] Activation health: hidden 레이어 포화/죽은 뉴런 모니터
    diag_act_health: bool = True
    act_health_n_sample: int = 512   # 버퍼에서 뽑을 샘플 수
    act_health_sat_thresh: float = 0.95   # tanh/gelu 포화 임계 (|post| 평균)
    act_health_dead_thresh: float = 0.05  # 모든 활성화: 거의 0 출력 임계 (|post| 최대)

    # [probe] Per-h activation regime + effective-gain (horizon 내부 fold-gain runaway 진단)
    #   붕괴 시그니처: 한 horizon 안에서 mean_gain/frac_pos가 fold(h) 따라 증가
    #   (건강하면 flat/감소). 매 h마다 forward가 추가되므로 반드시 cadence 게이팅.
    diag_act_regime: bool = True
    act_regime_every: int = 5     # N 에피소드마다만 프로브 (1이면 매 에피소드)
    act_regime_warmup: int = 0    # 이 에피소드 이후부터만 프로브 (후반 phase 집중용)

    # [probe] Sigma-spread activation (FV 전용): 시그마 클라우드가 레이어별 pre-act를
    #   얼마나 퍼뜨리고(spread) 활성화가 그 spread를 증폭/수축(amp)하는지 — UKF runaway
    #   메커니즘을 실제 시그마 포인트로 직접 관찰. 게이트 ON일 때만 시그마 forward 1회 추가.
    diag_sigma_spread: bool = False
    sigma_spread_every: int = 5
    sigma_spread_warmup: int = 0

    # [analysis] 로그 핵심원인 진단(VERDICT) + verbosity gating
    #   'auto'   = 콘솔에 요약(VERDICT/culprit/trend)만, 룰 발동 시에만 파일에 풀 덤프
    #   'always' = 기존 풀 덤프 유지 (요약은 위에 추가) / 'summary' = 요약만, 풀 덤프 항상 숨김
    diag_log_mode: str = 'always'
    collapse_amp_thresh: float = 1.0   # sigma-spread amp 이 값 초과 + 증가 → RUNAWAY
    cond_warn: float = 1e6             # cond(P_zz/Y) 경고
    dead_warn: float = 0.3             # dead 뉴런 비율 경고
    flip_warn: float = 0.4             # argmax flip rate 경고
    prior_ratio_warn: float = 3.0      # |H^Tθ|/|z-ẑ| 경고 (prior 지배)
    innov_warn: float = 1e3            # innovation max 폭발 경고
    save_file_log: bool = True

    def __post_init__(self):
        # node/layer + covariance도 이제 지원되므로 auto-fallback 제거됨
        # ── 옵션 값 검증 ──
        valid_modes = {'node', 'layer', 'fv'}
        if self.decoupling_mode not in valid_modes:
            raise ValueError(
                f"decoupling_mode='{self.decoupling_mode}' invalid. "
                f"Must be one of {valid_modes}."
            )
        valid_h0 = {'target', 'init', 'current'}
        if self.h0_prior_source not in valid_h0:
            raise ValueError(
                f"h0_prior_source='{self.h0_prior_source}' invalid. "
                f"Must be one of {valid_h0}."
            )
        valid_init = {'orthogonal', 'he', 'xavier'}
        if self.init_scheme not in valid_init:
            raise ValueError(
                f"init_scheme='{self.init_scheme}' invalid. "
                f"Must be one of {valid_init}."
            )
        valid_form = {'information', 'covariance'}
        if self.filter_form not in valid_form:
            raise ValueError(
                f"filter_form='{self.filter_form}' invalid. "
                f"Must be one of {valid_form}."
            )
        # [v7+] node/layer + covariance 지원 (rhukf_step, rhukf_step_error 추가됨)
        
        # [v7+] node_layer_other_source 검증
        valid_other = {'current', 'prior'}
        if self.node_layer_other_source not in valid_other:
            raise ValueError(
                f"node_layer_other_source='{self.node_layer_other_source}' invalid. "
                f"Must be one of {valid_other}."
            )
        
        # [v7] state_form 검증
        valid_state = {'absolute', 'error'}
        if self.state_form not in valid_state:
            raise ValueError(
                f"state_form='{self.state_form}' invalid. Must be one of {valid_state}."
            )

        # [v9+] measurement_mode 검증
        valid_meas = {'q_target', 'pure_reward'}
        if self.measurement_mode not in valid_meas:
            raise ValueError(
                f"measurement_mode='{self.measurement_mode}' invalid. "
                f"Must be one of {valid_meas}."
            )
        if self.measurement_mode == 'pure_reward':
            if self.decoupling_mode != 'fv':
                raise ValueError(
                    f"measurement_mode='pure_reward'는 decoupling_mode='fv'에서만 검증됨 "
                    f"(현재: '{self.decoupling_mode}'). pure_reward의 cross-covariance "
                    f"cancellation은 전체 weight covariance를 전제로 하므로 node/layer "
                    f"decoupled에선 이론적 근거가 약함. FV에서 먼저 실험하세요."
                )
            if self.r_init > 1.0:
                warnings.warn(
                    f"[v9+] pure_reward mode: r_init={self.r_init} is large for y=r. "
                    f"r_init ∈ [0.1, 0.5] is recommended (y=r has small dynamic range)."
                )

            if not self.use_per:
                warnings.warn(
                    f"[v9+] pure_reward mode: use_per=False. pure_reward는 환경 보상으로만 "
                    f"절대 Q 스케일을 anchor하므로 윈도우에 terminal transition이 충분히 "
                    f"들어와야 함. PER (--use_per) 활성화를 강력 권장."
                )

        # [v9+] PER 검증
        if self.use_per:
            if not (0.0 <= self.per_alpha <= 1.0):
                raise ValueError(f"per_alpha={self.per_alpha} must be in [0, 1]")
            if self.per_eps <= 0:
                raise ValueError(f"per_eps={self.per_eps} must be > 0")

        if self.state_form == 'error':
            # information & covariance 모두 모든 mode 지원
            if self.ddqn_argmax == 'spas' and self.decoupling_mode != 'fv':
                raise ValueError(
                    f"ddqn_argmax='spas'는 FV 전용 (현재 '{self.decoupling_mode}')."
                )
        valid_anchor = {'target', 'current', 'init'}
        if self.anchor_type not in valid_anchor:
            raise ValueError(
                f"anchor_type='{self.anchor_type}' invalid. Must be one of {valid_anchor}."
            )
        valid_argmax = {'target', 'online_frozen', 'online_moving', 'spas'}
        if self.ddqn_argmax not in valid_argmax:
            raise ValueError(
                f"ddqn_argmax='{self.ddqn_argmax}' invalid. Must be one of {valid_argmax}."
            )
        valid_h0_om = {'prev_est', 'theta_target', 'spas'}
        if self.h0_online_moving_init not in valid_h0_om:
            raise ValueError(
                f"h0_online_moving_init='{self.h0_online_moving_init}' invalid. "
                f"Must be one of {valid_h0_om}."
            )
        if (self.h0_online_moving_init == 'spas' and
                self.ddqn_argmax == 'online_moving' and
                self.decoupling_mode != 'fv'):
            raise ValueError(
                f"h0_online_moving_init='spas'는 FV 전용 (현재 decoupling_mode='{self.decoupling_mode}')."
            )
        
        # [v7+] activation_fn 검증
        valid_act = {'tanh', 'relu', 'leaky_relu', 'mish', 'gelu', 'silu'}
        if self.activation_fn not in valid_act:
            raise ValueError(
                f"activation_fn='{self.activation_fn}' invalid. Must be one of {valid_act}."
            )
        
        # [v7+] Twin-Q 검증
        if self.use_twin:
            if self.ddqn_argmax == 'spas':
                raise ValueError(
                    "use_twin=True는 ddqn_argmax='spas'와 호환되지 않음 "
                    "(min(Q1, Q2)으로 자동 max-bias 완화 → spas 불필요)."
                )
            if self.state_form != 'error':
                raise ValueError(
                    "use_twin=True는 state_form='error'에서만 지원 (Y_cache 공유 기반 구조). "
                    f"현재: '{self.state_form}'."
                )
        
        # [v9+] Train mode 검증
        valid_train_mode = {'filter', 'adam'}
        if self.train_mode not in valid_train_mode:
            raise ValueError(
                f"train_mode='{self.train_mode}' invalid. Must be one of {valid_train_mode}."
            )
        if self.train_mode == 'adam' and self.use_twin:
            raise ValueError(
                "train_mode='adam'는 use_twin=True와 비호환 "
                "(Twin은 filter Y_cache 공유 구조 전제)."
            )

        # [v9+] Adam warm-up 검증
        if self.use_adam_warmup:
            if self.use_twin:
                raise ValueError(
                    "use_adam_warmup=True는 use_twin=True와 비호환 "
                    "(Twin-Q는 filter Y_cache 공유 구조 전제). "
                    "Adam warm-up은 single network 경로에서만 지원."
                )
            if self.adam_lr <= 0:
                raise ValueError(f"adam_lr={self.adam_lr} must be > 0")

        # [v7+] target_update 검증
        valid_tgt_update = {'soft', 'hard'}
        if self.target_update_mode not in valid_tgt_update:
            raise ValueError(
                f"target_update_mode='{self.target_update_mode}' invalid. Must be one of {valid_tgt_update}."
            )
        if self.target_update_mode == 'hard' and self.target_update_period <= 0:
            raise ValueError(
                f"target_update_period={self.target_update_period} must be > 0 for hard update."
            )
        
        self.r_inv_sqrt = 1.0 / self.r_init
        self.r_inv = 1.0 / (self.r_init ** 2)
        duel_str = "D3QN" if self.use_dueling else "DDQN"
        nstep_str = f"n{self.n_step_size}" if self.use_n_step else "n1"
        if self.filter_form == 'covariance':
            form_str = 'rhukf'
        else:
            form_str = 'rhuif'
        if self.state_form == 'error':
            state_tag = f"ES_{self.anchor_type}_{self.ddqn_argmax}"
        else:
            state_tag = "ABS"
        # [v9+] measurement mode tag
        meas_tag = "MQ" if self.measurement_mode == 'q_target' else "MR"  # MQ = Q_target, MR = pure_Reward
        # [v9+] PER tag
        per_tag = f"_PER{self.per_alpha}" if self.use_per else ""
        # [v9+] Adam warm-up tag
        adam_tag = f"_adam{self.adam_lr:g}" if self.use_adam_warmup else ""
        # [LunarLander wind] wind 켜면 결과가 섞이지 않도록 태그 추가
        wind_tag = (f"_wind{self.wind_power:g}t{self.turbulence_power:g}"
                    if (self.enable_wind and self.env_name.startswith("LunarLander")) else "")
        # [v9+] Train mode tag (filter는 생략, adam은 명시)
        if self.train_mode == 'adam':
            # Adam은 항상 q_target form만 사용 → meas_tag는 폴더명에 포함 안 함
            self.param_str = (
                f"ADAM_lr{self.adam_lr:g}{per_tag}_{duel_str}_{self.init_scheme}_"
                f"b{self.batch_size}_{nstep_str}_s{self.network_seed}"
            )
        else:
            self.param_str = (
                f"{self.decoupling_mode}_{form_str}_{state_tag}_{meas_tag}{per_tag}{adam_tag}{wind_tag}_{duel_str}_{self.init_scheme}_"
                f"h0_{self.h0_prior_source}_a{self.alpha}_r{self.r_init}_"
                f"p{self.p_init}_pd{self.p_delta_init}_b{self.batch_size}_{nstep_str}_s{self.network_seed}"
            )
        # ── [env config] 환경별 설정 적용 ──
        env_cfg = ENV_CONFIGS.get(self.env_name, {})
        if self.obs_scale is None:
            self.obs_scale = env_cfg.get('obs_scale')
        if ('max_steps' in env_cfg) and (not self._max_steps_explicit):
            self.max_steps = env_cfg['max_steps']
        if ('max_episodes' in env_cfg) and (not self._max_episodes_explicit):
            self.max_episodes = env_cfg['max_episodes']
        if ('eps_decay_steps' in env_cfg) and (not self._eps_decay_steps_explicit):
            self.eps_decay_steps = env_cfg['eps_decay_steps']
        if ('buffer_size' in env_cfg) and (not self._buffer_size_explicit):
            self.buffer_size = env_cfg['buffer_size']
        results_dir = self.results_dir or env_cfg.get('results_dir', 'results')
        self.outdir = f"./{results_dir}/{self.param_str}"
        os.makedirs(self.outdir, exist_ok=True)

cfg = Config()

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default=cfg.env_name,
                    help=f"Gym env id. 등록된 환경: {list(ENV_CONFIGS.keys())} "
                         f"(미등록 환경은 obs_scale=None → --no... 정규화 주의)")
parser.add_argument('--max_steps', type=int, default=None,
                    help="에피소드당 최대 스텝. 미지정 시 ENV_CONFIGS의 env 기본값 사용.")
parser.add_argument('--enable_wind', dest='enable_wind', action='store_true', default=cfg.enable_wind,
                    help="[LunarLander-v3] 바람 활성화 (gym.make(enable_wind=True))")
parser.add_argument('--no_wind', dest='enable_wind', action='store_false',
                    help="[LunarLander-v3] 바람 비활성화")
parser.add_argument('--wind_power', type=float, default=cfg.wind_power,
                    help="[LunarLander-v3] 바람 세기 (gym 기본 %(default)s, 권장 0~20)")
parser.add_argument('--turbulence_power', type=float, default=cfg.turbulence_power,
                    help="[LunarLander-v3] 난기류 세기 (gym 기본 %(default)s, 권장 0~2)")
parser.add_argument('--tf32_forward', dest='use_tf32_forward', action='store_true', default=cfg.use_tf32_forward,
                    help="NN forward(matmul)만 TF32 허용 (Ampere+ GPU에서만 효과). 행렬연산은 FP32.")
parser.add_argument('--no_tf32_forward', dest='use_tf32_forward', action='store_false',
                    help="forward도 FP32로 (TF32 완전 비활성)")
parser.add_argument('--record_video', action='store_true', default=cfg.record_video,
                    help="매 --video_interval 에피소드마다 greedy rollout을 headless mp4로 녹화")
parser.add_argument('--video_interval', type=int, default=cfg.video_interval,
                    help="녹화 주기(에피소드). default %(default)s")
parser.add_argument('--video_dir', type=str, default=None,
                    help="mp4 저장 폴더. 미지정 시 {outdir}/videos")
parser.add_argument('--no_video_async', dest='video_async', action='store_false', default=cfg.video_async,
                    help="녹화를 데몬 스레드 대신 동기로 실행(학습이 잠깐 멈춤)")
parser.add_argument('--mode', type=str, default=cfg.decoupling_mode, choices=['node', 'layer', 'fv'],
                    help="'node' = per-neuron, 'layer' = per-layer joint (K-FAC-like), 'fv' = full vector")
parser.add_argument('--h0_prior', type=str, default=cfg.h0_prior_source,
                    choices=['target', 'init'],
                    help="h=0 prior source: 'target' (target net) or 'init' (frozen θ_init, FIR philosophy)")
parser.add_argument('--init_scheme', type=str, default=cfg.init_scheme,
                    choices=['orthogonal', 'he', 'xavier'])
parser.add_argument('--dueling', action='store_true', default=cfg.use_dueling)
parser.add_argument('--alpha', type=float, default=cfg.alpha)
parser.add_argument('--beta', type=float, default=cfg.beta)
parser.add_argument('--q_init', type=float, default=cfg.q_init,
                    help="Process noise std Q 초기값 (eps-decay 시작점, default %(default)s)")
parser.add_argument('--q_end', type=float, default=cfg.q_end,
                    help="Process noise std Q 최종값 (eps-decay 종료점, default %(default)s)")
parser.add_argument('--r_init', type=float, default=cfg.r_init)
parser.add_argument('--r_end', type=float, default=cfg.r_end,
                    help="Measurement noise std R 최종값 (eps-decay 종료점, default %(default)s)")
parser.add_argument('--p_init', type=float, default=cfg.p_init)
parser.add_argument('--episodes', type=int, default=None,
                    help="총 학습 에피소드 수. 미지정 시 ENV_CONFIGS의 env 기본값 사용.")
parser.add_argument('--batch', type=int, default=cfg.batch_size)
parser.add_argument('--buffer', type=int, default=None,
                    help="Replay buffer 크기. 미지정 시 ENV_CONFIGS의 env 기본값 사용.")
parser.add_argument('--N_horizon', type=int, default=cfg.N_horizon,
                    help="Receding horizon window size (default %(default)s)")
parser.add_argument('--gamma', type=float, default=cfg.gamma,
                    help="Discount factor (default %(default)s)")
parser.add_argument('--eps_decay_steps', type=int, default=None,
                    help="ε-greedy decay step count. 미지정 시 ENV_CONFIGS의 env 기본값 사용.")
parser.add_argument('--tau', type=float, default=cfg.tau_srrhuif)
parser.add_argument('--target_update_mode', type=str, default=cfg.target_update_mode,
                    choices=['soft', 'hard'],
                    help="Target net update: 'soft' (tau-blend each horizon) or 'hard' (full copy every N)")
parser.add_argument('--target_update_period', type=int, default=cfg.target_update_period,
                    help="Hard update period (호라이즌 업데이트 카운트 기준). soft 모드에서는 무시.")
parser.add_argument('--tikhonov', type=float, default=cfg.tikhonov_lambda)
parser.add_argument('--seed', type=int, default=cfg.seed)
parser.add_argument('--network_seed', type=int, default=cfg.network_seed)
parser.add_argument('--env_seed', type=int, default=cfg.env_seed)
parser.add_argument('--use_n_step', action='store_true', default=cfg.use_n_step,
                    help="Enable N-step return for TD target bootstrapping")
parser.add_argument('--no_n_step', dest='use_n_step', action='store_false',
                    help="Disable N-step (use 1-step bootstrap)")
parser.add_argument('--n_step', type=int, default=cfg.n_step_size,
                    help="N-step horizon (only used if use_n_step=True)")
parser.add_argument('--filter_form', type=str, default=cfg.filter_form,
                    choices=['information', 'covariance'],
                    help="FV mode filter form: 'information'=SRRHUIF, 'covariance'=RHUKF")

parser.add_argument('--state_form', type=str, default=cfg.state_form,
                    choices=['absolute', 'error'],
                    help="'absolute'=legacy θ filtering, 'error'=Error-State Δθ filtering (FV only)")
parser.add_argument('--measurement_mode', type=str, default=cfg.measurement_mode,
                    choices=['q_target', 'pure_reward'],
                    help="[v9+] 'q_target'=y=r+γQ(s',a*;θ_T), h(w)=Q(s,a;w) (기존). "
                         "'pure_reward'=y=r, h(w)=Q(s,a;w)-γQ(s',a*;w) (신규, Kalman-pure).")
parser.add_argument('--use_per', action='store_true',
                    help="[v9+] Prioritized Experience Replay 활성화. pure_reward 모드에서 "
                         "terminal transition을 oversampling하여 Q-floating을 방지.")
parser.add_argument('--per_alpha', type=float, default=cfg.per_alpha,
                    help="[v9+] PER priority exponent (default 0.6, Schaul 2016)")
parser.add_argument('--per_eps', type=float, default=cfg.per_eps,
                    help="[v9+] PER zero-priority offset (default 1e-6)")
parser.add_argument('--per_apply_is_weight', action='store_true',
                    help="[v9+] IS weight를 R inflation으로 반영 (실험적, off by default)")
parser.add_argument('--anchor_type', type=str, default=cfg.anchor_type,
                    choices=['target', 'current', 'init'],
                    help="Error-state anchor: 'target'=θ_target, 'current'=θ_active_prev, "
                         "'init'=frozen θ_init (임의 초기값)")
parser.add_argument('--ddqn_argmax', type=str, default=cfg.ddqn_argmax,
                    choices=['target', 'online_frozen', 'online_moving', 'spas'],
                    help="Error-state Y_batch argmax policy. 'online_frozen' = standard DDQN (recommended).")
parser.add_argument('--h0_online_moving_init', type=str, default=cfg.h0_online_moving_init,
                    choices=['prev_est', 'theta_target', 'spas'],
                    help="online_moving h=0 argmax 초기화: 'prev_est'=직전 active θ (default), "
                         "'theta_target'=θ_target, 'spas'=sigma ensemble mean (FV 전용)")
parser.add_argument('--p_delta_init', type=float, default=cfg.p_delta_init,
                    help="Error-state P_Δ initial scale (trust region)")
parser.add_argument('--use_twin', action='store_true',
                    help="Twin-Q (Clipped Double Q-Learning). 두 독립 (θ_1, θ_2)로 min target.")
parser.add_argument('--activation_fn', type=str, default=cfg.activation_fn,
                    choices=['tanh', 'relu', 'leaky_relu', 'mish', 'gelu', 'silu'],
                    help="히든 레이어 활성화 함수")
parser.add_argument('--node_layer_other_source', type=str, default=cfg.node_layer_other_source,
                    choices=['current', 'prior'],
                    help="Node/Layer 모드에서 OTHER 레이어들의 θ source: 'current' (running 추정치, 기존) / 'prior' (h=0 기준점)")
parser.add_argument('--use_residual', action='store_true',
                    help="Same-dim hidden layers에 residual (skip) connection 추가. UKF의 hidden layer vanishing 신호 문제 해결.")
parser.add_argument('--shared_layers', type=int, nargs='*', default=None,
                    help="Shared hidden layer sizes (예: --shared_layers 16 16). 미지정 시 Config default.")
parser.add_argument('--value_layers', type=int, nargs='*', default=None,
                    help="Value head hidden layer sizes (dueling 시)")
parser.add_argument('--advantage_layers', type=int, nargs='*', default=None,
                    help="Advantage head hidden layer sizes (dueling 시)")
parser.add_argument('--q_layers', type=int, nargs='*', default=None,
                    help="Q head hidden layer sizes (non-dueling 시)")
parser.add_argument('--train_mode', type=str, default=cfg.train_mode,
                    choices=['filter', 'adam'],
                    help="[v9+] 'filter'=SRRHUKF/RHUKF (기존), 'adam'=순수 Adam DDQN (compare용)")
parser.add_argument('--use_adam_warmup', dest='use_adam_warmup', action='store_true', default=cfg.use_adam_warmup,
                    help="[v9+] batch_hist가 가득 차기 전(filter 시작 전) 구간에 Adam으로 θ 업데이트")
parser.add_argument('--no_adam_warmup', dest='use_adam_warmup', action='store_false',
                    help="Adam warm-up 비활성 (기존 동작: 윈도우 채우는 동안 θ 변화 없음)")
parser.add_argument('--adam_lr', type=float, default=cfg.adam_lr,
                    help="[v9+] Adam warm-up learning rate (default %(default)s)")
args, _ = parser.parse_known_args()

cfg.env_name = args.env
cfg.enable_wind = args.enable_wind
cfg.wind_power = args.wind_power
cfg.turbulence_power = args.turbulence_power
cfg.use_tf32_forward = args.use_tf32_forward
if args.max_steps is not None:
    cfg.max_steps = args.max_steps
    cfg._max_steps_explicit = True
cfg.record_video = args.record_video
cfg.video_interval = args.video_interval
cfg.video_dir = args.video_dir
cfg.video_async = args.video_async
# env 변경 시 obs_scale/results_dir/max_steps는 __post_init__에서 새 env 기준으로 재계산되도록 리셋
cfg.obs_scale = None
cfg.results_dir = None
cfg.decoupling_mode = args.mode
cfg.h0_prior_source = args.h0_prior
cfg.init_scheme = args.init_scheme
cfg.use_dueling = args.dueling
cfg.alpha = args.alpha
cfg.beta = args.beta
cfg.q_init = args.q_init
cfg.q_end = args.q_end
cfg.r_init = args.r_init
cfg.r_end = args.r_end
cfg.p_init = args.p_init
if args.episodes is not None:
    cfg.max_episodes = args.episodes
    cfg._max_episodes_explicit = True
cfg.batch_size = args.batch
if args.buffer is not None:
    cfg.buffer_size = args.buffer
    cfg._buffer_size_explicit = True
cfg.N_horizon = args.N_horizon
cfg.gamma = args.gamma
if args.eps_decay_steps is not None:
    cfg.eps_decay_steps = args.eps_decay_steps
    cfg._eps_decay_steps_explicit = True
cfg.tau_srrhuif = args.tau
cfg.target_update_mode = args.target_update_mode
cfg.target_update_period = args.target_update_period
cfg.tikhonov_lambda = args.tikhonov
cfg.seed = args.seed
cfg.network_seed = args.network_seed
cfg.env_seed = args.env_seed
cfg.use_n_step = args.use_n_step
cfg.n_step_size = args.n_step
# ── filter_form 명시 여부 감지 (auto-fallback 보호용) ──
import sys as _sys
cfg._filter_form_explicit = any(a.startswith('--filter_form') for a in _sys.argv)
cfg.filter_form = args.filter_form
cfg.state_form = args.state_form
cfg.measurement_mode = args.measurement_mode
cfg.use_per = args.use_per
cfg.per_alpha = args.per_alpha
cfg.per_eps = args.per_eps
cfg.per_apply_is_weight = args.per_apply_is_weight
cfg.anchor_type = args.anchor_type
cfg.ddqn_argmax = args.ddqn_argmax
cfg.h0_online_moving_init = args.h0_online_moving_init
cfg.p_delta_init = args.p_delta_init
cfg.use_twin = args.use_twin
cfg.activation_fn = args.activation_fn
cfg.node_layer_other_source = args.node_layer_other_source
cfg.use_residual = args.use_residual
cfg.train_mode = args.train_mode
cfg.use_adam_warmup = args.use_adam_warmup
cfg.adam_lr = args.adam_lr

if args.shared_layers is not None:
    cfg.shared_layers = args.shared_layers
if args.value_layers is not None:
    cfg.value_layers = args.value_layers
if args.advantage_layers is not None:
    cfg.advantage_layers = args.advantage_layers
if args.q_layers is not None:
    cfg.q_layers = args.q_layers
cfg.__post_init__()

# ── TF32 정책 적용 (전역 FP32 고정 + GPU 지원 시 forward만 TF32) ──
_tf32_on, _tf32_sup = apply_tf32_config(cfg)
print(f"[TF32] forward TF32 = {'ON' if _tf32_on else 'off'} "
      f"(요청={cfg.use_tf32_forward}, GPU 지원={'yes' if _tf32_sup else 'no'}) | 행렬연산은 FP32 유지")

# =========================================================================
# 2. Network Info & Unified Cache
# =========================================================================
def create_network_info(dimS: int, nA: int, config: Config) -> Dict:
    info = {'dimS': dimS, 'nA': nA, 'layers': [], 'filter_layers': [], 'use_dueling': config.use_dueling,
            'act_fn': _get_act_fn(config.activation_fn), 'act_name': config.activation_fn,
            'use_residual': config.use_residual}
    idx, ld_idx = 0, 0
    def add_layers(sizes, type_str):
        nonlocal idx, ld_idx
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W_len = fan_out * fan_in
            b_len = fan_out
            param_len = W_len + b_len
            
            # [핵심 추상화] Node vs Layer vs FV 블록 크기 분기
            if config.decoupling_mode == 'node':
                block_size = fan_in + 1
                num_blocks = fan_out
            elif config.decoupling_mode == 'layer':
                block_size = param_len
                num_blocks = 1
            else:  # 'fv' - filter_layers는 안 쓰지만 forward용 layers 정보는 필요
                block_size = param_len  # placeholder (FV에선 사용 안 함)
                num_blocks = 1

            layer = {
                'type': type_str, 'layer_idx': i,
                'W_start': idx, 'W_len': W_len, 'W_shape': (fan_out, fan_in),
                'b_start': idx + W_len, 'b_len': b_len,
                'fan_in': fan_in, 'fan_out': fan_out,
            }
            idx += param_len
            info['layers'].append(layer)
            # FV 모드는 filter_layers 안 채움 (FilterCacheFV가 별도 처리)
            if config.decoupling_mode != 'fv':
                info['filter_layers'].append({
                    'global_idx': ld_idx, 'type': type_str, 'local_idx': i,
                    'fan_in': fan_in, 'fan_out': fan_out, 
                    'block_size': block_size, 'num_blocks': num_blocks, 'param_len': param_len,
                    'W_start': layer['W_start'], 'W_len': layer['W_len'],
                    'b_start': layer['b_start'], 'b_len': layer['b_len']})
                ld_idx += 1
            
    shared_out = config.shared_layers[-1] if config.shared_layers else dimS
    add_layers([dimS] + config.shared_layers, 'shared')
    info['shared_end_idx'] = len(info['layers'])
    
    if config.use_dueling:
        add_layers([shared_out] + config.value_layers + [1], 'value')
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + config.advantage_layers + [nA], 'advantage')
    else:
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + config.q_layers + [nA], 'q_layer')
        
    info['total_params'] = idx
    info['num_filter_layers'] = len(info['filter_layers'])
    return info

class FilterCache:
    def __init__(self, info: Dict, cfg: Config, device: str):
        self.layers = {}
        total_forwards = 0
        layer_fwd_slices = []
        
        for L, fl in enumerate(info['filter_layers']):
            block_size = fl['block_size']
            num_blocks = fl['num_blocks']
            num_sigma = 2 * block_size + 1
            count = num_blocks * num_sigma
            layer_fwd_slices.append((total_forwards, total_forwards + count))
            total_forwards += count
            
            lamb = cfg.alpha ** 2 * (block_size + cfg.kappa) - block_size
            gamma = float(np.sqrt(block_size + lamb))
            Wm = torch.zeros(num_sigma, dtype=DTYPE, device=device)
            Wc = torch.zeros(num_sigma, dtype=DTYPE, device=device)
            Wm[0] = lamb / (block_size + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha ** 2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (block_size + lamb)
            
            eye_block = torch.eye(block_size, dtype=DTYPE, device=device)
            eye_block_batch = eye_block.unsqueeze(0).expand(num_blocks, -1, -1).clone()
            S_Q_cached = cfg.q_init * eye_block_batch.clone()
            
            Wm_col_f32 = Wm.to(DTYPE_FWD).view(1, -1, 1).expand(num_blocks, -1, -1).clone()
            Wc_f32 = Wc.to(DTYPE_FWD)
            zero_col_f32 = torch.zeros(num_blocks, block_size, 1, dtype=DTYPE_FWD, device=device)
            
            layer_dict = {
                'eye_block': eye_block, 'eye_block_batch': eye_block_batch,
                'Wm': Wm, 'Wc': Wc, 'gamma': gamma,
                'block_size': block_size, 'num_blocks': num_blocks, 'num_sigma': num_sigma,
                'S_Q_cached': S_Q_cached,
                'Wm_col_f32': Wm_col_f32, 'Wc_f32': Wc_f32, 'zero_col_f32': zero_col_f32,
            }
            
            # Node 모드일 때만 흩뿌리기용 인덱스 필요 (Layer 모드는 그냥 연속 메모리 카피)
            if cfg.decoupling_mode == 'node':
                j_idx = torch.arange(fl['fan_out'], device=device).view(-1, 1, 1)
                k_idx = torch.arange(fl['fan_in'], device=device).view(1, 1, -1)
                layer_dict['w_col_idx'] = (fl['W_start'] + j_idx * fl['fan_in'] + k_idx).expand(fl['fan_out'], num_sigma, fl['fan_in']).contiguous()
                layer_dict['b_col_idx'] = (fl['b_start'] + j_idx.squeeze(-1)).expand(fl['fan_out'], num_sigma).unsqueeze(-1).contiguous()
                
            self.layers[L] = layer_dict
            
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards

        # 연산 최적화를 위해 block_size가 같은 층들끼리 묶기
        self.block_groups = {}
        for L, fl in enumerate(info['filter_layers']):
            bs = fl['block_size']
            if bs not in self.block_groups:
                self.block_groups[bs] = {'layers': [], 'num_blocks_list': [], 'total_blocks': 0}
            grp = self.block_groups[bs]
            grp['layers'].append(L)
            grp['num_blocks_list'].append(fl['num_blocks'])
            grp['total_blocks'] += fl['num_blocks']
        
        for bs, grp in self.block_groups.items():
            total_b = grp['total_blocks']
            grp['eye_grouped'] = torch.eye(bs, dtype=DTYPE, device=device).unsqueeze(0).expand(total_b, -1, -1).clone()
            grp['gamma'] = self.layers[grp['layers'][0]]['gamma']
            offsets = [0]
            for nb in grp['num_blocks_list']:
                offsets.append(offsets[-1] + nb)
            grp['offsets'] = offsets

    def get(self, layer_idx: int) -> Dict:
        return self.layers[layer_idx]

class FilterCacheFV:
    """Full Vector mode용 캐시. 전체 θ ∈ R^n_x를 하나의 블록으로."""
    def __init__(self, info: Dict, cfg: Config, device: str):
        n_x = info['total_params']
        self.n_x = n_x
        self.num_sigma = 2 * n_x + 1
        
        # UKF weights
        lam = cfg.alpha**2 * (n_x + cfg.kappa) - n_x
        self.gamma_sigma = float(np.sqrt(n_x + lam))
        Wm = np.full(self.num_sigma, 0.5 / (n_x + lam))
        Wc = Wm.copy()
        Wm[0] = lam / (n_x + lam)
        Wc[0] = Wm[0] + (1 - cfg.alpha**2 + cfg.beta)
        self.Wm = torch.tensor(Wm, dtype=DTYPE, device=device)  # [num_sigma]
        self.Wc = torch.tensor(Wc, dtype=DTYPE, device=device)
        
        # 자주 쓰는 buffer
        self.eye_n = torch.eye(n_x, dtype=DTYPE, device=device)
        # forward용 tensor (sigma points × n_x)
        self.unified_thetas = torch.empty(self.num_sigma, n_x, dtype=DTYPE_FWD, device=device)

class InputNormalizer:
    def __init__(self, device, scale=None):
        # scale 미지정 시 CartPole 기본값 (하위호환)
        if scale is None:
            scale = [2.4, 3.0, 0.21, 2.0]
        self.scale = torch.tensor(scale, dtype=DTYPE, device=device)
    def normalize(self, x):
        if x.dim() == 1: return x / self.scale
        elif x.shape[-1] == len(self.scale): return x / self.scale
        else: return x / self.scale.view(-1, 1)

# =========================================================================
# 3. Forward Functions & Replay Buffer
# =========================================================================
def _get_act_fn(name: str):
    """활성화 함수 이름 → callable. autograd 호환, float32 forward 호환."""
    if name == 'tanh':       return F.tanh
    elif name == 'relu':     return F.relu
    elif name == 'leaky_relu': return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == 'mish':     return F.mish
    elif name == 'gelu':     return F.gelu
    elif name == 'silu':     return F.silu
    else:
        raise ValueError(f"Unknown activation_fn: {name}")


@tf32_forward
def forward_single(theta, info, x):
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    use_resid = info.get('use_residual', False)
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = info['act_fn'](W @ h + b)
        # Residual: same-dim layers (fan_out == fan_in)에만 skip 추가
        if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
            h = h + z
        else:
            h = z
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ v + b
        is_final = (i == info['value_end_idx'] - 1)
        if is_final:
            v = z_lin  # 출력층은 activation/residual 둘 다 없음
        else:
            z = info['act_fn'](z_lin)
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                v = v + z
            else:
                v = z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ a + b
        is_final = (i == len(info['layers']) - 1)
        if is_final:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                a = a + z
            else:
                a = z

    if info['use_dueling']:
        return (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    else:
        return a.to(DTYPE)

@tf32_forward
def forward_single_with_shared(theta, info, x):
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    use_resid = info.get('use_residual', False)
    
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = info['act_fn'](W @ h + b)
        if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
            h = h + z
        else:
            h = z
    shared_out = h.clone()
    
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ v + b
        is_final = (i == info['value_end_idx'] - 1)
        if is_final:
            v = z_lin
        else:
            z = info['act_fn'](z_lin)
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                v = v + z
            else:
                v = z
        
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ a + b
        is_final = (i == len(info['layers']) - 1)
        if is_final:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                a = a + z
            else:
                a = z

    if info['use_dueling']:
        Q = (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    else:
        Q = a.to(DTYPE)
    return Q, shared_out.to(DTYPE)

@tf32_forward
def forward_bmm(thetas, info, x):
    thetas = thetas.to(DTYPE_FWD); x = x.to(DTYPE_FWD)
    num_sigma = thetas.shape[0]
    use_resid = info.get('use_residual', False)
    x_expanded = x.t().unsqueeze(0).expand(num_sigma, -1, -1)
    h = x_expanded
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = info['act_fn'](torch.bmm(W, h) + b)
        if use_resid and out_dim == in_dim:
            h = h + z
        else:
            h = z
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z_lin = torch.bmm(W, v) + b
        is_final = (i == info['value_end_idx'] - 1)
        if is_final:
            v = z_lin
        else:
            z = info['act_fn'](z_lin)
            if use_resid and out_dim == in_dim:
                v = v + z
            else:
                v = z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z_lin = torch.bmm(W, a) + b
        is_final = (i == len(info['layers']) - 1)
        if is_final:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            if use_resid and out_dim == in_dim:
                a = a + z
            else:
                a = z

    if info['use_dueling']:
        return (v + (a - a.mean(dim=1, keepdim=True))).to(DTYPE)
    else:
        return a.to(DTYPE)

class TensorReplayBuffer:
    """
    GPU 텐서 기반 replay buffer + 옵션 N-step bootstrap 캐시.

    use_n_step=False (or n_step_size=1):
        기존 1-step transition (s_t, a_t, r_t, s_{t+1}, done_{t+1})만 저장.

    use_n_step=True:
        매 push마다 길이 n_step_size deque에 쌓아두고, deque가 꽉 차면
        (s_t, a_t, R_n_t, s_{t+n}, done_{t+n}) 형태로 저장.
        R_n_t = Σ_{i=0..n-1} γ^i · r_{t+i}, 중간에 done 만나면 거기서 컷.
        srrhuif_step_*에서 z_measured = R_n + γ^n · (1-term) · Q_target(s_{t+n}).
    """
    def __init__(self, capacity: int, dimS: int, device: str, cfg: Config):
        self.capacity, self.count, self.device = capacity, 0, device
        self.S = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.ep_id = torch.zeros(capacity, dtype=torch.long, device=device)
        self.current_ep = 0

        # ─── N-step 캐시 ───
        self.use_n_step = cfg.use_n_step
        self.n_step = cfg.n_step_size if self.use_n_step else 1
        self.gamma = cfg.gamma
        self.n_step_cache = deque(maxlen=self.n_step)

        # ─── [v9+] PER 필드 ───
        self.use_per = cfg.use_per
        if self.use_per:
            # priorities[i] = p_i^alpha (이미 alpha 적용한 값을 저장하면 sampling 시 그대로 사용 가능)
            # 여기선 raw priority |TD|+eps 저장, sampling 시에 alpha 적용
            self.priorities = torch.ones(capacity, dtype=DTYPE, device=device)
            self.max_priority = 1.0
            self.per_alpha = cfg.per_alpha
            self.per_eps = cfg.per_eps
            self.per_apply_is_weight = cfg.per_apply_is_weight

    def _get_n_step_info(self):
        """deque에 쌓인 transition들로 (R_n, s_{t+n}, done_{t+n}) 계산."""
        reward = 0.0
        next_state = self.n_step_cache[-1][3]
        done = self.n_step_cache[-1][4]
        for i, transition in enumerate(self.n_step_cache):
            reward += (self.gamma ** i) * transition[2]
            if transition[4]:  # 중간에 에피소드 종료
                next_state, done = transition[3], True
                break
        return reward, next_state, done

    def push(self, s, a, r, s_next, done):
        if not self.use_n_step:
            # 1-step 경로: 즉시 저장
            self._push_tensor(s, a, r, s_next, done)
            return

        # N-step 경로
        self.n_step_cache.append((s, a, r, s_next, done))

        # deque가 꽉 차면 시작 시점 transition을 N-step return으로 저장 후 즉시 popleft
        # (v4 원본은 popleft 없어서 done=True 시 첫 flush iter와 중복 저장됨. 여기서 수정)
        if len(self.n_step_cache) == self.n_step:
            r_n, s_n, d_n = self._get_n_step_info()
            s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
            self._push_tensor(s_0, a_0, r_n, s_n, d_n)
            self.n_step_cache.popleft()

        # 에피소드 종료시 자투리 (길이 < n_step) 도 truncated N-step return으로 flush
        if done:
            while len(self.n_step_cache) > 0:
                r_n, s_n, d_n = self._get_n_step_info()
                s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
                self._push_tensor(s_0, a_0, r_n, s_n, d_n)
                self.n_step_cache.popleft()

    def _push_tensor(self, s, a, r, s_next, done):
        idx = self.count % self.capacity
        self.S[idx] = torch.as_tensor(s, dtype=DTYPE, device=self.device)
        self.A[idx] = a; self.R[idx] = r
        self.S_next[idx] = torch.as_tensor(s_next, dtype=DTYPE, device=self.device)
        self.term[idx] = float(done)
        self.ep_id[idx] = self.current_ep
        # [v9+] PER: 새 transition은 max priority로 초기화 (한 번은 반드시 샘플링되도록)
        if self.use_per:
            self.priorities[idx] = self.max_priority
        self.count += 1

    def set_current_episode(self, ep): self.current_ep = ep
    @property
    def current_size(self): return min(self.count, self.capacity)
    @property
    def is_saturated(self): return self.count >= self.capacity
    @property
    def fill_ratio(self): return self.current_size / self.capacity

    def sample_batch(self, batch_size: int) -> Dict:
        if not self.use_per:
            # ── 균일 샘플링 (기존 동작) ──
            indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
            return {
                's': self.S[indices].t(),
                'a': self.A[indices],
                'r': self.R[indices],
                's_next': self.S_next[indices].t(),
                'term': self.term[indices],
                'indices': indices,
                'is_weights': torch.ones(batch_size, dtype=DTYPE, device=self.device),
            }
        
        # ── [v9+] PER 샘플링 ──
        sz = self.current_size
        # priorities^alpha (alpha는 sampling 시 적용)
        probs_unnorm = self.priorities[:sz] ** self.per_alpha
        probs_sum = probs_unnorm.sum() + 1e-12
        probs = probs_unnorm / probs_sum
        indices = torch.multinomial(probs, batch_size, replacement=True)
        # IS weights: w_i = (N · P(i))^(-β). β=1로 두면 fully off-policy 보정.
        # 여기선 sampling만 PER, IS는 별도 toggle (per_apply_is_weight).
        sampling_prob = probs[indices].clamp(min=1e-12)
        is_weights = (sz * sampling_prob) ** (-1.0)
        is_weights = is_weights / is_weights.max().clamp(min=1e-12)  # normalize → max=1
        return {
            's': self.S[indices].t(),
            'a': self.A[indices],
            'r': self.R[indices],
            's_next': self.S_next[indices].t(),
            'term': self.term[indices],
            'indices': indices,
            'is_weights': is_weights.to(DTYPE),
        }

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """
        [v9+] PER: 필터 horizon 업데이트 종료 후 호출.
        td_errors: per-sample |residual| (또는 |z_measured - z_hat|), shape [batch_sz].
        priorities[indices] = |td| + eps  (alpha는 sampling 시 적용).
        """
        if not self.use_per:
            return
        # td_errors는 마지막 horizon step의 residual을 쓰는 게 가장 informative.
        new_p = (td_errors.detach().abs() + self.per_eps).to(self.priorities.dtype)
        # 안전한 in-place 업데이트 (indices 중복 허용 — 마지막 값으로 덮어씀)
        self.priorities[indices] = new_p
        cur_max = new_p.max().item()
        if cur_max > self.max_priority:
            self.max_priority = cur_max

# =========================================================================
# 4. Math Utilities (Batch QR & Triangular Solvers)
# =========================================================================
# =========================================================================
# 4. Math Utilities (Hybrid Batch QR & Triangular Solvers)
# =========================================================================
def tria_operation_batch(A):
    """
    [ND/LD 맞춤형 하이브리드 분해 엔진]
    - Layer Decoupled (LD): 거대 행렬의 병목 해소를 위해 고속 Cholesky 분해 사용
    - Node Decoupled (ND): 작은 행렬의 수치적 안정성을 위해 기존 QR 분해 사용
    """
    # 🚀 [TURBO MODE] Layer Decoupled일 때는 Cholesky 우선 시도
    if cfg.decoupling_mode == 'layer':
        try:
            # 1. 고속 행렬 곱셈: A * A^T 를 통해 양의 정부호 행렬(PD) Y 생성
            Y = torch.bmm(A, A.transpose(-2, -1))
            
            # 2. 수치적 안정성을 위한 미세 Jitter 추가
            jitter = JITTER_TRIA * torch.eye(Y.shape[-1], dtype=A.dtype, device=A.device)
            Y_safe = Y + jitter.unsqueeze(0)
            
            # 3. 고속 숄레스키 분해 (Lower Triangular 반환)
            s = torch.linalg.cholesky(Y_safe)
            return s
            
        except Exception:
            # 만약 특이 행렬(Singular) 문제로 Cholesky가 실패하면,
            # 당황하지 않고 아래의 안전한 QR 로직으로 폴백(Fallback)합니다.
            FALLBACK_COUNTS['tria_qr'] += 1
            pass

    # 🛡️ [SAFE MODE] Node Decoupled 이거나, LD에서 Cholesky가 실패했을 때의 QR 로직
    _, r = torch.linalg.qr(A.transpose(-2, -1).contiguous())
    s = r.transpose(-2, -1).contiguous()  # r은 Upper, s는 Lower Triangular
    
    # 부호 통일 (대각 성분을 양수로 맞춤)
    d = torch.diagonal(s, dim1=-2, dim2=-1)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    s = s * signs.unsqueeze(-2)
    
    # 대각 성분 Clamping (역행렬 계산 시 NaN 폭발 방지)
    d_positive = torch.diagonal(s, dim1=-2, dim2=-1)
    d_clamped = torch.clamp(d_positive, min=JITTER_TRIA)
    s = s - torch.diag_embed(d_positive) + torch.diag_embed(d_clamped)
    
    return s

def safe_inv_tril_batch(L_batch, eye_batch):
    return torch.linalg.solve_triangular(L_batch + JITTER * eye_batch, eye_batch, upper=False)

def robust_solve_spd_batch(S_tril_batch, y_batch, eye_batch):
    S_safe = S_tril_batch + JITTER * eye_batch
    z = torch.linalg.solve_triangular(S_safe, y_batch, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.transpose(-2, -1).contiguous(), z, upper=True)
    return theta

# =========================================================================
# 5. Diagnostic Utilities
# =========================================================================
@torch.no_grad()
def compute_pseudo_cond_from_S(S_batch):
    try:
        S_vals = torch.linalg.svdvals(S_batch)
        S_vals_clamped = S_vals.clamp(min=1e-8)
        Y_eigs = S_vals_clamped ** 2
        y_max_per_neuron = Y_eigs.max(dim=-1).values
        y_min_per_neuron = Y_eigs.min(dim=-1).values
        cond_per_neuron = y_max_per_neuron / y_min_per_neuron.clamp(min=1e-8)
        p_max_per_neuron = 1.0 / y_min_per_neuron.clamp(min=1e-8)
        return (cond_per_neuron.mean().item(), y_max_per_neuron.max().item(), 
                y_min_per_neuron.min().item(), p_max_per_neuron.max().item())
    except Exception:
        return -1.0, -1.0, -1.0, -1.0

@torch.no_grad()
def compute_full_cond_from_S(S_batch):
    try:
        SST = torch.bmm(S_batch, S_batch.transpose(-2, -1))
        eigvals_Y = torch.linalg.eigvalsh(SST)
        y_max = eigvals_Y[:, -1].clamp(min=1e-8)
        y_min = eigvals_Y[:, 0].clamp(min=1e-8)
        cond = y_max / y_min.clamp(min=1e-8)
        return cond.mean().item(), y_max.max().item()
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_effective_rank(X, tol_ratio=1e-3):
    if X.shape[0] > X.shape[1]: X = X.t()
    try:
        X_centered = X - X.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(X_centered)
        s_max = s.max()
        if s_max < 1e-8: return 0.0, 0.0
        eff_rank = (s > s_max * tol_ratio).sum().item()
        stable_rank = (s ** 2).sum().item() / (s_max ** 2).item()
        return float(eff_rank), float(stable_rank)
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_advantage_null_ratio(theta, info):
    adv_layers = [L for L in info['filter_layers'] if L['type'] in ('advantage', 'q_layer')]
    if not adv_layers: return 0.0, 0.0, 0.0
    a1_layer = adv_layers[-1]
    W_start, W_len = a1_layer['W_start'], a1_layer['W_len']
    b_start, b_len = a1_layer['b_start'], a1_layer['b_len']
    fan_in, fan_out = a1_layer['fan_in'], a1_layer['fan_out']
    
    theta_flat = theta.squeeze()
    W = theta_flat[W_start:W_start + W_len].view(fan_out, fan_in)
    b = theta_flat[b_start:b_start + b_len]
    W_mean = W.mean(dim=0)
    W_dev = W - W_mean.unsqueeze(0)
    
    null_norm = W_mean.norm().item()
    signal_norm = W_dev.norm().item()
    b_mean = b.mean().item()
    b_dev_norm = (b - b.mean()).norm().item()
    
    null_total = (null_norm ** 2 + b_mean ** 2) ** 0.5
    signal_total = (signal_norm ** 2 + b_dev_norm ** 2) ** 0.5
    ratio = null_total / (signal_total + 1e-8)
    return ratio, null_total, signal_total

@torch.no_grad()
def compute_layer_theta_norms(theta, info):
    norms = {}
    theta_flat = theta.squeeze()
    for L, fl in enumerate(info['filter_layers']):
        ltype = fl['type']
        lidx = fl['local_idx']
        label = f"{ltype[0].upper()}{lidx}"
        W_start, W_len = fl['W_start'], fl['W_len']
        b_start, b_len = fl['b_start'], fl['b_len']
        W_norm = theta_flat[W_start:W_start + W_len].norm().item()
        b_norm = theta_flat[b_start:b_start + b_len].norm().item()
        norms[label] = (W_norm ** 2 + b_norm ** 2) ** 0.5
    return norms

@torch.no_grad()
def compute_buffer_diversity(buffer, n_sample=512):
    if buffer.current_size < 32: return None
    n = min(n_sample, buffer.current_size)
    indices = torch.randperm(buffer.current_size, device=buffer.device)[:n]
    states = buffer.S[indices]
    rewards = buffer.R[indices]
    dones = buffer.term[indices]
    ep_ids = buffer.ep_id[indices].float()
    
    state_std = states.std(dim=0).mean().item()
    state_range = (states.max(dim=0).values - states.min(dim=0).values).mean().item()
    done_ratio = dones.mean().item()
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    
    age_min = ep_ids.min().item()
    age_max = ep_ids.max().item()
    age_range_val = age_max - age_min
    age_std = ep_ids.std().item() if n > 1 else 0.0
    
    fill_ratio = buffer.fill_ratio
    is_sat = buffer.is_saturated
    return {
        'state_std': state_std, 'state_range': state_range, 'done_ratio': done_ratio,
        'reward_mean': reward_mean, 'reward_std': reward_std, 'age_min': int(age_min),
        'age_max': int(age_max), 'age_range': age_range_val, 'age_std': age_std,
        'fill_ratio': fill_ratio, 'is_saturated': is_sat,
    }

@torch.no_grad()
def collect_hidden_activations(theta, info, x):
    """모든 hidden 레이어의 (pre-activation, post-activation) 수집. 출력층 제외.
    Returns: list of (label, pre [n_units, B], post [n_units, B])."""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    use_resid = info.get('use_residual', False)
    act_fn = info['act_fn']

    activations = []  # (label, pre, post)
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ h + b
        z = act_fn(z_lin)
        activations.append((f"S{i}", z_lin.to(DTYPE), z.to(DTYPE)))
        if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
            h = h + z
        else:
            h = z
    shared_out = h

    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        is_final = (i == info['value_end_idx'] - 1)
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ v + b
        if is_final:
            v = z_lin
        else:
            z = act_fn(z_lin)
            activations.append((f"V{i - info['shared_end_idx']}", z_lin.to(DTYPE), z.to(DTYPE)))
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                v = v + z
            else:
                v = z

    a_h = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        is_final = (i == len(info['layers']) - 1)
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ a_h + b
        if is_final:
            a_h = z_lin
        else:
            z = act_fn(z_lin)
            head_label = 'A' if info['use_dueling'] else 'Q'
            activations.append((f"{head_label}{i - info['value_end_idx']}", z_lin.to(DTYPE), z.to(DTYPE)))
            if use_resid and layer['W_shape'][0] == layer['W_shape'][1]:
                a_h = a_h + z
            else:
                a_h = z

    return activations


@torch.no_grad()
def compute_activation_health(theta, info, x, act_name, sat_thresh=0.95, dead_thresh=0.05):
    """Hidden 레이어 포화/죽은 뉴런 진단.

    정의 (per unit, batch averaged):
      tanh:        sat  = mean|tanh(z)| > sat_thresh   (gradient ~ 0)
                   dead = max|tanh(z)| < dead_thresh   (거의 0 출력)
      relu:        dead = pre_act <= 0 for entire batch (firing_rate == 0)
                   sat  = N/A (unbounded)
      leaky_relu:  dead = firing_rate < 1e-6 (음수 영역 고정, slope=0.01)
                   sat  = N/A
      gelu/mish/silu: dead = max|post| < dead_thresh (음수 saturation 근처)
                   sat  = N/A (unbounded above)

    Returns: dict { layer_label: {n_units, n_sat, n_dead, sat_ratio, dead_ratio,
                                  mean_abs, max_abs, fire_rate, pre_mean, pre_std},
                    '__total__': aggregated counts }
    """
    activations = collect_hidden_activations(theta, info, x)
    stats = {}
    total_units, total_sat, total_dead = 0, 0, 0

    for label, pre, post in activations:
        n_units, B = post.shape
        total_units += n_units

        abs_post = post.abs()                          # [n_units, B]
        unit_mean_abs = abs_post.mean(dim=1)
        unit_max_abs = abs_post.max(dim=1).values
        fire_rate = (post.abs() > 1e-6).float().mean(dim=1)  # 활성 비율
        pre_mean = pre.mean(dim=1)
        pre_std = pre.std(dim=1) if B > 1 else torch.zeros_like(pre_mean)

        if act_name == 'tanh':
            sat_mask = unit_mean_abs > sat_thresh
            dead_mask = unit_max_abs < dead_thresh
        elif act_name == 'relu':
            firing_pos = (post > 0).float().mean(dim=1)
            dead_mask = firing_pos < 1e-6
            sat_mask = torch.zeros_like(dead_mask, dtype=torch.bool)
        elif act_name == 'leaky_relu':
            firing_pos = (post > 0).float().mean(dim=1)
            dead_mask = firing_pos < 1e-6
            sat_mask = torch.zeros_like(dead_mask, dtype=torch.bool)
        elif act_name in ('gelu', 'mish', 'silu'):
            dead_mask = unit_max_abs < dead_thresh
            sat_mask = torch.zeros_like(dead_mask, dtype=torch.bool)
        else:
            dead_mask = torch.zeros(n_units, dtype=torch.bool, device=post.device)
            sat_mask = torch.zeros_like(dead_mask)

        n_sat = int(sat_mask.sum().item())
        n_dead = int(dead_mask.sum().item())
        total_sat += n_sat
        total_dead += n_dead

        stats[label] = {
            'n_units': n_units,
            'n_sat': n_sat,
            'n_dead': n_dead,
            'sat_ratio': n_sat / n_units,
            'dead_ratio': n_dead / n_units,
            'mean_abs': float(unit_mean_abs.mean().item()),
            'max_abs': float(unit_max_abs.max().item()),
            'fire_rate': float(fire_rate.mean().item()),
            'pre_mean': float(pre_mean.mean().item()),
            'pre_std': float(pre_std.mean().item()),
        }

    stats['__total__'] = {
        'n_units': total_units,
        'n_sat': total_sat,
        'n_dead': total_dead,
        'sat_ratio': total_sat / max(total_units, 1),
        'dead_ratio': total_dead / max(total_units, 1),
    }
    return stats


@torch.no_grad()
def act_deriv(name, x):
    """활성화 함수의 analytic 도함수 f'(z). 진단 프로브 전용 (필터엔 backprop 금지).
    x: pre-activation tensor."""
    if name == 'silu':
        s = torch.sigmoid(x)
        return s * (1 + x * (1 - s))
    if name == 'mish':
        sp = F.softplus(x)
        t = torch.tanh(sp)
        return t + x * torch.sigmoid(x) * (1 - t * t)
    if name == 'gelu':
        Phi = 0.5 * (1 + torch.erf(x / 2 ** 0.5))
        phi = torch.exp(-x * x / 2) / (2 * math.pi) ** 0.5
        return Phi + x * phi
    if name == 'tanh':
        return 1 - torch.tanh(x) ** 2
    if name == 'relu':
        return (x > 0).to(x.dtype)
    if name == 'leaky_relu':
        return torch.where(x > 0, torch.ones_like(x), 0.01 * torch.ones_like(x))
    return torch.ones_like(x)  # unknown → gain 1 가정 (linear)


@torch.no_grad()
def compute_act_regime(theta, info, x, act_name):
    """[Per-h probe] horizon 내부 fold마다 활성화 regime + effective-gain 측정.

    pre-activation z 기준:
      frac_pos  = mean(z > 0)    — unbounded-gain 영역 점유율 (SiLU runaway 1순위)
      frac_hi   = mean(z > 2.0)  — silu'>1 실제 증폭 구간 점유율
      mean_gain = mean(f'(z))    — fold당 유효 게인 (analytic 도함수)
      mean_z, max_abs_z

    Returns: dict { layer_label: {frac_pos, frac_hi, mean_gain, mean_z, max_abs_z},
                    '__total__': 전체 hidden pre-activation 집계 }
    """
    activations = collect_hidden_activations(theta, info, x)
    stats = {}
    all_pre = []
    for label, pre, post in activations:
        d = act_deriv(act_name, pre)
        stats[label] = {
            'frac_pos': float((pre > 0).float().mean().item()),
            'frac_hi': float((pre > 2.0).float().mean().item()),
            'mean_gain': float(d.mean().item()),
            'mean_z': float(pre.mean().item()),
            'max_abs_z': float(pre.abs().max().item()),
        }
        all_pre.append(pre.reshape(-1))

    if all_pre:
        cat = torch.cat(all_pre)
        d_all = act_deriv(act_name, cat)
        stats['__total__'] = {
            'frac_pos': float((cat > 0).float().mean().item()),
            'frac_hi': float((cat > 2.0).float().mean().item()),
            'mean_gain': float(d_all.mean().item()),
            'mean_z': float(cat.mean().item()),
            'max_abs_z': float(cat.abs().max().item()),
        }
    else:
        stats['__total__'] = {'frac_pos': 0.0, 'frac_hi': 0.0,
                              'mean_gain': 0.0, 'mean_z': 0.0, 'max_abs_z': 0.0}
    return stats


def _fv_layer_label(layer):
    """info['layers'] 항목 → 진단 라벨 (S0/V0/A0/Q0...). collect_hidden_activations와 동일 규칙."""
    return f"{layer['type'][0].upper()}{layer['layer_idx']}"


@torch.no_grad()
def fv_per_layer(info, vec, reduce='norm'):
    """[FV diag] 전체 파라미터 축(dim 0 = n_x) 양을 네트워크 레이어 구간으로 잘라 per-layer dict 반환.
       각 레이어 파라미터는 [W_start, b_start+b_len) 연속 구간.
       vec: [n_x] 또는 [n_x, m] (Δθ·Kalman gain은 [n_x], P_xz/H^T는 [n_x, m]).
       reduce='norm' → 구간 행들의 L2(Frobenius) norm, 'maxabs' → max(|.|), 'mean' → 평균."""
    out = {}
    for layer in info['layers']:
        s = layer['W_start']
        e = layer['b_start'] + layer['b_len']
        seg = vec[s:e]
        if reduce == 'maxabs':
            out[_fv_layer_label(layer)] = seg.abs().max().item()
        elif reduce == 'mean':
            out[_fv_layer_label(layer)] = seg.mean().item()
        else:
            out[_fv_layer_label(layer)] = torch.norm(seg).item()
    return out


def fv_broadcast(info, value):
    """[FV diag] 측정-공간 전역 스칼라(residual/cond 등)를 모든 레이어 라벨에 동일 값으로 복제.
       per-layer 컬럼 정렬(Tier-1/DIAGNOSTICS 블록 key 일치)을 위해 사용 — 레이어별로 의미가 갈리지 않는 양."""
    return {_fv_layer_label(layer): value for layer in info['layers']}


@torch.no_grad()
def compute_sigma_spread(unified_sigma, info, x, eps=1e-6):
    """[Sigma-spread probe / FV] 시그마 클라우드(unified_sigma: [num_sigma, n_x])를 forward_bmm과
    동일하게 전파하면서 hidden 레이어별로 '시그마 축(dim 0) 통계'를 측정.
      spread = mean( std_σ(z) )                  pre-activation이 시그마로 퍼진 정도
      amp    = mean(std_σ(f(z))) / spread        활성화가 그 spread에 가한 유효 게인
                                                 (>1 증폭=runaway, <1 수축=안정)
      frac_pos/frac_hi = 클라우드 전체 중 z>0 / z>2 비율 (증폭 구간 점유율)
      z_max  = max|z|
    forward_bmm과 동일한 residual/propagation 규칙을 그대로 따라 깊은 층 클라우드가 실제와 일치.
    Returns: {layer_label: {spread, amp, frac_pos, frac_hi, z_max}, '__total__': {...}}.
    """
    thetas = unified_sigma.to(DTYPE_FWD)
    x = x.to(DTYPE_FWD)
    num_sigma = thetas.shape[0]
    use_resid = info.get('use_residual', False)
    act_fn = info['act_fn']
    x_expanded = x.t().unsqueeze(0).expand(num_sigma, -1, -1)

    stats = {}

    def _record(layer, z_lin):
        z_std = z_lin.std(dim=0)                    # [out, B] 시그마 축 spread
        post_std = act_fn(z_lin).std(dim=0)
        spread = z_std.mean()
        stats[_fv_layer_label(layer)] = {
            'spread': spread.item(),
            'amp': (post_std.mean() / spread.clamp(min=eps)).item(),
            'frac_pos': (z_lin > 0).float().mean().item(),
            'frac_hi': (z_lin > 2.0).float().mean().item(),
            'z_max': z_lin.abs().max().item(),
        }

    def _W(layer):
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        return W, b, out_dim, in_dim

    h = x_expanded
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W, b, out_dim, in_dim = _W(layer)
        z_lin = torch.bmm(W, h) + b
        _record(layer, z_lin)
        z = act_fn(z_lin)
        h = h + z if (use_resid and out_dim == in_dim) else z
    shared_out = h

    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W, b, out_dim, in_dim = _W(layer)
        z_lin = torch.bmm(W, v) + b
        if i == info['value_end_idx'] - 1:
            v = z_lin  # 출력층 (activation 없음) — 진단 제외
        else:
            _record(layer, z_lin)
            z = act_fn(z_lin)
            v = v + z if (use_resid and out_dim == in_dim) else z

    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W, b, out_dim, in_dim = _W(layer)
        z_lin = torch.bmm(W, a) + b
        if i == len(info['layers']) - 1:
            a = z_lin  # 출력층 — 진단 제외
        else:
            _record(layer, z_lin)
            z = act_fn(z_lin)
            a = a + z if (use_resid and out_dim == in_dim) else z

    if stats:
        # 총계: 레이어 평균 (레이어마다 스케일이 달라 raw concat 대신 레이어 평균)
        stats['__total__'] = {
            'spread': float(np.mean([s['spread'] for s in stats.values()])),
            'amp': float(np.mean([s['amp'] for s in stats.values()])),
            'frac_pos': float(np.mean([s['frac_pos'] for s in stats.values()])),
            'frac_hi': float(np.mean([s['frac_hi'] for s in stats.values()])),
            'z_max': float(np.max([s['z_max'] for s in stats.values()])),
        }
    else:
        stats['__total__'] = {'spread': 0.0, 'amp': 0.0, 'frac_pos': 0.0, 'frac_hi': 0.0, 'z_max': 0.0}
    return stats


# =========================================================================
# 5b. Log Analysis Layer — "이 시점의 핵심 원인" 자동 진단
#   이미 계산된 last_h_* / 에피소드 스칼라를 룰로 해석. 추가 연산 거의 없음.
# =========================================================================
_PREV_FB = {'chol_1e5': 0, 'tria_qr': 0}  # FB 누적 카운터의 직전 스냅샷 (interval delta용)


def _traj_trend(traj):
    """궤적 → (방향기호, 배율). 앞 1/3 평균 대비 뒤 1/3 평균. 증가=위험, 감소=수축(건강)."""
    if traj is None or len(traj) < 2:
        return ('·', 1.0)
    n = len(traj)
    k = max(1, n // 3)
    a = float(np.mean(traj[:k]))
    b = float(np.mean(traj[-k:]))
    ratio = b / (abs(a) + 1e-8)
    if ratio >= 1.5:   sym = '↑↑'
    elif ratio >= 1.1: sym = '↑'
    elif ratio <= 0.67: sym = '↓↓'
    elif ratio <= 0.9: sym = '↓'
    else:              sym = '→'
    return (sym, ratio)


def _peak_per_layer(layer_dicts):
    """[{label:val}, …] (per-h 리스트) → {label: fold 최댓값}."""
    if not layer_dicts:
        return {}
    return {l: max(d[l] for d in layer_dicts) for l in layer_dicts[0].keys()}


def rank_culprit_layers(amp_h, ht_h, delta_h, cond_h):
    """per-h×per-layer dict 리스트들 → badness 점수로 레이어 랭킹.
    각 지표를 레이어별 fold-최댓값으로 환산 후 레이어축 max로 정규화해 합산.
    Returns: [(label, score, {amp,ht,delta,cond}), …] 내림차순."""
    sources = {'amp': _peak_per_layer(amp_h), 'ht': _peak_per_layer(ht_h),
               'delta': _peak_per_layer(delta_h), 'cond': _peak_per_layer(cond_h)}
    labels = set()
    for d in sources.values():
        labels |= set(d.keys())
    if not labels:
        return []
    maxes = {k: (max(d.values()) if d else 0.0) for k, d in sources.items()}
    scores = []
    for l in labels:
        s = sum((sources[k].get(l, 0.0) / maxes[k]) for k in sources if maxes[k] > 0)
        detail = {k: sources[k].get(l, 0.0) for k in sources}
        scores.append((l, s, detail))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def build_log_diagnosis(data, cfg):
    """이미 계산된 진단 자료(data dict)를 룰로 해석.
    Returns: (verdicts: list[str] severity 내림차순, culprit: str|None, trend: str)."""
    verdicts = []  # (severity, text)

    # 1) SIGMA_RUNAWAY: 어떤 레이어 peak amp > thresh 이고 amp 집계가 fold 따라 증가
    amp_h = data.get('amp_layer_h') or []
    amp_tot = [float(np.mean(list(d.values()))) for d in amp_h] if amp_h else []
    if amp_h:
        peak = _peak_per_layer(amp_h)
        top_l = max(peak, key=peak.get)
        sym, _ = _traj_trend(amp_tot)
        if peak[top_l] > cfg.collapse_amp_thresh and sym in ('↑', '↑↑'):
            verdicts.append((peak[top_l] - cfg.collapse_amp_thresh,
                             f"SIGMA_RUNAWAY({top_l}) amp {amp_tot[0]:.2f}→{amp_tot[-1]:.2f}"))

    # 2) GAIN_RUNAWAY: act-regime mean f' 가 1 초과 + 증가
    gain_h = data.get('gain_h') or []
    if gain_h:
        sym, _ = _traj_trend(gain_h)
        if gain_h[-1] > 1.0 and sym in ('↑', '↑↑'):
            verdicts.append((gain_h[-1] - 1.0, f"GAIN_RUNAWAY g {gain_h[0]:.2f}→{gain_h[-1]:.2f}"))

    # 3) COV_ILLCOND: cond 큼 / P_avg 급증 / FB 발동
    cond_h = data.get('cond_layer_h') or []
    cond_max = max((max(d.values()) for d in cond_h), default=0.0)
    p_sym, _ = _traj_trend(data.get('p_traj') or [])
    fb_delta = data.get('fb_delta', 0)
    if cond_max > cfg.cond_warn or p_sym == '↑↑' or fb_delta > 0:
        sev = 0.0
        if cond_max > cfg.cond_warn: sev += math.log10(cond_max / cfg.cond_warn)
        if fb_delta > 0: sev += 0.5 * fb_delta
        if p_sym == '↑↑': sev += 0.5
        verdicts.append((sev, f"COV_ILLCOND cond {cond_max:.0e} FB+{fb_delta} P {p_sym}"))

    # 4) PLASTICITY_LOSS: dead 비율 / eff_rank 저하
    dead = data.get('dead_ratio')
    eff_rank = data.get('eff_rank')
    eff_ref = data.get('eff_rank_ref')
    if (dead is not None and dead > cfg.dead_warn) or \
       (eff_rank is not None and eff_rank > 0 and eff_ref and eff_rank < eff_ref):
        rk = f"{eff_rank:.0f}" if (eff_rank and eff_rank > 0) else "?"
        verdicts.append(((dead or 0.0), f"PLASTICITY_LOSS dead {100*(dead or 0):.0f}% rank {rk}"))

    # 5) POLICY_THRASH: argmax flip
    flip = data.get('argmax_flip', 0.0)
    if flip > cfg.flip_warn:
        verdicts.append((flip, f"POLICY_THRASH flip {flip:.2f}"))

    # 6) PRIOR_DOMINATED: |H^Tθ| >> |z-ẑ|
    decomp = data.get('innov_decomp') or []
    if decomp:
        resid_m = float(np.mean([d[0] for d in decomp]))
        httheta_m = float(np.mean([d[1] for d in decomp]))
        if resid_m > 1e-9 and httheta_m / resid_m > cfg.prior_ratio_warn:
            verdicts.append((httheta_m / resid_m,
                             f"PRIOR_DOMINATED H^Tθ/resid {httheta_m / resid_m:.1f}"))

    # 7) INNOV_BLOWUP: innovation max 폭발
    max_innov = data.get('max_innov')
    if max_innov is not None and max_innov > cfg.innov_warn:
        verdicts.append((max_innov / cfg.innov_warn, f"INNOV_BLOWUP max {max_innov:.1f}"))

    verdicts.sort(key=lambda x: x[0], reverse=True)
    verdict_strs = [t for _, t in verdicts]

    # 범인 레이어 랭킹
    ranking = rank_culprit_layers(amp_h, data.get('ht_layer_h') or [],
                                  data.get('delta_layer_h') or [], cond_h)
    culprit = None
    if ranking:
        l, _, d = ranking[0]
        culprit = (f"{l} (amp{d['amp']:.2f}, ht{d['ht']:.2f}, "
                   f"Δθ{d['delta']:.3f}, cond{d['cond']:.0e})")

    # trend 시그니처
    def _t(name, traj):
        sym, r = _traj_trend(traj)
        return f"{name} {sym}(×{r:.1f})" if sym not in ('·', '→') else f"{name} {sym}"
    trend = "  ".join([_t('gain', gain_h), _t('amp', amp_tot),
                       _t('P_avg', data.get('p_traj') or []),
                       _t('K', data.get('k_traj') or [])])
    return verdict_strs, culprit, trend


REF_STATES = torch.tensor([
    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05, 0.0], [0.0, 0.0, -0.05, 0.0],
    [0.0, 0.0, 0.1, 0.5], [0.0, 0.0, -0.1, -0.5],
], dtype=DTYPE)
REF_NAMES = ["balance", "tilt_R", "tilt_L", "fall_R", "fall_L"]

@torch.no_grad()
def compute_ref_q_values(theta, info, normalizer, device):
    # REF_STATES는 CartPole 전용 (obs 4-dim, binary action: dq=q1-q0). 다른 env면 건너뜀.
    if info['dimS'] != REF_STATES.shape[1] or info['nA'] < 2:
        if not getattr(compute_ref_q_values, '_warned', False):
            print(f"[경고] diag_ref_states: REF_STATES는 CartPole(obs=4, nA=2) 전용 — "
                  f"현재 env(obs={info['dimS']}, nA={info['nA']})와 불일치하여 ref-state 진단을 건너뜁니다. "
                  f"(cfg.diag_ref_states=False 로 끄는 것을 권장)")
            compute_ref_q_values._warned = True
        return None
    ref = REF_STATES.to(device)
    ref_norm = normalizer.normalize(ref) if normalizer else ref
    Q = forward_single(theta.squeeze(), info, ref_norm.t())
    results = {}
    for i, name in enumerate(REF_NAMES):
        q0 = Q[0, i].item()
        q1 = Q[1, i].item()
        dq = q1 - q0
        argmax = 1 if dq > 0 else 0
        results[name] = {'q0': q0, 'q1': q1, 'dq': dq, 'argmax': argmax}
    return results

# =========================================================================
# 6. SRRHUIF Core Functions
# =========================================================================
def _time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    combined = torch.cat([P_sqrt_prev, S_Q_cached], dim=2)
    P_sqrt_pred = tria_operation_batch(combined)
    S_pred = safe_inv_tril_batch(P_sqrt_pred, eye_batch)
    
    temp_y = torch.bmm(S_pred.transpose(-2, -1), theta_3d)
    y_pred = torch.bmm(S_pred, temp_y)
    
    scaled_P = gamma_val * P_sqrt_pred
    theta_2d = theta_3d.squeeze(-1)
    X_sigma_all = torch.cat([
        theta_2d.unsqueeze(1),
        theta_2d.unsqueeze(1) + scaled_P.transpose(-2, -1),
        theta_2d.unsqueeze(1) - scaled_P.transpose(-2, -1),
    ], dim=1)
    
    return S_pred, None, y_pred, X_sigma_all, scaled_P

def _compute_ht_core(Z_sigma_T_fwd, Wm_col_fwd, Wc_fwd, zero_col_fwd,
                         scaled_P_fwd, z_measured_exp, S_pred): 
    Z_sigma_T_fwd = Z_sigma_T_fwd.to(DTYPE_FWD)
    Wm_col_fwd = Wm_col_fwd.to(DTYPE_FWD)
    
    z_hat_fwd = torch.bmm(Z_sigma_T_fwd, Wm_col_fwd)
    Z_dev_fwd = Z_sigma_T_fwd - z_hat_fwd
    X_dev_fwd = torch.cat([zero_col_fwd, scaled_P_fwd, -scaled_P_fwd], dim=2)
    P_xz_fwd = torch.bmm(X_dev_fwd * Wc_fwd.view(1, 1, -1), Z_dev_fwd.transpose(1, 2))
    
    z_hat = z_hat_fwd.to(DTYPE)
    residual_all = z_measured_exp.to(DTYPE) - z_hat
    P_xz = P_xz_fwd.to(DTYPE)
    S_pred = S_pred.to(DTYPE)
    
    temp_ht = torch.bmm(S_pred.transpose(-2, -1), P_xz)
    HT_all = torch.bmm(S_pred, temp_ht)
    
    ht_norm = torch.norm(HT_all, dim=1).mean().item()
    resid_norm = torch.norm(residual_all, dim=1).mean().item()
    
    return HT_all, residual_all, z_hat, ht_norm, resid_norm

def _meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all, 
                         r_inv_sqrt, r_inv, eye_batch, 
                         tikhonov_lambda=0.1, huber_c=2.0):
    res_abs = torch.abs(residual_all)
    adapt_factor = torch.clamp(res_abs / huber_c, min=1.0)
    
    r_inv_adapt = r_inv / adapt_factor
    r_inv_sqrt_adapt_for_HT = (r_inv_sqrt / torch.sqrt(adapt_factor)).transpose(1, 2)
    tikhonov_sqrt = float(np.sqrt(tikhonov_lambda))
    
    if tikhonov_lambda > 0:
        combined = torch.cat([S_pred, HT_all * r_inv_sqrt_adapt_for_HT, tikhonov_sqrt * eye_batch], dim=2)
    else:
        combined = torch.cat([S_pred, HT_all * r_inv_sqrt_adapt_for_HT], dim=2)

    S_new_all = tria_operation_batch(combined)
    
    ht_theta = torch.bmm(HT_all.transpose(1, 2), theta_3d)
    innov = residual_all + ht_theta
    y_new_all = y_pred + torch.bmm(HT_all, r_inv_adapt * innov)
    
    innov_abs = torch.abs(innov)
    innov_mean, innov_max = torch.mean(innov_abs).item(), torch.max(innov_abs).item()
    
    delta_y = torch.bmm(HT_all, r_inv_adapt * innov)
    delta_y_norm = torch.norm(delta_y, dim=1).mean()
    y_pred_norm = torch.norm(y_pred, dim=1).mean().item()
    y_new_norm = torch.norm(y_new_all, dim=1).mean()

    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    S_diag = torch.diagonal(S_new_all, dim1=-2, dim2=-1)
    avg_P_new = (1.0 / (S_diag ** 2 + 1e-8)).mean().item()
    
    meas_stats = {
        'innov_mean': innov_mean, 'innov_max': innov_max,
        'resid_in_innov': torch.mean(torch.abs(residual_all)).item(),
        'ht_theta_in_innov': torch.mean(torch.abs(ht_theta)).item(),
        'innov_norm': innov_mean,
        'delta_y': delta_y_norm.item(),
        'y_pred_norm': y_pred_norm,
        'y_new_norm': y_new_norm.item(),
        'avg_P': avg_P_new, 'adapt_ratio': torch.mean(adapt_factor).item()
    }
    return theta_new_all, S_new_all, meas_stats

# =========================================================================
# 7. Initialize theta (Orthogonal / He, config-selectable)
# =========================================================================
def initialize_theta(info, device, cfg):
    """
    Initialize θ vector based on cfg.init_scheme.

    'orthogonal': hidden gain=√2, final layer gain=0.1
    'he':         randn * √(2/fan_in), bias=0
    'xavier':     Xavier uniform, tanh gain=5/3, final layer gain=0.1
                  std = gain * sqrt(2/(fan_in+fan_out))  tanh 포화 방지에 최적
    """
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    TANH_GAIN = 5.0 / 3.0

    for layer in info['layers']:
        fan_in, fan_out = layer['W_shape'][1], layer['W_shape'][0]
        W_len = layer['W_len']
        l_type, l_idx = layer['type'], layer['layer_idx']
        is_final = (
            (l_type == 'value' and l_idx == len(cfg.value_layers)) or
            (l_type == 'advantage' and l_idx == len(cfg.advantage_layers)) or
            (l_type == 'q_layer' and l_idx == len(cfg.q_layers))
        )

        if cfg.init_scheme == 'orthogonal':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            gain = 0.1 if is_final else float(np.sqrt(2.0))
            torch.nn.init.orthogonal_(W_temp, gain=gain)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        elif cfg.init_scheme == 'xavier':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            gain = 0.1 if is_final else TANH_GAIN
            torch.nn.init.xavier_uniform_(W_temp, gain=gain)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        else:  # 'he'
            theta[layer['W_start']:layer['W_start'] + W_len] = \
                torch.randn(W_len, dtype=DTYPE, device=device) * float(np.sqrt(2.0 / fan_in))
        # bias = 0
    return theta


def analyze_initial_network(theta, info, env, cfg, normalizer=None, num_samples=100):
    print("\n" + "="*50)
    print(f" 🔍 [초기화 진단] Seed {cfg.seed} 네트워크 해부 리포트")
    print("="*50)
    
    theta_flat = theta.squeeze()
    print(" [1] 레이어별 가중치 분포 (Variance & Scale)")
    for L, fl in enumerate(info['filter_layers']):
        w_start, w_len = fl['W_start'], fl['W_len']
        W = theta_flat[w_start:w_start + w_len]
        
        w_std = W.std().item()
        w_mean = W.mean().item()
        w_max = W.max().item()
        w_min = W.min().item()
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        print(f"  ├─ {label:2s} Layer: Std = {w_std:.4f} | Mean = {w_mean:+.4f} | Range = [{w_min:+.3f}, {w_max:+.3f}]")

    print("\n [2] 초기 Q-Value 신호 대 잡음비 (Dueling Stream)")
    states = []
    for _ in range(num_samples):
        s, _ = env.reset()
        states.append(s)
    states_t = torch.tensor(np.array(states), dtype=DTYPE, device=cfg.device)
    if normalizer: states_t = normalizer.normalize(states_t)
        
    with torch.no_grad():
        Q_initial = forward_single(theta_flat, info, states_t.t()) 
        q0 = Q_initial[0, :]
        q1 = Q_initial[1, :]
        adv_diff = torch.abs(q0 - q1)
        print(f"  ├─ Q(a0) 평균: {q0.mean().item():.4f} (std: {q0.std().item():.4f})")
        print(f"  ├─ Q(a1) 평균: {q1.mean().item():.4f} (std: {q1.std().item():.4f})")
        print(f"  ├─ |Q(a0) - Q(a1)| 평균 차이: {adv_diff.mean().item():.4f} (이 값이 0에 가까우면 UKF 뇌사)")
        print(f"  └─ 초기 행동 쏠림 현상 (a0 선택 비율): {(q0 > q1).float().mean().item() * 100:.1f}%")
    print("="*50 + "\n")

# =========================================================================
# 7b. Background video recording (RecordVideo, headless rgb_array)
# =========================================================================
import threading

_VIDEO_THREADS: List[threading.Thread] = []

def _greedy_record_rollout(theta_cpu, info, env_name, max_steps, obs_scale,
                           video_folder, episode_idx, env_seed, env_kwargs=None):
    """현재 θ(=theta_cpu, CPU 사본)로 greedy(ε=0) rollout 1 에피소드를 rgb_array로
    렌더링하여 mp4 1개 저장. 학습 env와 완전히 분리된 독립 env에서 CPU forward로 동작."""
    try:
        from gymnasium.wrappers import RecordVideo
        rec_env = gym.make(env_name, render_mode="rgb_array", **(env_kwargs or {}))
        # 이 env는 정확히 1 에피소드만 돌리므로 episode_trigger는 항상 True.
        rec_env = RecordVideo(
            rec_env, video_folder=video_folder,
            episode_trigger=lambda e: True,
            name_prefix=f"ep{episode_idx:05d}",
            disable_logger=True,
        )
        scale = torch.tensor(obs_scale, dtype=DTYPE) if obs_scale else None
        seed = (env_seed + episode_idx) if env_seed is not None else None
        s, _ = rec_env.reset(seed=seed)
        done, steps, total_r = False, 0, 0.0
        with torch.no_grad():
            while not done and steps < max_steps:
                x = torch.as_tensor(s, dtype=DTYPE)
                if scale is not None:
                    x = x / scale
                q = forward_single(theta_cpu, info, x).squeeze()
                a = int(q.argmax().item())
                s, r, term, trunc, _ = rec_env.step(a)
                total_r += float(r); steps += 1
                done = term or trunc
        rec_env.close()
        print(f"  🎥 [video] ep{episode_idx:05d} 저장 완료 | reward={total_r:.1f} steps={steps} → {video_folder}")
    except Exception as e:
        print(f"  ⚠️ [video] ep{episode_idx:05d} 녹화 실패: {type(e).__name__}: {e}")


def maybe_record_video(theta, info, cfg, episode_idx):
    """cfg.record_video=True 이고 episode_idx가 video_interval 배수면 녹화 트리거.
    theta는 CPU로 detach-clone하여 스냅샷(학습이 이어서 θ를 갱신해도 안전)."""
    if not cfg.record_video:
        return
    if cfg.video_interval <= 0 or (episode_idx % cfg.video_interval != 0):
        return
    video_folder = cfg.video_dir or os.path.join(cfg.outdir, "videos")
    os.makedirs(video_folder, exist_ok=True)
    theta_cpu = theta.squeeze().detach().to('cpu').clone()
    env_seed = cfg.env_seed if cfg.env_seed is not None else cfg.seed
    # 학습 정책과 동일하게 정규화 적용 (use_input_norm=False면 스케일 미적용)
    obs_scale = cfg.obs_scale if cfg.use_input_norm else None
    args = (theta_cpu, info, cfg.env_name, cfg.max_steps, obs_scale,
            video_folder, episode_idx, env_seed, build_env_kwargs(cfg))
    if cfg.video_async:
        th = threading.Thread(target=_greedy_record_rollout, args=args, daemon=True)
        th.start()
        # 끝난 스레드는 정리하고 진행 중인 것만 추적
        _VIDEO_THREADS[:] = [t for t in _VIDEO_THREADS if t.is_alive()]
        _VIDEO_THREADS.append(th)
    else:
        _greedy_record_rollout(*args)


def finalize_videos(timeout=120):
    """학습 종료 시 진행 중인 async 녹화 스레드가 인코딩을 마치도록 대기."""
    pending = [t for t in _VIDEO_THREADS if t.is_alive()]
    if pending:
        print(f"  🎥 [video] 남은 녹화 {len(pending)}개 인코딩 대기 중...")
    for t in pending:
        t.join(timeout=timeout)

# =========================================================================
# 8. Main SRRHUIF Step (Unified Node/Layer Decoupling)
# =========================================================================
@torch.no_grad()
def srrhuif_step(theta_current_in, theta_target, filter_S_info, batch, sp,
                     is_first, p_init_val, f_cache):
    """
    Node/Layer Decoupled SRRHUIF step.

    [v5 변경] q_next_target_cached 제거. 매 horizon step 내부에서 새로 계산.
        - h=0:
            * use_spas=True  → sigma points의 mean Q로 argmax
            * use_spas=False → theta_target Q로 argmax (standard DDQN)
        - h≥1:
            * theta_current_in으로 argmax (직전 horizon에서 갱신된 θ)
        - value Q: 항상 theta_target

    [v5 N-step] z_measured = batch['r'] + γ^n_step · (1-term) · q_val_next
                (cfg.use_n_step=True 일 때 γ^n_step 사용, 아니면 γ)
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    
    # ──────────────────────────────────────────────────────────────────
    # [Prior 결정] h=0에서 cfg.h0_prior_source에 따라 분기
    #   'target' = target net (기존 동작)
    #   'init'   = 학습 시작시 frozen된 θ_init (FIR 정신)
    # h≥1: 항상 직전 추정치 (theta_current_in) 사용
    # ──────────────────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_prior = sp['theta_init'].clone()
        else:  # 'target'
            theta_prior = theta_target.clone()
    else:
        theta_prior = theta_current_in.clone()
    
    theta_current = theta_current_in.clone()
    new_S_info_dict = {}
    total_loss, layer_count = 0.0, 0

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    unified = f_cache.unified_thetas
    # ──────────────────────────────────────────────────────────────────
    # [v7+] OTHER 레이어들의 θ source는 config로 결정:
    #   'prior'   = theta_prior로 base (모든 블록이 동일 reference에서 sigma 평가)
    #   'current' = theta_current로 base (Gauss-Seidel, 직전 추정치 사용)
    # ──────────────────────────────────────────────────────────────────
    if cfg.node_layer_other_source == 'prior':
        unified[:] = theta_prior.squeeze().to(DTYPE_FWD)
    else:  # 'current'
        unified[:] = theta_current.squeeze().to(DTYPE_FWD)

    per_layer = {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        lc = f_cache.get(L)
        
        # [핵심 매핑] Mode에 따라 Prior를 추출
        if cfg.decoupling_mode == 'node':
            W_prior = theta_prior.squeeze()[fl['W_start']:fl['W_start'] + fl['W_len']].view(fl['fan_out'], fl['fan_in'])
            b_prior = theta_prior.squeeze()[fl['b_start']:fl['b_start'] + fl['b_len']]
            theta_all_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1) # [fan_out, fan_in+1]
        else:
            theta_all_prior = theta_prior.squeeze()[fl['W_start']:fl['W_start'] + fl['param_len']].unsqueeze(0) # [1, param_len]
            
        theta_all_prior_3d = theta_all_prior.unsqueeze(-1)

        S_3d = filter_S_info[L]
        if is_first or S_3d is None:
            P_sqrt_prev = np.sqrt(p_init_val) * lc['eye_block_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_block_batch'])

        per_layer[L] = {
            'fl': fl, 'lc': lc, 'theta_all_prior': theta_all_prior,
            'theta_all_prior_3d': theta_all_prior_3d, 'P_sqrt_prev': P_sqrt_prev,
        }
        
    current_q_std = sp.get('current_q_std', cfg.q_init)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    current_r_inv_sqrt = 1.0 / current_r_std
    current_r_inv = 1.0 / (current_r_std ** 2)
    
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev'] for L in layers_in_grp], dim=0)
        dynamic_S_Q = current_q_std * grp['eye_grouped']

        S_pred_g, _, y_pred_g, X_sigma_g, scaled_P_g = _time_update_core(
            all_theta_3d, all_P_sqrt, dynamic_S_Q, grp['eye_grouped'], grp['gamma'])
        
        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            per_layer[L]['S_pred'] = S_pred_g[s:e]
            per_layer[L]['y_pred'] = y_pred_g[s:e]
            per_layer[L]['X_sigma_all'] = X_sigma_g[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]

    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        X_sigma_f32 = pl['X_sigma_all'].to(DTYPE_FWD)
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        # [핵심 매핑] Mode에 따라 Sigma Point 흩뿌리기
        if cfg.decoupling_mode == 'node':
            layer_view = unified[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], -1)
            layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :fl['fan_in']])
            layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, fl['fan_in']:fl['fan_in'] + 1])
        else:
            # [버그 수정됨] X_sigma_f32[0] 을 그대로 복사 ([num_sigma, param_len])
            unified[fwd_start:fwd_end, fl['W_start']:fl['W_start'] + fl['param_len']] = X_sigma_f32[0]

    # ──────────────────────────────────────────────────────────────────
    # [v5] DDQN target Q (매 h-step에서 새로 계산, 사전 캐시 사용 안 함)
    #   action argmax:
    #     h=0 + use_spas    → sigma-mean Q
    #     h=0 + not use_spas → theta_target Q (= theta_prior with h0='target')
    #     h≥1                → theta_current_in Q (직전 horizon 추정치)
    #   value Q: 항상 theta_target Q
    # ──────────────────────────────────────────────────────────────────
    Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)
    Q_tgt = Q_tgt_f32[0]  # [nA, batch_sz]

    if is_first:
        if cfg.use_spas:
            Q_sigma_f32 = forward_bmm(unified, info, s_next)
            a_best_next = Q_sigma_f32.mean(dim=0).argmax(dim=0)
        else:
            a_best_next = Q_tgt.argmax(dim=0)
    else:
        Q_curr = forward_single(theta_current.squeeze(), info, s_next)
        a_best_next = Q_curr.argmax(dim=0)

    q_val_next = Q_tgt[a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)

    # [N-step] target gamma
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    z_measured = (batch['r'] + target_gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    target_var = torch.var(z_measured).item()

    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
        z_measured_exp = z_measured.unsqueeze(0).expand(lc['num_blocks'], -1, -1)

        HT_all, residual_all, z_hat, ht_norm, resid_norm = _compute_ht_core(
            Z_sigma_T_f32, lc['Wm_col_f32'], lc['Wc_f32'], lc['zero_col_f32'],
            pl['scaled_P'].to(DTYPE_FWD), z_measured_exp, pl['S_pred'])

        per_layer[L]['HT_all'] = HT_all
        per_layer[L]['residual_all'] = residual_all
        per_layer[L]['loss'] = torch.mean(residual_all ** 2)
        per_layer[L]['ht_norm'] = ht_norm
        per_layer[L]['resid_norm'] = resid_norm
        per_layer[L]['resid_max'] = torch.max(torch.abs(residual_all)).item()
        layer_count += 1

    total_innov_mean, total_innov_max = 0.0, 0.0
    total_ht_norm, total_resid_norm = 0.0, 0.0
    total_delta_y, total_y_new, total_avg_P = 0.0, 0.0, 0.0
    total_resid_in_innov, total_ht_theta_in_innov = 0.0, 0.0
    total_innov_norm, total_y_pred_norm, total_adapt_ratio = 0.0, 0.0, 0.0
    group_count = 0
    
    per_layer_cond, per_layer_ymax, per_layer_cond_full = {}, {}, {}

    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_S_pred = torch.cat([per_layer[L]['S_pred'] for L in layers_in_grp], dim=0)
        all_y_pred = torch.cat([per_layer[L]['y_pred'] for L in layers_in_grp], dim=0)
        all_HT = torch.cat([per_layer[L]['HT_all'] for L in layers_in_grp], dim=0)
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_residual = torch.cat([per_layer[L]['residual_all'] for L in layers_in_grp], dim=0)

        # [Trust region 제거] 항상 표준 measurement update만 사용
        theta_new_g, S_new_g, meas_stats = _meas_update_core(
            all_S_pred, all_y_pred, all_HT, all_theta_3d,
            all_residual, current_r_inv_sqrt, current_r_inv, grp['eye_grouped'],
            tikhonov_lambda=cfg.tikhonov_lambda, huber_c=cfg.huber_c)

        total_innov_mean += meas_stats['innov_mean']
        total_innov_max = max(total_innov_max, meas_stats['innov_max'])
        total_delta_y += meas_stats['delta_y']
        total_y_new += meas_stats['y_new_norm']
        total_avg_P += meas_stats['avg_P']
        total_resid_in_innov += meas_stats['resid_in_innov']
        total_ht_theta_in_innov += meas_stats['ht_theta_in_innov']
        total_innov_norm += meas_stats['innov_norm']
        total_y_pred_norm += meas_stats['y_pred_norm']
        total_adapt_ratio += meas_stats['adapt_ratio']
        group_count += 1
            
        for L in layers_in_grp:
            total_ht_norm += per_layer[L]['ht_norm']
            total_resid_norm += per_layer[L]['resid_norm']

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            pl = per_layer[L]
            fl = pl['fl']
            theta_new_L = theta_new_g[s:e]
            S_new_L = S_new_g[s:e]
            
            if cfg.diag_horizon_cond:
                label = f"{fl['type'][0].upper()}{fl['local_idx']}"
                cond_val, ymax_val, _, _ = compute_pseudo_cond_from_S(S_new_L)
                per_layer_cond[label] = cond_val
                per_layer_ymax[label] = ymax_val
                if cfg.use_full_eigvalsh:
                    full_cond, _ = compute_full_cond_from_S(S_new_L)
                    per_layer_cond_full[label] = full_cond

            invalid = ~torch.isfinite(theta_new_L).all(dim=(1, 2))
            if invalid.any(): theta_new_L[invalid] = pl['theta_all_prior'][invalid].unsqueeze(-1)

            theta_flat = theta_current.squeeze()
            
            if cfg.decoupling_mode == 'node':
                W_new = theta_new_L[:, :fl['fan_in'], 0]
                b_new = theta_new_L[:, fl['fan_in'], 0]
                
                if cfg.max_layer_step > 0:
                    W_curr = theta_flat[fl['W_start']:fl['W_start']+fl['W_len']].view(fl['fan_out'], fl['fan_in'])
                    b_curr = theta_flat[fl['b_start']:fl['b_start']+fl['b_len']]
                    delta_norm = torch.sqrt(torch.norm(W_new - W_curr)**2 + torch.norm(b_new - b_curr)**2)
                    if delta_norm > cfg.max_layer_step:
                        scale = cfg.max_layer_step / (delta_norm + 1e-8)
                        W_new = W_curr + (W_new - W_curr) * scale
                        b_new = b_curr + (b_new - b_curr) * scale
                        
                theta_flat[fl['W_start']:fl['W_start'] + fl['W_len']] = W_new.reshape(-1)
                theta_flat[fl['b_start']:fl['b_start'] + fl['b_len']] = b_new
            else:
                theta_new_flat = theta_new_L[0, :, 0]
                if cfg.max_layer_step > 0:
                    theta_curr = theta_flat[fl['W_start']:fl['W_start'] + fl['param_len']]
                    delta_norm = torch.norm(theta_new_flat - theta_curr)
                    if delta_norm > cfg.max_layer_step:
                        scale = cfg.max_layer_step / (delta_norm + 1e-8)
                        theta_new_flat = theta_curr + (theta_new_flat - theta_curr) * scale
                theta_flat[fl['W_start']:fl['W_start'] + fl['param_len']] = theta_new_flat
                
            theta_current = theta_flat.view(-1, 1)
            new_S_info_dict[L] = S_new_L.permute(1, 2, 0)
            total_loss += pl['loss']
            
    new_S_info = [new_S_info_dict[L] for L in range(info['num_filter_layers'])]
    delta_theta = theta_current.squeeze() - theta_current_in.squeeze()
    k_gain_norm = torch.norm(delta_theta).item()
    
    if cfg.max_k_gain > 0 and k_gain_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / k_gain_norm
        theta_current = (theta_current_in.squeeze() + delta_theta * scale).view(-1, 1)
        k_gain_norm = cfg.max_k_gain

    per_layer_ht, per_layer_delta, per_layer_resid_max = {}, {}, {}
    theta_new_flat = theta_current.squeeze()
    theta_old_flat = theta_current_in.squeeze()
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        per_layer_ht[label] = per_layer[L]['ht_norm']
        per_layer_resid_max[label] = per_layer[L]['resid_max']
        s, p_len = fl['W_start'], fl['param_len']
        per_layer_delta[label] = torch.norm(theta_new_flat[s:s+p_len] - theta_old_flat[s:s+p_len]).item()
        
    n_layers = info['num_filter_layers']
    gc = max(group_count, 1)
    dbg = {
        'innov_mean': total_innov_mean / gc,
        'innov_max': total_innov_max,
        'ht_norm': total_ht_norm / n_layers,
        'resid_norm': total_resid_norm / n_layers,
        'delta_y': total_delta_y / gc,
        'y_pred_norm': total_y_pred_norm / gc,
        'y_new': total_y_new / gc,
        'avg_P': total_avg_P / gc,
        'resid_in_innov': total_resid_in_innov / gc,
        'ht_theta_in_innov': total_ht_theta_in_innov / gc,
        'innov_norm': total_innov_norm / gc,
        'per_layer_ht': per_layer_ht,
        'per_layer_delta': per_layer_delta,
        'per_layer_resid_max': per_layer_resid_max,
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond_full,
        'adapt_ratio': total_adapt_ratio / gc,
    }
    
    return theta_current, new_S_info, (total_loss / layer_count).item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8b. SRRHUIF Full Vector Mode
# =========================================================================
def srrhuif_step_fv(theta_current_in, theta_target, filter_S_info, batch, sp,
                    is_first, p_init_val, fv_cache):
    """
    Full Vector mode: 전체 θ ∈ R^n_x를 하나의 블록으로 다룸.
    - layer/node 분리 없음 (모든 파라미터 covariance가 한 행렬에 들어감)
    - 비용: O(n_x²) 메모리, O(n_x³) QR. CartPole 토이 스케일에서만 권장.
    - 진동 측면에선 가장 정확 (within-layer + between-layer correlation 모두 잡힘)

    [v5 변경] q_next_target_cached 제거. 매 horizon step 내부에서 새로 계산.
        - h=0:
            * use_spas=True  → sigma points의 mean Q로 argmax (SPAS)
            * use_spas=False → theta_target로 argmax (standard DDQN at h=0)
          (value Q는 항상 theta_target 사용)
        - h≥1:
            * theta_current_in (직전 horizon step에서 갱신된 θ, == theta_pred) 로 argmax
            * value Q는 theta_target

    [v5 N-step] z_measured = R_n + γ^n · (1-term) · Q_target(s_{t+n}).
        cfg.use_n_step=True 이면 γ^n_step_size 사용, False 이면 γ.
        (batch['r']와 batch['s_next']는 이미 buffer 단계에서 N-step return으로 가공돼 들어옴)
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    
    # ─────────────────────────────────────────────────────────────
    # [Prior 결정] h=0이면 cfg.h0_prior_source에 따라 분기
    # ─────────────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_pred = sp['theta_init'].clone()
        else:  # 'target'
            theta_pred = theta_target.clone()
    else:
        theta_pred = theta_current_in.clone()
    
    theta_pred_flat = theta_pred.squeeze()  # [n_x]
    
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    # ─────────────────────────────────────────────────────────────
    # [A] Time Update (prediction)
    # ─────────────────────────────────────────────────────────────
    eye_n = fv_cache.eye_n
    if is_first or filter_S_info is None:
        P_sqrt_prev = float(np.sqrt(p_init_val)) * eye_n
    else:
        # filter_S_info: [n_x, n_x] lower-triangular S (information factor)
        P_sqrt_prev = safe_inv_tril_batch(
            filter_S_info.unsqueeze(0), eye_n.unsqueeze(0)
        ).squeeze(0)
    
    S_Q = cfg.q_init * eye_n
    P_sqrt_pred = tria_operation_batch(
        torch.cat([P_sqrt_prev, S_Q], dim=1).unsqueeze(0)
    ).squeeze(0)  # [n_x, n_x]
    
    S_pred = safe_inv_tril_batch(
        P_sqrt_pred.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)  # [n_x, n_x] = P^{-1/2}_{t|t-1}
    Y_pred = S_pred @ S_pred.t()
    y_pred = Y_pred @ theta_pred_flat.unsqueeze(-1)  # [n_x, 1]
    
    # ─────────────────────────────────────────────────────────────
    # [B] Sigma Point Generation: 2n_x+1 points around theta_pred
    # ─────────────────────────────────────────────────────────────
    scaled_P = fv_cache.gamma_sigma * P_sqrt_pred  # [n_x, n_x]
    unified = fv_cache.unified_thetas  # [num_sigma, n_x]
    unified[0] = theta_pred_flat.to(DTYPE_FWD)
    unified[1:n_x+1] = (theta_pred_flat.unsqueeze(0) + scaled_P.t()).to(DTYPE_FWD)
    unified[n_x+1:] = (theta_pred_flat.unsqueeze(0) - scaled_P.t()).to(DTYPE_FWD)
    
    # ─────────────────────────────────────────────────────────────
    # [C] Forward all sigma points → measurement statistics
    # ─────────────────────────────────────────────────────────────
    Q_all_f32 = forward_bmm(unified, info, s_batch)  # [num_sigma, nA, batch_sz]
    Z_sigma_T_f32 = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)]  # [num_sigma, batch_sz]
    Z_sigma_T = Z_sigma_T_f32.to(DTYPE)
    
    # ─────────────────────────────────────────────────────────────
    # [D] DDQN target value (매 h-step에서 새로 계산)
    #     - action argmax: h=0+SPAS → sigma-mean Q
    #                      h=0+not SPAS → theta_target Q
    #                      h≥1 → theta_pred(=theta_current_in) Q
    #     - value: q_target 모드 → theta_target Q
    #             pure_reward 모드 → sigma point의 Q(s', a*; chi) (helper에서 처리)
    # ─────────────────────────────────────────────────────────────
    Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)  # [1, nA, batch_sz]
    Q_tgt = Q_tgt_f32[0]  # [nA, batch_sz]

    # [v9+] Q_sigma_next 캐싱 (SPAS에서 재사용)
    Q_sigma_next_cache = None
    if is_first:
        if cfg.use_spas:
            # 이미 계산된 sigma points의 s_next 평가
            Q_sigma_next_cache = forward_bmm(unified, info, s_next)  # [num_sigma, nA, batch_sz]
            a_best_next = Q_sigma_next_cache.mean(dim=0).argmax(dim=0)
        else:
            # h=0 + not SPAS → theta_target (== theta_pred when h0_prior_source='target')
            a_best_next = Q_tgt.argmax(dim=0)
    else:
        # h≥1 → theta_pred(=직전 추정 theta_current_in)으로 argmax
        Q_curr = forward_single(theta_pred_flat, info, s_next)  # [nA, batch_sz]
        a_best_next = Q_curr.argmax(dim=0)

    # [N-step] target_gamma = γ^n_step_size if use_n_step else γ
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    
    # [v9+] Mode-dispatched: Z_sigma_T 변형 + z_measured 계산
    Z_sigma_T, z_measured, _ = _resolve_measurement(
        Q_sigma_at_s_a=Z_sigma_T,
        unified_sigma=unified, info=info, s_next=s_next,
        a_best_next=a_best_next, Q_tgt_next=Q_tgt,
        reward=batch['r'], term_mask=batch['term'],
        target_gamma=target_gamma, device=device,
        Q_sigma_next_cache=Q_sigma_next_cache,
        fwd_fn=forward_bmm,
    )
    target_var = torch.var(z_measured).item()
    
    # z_hat = Σ Wm·Z_sigma  (FINAL Z_sigma_T로 계산 — pure_reward 모드에선 차분 신호의 mean)
    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [batch_sz, 1]
    
    # ─────────────────────────────────────────────────────────────
    # [E] Statistical linearization → H^T
    # ─────────────────────────────────────────────────────────────
    # X_dev: deviation of sigma points from center. [num_sigma, n_x]
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x+1] = scaled_P.t()
    X_dev[n_x+1:] = -scaled_P.t()
    
    Z_dev = Z_sigma_T - z_hat.t()  # [num_sigma, batch_sz]
    P_xz = (X_dev * fv_cache.Wc.view(-1, 1)).t() @ Z_dev  # [n_x, batch_sz]
    HT = Y_pred @ P_xz  # [n_x, batch_sz]
    
    residual = z_measured - z_hat  # [batch_sz, 1]
    loss = torch.mean(residual ** 2)
    
    # ─────────────────────────────────────────────────────────────
    # [F] Information form measurement update
    # ─────────────────────────────────────────────────────────────
    # Adaptive R (스케줄링) — sp['current_r_std']가 있으면 그걸, 없으면 cfg.r_init
    current_r_std = sp.get('current_r_std', cfg.r_init)
    r_inv = 1.0 / (current_r_std ** 2)
    r_inv_sqrt = 1.0 / current_r_std
    
    # Huber-style adaptive R
    res_abs = torch.abs(residual)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)  # [batch_sz, 1]
    r_inv_adapt = r_inv / adapt_factor  # [batch_sz, 1]
    r_inv_sqrt_adapt = (r_inv_sqrt / torch.sqrt(adapt_factor)).t()  # [1, batch_sz]
    
    # S_new = QR([S_pred | HT * r_inv_sqrt_adapt | sqrt(λ)·I])
    tikhonov_sqrt = float(np.sqrt(cfg.tikhonov_lambda))
    if cfg.tikhonov_lambda > 0:
        combined = torch.cat([S_pred, HT * r_inv_sqrt_adapt, tikhonov_sqrt * eye_n], dim=1)
    else:
        combined = torch.cat([S_pred, HT * r_inv_sqrt_adapt], dim=1)
    S_new = tria_operation_batch(combined.unsqueeze(0)).squeeze(0)
    
    # y_new = y_pred + HT · r_inv_adapt · (residual + HT^T · θ_pred)
    ht_theta = HT.t() @ theta_pred_flat.unsqueeze(-1)  # [batch_sz, 1]
    innov = residual + ht_theta
    y_new = y_pred + HT @ (r_inv_adapt * innov)
    
    # Recover θ from information form
    theta_new = robust_solve_spd_batch(
        S_new.unsqueeze(0), y_new.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)  # [n_x, 1]
    
    if not torch.isfinite(theta_new).all():
        theta_new = theta_pred.clone()
    
    # ─────────────────────────────────────────────────────────────
    # [G] Diagnostics (LD/ND와 호환되는 dbg dict)
    # ─────────────────────────────────────────────────────────────
    # K-gain norm: ||HT · r_inv||
    k_gain = HT * r_inv_sqrt_adapt
    k_gain_norm = torch.norm(k_gain).item()
    
    # avg_P: 1/diag(Y_new)의 평균 (대략 평균 분산)
    Y_new = S_new @ S_new.t()
    Y_diag = torch.diagonal(Y_new)
    avg_P = (1.0 / (Y_diag + 1e-8)).mean().item()
    
    # innov 분해 stats
    innov_abs = torch.abs(innov)
    resid_abs = torch.abs(residual)
    ht_theta_abs = torch.abs(ht_theta)
    
    delta_theta_norm = torch.norm(theta_new - theta_pred).item()
    
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'innov_norm': innov_abs.mean().item(),
        'resid_in_innov': resid_abs.mean().item(),
        'ht_theta_in_innov': ht_theta_abs.mean().item(),
        'avg_P': avg_P,
        'ht_norm': torch.norm(HT).item(),
        'resid_norm': torch.norm(residual).item(),
        'delta_y': torch.norm(HT @ (r_inv_adapt * innov)).item(),
        'y_pred_norm': torch.norm(y_pred).item(),
        'y_new_norm': torch.norm(y_new).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        # FV는 layer 분리 없으니 per_layer dict는 비움 (training loop이 .get() 패턴 사용)
        # FV: 전체 벡터를 네트워크 레이어 구간으로 분해 (가로=layer, 세로=horizon 정밀 진단)
        'per_layer_ht': fv_per_layer(info, HT, 'norm'),
        'per_layer_delta': fv_per_layer(info, (theta_new - theta_pred).squeeze(-1), 'norm'),
        'per_layer_resid_max': fv_broadcast(info, resid_abs.max().item()),  # 측정-공간 전역
        'per_layer_cond': fv_broadcast(info, 1.0),        # placeholder (정보형 cond 미산출)
        'per_layer_ymax': fv_per_layer(info, y_new, 'maxabs'),  # per-layer 정보벡터 max
        'per_layer_cond_full': fv_broadcast(info, 1.0),
    }
    
    if sp.get('_do_sigma_spread', False):
        dbg['sigma_spread'] = compute_sigma_spread(unified, info, s_batch)
    return theta_new, S_new, loss.item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8c. RHUKF — Full Vector Mode, Covariance Form (Kim et al. 2010 Alg 1)
# =========================================================================
@torch.no_grad()
def rhukf_step_fv(theta_current_in, theta_target, filter_P_cov, batch, sp,
                  is_first, p_init_val, fv_cache):
    """
    Receding Horizon UKF, full vector, COVARIANCE form (full P, no sqrt).
    
    Paper: Kim, Yang, Jeon, Shin (2010) "Receding Horizon Estimation for Hybrid
    Particle Filters...", ICPR. Algorithm 1.
    
    Adaptation to RL parameter estimation:
        - state x ≡ θ ∈ R^{n_x} (network parameters)
        - dynamics: random walk θ_{s+1} = θ_s + v_s, v_s ~ N(0, Q), Q = q_init²·I
        - measurement y ≡ z_measured ∈ R^{batch_sz}
          (TD target: z = r + γⁿ·(1-term)·Q_target(s_next, a_best))
        - observation function h(θ; s_batch, a_batch) ≡ Q_network output
    
    Storage: full P (n_x × n_x). NOT square-root.
        - 이유: sqrt 저장하면 매 step마다 sigma point 위해 Cholesky(P)는 어차피 필요,
          measurement update에선 P 명시 형태로 다뤄야 batch downdate가 안정적.
          저장만 sqrt하면 분해/조합 비용만 더해짐 (사용자 지적대로).
        - 따라서 P 자체를 저장하고, sigma point용으로만 chol(P_pred) 1회 계산.
    
    초기조건 (h=0):
        θ_0 = θ_prior   (cfg.h0_prior_source에 따라 target 또는 init)
        P_0 = p_init · I    ← p_init이 그대로 covariance scale로 들어감 (직관적)
    
    Huber-adaptive R: 정보형 그대로의 의미. 샘플별 R_eff[i] = r_init² · adapt_factor[i]
        adapt_factor[i] = max(|residual[i]|/huber_c, 1) → outlier 샘플의 영향 감소.
        P_zz의 diagonal에 들어가서 batch×batch innovation cov를 outlier-robust하게 만듦.
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    
    # ─────────────────────────────────────────────────────────────
    # [Prior] h=0이면 cfg.h0_prior_source, h≥1이면 직전 추정치
    # ─────────────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_pred = sp['theta_init'].clone()
        else:  # 'target'
            theta_pred = theta_target.clone()
    else:
        theta_pred = theta_current_in.clone()
    theta_pred_flat = theta_pred.squeeze()  # [n_x]
    
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    # ═════════════════════════════════════════════════════════════
    # [A] Time Update (covariance form): P_pred = P_prev + Q
    #     random walk이라 propagation 자체는 trivial (덧셈 한 번).
    # ═════════════════════════════════════════════════════════════
    eye_n = fv_cache.eye_n
    if is_first or filter_P_cov is None:
        P_prev = p_init_val * eye_n
    else:
        P_prev = filter_P_cov
    
    Q_proc = (cfg.q_init ** 2) * eye_n
    P_pred = P_prev + Q_proc
    P_pred = 0.5 * (P_pred + P_pred.t())  # symmetrize (수치 안전망)
    
    # ═════════════════════════════════════════════════════════════
    # [B] Sigma points: chol(P_pred)으로 spread
    # ═════════════════════════════════════════════════════════════
    S_P_pred = safe_cholesky_fallback(P_pred, eye_n, JITTER_TRIA)
    
    scaled_P = fv_cache.gamma_sigma * S_P_pred  # [n_x, n_x]
    unified = fv_cache.unified_thetas  # [num_sigma, n_x]
    unified[0] = theta_pred_flat.to(DTYPE_FWD)
    unified[1:n_x+1] = (theta_pred_flat.unsqueeze(0) + scaled_P.t()).to(DTYPE_FWD)
    unified[n_x+1:] = (theta_pred_flat.unsqueeze(0) - scaled_P.t()).to(DTYPE_FWD)
    
    # ═════════════════════════════════════════════════════════════
    # [C] Forward sigma points → measurement statistics
    # ═════════════════════════════════════════════════════════════
    Q_all_f32 = forward_bmm(unified, info, s_batch)  # [num_sigma, nA, batch_sz]
    Z_sigma_T_f32 = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)]
    Z_sigma_T = Z_sigma_T_f32.to(DTYPE)  # [num_sigma, batch_sz]
    
    # ═════════════════════════════════════════════════════════════
    # [D] DDQN target value (per-horizon recompute, v5 logic)
    # ═════════════════════════════════════════════════════════════
    Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)
    Q_tgt = Q_tgt_f32[0]  # [nA, batch_sz]
    
    # [v9+] Q_sigma_next 캐싱 (SPAS에서 재사용)
    Q_sigma_next_cache = None
    if is_first:
        if cfg.use_spas:
            Q_sigma_next_cache = forward_bmm(unified, info, s_next)
            a_best_next = Q_sigma_next_cache.mean(dim=0).argmax(dim=0)
        else:
            a_best_next = Q_tgt.argmax(dim=0)
    else:
        Q_curr = forward_single(theta_pred_flat, info, s_next)
        a_best_next = Q_curr.argmax(dim=0)
    
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    
    # [v9+] Mode-dispatched: Z_sigma_T 변형 + z_measured 계산
    Z_sigma_T, z_measured, _ = _resolve_measurement(
        Q_sigma_at_s_a=Z_sigma_T,
        unified_sigma=unified, info=info, s_next=s_next,
        a_best_next=a_best_next, Q_tgt_next=Q_tgt,
        reward=batch['r'], term_mask=batch['term'],
        target_gamma=target_gamma, device=device,
        Q_sigma_next_cache=Q_sigma_next_cache,
        fwd_fn=forward_bmm,
    )
    target_var = torch.var(z_measured).item()
    
    # z_hat (FINAL Z_sigma_T로 계산)
    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [batch_sz, 1]
    
    residual = z_measured - z_hat  # [batch_sz, 1]
    loss = torch.mean(residual ** 2)
    
    # ═════════════════════════════════════════════════════════════
    # [E] Cross covariance P_xz, innovation cov P_zz (UT 직접)
    # ═════════════════════════════════════════════════════════════
    Wc_col = fv_cache.Wc.view(-1, 1)  # [num_sigma, 1]
    
    Z_dev = Z_sigma_T - z_hat.t()  # [num_sigma, batch_sz]
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x+1] = scaled_P.t()
    X_dev[n_x+1:] = -scaled_P.t()
    
    # P_zz의 sigma 부분 (R 추가 전): Σ Wc·(Z-z̄)(Z-z̄)^T
    P_zz_sigma = Z_dev.t() @ (Wc_col * Z_dev)  # [batch_sz, batch_sz]
    
    # P_xz: Σ Wc·(X-x̄)(Z-z̄)^T
    P_xz = X_dev.t() @ (Wc_col * Z_dev)  # [n_x, batch_sz]
    
    # ═════════════════════════════════════════════════════════════
    # [F] Huber-adaptive R: 샘플별 R 인플레이션
    #     큰 |residual| 샘플은 R_eff↑ → P_zz_diag↑ → K_col↓ → 영향력 감소
    # ═════════════════════════════════════════════════════════════
    res_abs = torch.abs(residual).squeeze(-1)  # [batch_sz]
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)  # [batch_sz]
    
    current_r_std = sp.get('current_r_std', cfg.r_init)
    R_diag_eff = (current_r_std ** 2) * adapt_factor  # [batch_sz], per-sample variance
    
    P_zz = P_zz_sigma + torch.diag(R_diag_eff)
    P_zz = 0.5 * (P_zz + P_zz.t())  # symmetrize
    
    # ═════════════════════════════════════════════════════════════
    # [G] Kalman gain K = P_xz · P_zz⁻¹ via Cholesky(P_zz)
    # ═════════════════════════════════════════════════════════════
    eye_batch = torch.eye(batch_sz, dtype=DTYPE, device=device)
    L_zz = safe_cholesky_fallback(P_zz, eye_batch)
    
    # K = P_xz @ inv(P_zz) = P_xz @ inv(L_zz^T) @ inv(L_zz)
    # 두 번의 triangular solve로 수치 안정적으로
    tmp = torch.linalg.solve_triangular(L_zz, P_xz.t(), upper=False)
    K_t = torch.linalg.solve_triangular(L_zz.t(), tmp, upper=True)
    K = K_t.t()  # [n_x, batch_sz]
    
    # ═════════════════════════════════════════════════════════════
    # [H] State update: θ_new = θ_pred + K · innovation
    #     KF form에선 innovation == residual (z - ẑ), 정보형의 H^T·θ 항 없음
    # ═════════════════════════════════════════════════════════════
    theta_new_flat = theta_pred_flat + (K @ residual).squeeze(-1)
    
    if not torch.isfinite(theta_new_flat).all():
        theta_new_flat = theta_pred_flat.clone()
    theta_new = theta_new_flat.view(-1, 1)
    
    # ═════════════════════════════════════════════════════════════
    # [I] Covariance update: P_new = P_pred - K · P_zz · K^T
    #     K · P_zz · K^T = (K · L_zz)(K · L_zz)^T 형태로 PSD 보장
    # ═════════════════════════════════════════════════════════════
    K_L = K @ L_zz  # [n_x, batch_sz]
    P_new = P_pred - K_L @ K_L.t()
    
    # Tikhonov + symmetrize (수치 PSD 안전)
    P_new = 0.5 * (P_new + P_new.t())
    if cfg.tikhonov_lambda > 0:
        P_new = P_new + cfg.tikhonov_lambda * eye_n
    
    # ═════════════════════════════════════════════════════════════
    # [J] Diagnostics — covariance form 의미에 맞게
    # ═════════════════════════════════════════════════════════════
    P_diag = torch.diagonal(P_new)
    avg_P = P_diag.mean().item()
    max_P = P_diag.max().item()
    
    # cond(P_zz) ≈ cond(L_zz)² (cheap proxy)
    L_zz_diag = torch.diagonal(L_zz)
    cond_P_zz = ((L_zz_diag.max() / L_zz_diag.min().clamp(min=1e-12)) ** 2).item()
    
    k_gain_norm = torch.norm(K).item()
    delta_theta_norm = torch.norm(theta_new_flat - theta_pred_flat).item()
    innov_abs = torch.abs(residual)
    
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'innov_norm': innov_abs.mean().item(),
        # KF form: innovation == residual (no H^T·θ_prior term)
        'resid_in_innov': innov_abs.mean().item(),
        'ht_theta_in_innov': 0.0,
        'avg_P': avg_P,
        'max_P': max_P,
        # P_xz는 KF form에서 H^T (cross covariance)의 직접 대응. 
        # ‖P_xz‖는 측정-상태 민감도 (statistical Jacobian 크기)의 proxy.
        'ht_norm': torch.norm(P_xz).item(),
        'resid_norm': torch.norm(residual).item(),
        'delta_y': torch.norm(K @ residual).item(),  # ‖Δθ_correction‖ = ‖K·innov‖
        # KF form엔 y_pred 개념이 없음 (정보형 전용). theta_pred norm으로 대체.
        'y_pred_norm': torch.norm(theta_pred_flat).item(),
        'y_new_norm': torch.norm(theta_new_flat).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        # FV: 전체 벡터를 네트워크 레이어 구간으로 분해 (가로=layer, 세로=horizon 정밀 진단)
        'per_layer_ht': fv_per_layer(info, P_xz, 'norm'),  # ||P_xz|| 행 = 레이어별 측정-상태 민감도
        'per_layer_delta': fv_per_layer(info, theta_new_flat - theta_pred_flat, 'norm'),
        'per_layer_resid_max': fv_broadcast(info, innov_abs.max().item()),  # 측정-공간 전역
        # 정보형 'cond(Y)' 자리에 cond(P_zz) (innovation cov 조건수, 측정-공간 전역)
        'per_layer_cond': fv_broadcast(info, cond_P_zz),
        # 정보형 'Y_max' 자리에 per-layer max diag(P) (레이어별 파라미터 불확실성)
        'per_layer_ymax': fv_per_layer(info, P_diag, 'maxabs'),
        'per_layer_cond_full': fv_broadcast(info, cond_P_zz),
    }
    
    if sp.get('_do_sigma_spread', False):
        dbg['sigma_spread'] = compute_sigma_spread(unified, info, s_batch)
    return theta_new, P_new, loss.item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8d. [v7] Error-State Horizon Setup
#     - 호라이즌 직전 1회 호출. Anchor 결정 + Y_cache 계산.
#     - Y_cache는 ddqn_argmax 정책에 따라:
#         'target'        : argmax & value 모두 θ_target → 완전 캐싱
#         'online_frozen' : argmax = θ_active(호라이즌 시작 직전 동결), value = θ_target → 완전 캐싱
#         'online_moving' : argmax이 h마다 변함 → Y_cache=None (루프 안에서 즉석 계산)
#     - Anchor는 호라이즌 동안 frozen.
# =========================================================================
@torch.no_grad()
def compute_twin_y_cache(theta_active_1, theta_target_1, theta_target_2,
                          batch_hist, sp, cfg):
    """
    Twin-Q (Clipped Double Q-Learning, TD3) Y_cache 계산:
        a* = argmax Q_1_active(s')  (TD3 표준; ddqn_argmax='target'이면 Q_1_target)
        Y  = r + γ · min(Q_1_tgt(s', a*), Q_2_tgt(s', a*))
    
    동일한 Y를 두 필터(θ_1, θ_2)가 공유하여 병렬 FIR 업데이트.
    
    Returns: Y_cache [N, B] tensor
    """
    device, info = sp['device'], sp['info']
    B = cfg.batch_size
    N = cfg.N_horizon
    NB = N * B
    
    s_next_all = torch.cat([b['s_next'] for b in batch_hist], dim=1)
    if sp.get('normalizer'):
        s_next_all = sp['normalizer'].normalize(s_next_all)
    
    idx_all = torch.arange(NB, device=device)
    theta_active_1_flat = theta_active_1.squeeze().detach()
    theta_target_1_flat = theta_target_1.squeeze().detach()
    theta_target_2_flat = theta_target_2.squeeze().detach()
    
    # 두 target net의 Q값
    Q_tgt_1 = forward_single(theta_target_1_flat, info, s_next_all).to(DTYPE)  # [nA, NB]
    Q_tgt_2 = forward_single(theta_target_2_flat, info, s_next_all).to(DTYPE)  # [nA, NB]
    
    # Argmax 정책 (Twin-Q에서는 일반적으로 Q_1_active 사용 — TD3 식)
    if cfg.ddqn_argmax == 'target':
        a_best_all = Q_tgt_1.argmax(dim=0)
    elif cfg.ddqn_argmax == 'online_frozen':
        Q_active_1 = forward_single(theta_active_1_flat, info, s_next_all).to(DTYPE)
        a_best_all = Q_active_1.argmax(dim=0)
    elif cfg.ddqn_argmax == 'online_moving':
        # online_moving은 caching 깨므로 Twin과 함께 사용시 의미 모호
        # 호라이즌 시작 시점의 active로 fallback
        Q_active_1 = forward_single(theta_active_1_flat, info, s_next_all).to(DTYPE)
        a_best_all = Q_active_1.argmax(dim=0)
    else:
        raise RuntimeError(f"Twin-Q + ddqn_argmax='{cfg.ddqn_argmax}' not supported")
    
    # ★ Clipped Double Q-Learning: min(Q_1, Q_2)
    Q1_at_a = Q_tgt_1[a_best_all, idx_all]
    Q2_at_a = Q_tgt_2[a_best_all, idx_all]
    q_val_next_all = torch.minimum(Q1_at_a, Q2_at_a)  # [NB]
    
    r_all = torch.cat([b['r'] for b in batch_hist], dim=0).to(DTYPE)
    term_all = torch.cat([b['term'] for b in batch_hist], dim=0).to(DTYPE)
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    # [v9+] measurement_mode 분기
    if cfg.measurement_mode == 'q_target':
        Y_flat = r_all + target_gamma * (1.0 - term_all) * q_val_next_all
    elif cfg.measurement_mode == 'pure_reward':
        # pure_reward + twin: y = r (각 필터의 h(w) 안에서 sigma point가 차분 처리).
        # Twin의 min(Q1,Q2) 효과는 사라지지만, pure_reward 자체의 (1-γ) attenuation이
        # 이미 max-bias를 누름. 두 필터는 독립적으로 동일 y=r을 학습.
        Y_flat = r_all
    else:
        raise RuntimeError(f"measurement_mode={cfg.measurement_mode}")
    return Y_flat.view(N, B)


def compute_adam_td_loss(theta_param, theta_target, batch, sp, cfg):
    """
    [v9+] Adam TD loss. cfg.measurement_mode와 무관하게 항상 q_target
    (semi-gradient TD) form 사용:
        residual = Q(s,a;θ) - (r + γ(1-term) Q(s',a*;θ_T))
        a* = argmax_a Q(s',a;θ) (DDQN, online θ argmax + detach)

    왜 pure_reward를 안 쓰는가:
      pure_reward (h(θ)=Q(s,a;θ)-γQ(s',a*;θ), y=r) 는 Kalman의 sigma-point
      cross-covariance cancellation을 전제로 한 측정 모델. gradient 기반으로
      풀면 residual gradient (Baird 1995) 가 되어 분산 폭증 / 수렴 불안정.
      따라서 Adam 경로는 항상 q_target form만 사용.

    cfg.huber_c>0이면 Huber, else MSE.
    """
    info, normalizer, device = sp['info'], sp['normalizer'], sp['device']
    s = batch['s'].t()
    s_next = batch['s_next'].t()
    a = batch['a']
    r = batch['r'].to(DTYPE)
    term = batch['term'].to(DTYPE)
    if normalizer:
        s = normalizer.normalize(s)
        s_next = normalizer.normalize(s_next)

    B = a.shape[0]
    idx_arange = torch.arange(B, device=device)
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma

    # Q(s, a; θ) — grad 흐름
    Q_curr = forward_single(theta_param, info, s).to(DTYPE)            # [nA, B]
    q_sa = Q_curr[a, idx_arange]                                       # [B]

    # DDQN target — argmax는 online θ로, value는 θ_T로. 모두 grad 차단.
    with torch.no_grad():
        Q_next_nograd = forward_single(theta_param.detach(), info, s_next).to(DTYPE)
        a_star = Q_next_nograd.argmax(dim=0)
        Q_tgt_next = forward_single(theta_target.squeeze(), info, s_next).to(DTYPE)
        y = r + target_gamma * (1.0 - term) * Q_tgt_next[a_star, idx_arange]
    residual = q_sa - y

    if cfg.huber_c > 0:
        loss = F.huber_loss(residual, torch.zeros_like(residual),
                            delta=cfg.huber_c, reduction='mean')
    else:
        loss = (residual ** 2).mean()
    return loss


def _compute_per_priorities(theta, theta_target, batch_hist, sp, cfg, normalizer):
    """
    [v9+] PER: horizon 종료 후 최신 theta로 |TD error| 재계산해서 priority 업데이트용.
    
    Returns:
        idx_all: [N*B] long tensor — buffer indices
        td_abs:  [N*B] float tensor — |TD error|
    """
    if not cfg.use_per:
        return None, None
    device, info = sp['device'], sp['info']
    
    # Concatenate all horizon transitions
    s_all = torch.cat([b['s'] for b in batch_hist], dim=1)        # [dim_s, N*B]
    a_all = torch.cat([b['a'] for b in batch_hist], dim=0)        # [N*B]
    r_all = torch.cat([b['r'] for b in batch_hist], dim=0).to(DTYPE)
    s_next_all = torch.cat([b['s_next'] for b in batch_hist], dim=1)
    term_all = torch.cat([b['term'] for b in batch_hist], dim=0).to(DTYPE)
    idx_all = torch.cat([b['indices'] for b in batch_hist], dim=0)
    
    if normalizer:
        s_all = normalizer.normalize(s_all)
        s_next_all = normalizer.normalize(s_next_all)
    
    NB = a_all.shape[0]
    idx_arange = torch.arange(NB, device=device)
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    
    with torch.no_grad():
        theta_flat = theta.squeeze().detach()
        theta_target_flat = theta_target.squeeze().detach()
        
        Q_curr_all = forward_single(theta_flat, info, s_all).to(DTYPE)        # [nA, NB]
        q_sa = Q_curr_all[a_all, idx_arange]
        
        Q_curr_next = forward_single(theta_flat, info, s_next_all).to(DTYPE)
        a_star = Q_curr_next.argmax(dim=0)
        
        if cfg.measurement_mode == 'q_target':
            Q_target_next = forward_single(theta_target_flat, info, s_next_all).to(DTYPE)
            q_next_v = Q_target_next[a_star, idx_arange]
            target_y = r_all + target_gamma * (1.0 - term_all) * q_next_v
            td = target_y - q_sa
        else:  # pure_reward
            q_next_h = Q_curr_next[a_star, idx_arange]
            h_w = q_sa - target_gamma * (1.0 - term_all) * q_next_h
            td = r_all - h_w
    
    return idx_all, td.abs()


def _resolve_measurement(
    Q_sigma_at_s_a,         # [num_sigma, B] — Q(s, a; chi_i)  (이미 계산됨)
    unified_sigma,          # [num_sigma, n_x] — sigma points (pure_reward 시 필요)
    info,                   # network info dict
    s_next,                 # [dim_s, B] — 다음 상태
    a_best_next,            # [B] — 고정된 a*
    Q_tgt_next,             # [nA, B] 또는 None — Q(s', :; θ_T) (q_target 시 필요)
    reward,                 # [B]
    term_mask,              # [B]
    target_gamma,           # γ 또는 γ^n
    device,
    Q_sigma_next_cache=None,   # [num_sigma, nA, B] 캐시 (SPAS 등에서 이미 계산했으면 재사용)
    fwd_fn=None,               # forward 함수 (보통 forward_bmm)
    q_val_next_override=None,  # Twin-Q 등에서 min(Q1, Q2) 외부 주입용 (q_target 모드 전용)
):
    """
    [v9+] Mode-dispatched measurement / target computation.
    
    Returns:
        Z_sigma_T:  [num_sigma, B]  measurement function 값 per sigma
        z_measured: [B, 1]          target measurement
        Q_sigma_next_cache: [num_sigma, nA, B] 또는 None (재사용 가능하도록 반환)
    
    q_target 모드 (기존):
        Z_sigma_T  = Q(s, a; chi_i)               (입력 Q_sigma_at_s_a 그대로)
        z_measured = r + γ (1-term) Q(s', a*; θ_T)
    
    pure_reward 모드 (신규):
        Z_sigma_T  = Q(s, a; chi_i) - γ (1-term) Q(s', a*; chi_i)
        z_measured = r
    """
    B = reward.shape[0]
    dtype_z = Q_sigma_at_s_a.dtype
    not_term = (1.0 - term_mask).to(dtype_z)
    idx = torch.arange(B, device=device)
    
    if cfg.measurement_mode == 'q_target':
        # ── 기존 동작 ─────────────────────────────────────────────────
        Z_sigma_T = Q_sigma_at_s_a  # 그대로
        if q_val_next_override is not None:
            q_val_next = q_val_next_override.to(dtype_z)
        else:
            q_val_next = Q_tgt_next[a_best_next, idx].to(dtype_z)
        z_measured = (reward.to(dtype_z) + target_gamma * not_term * q_val_next).view(-1, 1)
        return Z_sigma_T, z_measured, Q_sigma_next_cache
    
    elif cfg.measurement_mode == 'pure_reward':
        # ── 새 동작: h(chi) = Q(s,a;chi) - γ Q(s',a*;chi) ──────────────
        if Q_sigma_next_cache is None:
            if fwd_fn is None:
                raise ValueError("pure_reward 모드는 Q_sigma_next_cache 또는 fwd_fn 필요")
            Q_sigma_next_cache = fwd_fn(unified_sigma, info, s_next)  # [num_sigma, nA, B]
        # 각 sigma point의 Q(s', a*; chi) gather
        q_next_per_sigma = Q_sigma_next_cache[:, a_best_next, idx].to(dtype_z)  # [num_sigma, B]
        # terminal masking
        q_next_per_sigma = q_next_per_sigma * not_term.unsqueeze(0)
        Z_sigma_T = Q_sigma_at_s_a - target_gamma * q_next_per_sigma
        z_measured = reward.to(dtype_z).view(-1, 1)
        return Z_sigma_T, z_measured, Q_sigma_next_cache
    
    else:
        raise ValueError(f"Unknown measurement_mode: {cfg.measurement_mode!r}")


def init_error_horizon(theta_active, theta_target, batch_hist, sp, cfg, fv_cache,
                       Y_cache_external=None):
    """
    호라이즌 시작 직전 1회 호출.
    
    Y_cache_external: Twin-Q 등에서 미리 계산한 Y_cache를 주입하는 용도.
                      None이면 내부에서 Q(θ_target)[a*]로 계산 (기존 single 모드).
    
    Returns:
        ctx: dict with keys:
            'theta_anchor'    : [n_x] tensor (frozen)
            'theta_target_ref': [n_x] tensor
            'Y_cache'         : [N, B] tensor or None
            'p_delta_init'    : float
    """
    device, info = sp['device'], sp['info']
    B = cfg.batch_size
    N = cfg.N_horizon
    
    # ── 1. Anchor 결정 (frozen) ──────────────────────────────────────
    theta_active_flat = theta_active.squeeze().detach().clone()  # [n_x]
    theta_target_flat = theta_target.squeeze().detach().clone()  # [n_x]
    if cfg.anchor_type == 'target':
        theta_anchor = theta_target_flat.clone()
    elif cfg.anchor_type == 'init':  # frozen θ_init (임의 초기값)
        theta_anchor = sp['theta_init'].squeeze().detach().clone()
    else:  # 'current'
        theta_anchor = theta_active_flat.clone()
    
    # ── 2. Y_cache (가능하면 미리 계산) ───────────────────────────────
    if cfg.ddqn_argmax == 'online_moving':
        Y_cache = None  # 루프 안에서 매 h마다 계산
    else:
        # 모든 h의 s_next를 한 텐서로 스택. batch['s_next']는 [dim_s, B].
        s_next_all = torch.cat([b['s_next'] for b in batch_hist], dim=1)  # [dim_s, N*B]
        if sp.get('normalizer'):
            s_next_all = sp['normalizer'].normalize(s_next_all)
        
        NB = N * B
        idx_all = torch.arange(NB, device=device)
        
        # Q_target (단일점) — value='single' OR argmax='target' 시 필요
        Q_tgt_all = forward_single(theta_target_flat, info, s_next_all).to(DTYPE)  # [nA, NB]
        
        # ── Sigma ensemble forward (argmax='spas' 시만) ──
        need_sigma = (cfg.ddqn_argmax == 'spas')
        Q_sigma_all = None  # [num_sigma, nA, NB]
        if need_sigma:
            n_x = info['total_params']
            num_sigma = 2 * n_x + 1
            lam_ut = cfg.alpha**2 * (n_x + cfg.kappa) - n_x
            gamma_fv = float(np.sqrt(n_x + lam_ut))
            spread = gamma_fv * float(np.sqrt(cfg.p_delta_init))
            eye_n_local = torch.eye(n_x, dtype=DTYPE, device=device)
            sigma_thetas = torch.empty(num_sigma, n_x, dtype=DTYPE_FWD, device=device)
            sigma_thetas[0] = theta_anchor.to(DTYPE_FWD)
            sigma_thetas[1:n_x+1] = (theta_anchor.unsqueeze(0) + spread * eye_n_local).to(DTYPE_FWD)
            sigma_thetas[n_x+1:] = (theta_anchor.unsqueeze(0) - spread * eye_n_local).to(DTYPE_FWD)
            Q_sigma_all_f32 = forward_bmm(sigma_thetas, info, s_next_all.t())
            Q_sigma_all = Q_sigma_all_f32.to(DTYPE)
        
        # ── Argmax (a* 선택) ──
        if cfg.ddqn_argmax == 'target':
            a_best_all = Q_tgt_all.argmax(dim=0)
        elif cfg.ddqn_argmax == 'online_frozen':
            Q_online_all = forward_single(theta_active_flat, info, s_next_all).to(DTYPE)
            a_best_all = Q_online_all.argmax(dim=0)
        elif cfg.ddqn_argmax == 'spas':
            Q_sigma_mean_for_argmax = Q_sigma_all.mean(dim=0)
            a_best_all = Q_sigma_mean_for_argmax.argmax(dim=0)
        else:
            raise RuntimeError(f"Unreachable: ddqn_argmax={cfg.ddqn_argmax}")
        
        # ── Value: Q(s', a*) — Twin이면 외부에서 Y_cache_external 주입, 아니면 single ──
        if Y_cache_external is not None:
            Y_cache = Y_cache_external
        else:
            r_all = torch.cat([b['r'] for b in batch_hist], dim=0).to(DTYPE)
            term_all = torch.cat([b['term'] for b in batch_hist], dim=0).to(DTYPE)
            target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
            # [v9+] measurement_mode 분기
            if cfg.measurement_mode == 'q_target':
                q_val_next_all = Q_tgt_all[a_best_all, idx_all]
                Y_flat = r_all + target_gamma * (1.0 - term_all) * q_val_next_all
            elif cfg.measurement_mode == 'pure_reward':
                # y = r (Q_target은 measurement에서 빠지고 h(w) 안으로 흡수됨)
                Y_flat = r_all
            else:
                raise RuntimeError(f"measurement_mode={cfg.measurement_mode}")
            Y_cache = Y_flat.view(N, B)
    
    return {
        'theta_anchor': theta_anchor,
        'theta_target_ref': theta_target_flat,
        'theta_active_ref': theta_active_flat,   # online_moving h=0용: horizon 직전 active θ
        'Y_cache': Y_cache,
        'p_delta_init': cfg.p_delta_init,
        # [v9+] pure_reward 모드에서 sigma point가 Q(s', a*; chi)를 평가할 때 필요
        'a_best_per_step': (a_best_all.view(N, B) if cfg.ddqn_argmax != 'online_moving' else None),
    }


# =========================================================================
# 8e. [v7] Error-State SRRHUIF — Full Vector, Information Form
#
#  상태:   Δμ ∈ R^{n_x}, S_{YΔ} (information sqrt, lower-tri [n_x × n_x])
#  Prior(h=0): Δμ⁻ = 0, P_Δ⁻ = p_delta_init·I  →  S_{YΔ}⁻ = (1/√p_delta_init)·I
#  Anchor: ctx['theta_anchor'] (호라이즌 동안 frozen)
#  Sigma:  χ^(Δ)_j = Δμ⁻ ± γ√P_Δ⁻ (error space)
#          forward 시 projection: θ_anchor + χ^(Δ)_j (절대공간)
#  Update: y_{Δ,new} = y_{Δ,pred} + H·R⁻¹·(residual + H^T·Δμ⁻)
#          residual = Y_cache[h] - ẑ (또는 즉석 계산)
#          Δμ_new   = (S_{YΔ,new} S_{YΔ,new}^T)⁻¹ y_{Δ,new}
#  Final:  θ_active = θ_anchor + Δμ_new
# =========================================================================
@torch.no_grad()
def srrhuif_step_fv_error(filter_state, ctx, batch, h_idx, sp, cfg, fv_cache):
    """
    Error-state SRRHUIF (Information form) — one horizon h-step.
    
    Args:
        filter_state: None at h=0, else dict {'mu_delta', 'S_Y_delta'} from h-1
        ctx:        from init_error_horizon (anchor, Y_cache, etc.)
        batch:      batch_hist[h]
        h_idx:      horizon step index (0..N-1)
    
    Returns:
        theta_active [n_x, 1], filter_state_new (dict), loss, target_var, k_gain_norm, dbg
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    eye_n = fv_cache.eye_n
    
    theta_anchor = ctx['theta_anchor']      # [n_x] frozen
    Y_cache = ctx['Y_cache']                # [N, B] or None
    p_delta_init_val = ctx['p_delta_init']  # float
    
    is_first = (h_idx == 0) or (filter_state is None)
    
    # ── Prior 결정 ───────────────────────────────────────────────────
    if is_first:
        mu_delta_prev = torch.zeros(n_x, dtype=DTYPE, device=device)
        # P_Δ⁻ = p_delta_init·I  →  P_sqrt⁻ = √p_delta_init·I
        P_sqrt_prev_delta = float(np.sqrt(p_delta_init_val)) * eye_n
    else:
        mu_delta_prev = filter_state['mu_delta']
        S_Y_delta_prev = filter_state['S_Y_delta']
        # P_sqrt = inv(S_Y) (lower-tri)
        P_sqrt_prev_delta = safe_inv_tril_batch(
            S_Y_delta_prev.unsqueeze(0), eye_n.unsqueeze(0)
        ).squeeze(0)
    
    # ── Time update: P_Δ_pred = P_Δ_prev + Q_proc ───────────────────
    S_Q = cfg.q_init * eye_n
    P_sqrt_pred_delta = tria_operation_batch(
        torch.cat([P_sqrt_prev_delta, S_Q], dim=1).unsqueeze(0)
    ).squeeze(0)  # [n_x, n_x]
    
    # S_Y_pred = inv(P_sqrt_pred); Y_pred = S_Y_pred · S_Y_pred^T
    S_Y_pred_delta = safe_inv_tril_batch(
        P_sqrt_pred_delta.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)
    Y_pred_delta = S_Y_pred_delta @ S_Y_pred_delta.t()
    y_pred_delta = Y_pred_delta @ mu_delta_prev.unsqueeze(-1)  # [n_x, 1]
    
    # ── Sigma points in error space, project to absolute ────────────
    scaled_P = fv_cache.gamma_sigma * P_sqrt_pred_delta  # [n_x, n_x]
    unified = fv_cache.unified_thetas  # [num_sigma, n_x]
    # 시그마는 에러공간 Δμ_prev 중심, forward 시 anchor 추가
    theta_center = theta_anchor + mu_delta_prev  # [n_x]
    unified[0] = theta_center.to(DTYPE_FWD)
    unified[1:n_x+1] = (theta_center.unsqueeze(0) + scaled_P.t()).to(DTYPE_FWD)
    unified[n_x+1:] = (theta_center.unsqueeze(0) - scaled_P.t()).to(DTYPE_FWD)
    
    # ── Forward ─────────────────────────────────────────────────────
    s_batch = batch['s'].t()  # [B, dim_s]
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
    
    Q_all_f32 = forward_bmm(unified, info, s_batch)  # [num_sigma, nA, B]
    Z_sigma_T_f32 = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)]
    Z_sigma_T = Z_sigma_T_f32.to(DTYPE)  # [num_sigma, B]
    
    # ── [v9+] Y_target + Z_sigma_T 변형 (mode-dispatched) ───────────
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    s_next_for_y = batch['s_next']  # [dim_s, B]
    if sp.get('normalizer'):
        s_next_for_y = sp['normalizer'].normalize(s_next_for_y)
    # forward_bmm은 [B, dim_s] 입력 기대; forward_single은 [dim_s, B] 그대로
    s_next_bmm = s_next_for_y.t()  # [B, dim_s] — for sigma point batched forward
    
    if Y_cache is not None:
        # 캐싱 경로 (online_frozen / target / spas) — a_best는 ctx에서 가져옴
        a_best_for_step = ctx['a_best_per_step'][h_idx]  # [B]
        
        if cfg.measurement_mode == 'q_target':
            # 기존 동작: Z_sigma_T 그대로, z_measured는 Y_cache 사용
            z_measured = Y_cache[h_idx].view(-1, 1).to(DTYPE)
        elif cfg.measurement_mode == 'pure_reward':
            # Z_sigma_T 차분 적용: 새 sigma forward 필요
            Z_sigma_T, z_measured, _ = _resolve_measurement(
                Q_sigma_at_s_a=Z_sigma_T,
                unified_sigma=unified, info=info, s_next=s_next_bmm,
                a_best_next=a_best_for_step, Q_tgt_next=None,
                reward=batch['r'], term_mask=batch['term'],
                target_gamma=target_gamma, device=device,
                Q_sigma_next_cache=None, fwd_fn=forward_bmm,
            )
        else:
            raise RuntimeError(f"measurement_mode={cfg.measurement_mode}")
    else:
        # 'online_moving': h=0은 cfg.h0_online_moving_init에 따라, h≥1은 θ_anchor+Δμ_prev
        Q_tgt = forward_single(ctx['theta_target_ref'], info, s_next_for_y).to(DTYPE)  # [nA, B]
        if is_first:
            h0_init = cfg.h0_online_moving_init
            if h0_init == 'theta_target':
                a_best = Q_tgt.argmax(dim=0)
            elif h0_init == 'spas':
                # unified는 이미 sigma points로 채워진 상태 (theta_anchor 중심)
                Q_sigma_h0 = forward_bmm(unified, info, s_next_bmm)  # [num_sigma, nA, B]
                a_best = Q_sigma_h0.mean(dim=0).argmax(dim=0)
            else:  # 'prev_est': horizon 직전 active θ
                Q_prev = forward_single(ctx['theta_active_ref'], info, s_next_for_y).to(DTYPE)
                a_best = Q_prev.argmax(dim=0)
        else:
            theta_current = theta_anchor + mu_delta_prev
            Q_curr = forward_single(theta_current, info, s_next_for_y).to(DTYPE)
            a_best = Q_curr.argmax(dim=0)
        Z_sigma_T, z_measured, _ = _resolve_measurement(
            Q_sigma_at_s_a=Z_sigma_T,
            unified_sigma=unified, info=info, s_next=s_next_bmm,
            a_best_next=a_best, Q_tgt_next=Q_tgt,
            reward=batch['r'], term_mask=batch['term'],
            target_gamma=target_gamma, device=device,
            Q_sigma_next_cache=None, fwd_fn=forward_bmm,
        )

    # z_hat (FINAL Z_sigma_T로 계산)
    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [B, 1]

    target_var = torch.var(z_measured).item()

    # ── Statistical linearization → H^T ─────────────────────────────
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x+1] = scaled_P.t()
    X_dev[n_x+1:] = -scaled_P.t()
    Z_dev = Z_sigma_T - z_hat.t()  # [num_sigma, B]
    P_xz_delta = (X_dev * fv_cache.Wc.view(-1, 1)).t() @ Z_dev  # [n_x, B]
    H_T = Y_pred_delta @ P_xz_delta  # [n_x, B]
    
    residual = z_measured - z_hat  # [B, 1]
    loss = torch.mean(residual ** 2)
    
    # ── Huber-adaptive R ────────────────────────────────────────────
    current_r_std = sp.get('current_r_std', cfg.r_init)
    r_inv = 1.0 / (current_r_std ** 2)
    r_inv_sqrt = 1.0 / current_r_std
    res_abs = torch.abs(residual)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)  # [B, 1]
    r_inv_adapt = r_inv / adapt_factor  # [B, 1]
    r_inv_sqrt_adapt = (r_inv_sqrt / torch.sqrt(adapt_factor)).t()  # [1, B]
    
    # ── Information form measurement update ─────────────────────────
    # S_Y_new = QR-tria([S_Y_pred | H^T·r_inv_sqrt_adapt | √λ·I])
    tikhonov_sqrt = float(np.sqrt(cfg.tikhonov_lambda))
    if cfg.tikhonov_lambda > 0:
        combined = torch.cat([S_Y_pred_delta, H_T * r_inv_sqrt_adapt, tikhonov_sqrt * eye_n], dim=1)
    else:
        combined = torch.cat([S_Y_pred_delta, H_T * r_inv_sqrt_adapt], dim=1)
    S_Y_delta_new = tria_operation_batch(combined.unsqueeze(0)).squeeze(0)
    
    # Pseudo-measurement innovation in error-state info form:
    #   z̃ = residual + H^T·Δμ⁻
    # At h=0: Δμ⁻=0 ⇒ z̃ = residual (깔끔)
    ht_mu_delta = H_T.t() @ mu_delta_prev.unsqueeze(-1)  # [B, 1]
    innov = residual + ht_mu_delta
    y_delta_new = y_pred_delta + H_T @ (r_inv_adapt * innov)
    
    # Recover Δμ from info form: Δμ = inv(Y_new) · y_new
    mu_delta_new_col = robust_solve_spd_batch(
        S_Y_delta_new.unsqueeze(0), y_delta_new.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)  # [n_x, 1]
    mu_delta_new = mu_delta_new_col.squeeze(-1)
    
    if not torch.isfinite(mu_delta_new).all():
        mu_delta_new = mu_delta_prev.clone()
    
    # ── Final θ_active = θ_anchor + Δμ_new ───────────────────────────
    theta_active = (theta_anchor + mu_delta_new).view(-1, 1)
    
    # ── Diagnostics (기존 dbg dict 호환) ────────────────────────────
    k_gain = H_T * r_inv_sqrt_adapt
    k_gain_norm = torch.norm(k_gain).item()
    Y_new = S_Y_delta_new @ S_Y_delta_new.t()
    Y_diag = torch.diagonal(Y_new)
    avg_P = (1.0 / (Y_diag + 1e-8)).mean().item()
    
    innov_abs = torch.abs(innov)
    resid_abs = torch.abs(residual)
    ht_mu_abs = torch.abs(ht_mu_delta)
    delta_correction_norm = torch.norm(mu_delta_new - mu_delta_prev).item()
    
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'innov_norm': innov_abs.mean().item(),
        'resid_in_innov': resid_abs.mean().item(),
        'ht_theta_in_innov': ht_mu_abs.mean().item(),  # error-state: H^T·Δμ_prev
        'avg_P': avg_P,
        'ht_norm': torch.norm(H_T).item(),
        'resid_norm': torch.norm(residual).item(),
        'delta_y': torch.norm(H_T @ (r_inv_adapt * innov)).item(),
        'y_pred_norm': torch.norm(y_pred_delta).item(),
        'y_new_norm': torch.norm(y_delta_new).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        # FV: 전체 벡터를 네트워크 레이어 구간으로 분해 (가로=layer, 세로=horizon 정밀 진단)
        'per_layer_ht': fv_per_layer(info, H_T, 'norm'),
        'per_layer_delta': fv_per_layer(info, mu_delta_new - mu_delta_prev, 'norm'),
        'per_layer_resid_max': fv_broadcast(info, resid_abs.max().item()),  # 측정-공간 전역
        'per_layer_cond': fv_broadcast(info, 1.0),
        'per_layer_ymax': fv_per_layer(info, y_delta_new, 'maxabs'),  # per-layer 정보벡터 max
        'per_layer_cond_full': fv_broadcast(info, 1.0),
        # error-state 전용 추가
        'mu_delta_norm': torch.norm(mu_delta_new).item(),
    }
    
    filter_state_new = {
        'mu_delta': mu_delta_new,
        'S_Y_delta': S_Y_delta_new,
    }

    if sp.get('_do_sigma_spread', False):
        dbg['sigma_spread'] = compute_sigma_spread(unified, info, s_batch)
    return theta_active, filter_state_new, loss.item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8f. [v7] Error-State RHUKF — Full Vector, Covariance Form
#
#  상태:   Δμ ∈ R^{n_x}, P_Δ ∈ R^{n_x × n_x} (full symmetric)
#  Prior(h=0): Δμ⁻ = 0, P_Δ⁻ = p_delta_init·I
#  Anchor: ctx['theta_anchor'] (호라이즌 동안 frozen)
#  Sigma:  χ^(Δ)_j = Δμ⁻ ± γ·chol(P_Δ_pred) (error space)
#          forward 시 projection: θ_anchor + χ^(Δ)_j (절대공간)
#  Update: Δμ_new = Δμ⁻ + K·(Y_cache[h] - ẑ),  P_Δ_new = P_Δ_pred - K·P_zz·K^T
#  Final:  θ_active = θ_anchor + Δμ_new
# =========================================================================
@torch.no_grad()
def rhukf_step_fv_error(filter_state, ctx, batch, h_idx, sp, cfg, fv_cache):
    """
    Error-state RHUKF (Covariance form) — one horizon h-step.
    
    Args:
        filter_state: None at h=0, else dict {'mu_delta', 'P_delta'} from h-1
        ctx:        from init_error_horizon
        batch:      batch_hist[h]
        h_idx:      horizon step index
    
    Returns:
        theta_active [n_x, 1], filter_state_new (dict), loss, target_var, k_gain_norm, dbg
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    eye_n = fv_cache.eye_n
    
    theta_anchor = ctx['theta_anchor']
    Y_cache = ctx['Y_cache']
    p_delta_init_val = ctx['p_delta_init']
    
    is_first = (h_idx == 0) or (filter_state is None)
    
    # ── Prior ───────────────────────────────────────────────────────
    if is_first:
        mu_delta_prev = torch.zeros(n_x, dtype=DTYPE, device=device)
        P_delta_prev = p_delta_init_val * eye_n
    else:
        mu_delta_prev = filter_state['mu_delta']
        P_delta_prev = filter_state['P_delta']
    
    # ── Time update ─────────────────────────────────────────────────
    Q_proc = (cfg.q_init ** 2) * eye_n
    P_delta_pred = P_delta_prev + Q_proc
    P_delta_pred = 0.5 * (P_delta_pred + P_delta_pred.t())
    
    # ── Sigma in error space ────────────────────────────────────────
    S_P_pred = safe_cholesky_fallback(P_delta_pred, eye_n, JITTER_TRIA)
    
    scaled_P = fv_cache.gamma_sigma * S_P_pred
    unified = fv_cache.unified_thetas
    theta_center = theta_anchor + mu_delta_prev  # absolute center
    unified[0] = theta_center.to(DTYPE_FWD)
    unified[1:n_x+1] = (theta_center.unsqueeze(0) + scaled_P.t()).to(DTYPE_FWD)
    unified[n_x+1:] = (theta_center.unsqueeze(0) - scaled_P.t()).to(DTYPE_FWD)
    
    # ── Forward ─────────────────────────────────────────────────────
    s_batch = batch['s'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
    
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    Z_sigma_T_f32 = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)]
    Z_sigma_T = Z_sigma_T_f32.to(DTYPE)
    
    # ── [v9+] Y target + Z_sigma_T (mode-dispatched) ───────────────
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    s_next_for_y = batch['s_next']
    if sp.get('normalizer'):
        s_next_for_y = sp['normalizer'].normalize(s_next_for_y)
    # forward_bmm은 [B, dim_s] 기대
    s_next_bmm = s_next_for_y.t()
    
    if Y_cache is not None:
        a_best_for_step = ctx['a_best_per_step'][h_idx]
        if cfg.measurement_mode == 'q_target':
            z_measured = Y_cache[h_idx].view(-1, 1).to(DTYPE)
        elif cfg.measurement_mode == 'pure_reward':
            Z_sigma_T, z_measured, _ = _resolve_measurement(
                Q_sigma_at_s_a=Z_sigma_T,
                unified_sigma=unified, info=info, s_next=s_next_bmm,
                a_best_next=a_best_for_step, Q_tgt_next=None,
                reward=batch['r'], term_mask=batch['term'],
                target_gamma=target_gamma, device=device,
                Q_sigma_next_cache=None, fwd_fn=forward_bmm,
            )
        else:
            raise RuntimeError(f"measurement_mode={cfg.measurement_mode}")
    else:
        # 'online_moving': h=0은 cfg.h0_online_moving_init에 따라, h≥1은 θ_anchor+Δμ_prev
        Q_tgt = forward_single(ctx['theta_target_ref'], info, s_next_for_y).to(DTYPE)
        if is_first:
            h0_init = cfg.h0_online_moving_init
            if h0_init == 'theta_target':
                a_best = Q_tgt.argmax(dim=0)
            elif h0_init == 'spas':
                Q_sigma_h0 = forward_bmm(unified, info, s_next_bmm)  # [num_sigma, nA, B]
                a_best = Q_sigma_h0.mean(dim=0).argmax(dim=0)
            else:  # 'prev_est'
                Q_prev = forward_single(ctx['theta_active_ref'], info, s_next_for_y).to(DTYPE)
                a_best = Q_prev.argmax(dim=0)
        else:
            theta_current = theta_anchor + mu_delta_prev
            Q_curr = forward_single(theta_current, info, s_next_for_y).to(DTYPE)
            a_best = Q_curr.argmax(dim=0)
        Z_sigma_T, z_measured, _ = _resolve_measurement(
            Q_sigma_at_s_a=Z_sigma_T,
            unified_sigma=unified, info=info, s_next=s_next_bmm,
            a_best_next=a_best, Q_tgt_next=Q_tgt,
            reward=batch['r'], term_mask=batch['term'],
            target_gamma=target_gamma, device=device,
            Q_sigma_next_cache=None, fwd_fn=forward_bmm,
        )
    
    # z_hat (FINAL Z_sigma_T)
    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [B, 1]
    
    target_var = torch.var(z_measured).item()
    residual = z_measured - z_hat  # [B, 1]
    loss = torch.mean(residual ** 2)
    
    # ── Cross-cov in error space, P_zz ──────────────────────────────
    Wc_col = fv_cache.Wc.view(-1, 1)
    Z_dev = Z_sigma_T - z_hat.t()  # [num_sigma, B]
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x+1] = scaled_P.t()
    X_dev[n_x+1:] = -scaled_P.t()
    
    P_zz_sigma = Z_dev.t() @ (Wc_col * Z_dev)            # [B, B]
    P_delta_z = X_dev.t() @ (Wc_col * Z_dev)             # [n_x, B]
    
    # Huber-adaptive R (per-sample variance inflation)
    res_abs = torch.abs(residual).squeeze(-1)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    R_diag_eff = (current_r_std ** 2) * adapt_factor
    P_zz = P_zz_sigma + torch.diag(R_diag_eff)
    P_zz = 0.5 * (P_zz + P_zz.t())
    
    # ── Kalman gain ─────────────────────────────────────────────────
    eye_batch = torch.eye(batch_sz, dtype=DTYPE, device=device)
    L_zz = safe_cholesky_fallback(P_zz, eye_batch)
    
    tmp = torch.linalg.solve_triangular(L_zz, P_delta_z.t(), upper=False)
    K_t = torch.linalg.solve_triangular(L_zz.t(), tmp, upper=True)
    K = K_t.t()  # [n_x, B]
    
    # ── State update in error space ─────────────────────────────────
    mu_delta_new = mu_delta_prev + (K @ residual).squeeze(-1)
    if not torch.isfinite(mu_delta_new).all():
        mu_delta_new = mu_delta_prev.clone()
    
    # Covariance update: P_new = P_pred - K·P_zz·K^T
    K_L = K @ L_zz
    P_delta_new = P_delta_pred - K_L @ K_L.t()
    P_delta_new = 0.5 * (P_delta_new + P_delta_new.t())
    if cfg.tikhonov_lambda > 0:
        P_delta_new = P_delta_new + cfg.tikhonov_lambda * eye_n
    
    # ── Final ───────────────────────────────────────────────────────
    theta_active = (theta_anchor + mu_delta_new).view(-1, 1)
    
    # ── Diagnostics ────────────────────────────────────────────────
    P_diag = torch.diagonal(P_delta_new)
    avg_P = P_diag.mean().item()
    max_P = P_diag.max().item()
    L_zz_diag = torch.diagonal(L_zz)
    cond_P_zz = ((L_zz_diag.max() / L_zz_diag.min().clamp(min=1e-12)) ** 2).item()
    k_gain_norm = torch.norm(K).item()
    delta_correction_norm = torch.norm(mu_delta_new - mu_delta_prev).item()
    innov_abs = torch.abs(residual)
    
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'innov_norm': innov_abs.mean().item(),
        'resid_in_innov': innov_abs.mean().item(),
        'ht_theta_in_innov': 0.0,  # KF form: 항상 0 (정보형의 H^T·θ_pred 항 없음)
        'avg_P': avg_P,
        'max_P': max_P,
        'ht_norm': torch.norm(P_delta_z).item(),
        'resid_norm': torch.norm(residual).item(),
        'delta_y': torch.norm(K @ residual).item(),
        'y_pred_norm': torch.norm(mu_delta_prev).item(),  # error-state: Δμ_prev norm
        'y_new_norm': torch.norm(mu_delta_new).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        # FV: 전체 벡터를 네트워크 레이어 구간으로 분해 (가로=layer, 세로=horizon 정밀 진단)
        'per_layer_ht': fv_per_layer(info, P_delta_z, 'norm'),  # ||P_xz|| 행 = 레이어별 측정-상태 민감도
        'per_layer_delta': fv_per_layer(info, mu_delta_new - mu_delta_prev, 'norm'),
        'per_layer_resid_max': fv_broadcast(info, innov_abs.max().item()),  # 측정-공간 전역
        'per_layer_cond': fv_broadcast(info, cond_P_zz),  # innovation cov 조건수, 측정-공간 전역
        'per_layer_ymax': fv_per_layer(info, P_diag, 'maxabs'),  # per-layer max diag(P) 불확실성
        'per_layer_cond_full': fv_broadcast(info, cond_P_zz),
        'mu_delta_norm': torch.norm(mu_delta_new).item(),
    }
    
    filter_state_new = {
        'mu_delta': mu_delta_new,
        'P_delta': P_delta_new,
    }

    if sp.get('_do_sigma_spread', False):
        dbg['sigma_spread'] = compute_sigma_spread(unified, info, s_batch)
    return theta_active, filter_state_new, loss.item(), target_var, k_gain_norm, dbg


def _split_anchor_per_layer(theta_anchor, info, decoupling_mode):
    """Returns list of [num_blocks, param_len_per_block] anchor tensors per filter layer."""
    anchor_per_L = []
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        if decoupling_mode == 'node':
            W = theta_anchor[fl['W_start']:fl['W_start']+fl['W_len']].view(fl['fan_out'], fl['fan_in'])
            b = theta_anchor[fl['b_start']:fl['b_start']+fl['b_len']]
            anchor_L = torch.cat([W, b.unsqueeze(1)], dim=1)  # [fan_out, fan_in+1]
        else:  # 'layer'
            anchor_L = theta_anchor[fl['W_start']:fl['W_start']+fl['param_len']].unsqueeze(0)  # [1, param_len]
        anchor_per_L.append(anchor_L)
    return anchor_per_L


def _compose_theta_from_delta(theta_anchor, mu_delta_per_L, info, decoupling_mode):
    """
    Compose absolute θ = θ_anchor + Δμ (with Δμ scattered back to flat layout).
    mu_delta_per_L: list of [num_blocks, param_len_per_block] tensors.
    """
    theta = theta_anchor.clone()
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        mu_L = mu_delta_per_L[L]
        if decoupling_mode == 'node':
            # mu_L: [fan_out, fan_in+1]
            W_delta = mu_L[:, :fl['fan_in']]                # [fan_out, fan_in]
            b_delta = mu_L[:, fl['fan_in']]                  # [fan_out]
            theta[fl['W_start']:fl['W_start']+fl['W_len']] += W_delta.reshape(-1)
            theta[fl['b_start']:fl['b_start']+fl['b_len']] += b_delta
        else:  # 'layer'
            theta[fl['W_start']:fl['W_start']+fl['param_len']] += mu_L[0]
    return theta


@torch.no_grad()
def srrhuif_step_error(filter_state, ctx, batch, h_idx, sp, cfg, f_cache):
    """
    Error-state SRRHUIF (Information form), Node/Layer Decoupled.
    
    filter_state:
      - h=0 또는 None: 각 block마다 Δμ=0, P_Δ=p_delta_init·I로 초기화
      - h≥1: dict {'S_Y_delta': [list per layer], 'mu_delta': [list per layer]}
    ctx: init_error_horizon() 결과
    
    Returns:
      theta_active [n_x, 1], filter_state_new (dict), loss, target_var, k_gain_norm, dbg
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    is_first = (h_idx == 0) or (filter_state is None)
    
    theta_anchor = ctx['theta_anchor'].detach()  # [n_x] frozen
    Y_cache = ctx['Y_cache']
    p_delta_init_val = ctx['p_delta_init']
    
    # ── Anchor split per layer (frozen) ──────────────────────────
    anchor_per_L = _split_anchor_per_layer(theta_anchor, info, cfg.decoupling_mode)
    
    # ── Δμ_prev, P_sqrt_prev per layer ────────────────────────────
    if is_first:
        mu_delta_prev_per_L = [torch.zeros_like(a) for a in anchor_per_L]
        S_Y_delta_prev_per_L = None  # generate from p_delta_init on the fly
    else:
        mu_delta_prev_per_L = filter_state['mu_delta']
        S_Y_delta_prev_per_L = filter_state['S_Y_delta']
    
    # ── Compose theta_current = anchor + Δμ_prev (composed) ─────
    # 호라이즌 base. 호라이즌 안에서 OTHER layers 그대로 유지 (frozen anchor + 누적 Δμ).
    theta_current_flat = _compose_theta_from_delta(
        theta_anchor, mu_delta_prev_per_L, info, cfg.decoupling_mode)
    
    # ── Y target ────────────────────────────────────────────────
    if Y_cache is not None:
        z_measured = Y_cache[h_idx].view(-1, 1).to(DTYPE)
    else:
        # 'online_moving': h=0은 cfg.h0_online_moving_init에 따라, h≥1은 composed theta
        s_next = batch['s_next']  # [dim_s, B]
        if sp.get('normalizer'):
            s_next = sp['normalizer'].normalize(s_next)
        Q_tgt = forward_single(ctx['theta_target_ref'], info, s_next).to(DTYPE)
        if is_first:
            h0_init = cfg.h0_online_moving_init
            if h0_init == 'theta_target':
                a_best = Q_tgt.argmax(dim=0)
            else:  # 'prev_est' (spas는 FV 전용, node/layer에서는 prev_est로 동작)
                Q_prev = forward_single(ctx['theta_active_ref'], info, s_next).to(DTYPE)
                a_best = Q_prev.argmax(dim=0)
        else:
            Q_curr = forward_single(theta_current_flat, info, s_next).to(DTYPE)
            a_best = Q_curr.argmax(dim=0)
        q_val_next = Q_tgt[a_best, torch.arange(batch_sz, device=device)]
        target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
        z_measured = (batch['r'] + target_gamma * (1 - batch['term']) * q_val_next).view(-1, 1).to(DTYPE)
    
    target_var = torch.var(z_measured).item()
    
    # ── Per-layer setup ─────────────────────────────────────────
    per_layer = {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        lc = f_cache.get(L)
        mu_delta_L = mu_delta_prev_per_L[L]  # [num_blocks, param_len]
        
        if is_first:
            P_sqrt_prev_L = float(np.sqrt(p_delta_init_val)) * lc['eye_block_batch'].clone()
        else:
            S_3d_L = S_Y_delta_prev_per_L[L]
            P_sqrt_prev_L = safe_inv_tril_batch(S_3d_L.permute(2, 0, 1), lc['eye_block_batch'])
        
        per_layer[L] = {
            'fl': fl, 'lc': lc,
            'theta_anchor_L': anchor_per_L[L],
            'mu_delta_L_3d': mu_delta_L.unsqueeze(-1),  # [num_blocks, param_len, 1]
            'mu_delta_L_2d': mu_delta_L,
            'P_sqrt_prev_L': P_sqrt_prev_L,
        }
    
    # ── Sample data setup ───────────────────────────────────────
    s_batch = batch['s'].t()  # [B, dim_s]
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
    
    current_q_std = sp.get('current_q_std', cfg.q_init)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    current_r_inv_sqrt = 1.0 / current_r_std
    current_r_inv = 1.0 / (current_r_std ** 2)
    
    # ── Block-group time update (sigma in ERROR space) ─────────
    # _time_update_core을 Δμ에 적용: theta_3d=Δμ_prev → 출력 sigma는 error space, 
    # y_pred = Y_pred @ Δμ_prev (h=0에선 0).
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        
        all_mu_delta_3d = torch.cat([per_layer[L]['mu_delta_L_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev_L'] for L in layers_in_grp], dim=0)
        dynamic_S_Q = current_q_std * grp['eye_grouped']
        
        S_pred_g, _, y_pred_g, X_sigma_err_g, scaled_P_g = _time_update_core(
            all_mu_delta_3d, all_P_sqrt, dynamic_S_Q, grp['eye_grouped'], grp['gamma'])
        
        for i, L in enumerate(layers_in_grp):
            s_idx, e_idx = offsets[i], offsets[i + 1]
            per_layer[L]['S_pred'] = S_pred_g[s_idx:e_idx]
            per_layer[L]['y_pred'] = y_pred_g[s_idx:e_idx]
            per_layer[L]['X_sigma_err'] = X_sigma_err_g[s_idx:e_idx]  # error space
            per_layer[L]['scaled_P'] = scaled_P_g[s_idx:e_idx]
    
    # ── Sigma scatter: error → absolute (add anchor) ───────────
    unified = f_cache.unified_thetas
    # [v7+] OTHER 레이어들의 base: 'current' = anchor + composed Δμ_prev (기존),
    #                              'prior'   = anchor만 (Δμ 무시)
    if cfg.node_layer_other_source == 'prior':
        unified[:] = theta_anchor.to(DTYPE_FWD)
    else:  # 'current'
        unified[:] = theta_current_flat.to(DTYPE_FWD)
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        X_sigma_err_f32 = pl['X_sigma_err'].to(DTYPE_FWD)  # [num_blocks, num_sigma, param_len]
        # Convert to absolute: anchor_L + sigma_error (broadcast anchor over sigma dim)
        anchor_L_3d_f32 = pl['theta_anchor_L'].unsqueeze(1).to(DTYPE_FWD)  # [num_blocks, 1, param_len]
        X_sigma_abs_f32 = anchor_L_3d_f32 + X_sigma_err_f32
        
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        if cfg.decoupling_mode == 'node':
            layer_view = unified[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], -1)
            layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_abs_f32[:, :, :fl['fan_in']])
            layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_abs_f32[:, :, fl['fan_in']:fl['fan_in']+1])
        else:
            unified[fwd_start:fwd_end, fl['W_start']:fl['W_start'] + fl['param_len']] = X_sigma_abs_f32[0]
    
    # ── Forward through unified sigma copies ────────────────────
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    # ── Per-layer H^T, residual ─────────────────────────────────
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
        z_measured_exp = z_measured.unsqueeze(0).expand(lc['num_blocks'], -1, -1)
        
        HT_all, residual_all, z_hat, ht_norm, resid_norm = _compute_ht_core(
            Z_sigma_T_f32, lc['Wm_col_f32'], lc['Wc_f32'], lc['zero_col_f32'],
            pl['scaled_P'].to(DTYPE_FWD), z_measured_exp, pl['S_pred'])
        
        per_layer[L]['HT_all'] = HT_all
        per_layer[L]['residual_all'] = residual_all
        per_layer[L]['loss'] = torch.mean(residual_all ** 2)
        per_layer[L]['ht_norm'] = ht_norm
        per_layer[L]['resid_norm'] = resid_norm
        per_layer[L]['resid_max'] = torch.max(torch.abs(residual_all)).item()
    
    # ── Block-group measurement update on Δμ ───────────────────
    new_S_Y_delta_dict = {}
    new_mu_delta_dict = {}
    total_loss = 0.0
    layer_count = info['num_filter_layers']
    
    total_innov_mean = total_innov_max = 0.0
    total_ht_norm = total_resid_norm = 0.0
    total_delta_y = total_y_new = total_avg_P = 0.0
    total_resid_in_innov = total_ht_theta_in_innov = 0.0
    total_innov_norm = total_y_pred_norm = total_adapt_ratio = 0.0
    group_count = 0
    per_layer_cond, per_layer_ymax, per_layer_cond_full = {}, {}, {}
    
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        
        all_S_pred = torch.cat([per_layer[L]['S_pred'] for L in layers_in_grp], dim=0)
        all_y_pred = torch.cat([per_layer[L]['y_pred'] for L in layers_in_grp], dim=0)
        all_HT = torch.cat([per_layer[L]['HT_all'] for L in layers_in_grp], dim=0)
        all_mu_delta_3d = torch.cat([per_layer[L]['mu_delta_L_3d'] for L in layers_in_grp], dim=0)
        all_residual = torch.cat([per_layer[L]['residual_all'] for L in layers_in_grp], dim=0)
        
        # _meas_update_core: theta_3d 자리에 Δμ_3d 들어감 → 출력 theta_new = Δμ_new
        # 수식 동일 (info form: y_new = y_pred + H R⁻¹ (resid + H^T·Δμ_prev), recover Δμ)
        mu_delta_new_g, S_new_g, meas_stats = _meas_update_core(
            all_S_pred, all_y_pred, all_HT, all_mu_delta_3d,
            all_residual, current_r_inv_sqrt, current_r_inv, grp['eye_grouped'],
            tikhonov_lambda=cfg.tikhonov_lambda, huber_c=cfg.huber_c)
        
        total_innov_mean += meas_stats['innov_mean']
        total_innov_max = max(total_innov_max, meas_stats['innov_max'])
        total_delta_y += meas_stats['delta_y']
        total_y_new += meas_stats['y_new_norm']
        total_avg_P += meas_stats['avg_P']
        total_resid_in_innov += meas_stats['resid_in_innov']
        total_ht_theta_in_innov += meas_stats['ht_theta_in_innov']
        total_innov_norm += meas_stats['innov_norm']
        total_y_pred_norm += meas_stats['y_pred_norm']
        total_adapt_ratio += meas_stats['adapt_ratio']
        group_count += 1
        
        for L in layers_in_grp:
            total_ht_norm += per_layer[L]['ht_norm']
            total_resid_norm += per_layer[L]['resid_norm']
        
        for i, L in enumerate(layers_in_grp):
            s_idx, e_idx = offsets[i], offsets[i + 1]
            pl = per_layer[L]
            fl = pl['fl']
            mu_delta_new_L_3d = mu_delta_new_g[s_idx:e_idx]  # [num_blocks, param_len, 1]
            S_new_L = S_new_g[s_idx:e_idx]
            
            # NaN check
            invalid = ~torch.isfinite(mu_delta_new_L_3d).all(dim=(1, 2))
            if invalid.any():
                mu_delta_new_L_3d[invalid] = pl['mu_delta_L_3d'][invalid]
            
            mu_delta_new_L = mu_delta_new_L_3d.squeeze(-1)  # [num_blocks, param_len]
            
            # max_layer_step: bound per-h-step change ||Δμ_new - Δμ_prev||
            if cfg.max_layer_step > 0:
                if cfg.decoupling_mode == 'node':
                    dW = mu_delta_new_L[:, :fl['fan_in']] - pl['mu_delta_L_2d'][:, :fl['fan_in']]
                    db = mu_delta_new_L[:, fl['fan_in']] - pl['mu_delta_L_2d'][:, fl['fan_in']]
                    step_norm = torch.sqrt(torch.norm(dW)**2 + torch.norm(db)**2)
                else:
                    step_norm = torch.norm(mu_delta_new_L - pl['mu_delta_L_2d'])
                if step_norm > cfg.max_layer_step:
                    scale = cfg.max_layer_step / (step_norm + 1e-8)
                    mu_delta_new_L = pl['mu_delta_L_2d'] + (mu_delta_new_L - pl['mu_delta_L_2d']) * scale
            
            new_mu_delta_dict[L] = mu_delta_new_L
            new_S_Y_delta_dict[L] = S_new_L.permute(1, 2, 0)
            total_loss += pl['loss']
            
            if cfg.diag_horizon_cond:
                label = f"{fl['type'][0].upper()}{fl['local_idx']}"
                cond_val, ymax_val, _, _ = compute_pseudo_cond_from_S(S_new_L)
                per_layer_cond[label] = cond_val
                per_layer_ymax[label] = ymax_val
                if cfg.use_full_eigvalsh:
                    full_cond, _ = compute_full_cond_from_S(S_new_L)
                    per_layer_cond_full[label] = full_cond
    
    # ── Compose final theta_active = anchor + Δμ_new ────────────
    new_mu_delta_per_L = [new_mu_delta_dict[L] for L in range(info['num_filter_layers'])]
    theta_active_flat = _compose_theta_from_delta(
        theta_anchor, new_mu_delta_per_L, info, cfg.decoupling_mode)
    
    # max_k_gain: bound total ||Δμ_new - Δμ_prev|| (호라이즌-step 변동)
    delta_change = theta_active_flat - theta_current_flat
    k_gain_norm = torch.norm(delta_change).item()
    if cfg.max_k_gain > 0 and k_gain_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / k_gain_norm
        theta_active_flat = theta_current_flat + delta_change * scale
        # Update mu_delta_dict to reflect clamping
        scaled_mu_delta_per_L = []
        for L in range(info['num_filter_layers']):
            mu_prev = mu_delta_prev_per_L[L]
            mu_new = new_mu_delta_per_L[L]
            scaled_mu_delta_per_L.append(mu_prev + (mu_new - mu_prev) * scale)
        new_mu_delta_per_L = scaled_mu_delta_per_L
        for L in range(info['num_filter_layers']):
            new_mu_delta_dict[L] = new_mu_delta_per_L[L]
        k_gain_norm = cfg.max_k_gain
    
    theta_active = theta_active_flat.view(-1, 1)
    
    # ── Per-layer diagnostics ───────────────────────────────────
    per_layer_ht_dict, per_layer_delta_dict, per_layer_resid_max_dict = {}, {}, {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        per_layer_ht_dict[label] = per_layer[L]['ht_norm']
        per_layer_resid_max_dict[label] = per_layer[L]['resid_max']
        # per_layer_delta: ||Δμ_new - Δμ_prev|| (이 h-step의 변화량)
        delta_change_L = new_mu_delta_dict[L] - per_layer[L]['mu_delta_L_2d']
        per_layer_delta_dict[label] = torch.norm(delta_change_L).item()
    
    gc = max(group_count, 1)
    dbg = {
        'innov_mean': total_innov_mean / gc,
        'innov_max': total_innov_max,
        'ht_norm': total_ht_norm / layer_count,
        'resid_norm': total_resid_norm / layer_count,
        'delta_y': total_delta_y / gc,
        'y_pred_norm': total_y_pred_norm / gc,
        'y_new': total_y_new / gc,
        'avg_P': total_avg_P / gc,
        'resid_in_innov': total_resid_in_innov / gc,
        'ht_theta_in_innov': total_ht_theta_in_innov / gc,
        'innov_norm': total_innov_norm / gc,
        'per_layer_ht': per_layer_ht_dict,
        'per_layer_delta': per_layer_delta_dict,
        'per_layer_resid_max': per_layer_resid_max_dict,
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond_full,
        'adapt_ratio': total_adapt_ratio / gc,
    }
    
    filter_state_new = {
        'S_Y_delta': [new_S_Y_delta_dict[L] for L in range(info['num_filter_layers'])],
        'mu_delta': [new_mu_delta_dict[L] for L in range(info['num_filter_layers'])],
    }
    
    return theta_active, filter_state_new, (total_loss / layer_count).item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8i. [v7+] RHUKF — Covariance Form, Node/Layer Decoupled (Absolute)
#
#   각 block(node) 또는 layer마다 (θ_block, P_block) 추적.
#   FV cov(rhukf_step_fv)와 정보 폼 node/layer(srrhuif_step)의 결합.
#
#   - State per block: (θ_3d [num_blocks, param_len, 1], P_3d [..., param_len, param_len])
#   - Time update: P_pred = P_prev + Q  (random walk, trivial)
#   - Sigma points: chol(P_pred) · γ
#   - Measurement: K = P_xz · P_zz⁻¹, θ_new = θ_pred + K·residual, P_new = P_pred - K·P_zz·K^T
# =========================================================================
@torch.no_grad()
def rhukf_step(theta_current_in, theta_target, filter_P_cov_list, batch, sp,
               is_first, p_init_val, f_cache):
    """
    RHUKF (covariance form) node/layer 디커플링.
    
    filter_P_cov_list: list of P_3d per layer (각 P_3d: [num_blocks, param_len, param_len]),
                       또는 None (h=0 초기화).
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    cfg = sp['cfg']
    
    # ── Prior 결정 ──
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_prior = sp['theta_init'].clone()
        else:
            theta_prior = theta_target.clone()
    else:
        theta_prior = theta_current_in.clone()
    theta_current = theta_current_in.clone()
    
    # ── Per-layer setup: θ extract + P prior ──
    per_layer = {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        lc = f_cache.get(L)
        
        if cfg.decoupling_mode == 'node':
            W_p = theta_prior.squeeze()[fl['W_start']:fl['W_start']+fl['W_len']].view(fl['fan_out'], fl['fan_in'])
            b_p = theta_prior.squeeze()[fl['b_start']:fl['b_start']+fl['b_len']]
            theta_L = torch.cat([W_p, b_p.unsqueeze(1)], dim=1)
        else:
            theta_L = theta_prior.squeeze()[fl['W_start']:fl['W_start']+fl['param_len']].unsqueeze(0)
        theta_L_3d = theta_L.unsqueeze(-1)  # [num_blocks, param_len, 1]
        
        if is_first or filter_P_cov_list is None:
            P_prev = p_init_val * lc['eye_block_batch'].clone()
        else:
            P_prev = filter_P_cov_list[L]
        
        per_layer[L] = {'fl': fl, 'lc': lc, 'theta_3d': theta_L_3d, 'P_prev': P_prev}
    
    # ── Sample setup ──
    s_batch = batch['s'].t()
    s_next = batch['s_next'].t() if batch['s_next'].shape[0] == info['dimS'] else batch['s_next']
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    current_q_std = sp.get('current_q_std', cfg.q_init)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    
    # ── Block-group time update + sigma points ──
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        
        all_theta_3d = torch.cat([per_layer[L]['theta_3d'] for L in layers_in_grp], dim=0)
        all_P_prev = torch.cat([per_layer[L]['P_prev'] for L in layers_in_grp], dim=0)
        
        eye_grp = grp['eye_grouped']  # [num_in_grp, bs, bs]
        Q_proc = (current_q_std ** 2) * eye_grp
        all_P_pred = all_P_prev + Q_proc
        all_P_pred = 0.5 * (all_P_pred + all_P_pred.transpose(-1, -2))
        
        all_L_chol = safe_cholesky_fallback(all_P_pred, eye_grp, JITTER_TRIA)
        
        scaled_P_g = grp['gamma'] * all_L_chol  # [num_in_grp, bs, bs]
        
        # Sigma points
        num_sigma_g = 2 * bs_val + 1
        X_sigma_g = torch.zeros(all_theta_3d.shape[0], num_sigma_g, bs_val,
                                 dtype=DTYPE, device=device)
        theta_2d_g = all_theta_3d.squeeze(-1)  # [num_in_grp, bs]
        X_sigma_g[:, 0, :] = theta_2d_g
        # scaled_P^T row k = γ·sqrt(P) k-th column
        scaled_P_T_g = scaled_P_g.transpose(-1, -2)  # [num_in_grp, bs, bs]
        X_sigma_g[:, 1:bs_val+1, :] = theta_2d_g.unsqueeze(1) + scaled_P_T_g
        X_sigma_g[:, bs_val+1:, :] = theta_2d_g.unsqueeze(1) - scaled_P_T_g
        
        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i+1]
            per_layer[L]['P_pred'] = all_P_pred[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]
            per_layer[L]['X_sigma'] = X_sigma_g[s:e]
    
    # ── Sigma scatter into unified ──
    unified = f_cache.unified_thetas
    if cfg.node_layer_other_source == 'prior':
        unified[:] = theta_prior.squeeze().to(DTYPE_FWD)
    else:
        unified[:] = theta_current.squeeze().to(DTYPE_FWD)
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        X_sigma_f32 = pl['X_sigma'].to(DTYPE_FWD)
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        if cfg.decoupling_mode == 'node':
            layer_view = unified[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], -1)
            layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :fl['fan_in']])
            layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, fl['fan_in']:fl['fan_in']+1])
        else:
            unified[fwd_start:fwd_end, fl['W_start']:fl['W_start']+fl['param_len']] = X_sigma_f32[0]
    
    # ── Forward all sigma copies ──
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    # ── DDQN target Y ──
    Q_tgt = forward_single(theta_target.squeeze(), info, s_next).to(DTYPE)
    if is_first:
        a_best_next = Q_tgt.argmax(dim=0)
    else:
        Q_curr = forward_single(theta_current.squeeze(), info, s_next).to(DTYPE)
        a_best_next = Q_curr.argmax(dim=0)
    q_val_next = Q_tgt[a_best_next, torch.arange(batch_sz, device=device)]
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    z_measured = (batch['r'] + target_gamma * (1.0 - batch['term']) * q_val_next).view(-1, 1).to(DTYPE)
    target_var = torch.var(z_measured).item()
    
    # ── Per-layer measurement update ──
    new_P_dict = {}
    new_theta_dict = {}
    total_loss = 0.0
    layer_count = info['num_filter_layers']
    total_innov_mean = total_innov_max = total_k_norm = 0.0
    total_ht_norm = total_resid_norm = total_avg_P = 0.0
    per_layer_ht_dict, per_layer_delta_dict, per_layer_resid_max_dict = {}, {}, {}
    per_layer_cond, per_layer_ymax = {}, {}
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        # Extract Z_sigma per block
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].to(DTYPE)
        # Z_sigma_T: [num_blocks, num_sigma, batch_sz]
        
        # z_hat per block: weighted mean over sigma dim
        Wm_col = lc['Wm_col_f32'].to(DTYPE)  # [num_sigma, 1]
        z_hat = (Wm_col * Z_sigma_T).sum(dim=1)  # [num_blocks, batch_sz]
        
        residual = z_measured.squeeze(-1).unsqueeze(0) - z_hat  # [num_blocks, batch_sz]
        
        # Deviations
        Z_dev = Z_sigma_T - z_hat.unsqueeze(1)  # [num_blocks, num_sigma, batch_sz]
        # X_dev: from scaled_P
        nb = lc['num_blocks']
        plen = fl['fan_in']+1 if cfg.decoupling_mode == 'node' else fl['param_len']
        X_dev = torch.zeros(nb, lc['num_sigma'], plen, dtype=DTYPE, device=device)
        scaled_P = pl['scaled_P']  # [num_blocks, bs, bs] — wait, bs = block_size here
        # Actually scaled_P is [num_blocks, param_len, param_len] in this batched per-block context
        # Let me re-check
        scaled_P_T = scaled_P.transpose(-1, -2)  # [num_blocks, param_len, param_len]
        X_dev[:, 1:plen+1, :] = scaled_P_T
        X_dev[:, plen+1:, :] = -scaled_P_T
        
        Wc_col = lc['Wc_f32'].to(DTYPE).view(1, -1, 1)  # [1, num_sigma, 1]
        
        # P_zz = Σ Wc (Z-z̄)(Z-z̄)^T per block
        P_zz_sigma = torch.einsum('bsj,bsi->bji', Wc_col * Z_dev, Z_dev)  # [num_blocks, bs_sz, bs_sz]
        
        # P_xz = Σ Wc (X-x̄)(Z-z̄)^T per block
        P_xz = torch.einsum('bsp,bsj->bpj', Wc_col * X_dev, Z_dev)  # [num_blocks, param_len, batch_sz]
        
        # Huber-adaptive R
        res_abs = torch.abs(residual)  # [num_blocks, batch_sz]
        adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)
        R_diag_eff = (current_r_std ** 2) * adapt_factor  # [num_blocks, batch_sz]
        R_diag_mat = torch.diag_embed(R_diag_eff)  # [num_blocks, batch_sz, batch_sz]
        
        P_zz = P_zz_sigma + R_diag_mat
        P_zz = 0.5 * (P_zz + P_zz.transpose(-1, -2))
        
        # Kalman gain via batched Cholesky
        eye_bs = torch.eye(batch_sz, dtype=DTYPE, device=device).unsqueeze(0).expand(nb, -1, -1)
        L_zz = safe_cholesky_fallback(P_zz, eye_bs)
        
        # K = P_xz @ P_zz⁻¹
        # K^T = (P_zz⁻¹)^T @ P_xz^T = P_zz⁻¹ @ P_xz^T (P_zz symmetric)
        tmp = torch.linalg.solve_triangular(L_zz, P_xz.transpose(-1, -2), upper=False)
        K_t = torch.linalg.solve_triangular(L_zz.transpose(-1, -2), tmp, upper=True)
        K = K_t.transpose(-1, -2)  # [num_blocks, param_len, batch_sz]
        
        # State update per block
        theta_pred_block = pl['theta_3d']  # [num_blocks, param_len, 1]
        delta_theta = torch.einsum('bpj,bj->bp', K, residual)  # [num_blocks, param_len]
        theta_new_block = theta_pred_block.squeeze(-1) + delta_theta
        
        # NaN check
        invalid = ~torch.isfinite(theta_new_block).all(dim=1)
        if invalid.any():
            theta_new_block[invalid] = theta_pred_block.squeeze(-1)[invalid]
        
        # Covariance update: P_new = P_pred - K @ P_zz @ K^T = P_pred - (K L_zz)(K L_zz)^T
        K_L = torch.bmm(K, L_zz)  # [num_blocks, param_len, batch_sz]
        P_new = pl['P_pred'] - torch.bmm(K_L, K_L.transpose(-1, -2))
        P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
        if cfg.tikhonov_lambda > 0:
            eye_p = lc['eye_block_batch']
            P_new = P_new + cfg.tikhonov_lambda * eye_p
        
        new_P_dict[L] = P_new
        new_theta_dict[L] = theta_new_block
        
        # Loss & diagnostics
        loss_L = torch.mean(residual ** 2)
        total_loss += loss_L
        ht_norm = torch.norm(P_xz).item()
        resid_norm = torch.norm(residual).item()
        k_norm = torch.norm(K).item()
        
        total_innov_mean += res_abs.mean().item()
        total_innov_max = max(total_innov_max, res_abs.max().item())
        total_k_norm += k_norm
        total_ht_norm += ht_norm
        total_resid_norm += resid_norm
        total_avg_P += torch.diagonal(P_new, dim1=-2, dim2=-1).mean().item()
        
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        per_layer_ht_dict[label] = ht_norm
        per_layer_resid_max_dict[label] = res_abs.max().item()
        per_layer_delta_dict[label] = torch.norm(delta_theta).item()
        per_layer_cond[label] = 1.0
        per_layer_ymax[label] = torch.max(torch.abs(z_measured)).item()
    
    # ── Compose final θ ──
    theta_new_flat = theta_current_in.squeeze().clone()
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        new_block = new_theta_dict[L]  # [num_blocks, param_len]
        if cfg.decoupling_mode == 'node':
            W_new = new_block[:, :fl['fan_in']]
            b_new = new_block[:, fl['fan_in']]
            theta_new_flat[fl['W_start']:fl['W_start']+fl['W_len']] = W_new.reshape(-1)
            theta_new_flat[fl['b_start']:fl['b_start']+fl['b_len']] = b_new
        else:
            theta_new_flat[fl['W_start']:fl['W_start']+fl['param_len']] = new_block[0]
    
    theta_new = theta_new_flat.view(-1, 1)
    
    # max_k_gain
    total_delta_norm = torch.norm(theta_new.squeeze() - theta_current_in.squeeze()).item()
    if cfg.max_k_gain > 0 and total_delta_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / total_delta_norm
        theta_new = (theta_current_in.squeeze() + (theta_new.squeeze() - theta_current_in.squeeze()) * scale).view(-1, 1)
        total_delta_norm = cfg.max_k_gain
    
    filter_P_cov_new = [new_P_dict[L] for L in range(info['num_filter_layers'])]
    
    dbg = {
        'innov_mean': total_innov_mean / layer_count,
        'innov_max': total_innov_max,
        'ht_norm': total_ht_norm / layer_count,
        'resid_norm': total_resid_norm / layer_count,
        'avg_P': total_avg_P / layer_count,
        'max_P': total_avg_P / layer_count,
        'delta_y': total_delta_norm,
        'y_pred_norm': torch.norm(z_measured).item(),
        'y_new': 0.0,
        'innov_norm': total_innov_mean / layer_count,
        'resid_in_innov': total_resid_norm / layer_count,
        'ht_theta_in_innov': 0.0,
        'adapt_ratio': 1.0,
        'per_layer_ht': per_layer_ht_dict,
        'per_layer_delta': per_layer_delta_dict,
        'per_layer_resid_max': per_layer_resid_max_dict,
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond,
    }
    
    return theta_new, filter_P_cov_new, (total_loss / layer_count).item(), target_var, total_k_norm / layer_count, dbg


# =========================================================================
# 8j. [v7+] RHUKF — Covariance Form, Node/Layer, Error-State
#   srrhuif_step_error의 covariance 버전. P_3d 직접 저장, sigma는 chol(P).
# =========================================================================
@torch.no_grad()
def rhukf_step_error(filter_state, ctx, batch, h_idx, sp, cfg, f_cache):
    """
    Error-state RHUKF (covariance) for node/layer decoupling.
    
    filter_state:
        h=0: None
        h≥1: dict {'P_delta': [list per layer], 'mu_delta': [list per layer]}
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    is_first = (h_idx == 0) or (filter_state is None)
    
    theta_anchor = ctx['theta_anchor'].detach()
    Y_cache = ctx['Y_cache']
    p_delta_init_val = ctx['p_delta_init']
    
    anchor_per_L = _split_anchor_per_layer(theta_anchor, info, cfg.decoupling_mode)
    
    if is_first:
        mu_delta_prev_per_L = [torch.zeros_like(a) for a in anchor_per_L]
        P_delta_prev_per_L = None
    else:
        mu_delta_prev_per_L = filter_state['mu_delta']
        P_delta_prev_per_L = filter_state['P_delta']
    
    theta_current_flat = _compose_theta_from_delta(
        theta_anchor, mu_delta_prev_per_L, info, cfg.decoupling_mode)
    
    # Y target
    if Y_cache is not None:
        z_measured = Y_cache[h_idx].view(-1, 1).to(DTYPE)
    else:
        # 'online_moving': h=0은 cfg.h0_online_moving_init에 따라, h≥1은 composed theta
        s_next = batch['s_next']
        if sp.get('normalizer'):
            s_next = sp['normalizer'].normalize(s_next)
        Q_tgt = forward_single(ctx['theta_target_ref'], info, s_next).to(DTYPE)
        if is_first:
            h0_init = cfg.h0_online_moving_init
            if h0_init == 'theta_target':
                a_best = Q_tgt.argmax(dim=0)
            else:  # 'prev_est' (spas는 FV 전용, node/layer에서는 prev_est로 동작)
                Q_prev = forward_single(ctx['theta_active_ref'], info, s_next).to(DTYPE)
                a_best = Q_prev.argmax(dim=0)
        else:
            Q_curr = forward_single(theta_current_flat, info, s_next).to(DTYPE)
            a_best = Q_curr.argmax(dim=0)
        q_val_next = Q_tgt[a_best, torch.arange(batch_sz, device=device)]
        target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
        z_measured = (batch['r'] + target_gamma * (1 - batch['term']) * q_val_next).view(-1, 1).to(DTYPE)
    target_var = torch.var(z_measured).item()

    # Per-layer setup
    per_layer = {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        lc = f_cache.get(L)
        mu_L = mu_delta_prev_per_L[L]
        if is_first:
            P_prev = p_delta_init_val * lc['eye_block_batch'].clone()
        else:
            P_prev = P_delta_prev_per_L[L]
        per_layer[L] = {
            'fl': fl, 'lc': lc,
            'theta_anchor_L': anchor_per_L[L],
            'mu_delta_3d': mu_L.unsqueeze(-1),
            'mu_delta_2d': mu_L,
            'P_prev': P_prev,
        }
    
    s_batch = batch['s'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
    
    current_q_std = sp.get('current_q_std', cfg.q_init)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    
    # Block-group time update (sigma in ERROR space)
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        
        all_mu_3d = torch.cat([per_layer[L]['mu_delta_3d'] for L in layers_in_grp], dim=0)
        all_P_prev = torch.cat([per_layer[L]['P_prev'] for L in layers_in_grp], dim=0)
        
        eye_grp = grp['eye_grouped']
        Q_proc = (current_q_std ** 2) * eye_grp
        all_P_pred = all_P_prev + Q_proc
        all_P_pred = 0.5 * (all_P_pred + all_P_pred.transpose(-1, -2))
        
        all_L_chol = safe_cholesky_fallback(all_P_pred, eye_grp, JITTER_TRIA)
        
        scaled_P_g = grp['gamma'] * all_L_chol
        num_sigma_g = 2 * bs_val + 1
        X_sigma_g = torch.zeros(all_mu_3d.shape[0], num_sigma_g, bs_val, dtype=DTYPE, device=device)
        mu_2d = all_mu_3d.squeeze(-1)
        scaled_P_T = scaled_P_g.transpose(-1, -2)
        X_sigma_g[:, 0, :] = mu_2d
        X_sigma_g[:, 1:bs_val+1, :] = mu_2d.unsqueeze(1) + scaled_P_T
        X_sigma_g[:, bs_val+1:, :] = mu_2d.unsqueeze(1) - scaled_P_T
        
        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i+1]
            per_layer[L]['P_pred'] = all_P_pred[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]
            per_layer[L]['X_sigma_err'] = X_sigma_g[s:e]
    
    # Sigma scatter: error → absolute
    unified = f_cache.unified_thetas
    if cfg.node_layer_other_source == 'prior':
        unified[:] = theta_anchor.to(DTYPE_FWD)
    else:
        unified[:] = theta_current_flat.to(DTYPE_FWD)
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        X_sigma_err_f32 = pl['X_sigma_err'].to(DTYPE_FWD)
        anchor_L_3d_f32 = pl['theta_anchor_L'].unsqueeze(1).to(DTYPE_FWD)
        X_sigma_abs_f32 = anchor_L_3d_f32 + X_sigma_err_f32
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        if cfg.decoupling_mode == 'node':
            layer_view = unified[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], -1)
            layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_abs_f32[:, :, :fl['fan_in']])
            layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_abs_f32[:, :, fl['fan_in']:fl['fan_in']+1])
        else:
            unified[fwd_start:fwd_end, fl['W_start']:fl['W_start']+fl['param_len']] = X_sigma_abs_f32[0]
    
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    # Per-layer measurement update
    new_P_dict = {}
    new_mu_delta_dict = {}
    total_loss = 0.0
    layer_count = info['num_filter_layers']
    total_innov_mean = total_innov_max = total_k_norm = 0.0
    total_ht_norm = total_resid_norm = total_avg_P = 0.0
    per_layer_ht_dict, per_layer_delta_dict, per_layer_resid_max_dict = {}, {}, {}
    per_layer_cond, per_layer_ymax = {}, {}
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].to(DTYPE)
        
        Wm_col = lc['Wm_col_f32'].to(DTYPE)
        z_hat = (Wm_col * Z_sigma_T).sum(dim=1)  # [num_blocks, batch_sz]
        residual = z_measured.squeeze(-1).unsqueeze(0) - z_hat
        
        Z_dev = Z_sigma_T - z_hat.unsqueeze(1)
        nb = lc['num_blocks']
        plen = fl['fan_in']+1 if cfg.decoupling_mode == 'node' else fl['param_len']
        X_dev = torch.zeros(nb, lc['num_sigma'], plen, dtype=DTYPE, device=device)
        scaled_P_T = pl['scaled_P'].transpose(-1, -2)
        X_dev[:, 1:plen+1, :] = scaled_P_T
        X_dev[:, plen+1:, :] = -scaled_P_T
        
        Wc_col = lc['Wc_f32'].to(DTYPE).view(1, -1, 1)
        P_zz_sigma = torch.einsum('bsj,bsi->bji', Wc_col * Z_dev, Z_dev)
        P_xz = torch.einsum('bsp,bsj->bpj', Wc_col * X_dev, Z_dev)
        
        res_abs = torch.abs(residual)
        adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)
        R_diag_eff = (current_r_std ** 2) * adapt_factor
        R_diag_mat = torch.diag_embed(R_diag_eff)
        
        P_zz = P_zz_sigma + R_diag_mat
        P_zz = 0.5 * (P_zz + P_zz.transpose(-1, -2))
        
        eye_bs = torch.eye(batch_sz, dtype=DTYPE, device=device).unsqueeze(0).expand(nb, -1, -1)
        L_zz = safe_cholesky_fallback(P_zz, eye_bs)
        
        tmp = torch.linalg.solve_triangular(L_zz, P_xz.transpose(-1, -2), upper=False)
        K_t = torch.linalg.solve_triangular(L_zz.transpose(-1, -2), tmp, upper=True)
        K = K_t.transpose(-1, -2)
        
        # Δμ update: Δμ_new = Δμ_prev + K · residual
        mu_delta_prev_block = pl['mu_delta_3d'].squeeze(-1)
        delta_correction = torch.einsum('bpj,bj->bp', K, residual)
        mu_delta_new_block = mu_delta_prev_block + delta_correction
        
        invalid = ~torch.isfinite(mu_delta_new_block).all(dim=1)
        if invalid.any():
            mu_delta_new_block[invalid] = mu_delta_prev_block[invalid]
        
        K_L = torch.bmm(K, L_zz)
        P_new = pl['P_pred'] - torch.bmm(K_L, K_L.transpose(-1, -2))
        P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
        if cfg.tikhonov_lambda > 0:
            P_new = P_new + cfg.tikhonov_lambda * lc['eye_block_batch']
        
        new_P_dict[L] = P_new
        new_mu_delta_dict[L] = mu_delta_new_block
        
        loss_L = torch.mean(residual ** 2)
        total_loss += loss_L
        
        ht_norm = torch.norm(P_xz).item()
        k_norm = torch.norm(K).item()
        delta_change_norm = torch.norm(delta_correction).item()
        
        total_innov_mean += res_abs.mean().item()
        total_innov_max = max(total_innov_max, res_abs.max().item())
        total_k_norm += k_norm
        total_ht_norm += ht_norm
        total_resid_norm += torch.norm(residual).item()
        total_avg_P += torch.diagonal(P_new, dim1=-2, dim2=-1).mean().item()
        
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        per_layer_ht_dict[label] = ht_norm
        per_layer_resid_max_dict[label] = res_abs.max().item()
        per_layer_delta_dict[label] = delta_change_norm
        per_layer_cond[label] = 1.0
        per_layer_ymax[label] = torch.max(torch.abs(z_measured)).item()
    
    # Compose final θ_active
    new_mu_delta_per_L = [new_mu_delta_dict[L] for L in range(info['num_filter_layers'])]
    theta_active_flat = _compose_theta_from_delta(
        theta_anchor, new_mu_delta_per_L, info, cfg.decoupling_mode)
    
    delta_change = theta_active_flat - theta_current_flat
    k_gain_norm = torch.norm(delta_change).item()
    if cfg.max_k_gain > 0 and k_gain_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / k_gain_norm
        theta_active_flat = theta_current_flat + delta_change * scale
        for L in range(info['num_filter_layers']):
            mu_prev = mu_delta_prev_per_L[L]
            mu_new = new_mu_delta_dict[L]
            new_mu_delta_dict[L] = mu_prev + (mu_new - mu_prev) * scale
        k_gain_norm = cfg.max_k_gain
    
    theta_active = theta_active_flat.view(-1, 1)
    
    filter_state_new = {
        'P_delta': [new_P_dict[L] for L in range(info['num_filter_layers'])],
        'mu_delta': [new_mu_delta_dict[L] for L in range(info['num_filter_layers'])],
    }
    
    dbg = {
        'innov_mean': total_innov_mean / layer_count,
        'innov_max': total_innov_max,
        'ht_norm': total_ht_norm / layer_count,
        'resid_norm': total_resid_norm / layer_count,
        'avg_P': total_avg_P / layer_count,
        'max_P': total_avg_P / layer_count,
        'delta_y': k_gain_norm,
        'y_pred_norm': torch.norm(z_measured).item(),
        'y_new': 0.0,
        'innov_norm': total_innov_mean / layer_count,
        'resid_in_innov': total_resid_norm / layer_count,
        'ht_theta_in_innov': 0.0,
        'adapt_ratio': 1.0,
        'per_layer_ht': per_layer_ht_dict,
        'per_layer_delta': per_layer_delta_dict,
        'per_layer_resid_max': per_layer_resid_max_dict,
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond,
    }
    
    return theta_active, filter_state_new, (total_loss / layer_count).item(), target_var, total_k_norm / layer_count, dbg


# =========================================================================
# 9. Live Plotter
# =========================================================================
class LivePlotter:
    def __init__(self, method_name: str, max_episodes: int, param_str: str = "",
                 filter_form: str = 'information'):
        self.method_name = method_name
        self.outdir = cfg.outdir
        self.filter_form = filter_form  # 'information' (SRRHUIF) | 'covariance' (RHUKF)
        
        self.rewards, self.losses, self.p_inits, self.z_vars = [], [], [], []
        self.k_gains = [] 
        self.q_vals_0, self.q_vals_1 = [], []
        self.total_time, self.avg_step_time = 0.0, 0.0
        
        self.cond_history = {}
        self.ymax_history = {}
        self.theta_norm_history = {}
        self.null_ratio_history = []
        self.eff_rank_history = []
        self.stable_rank_history = []
        self.argmax_flip_history = []
        self.ref_dq_history = {name: [] for name in REF_NAMES}
        
        self.buf_state_std, self.buf_state_range, self.buf_done_ratio = [], [], []
        self.buf_reward_std, self.buf_fill_ratio, self.buf_age_range, self.buf_age_std = [], [], [], []
        self.buf_saturated_ep = None  
        
        self.fig, self.axes = plt.subplots(1, 6, figsize=(30, 4))
        
        # [env-aware] 보상 그래프 설정 (CartPole / LunarLander 등)
        _env_cfg = ENV_CONFIGS.get(cfg.env_name, {})
        self.reward_threshold = _env_cfg.get('reward_threshold', None)
        self.reward_ylim = _env_cfg.get('reward_ylim', None)

        self.ax_r = self.axes[0]
        self.line_r_raw, = self.ax_r.plot([], [], 'b-', alpha=0.3)
        self.line_r_ma, = self.ax_r.plot([], [], 'b-', linewidth=2)
        if self.reward_threshold is not None:
            self.ax_r.axhline(y=self.reward_threshold, color='g', linestyle='--', alpha=0.5)
        self.ax_r.axhline(y=0, color='gray', linestyle=':', alpha=0.4)  # 음수보상 환경 기준선
        self.ax_r.set_xlim(0, max_episodes)
        if self.reward_ylim is not None:
            self.ax_r.set_ylim(*self.reward_ylim)
        self.ax_r.set_title(f'Reward ({method_name})')
        
        self.ax_l = self.axes[1]
        self.line_l, = self.ax_l.plot([], [], 'r-', linewidth=1.5)
        self.ax_l.set_title('TD Loss'); self.ax_l.set_xlim(0, max_episodes)
        
        # ──────────────────────────────────────────────────────────────
        # [Plot 3] avg posterior P diag (둘 다 dbg['avg_P']에서 옴)
        #   - information: avg P = mean(1/diag(Y)). RHUIF에서 동적값.
        #   - covariance:  avg P = mean(diag(P)) 직접.
        #   둘 다 "평균 파라미터 분산"이라는 같은 의미. 동적으로 추적됨.
        # ──────────────────────────────────────────────────────────────
        self.ax_p = self.axes[2]
        self.line_p, = self.ax_p.plot([], [], 'g-', linewidth=2)
        title_p = 'Avg P diag (RHUKF)' if filter_form == 'covariance' else 'Avg P diag (1/diag(Y), RHUIF)'
        self.ax_p.set_title(title_p); self.ax_p.set_xlim(0, max_episodes)
        # ylim 자동 (avg_P는 cfg.p_init보다 작아져야 정상 — 정보 누적 → P 감소)
            
        self.ax_z = self.axes[3]
        self.line_z, = self.ax_z.plot([], [], 'm-', linewidth=1.5)
        self.ax_z.set_title('TD Target Variance (Z_var)'); self.ax_z.set_xlim(0, max_episodes)

        self.ax_k = self.axes[4]
        self.line_k, = self.ax_k.plot([], [], 'darkorange', linewidth=1.5)
        title_k = '||K||_F (RHUKF gain)' if filter_form == 'covariance' else 'Weight Update Norm ||Δθ||'
        self.ax_k.set_title(title_k); self.ax_k.set_xlim(0, max_episodes)

        self.ax_q = self.axes[5]
        self.line_q0, = self.ax_q.plot([], [], 'c-', linewidth=1.5, label='Q(a=0) Left')
        self.line_q1, = self.ax_q.plot([], [], 'm-', linewidth=1.5, label='Q(a=1) Right')
        self.ax_q.set_title('Avg Q-Values'); self.ax_q.set_xlim(0, max_episodes)
        self.ax_q.legend(loc='upper left')
        
        plt.tight_layout()
        # [Windows MAX_PATH fix] outdir에 이미 param_str이 들어있으므로 파일명에 다시 넣지 않음
        # (그렇지 않으면 path가 ~280자 되어 Windows 260자 제한 초과)
        clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        self.filename = os.path.join(self.outdir, clean_name)
    
    def add(self, reward, loss, avg_P=0.0, z_var=0.0, k_gain=0.0, q0=0.0, q1=0.0): 
        """
        avg_P: mean posterior parameter variance (dbg['avg_P']).
            - RHUIF form: (1/diag(Y)).mean(), 정보 누적될수록 감소
            - RHUKF form: diag(P).mean(), 동일하게 정보 누적될수록 감소
        """
        self.rewards.append(reward)
        self.losses.append(max(loss, 1e-8))
        self.p_inits.append(avg_P)
        self.z_vars.append(z_var)
        self.k_gains.append(k_gain) 
        self.q_vals_0.append(q0)
        self.q_vals_1.append(q1)
    
    def add_diagnostics(self, cond_dict, ymax_dict, theta_norms, null_ratio,
                        eff_rank, stable_rank, argmax_flip, ref_q):
        if cond_dict:
            for k, v in cond_dict.items(): self.cond_history.setdefault(k, []).append(v)
        if ymax_dict:
            for k, v in ymax_dict.items(): self.ymax_history.setdefault(k, []).append(v)
        if theta_norms:
            for k, v in theta_norms.items(): self.theta_norm_history.setdefault(k, []).append(v)
        self.null_ratio_history.append(null_ratio)
        self.eff_rank_history.append(eff_rank)
        self.stable_rank_history.append(stable_rank)
        self.argmax_flip_history.append(argmax_flip)
        if ref_q:
            for name in REF_NAMES: self.ref_dq_history[name].append(ref_q[name]['dq'])
    
    def add_buffer_diag(self, buf_info, ep):
        if buf_info is None:
            self.buf_state_std.append(float('nan')); self.buf_state_range.append(float('nan'))
            self.buf_done_ratio.append(float('nan')); self.buf_reward_std.append(float('nan'))
            self.buf_fill_ratio.append(0.0); self.buf_age_range.append(0); self.buf_age_std.append(0.0)
            return
        self.buf_state_std.append(buf_info['state_std']); self.buf_state_range.append(buf_info['state_range'])
        self.buf_done_ratio.append(buf_info['done_ratio']); self.buf_reward_std.append(buf_info['reward_std'])
        self.buf_fill_ratio.append(buf_info['fill_ratio']); self.buf_age_range.append(buf_info['age_range'])
        self.buf_age_std.append(buf_info['age_std'])
        if buf_info['is_saturated'] and self.buf_saturated_ep is None: self.buf_saturated_ep = ep
    
    def refresh(self):
        ep_range = range(len(self.rewards))
        self.line_r_raw.set_data(ep_range, self.rewards)
        if len(self.rewards) >= 20:
            ma = np.convolve(self.rewards, np.ones(20)/20, 'valid')
            self.line_r_ma.set_data(range(19, len(self.rewards)), ma)
        self.line_l.set_data(ep_range, self.losses)
        self.line_p.set_data(ep_range, self.p_inits)
        self.line_z.set_data(ep_range, self.z_vars) 
        self.line_k.set_data(ep_range, self.k_gains) 
        self.line_q0.set_data(ep_range, self.q_vals_0)
        self.line_q1.set_data(ep_range, self.q_vals_1)
        
        for ax in self.axes: ax.relim(); ax.autoscale_view()
        # [env-aware] 보상 y축: 음수보상(LunarLander 추락 등)도 잘리지 않도록 데이터 기반으로 설정
        if self.rewards:
            r_min, r_max = min(self.rewards), max(self.rewards)
            if self.reward_ylim is not None:
                lo = min(self.reward_ylim[0], r_min - abs(r_min) * 0.1 - 1)
                hi = max(self.reward_ylim[1], r_max * 1.1)
            else:
                pad = (r_max - r_min) * 0.1 + 1.0
                lo, hi = r_min - pad, r_max + pad
            self.axes[0].set_ylim(lo, hi)
        plt.savefig(f'{self.filename}_live.png', dpi=100)
    
    def save_diagnostic_plots(self):
        if not self.cond_history and not self.theta_norm_history: return
        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
        ax = axes[0, 0]
        for label, vals in sorted(self.cond_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log')
        cond_title = ('cond(P_zz) per Layer (RHUKF innovation cov)' 
                      if self.filter_form == 'covariance' 
                      else 'Pseudo Condition Number per Layer (cond(Y), RHUIF)')
        ax.set_title(cond_title)
        ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        for label, vals in sorted(self.ymax_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log')
        ymax_title = ('max(diag P) per Layer (RHUKF, param uncertainty)' 
                      if self.filter_form == 'covariance' 
                      else 'Y_max per Layer (max info eigenvalue, RHUIF)')
        ax.set_title(ymax_title)
        ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        for label, vals in sorted(self.theta_norm_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_title('Layer ||θ|| Evolution'); ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(self.null_ratio_history, 'r-', linewidth=2)
        ax.set_title('Advantage/Q-Layer Null/Signal Ratio'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(self.eff_rank_history, 'b-', linewidth=2, label='effective rank')
        ax.plot(self.stable_rank_history, 'g--', linewidth=2, label='stable rank')
        ax.set_title('Shared Output Rank'); ax.set_xlabel('Episode'); ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        for name in REF_NAMES: ax.plot(self.ref_dq_history[name], label=name, linewidth=1.5)
        ax.set_title('ΔQ = Q(right) - Q(left) at Reference States'); ax.set_xlabel('Episode')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3); ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.savefig(f'{self.filename}_diagnostics.png', dpi=120, bbox_inches='tight'); plt.close(fig)
        
        if self.argmax_flip_history:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(self.argmax_flip_history, 'orange', linewidth=1.5)
            ax2.set_title('Argmax Flip Rate (update-induced policy instability)'); ax2.set_xlabel('Episode')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
            ax2.grid(True, alpha=0.3); ax2.legend(); plt.tight_layout()
            plt.savefig(f'{self.filename}_argmax_flip.png', dpi=120); plt.close(fig2)
        
        if self.buf_state_std and any(not (np.isnan(v)) for v in self.buf_state_std):
            fig3, axes3 = plt.subplots(2, 3, figsize=(21, 10))
            ax = axes3[0, 0]
            ax.plot(self.buf_fill_ratio, 'b-', linewidth=2)
            if self.buf_saturated_ep is not None:
                ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', label=f'First saturation: Ep {self.buf_saturated_ep}')
                ax.legend()
            ax.set_title('Buffer Fill Ratio'); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
            
            ax = axes3[0, 1]
            ax.plot(self.buf_state_std, 'g-', linewidth=2, label='state_std')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('State Std (diversity of sampled states)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes3[0, 2]
            ax.plot(self.buf_state_range, 'm-', linewidth=2, label='state_range')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('State Range (max-min)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 0]
            ax.plot(self.buf_done_ratio, 'orange', linewidth=2)
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Done Ratio in Buffer'); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 1]
            ax.plot(self.buf_reward_std, 'purple', linewidth=2)
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Reward Std in Buffer'); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 2]
            ax.plot(self.buf_age_range, 'teal', linewidth=2, label='age range')
            ax.plot(self.buf_age_std, 'brown', linewidth=2, label='age std')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Buffer Age Diversity (ep_id range/std)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            plt.tight_layout(); plt.savefig(f'{self.filename}_buffer_diag.png', dpi=120, bbox_inches='tight'); plt.close(fig3)
    
    def close(self):
        plt.close(self.fig)

# =========================================================================
# 10. Landscape Visualization
# =========================================================================
def plot_cartpole_state_landscape(theta_star, info, cfg, normalizer, method_name, param_str, resolution=50):
    print(f"\n[Landscape] {method_name} 상태 공간 Q-지형 분석 중...")
    device = cfg.device
    theta_range = np.linspace(-0.25, 0.25, resolution)
    theta_dot_range = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(theta_range, theta_dot_range)
    states = np.zeros((resolution * resolution, 4))
    states[:, 2] = X.flatten(); states[:, 3] = Y.flatten()
    states_t = torch.tensor(states, dtype=torch.float64, device=device)
    if normalizer: states_t = normalizer.normalize(states_t)
        
    with torch.no_grad():
        q_vals_f32 = forward_single(theta_star.squeeze(), info, states_t.t())
        max_q = q_vals_f32.max(dim=0).values.cpu().numpy()
        
    Z = max_q.reshape(resolution, resolution)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
    
    z_min, z_max = np.min(Z), np.max(Z)
    z_floor = z_min - (z_max - z_min) * 0.15 
    ax.contourf(X, Y, Z, zdir='z', offset=z_floor, cmap='plasma', alpha=0.5)
    ax.set_zlim(z_floor, z_max)
    ax.view_init(elev=25, azim=230)

    ax.set_title(f'State-Space Q-Landscape: {method_name}\n({param_str})')
    ax.set_xlabel('Pole Angle (rad)'); ax.set_ylabel('Angular Velocity (rad/s)'); ax.set_zlabel('Max Q-value')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    # [Windows MAX_PATH fix] outdir에 이미 param_str이 있으므로 파일명에서는 제외
    filename = os.path.join(cfg.outdir, f"{clean_name}_State_Land.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 11. Main Loop with Full Logging
# =========================================================================
def train_srrhuif():
    net_seed = cfg.network_seed if cfg.network_seed is not None else cfg.seed
    env_seed = cfg.env_seed if cfg.env_seed is not None else cfg.seed
    set_all_seeds(net_seed)
    apply_tf32_config(cfg)  # cfg가 코드에서 바뀐 경우에도 반영 (idempotent)
    env = gym.make(cfg.env_name, **build_env_kwargs(cfg))
    env.action_space.seed(net_seed)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)

    method_title = f"{'D3QN' if cfg.use_dueling else 'DDQN'} + {cfg.decoupling_mode.upper()} Decoupled"
    
    # ── 실제 작동 prior 선택 ──
    #   state_form='error'이면 prior는 P_Δ⁻ = p_delta_init·I 로 시작하므로
    #   로그에 p_init이 아니라 p_delta_init을 찍어야 한다 (p_init은 absolute 전용, error에선 dead value).
    eff_prior = cfg.p_delta_init if cfg.state_form == 'error' else cfg.p_init
    eff_prior_name = 'p_Δ_init' if cfg.state_form == 'error' else 'p_init'

    print(f"\n{'='*60}")
    form_short = "RHUKF" if cfg.filter_form == 'covariance' else "SRRHUIF"
    print(f"  {form_short}-{method_title} v6.0 Robust Session")
    print(f"  Env: {cfg.env_name} | obs_dim={dimS} | nA={nA} | max_steps={cfg.max_steps} "
          f"| input_norm={'on' if (cfg.use_input_norm and cfg.obs_scale) else 'off'}")
    if cfg.env_name.startswith("LunarLander"):
        print(f"  Wind: {'ON' if cfg.enable_wind else 'off'}"
              + (f" (wind_power={cfg.wind_power:g}, turbulence_power={cfg.turbulence_power:g})" if cfg.enable_wind else ""))
    print(f"  Filter form: {cfg.filter_form} ({'Kim et al. 2010 Alg 1' if cfg.filter_form == 'covariance' else 'sqrt-information'})")
    print(f"  Horizon: {cfg.N_horizon} | Batch: {cfg.batch_size} | Params: {info['total_params']}")
    print(f"  Settings: {eff_prior_name}={eff_prior} (effective prior), Tikhonov={cfg.tikhonov_lambda}, Huber_c={cfg.huber_c}")
    print(f"  N-step: use={cfg.use_n_step}, size={cfg.n_step_size} "
          f"(target γ = {cfg.gamma ** cfg.n_step_size if cfg.use_n_step else cfg.gamma:.4f})")
    print(f"  Output Dir: {cfg.outdir}")
    print(f"  Seeds: network={net_seed}, env={env_seed}")
    print(f"{'='*60}\n")

    normalizer = InputNormalizer(cfg.device, cfg.obs_scale) if (cfg.use_input_norm and cfg.obs_scale) else None
    
    # ──────────────────────────────────────────────────────────────────
    # [FV 분기] 모드별 캐시 생성
    # ──────────────────────────────────────────────────────────────────
    is_fv = (cfg.decoupling_mode == 'fv')
    is_rhukf = is_fv and (cfg.filter_form == 'covariance')
    if is_fv:
        f_cache = FilterCacheFV(info, cfg, cfg.device)
    else:
        f_cache = FilterCache(info, cfg, cfg.device)
    
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size, 'normalizer': normalizer, 'device': cfg.device, 'cfg': cfg}

    theta = initialize_theta(info, cfg.device, cfg).view(-1, 1)
    # ──────────────────────────────────────────────────────────────────
    # [FIR 철학] theta_init: 학습 시작시 한 번 뽑은 frozen 값.
    #   cfg.h0_prior_source == 'init'이면 매 horizon의 h=0에서 prior로 사용.
    # ──────────────────────────────────────────────────────────────────
    theta_init = theta.clone()
    sp['theta_init'] = theta_init
    
    theta_target = theta.clone()
    param_update_count = 0  # [v7+] target update 카운터 (hard mode 트리거 + 진단)

    # ──────────────────────────────────────────────────────────────────
    # [v9+] Adam warm-up: batch_hist가 N_horizon에 도달하기 전까지
    #   theta_param (Parameter, 1D) ↔ theta (data tensor, [n_x, 1])
    #   매 Adam step 후 theta.data ← theta_param.data로 동기화.
    #   필터가 시작되면 (len(batch_hist) == N_horizon) 이후로는 사용 안 함.
    # ──────────────────────────────────────────────────────────────────
    if cfg.use_adam_warmup:
        theta_param = nn.Parameter(theta.squeeze().clone().detach(), requires_grad=True)
        adam_opt = torch.optim.Adam([theta_param], lr=cfg.adam_lr)
        adam_steps_taken = 0
        print(f"[Adam Warm-up] enabled (lr={cfg.adam_lr:g}, "
              f"until len(batch_hist)=={cfg.N_horizon}, loss=Huber(c={cfg.huber_c}))")
        if cfg.measurement_mode == 'pure_reward':
            print(f"[Adam Warm-up] NOTE: cfg.measurement_mode='pure_reward'이지만 "
                  f"Adam은 항상 q_target (semi-gradient TD) form만 사용 "
                  f"(pure_reward는 residual gradient라 gradient 기반에선 불안정). "
                  f"필터 전환 후엔 pure_reward로 복귀.")
    else:
        theta_param = None
        adam_opt = None
        adam_steps_taken = 0
    
    # [v7+] Twin-Q: 두 번째 네트워크 (독립 초기화로 함수 diversity 확보)
    theta_2 = None
    theta_target_2 = None
    if cfg.use_twin:
        # theta_2를 다른 seed로 독립 초기화 — TD3에서 함수 다양성 필수
        _twin_seed = cfg.network_seed + 10000
        _saved_state = torch.random.get_rng_state()
        torch.manual_seed(_twin_seed)
        theta_2 = initialize_theta(info, cfg.device, cfg).view(-1, 1)
        torch.random.set_rng_state(_saved_state)
        theta_target_2 = theta_2.clone()
        print(f"[Twin-Q] θ_2 독립 초기화 완료 (twin_seed={_twin_seed}, "
              f"‖θ_2 - θ_1‖ = {torch.norm(theta_2 - theta).item():.4f})")
    
    if cfg.filter_form == 'covariance':
        form_label = "RHUKF (covariance)"
    elif is_fv:
        form_label = "SRRHUIF (information)"
    else:
        form_label = "SRRHUIF-decoupled"
    if cfg.state_form == 'error':
        state_label = (f"Error-State[anchor={cfg.anchor_type}, argmax={cfg.ddqn_argmax}, "
                       f"p_Δ={cfg.p_delta_init}]")
    else:
        state_label = "Absolute (legacy)"
    print(f"[Init] scheme='{cfg.init_scheme}' | h0_prior_source='{cfg.h0_prior_source}' | "
          f"mode='{cfg.decoupling_mode}' | filter_form='{cfg.filter_form}' ({form_label})")
    print(f"[Init] state_form='{cfg.state_form}' → {state_label}")
    print(f"[Init] activation_fn='{cfg.activation_fn}' | use_twin={cfg.use_twin}"
          + (" (Clipped Double Q, TD3-style)" if cfg.use_twin else ""))
    if cfg.use_residual:
        # 어느 레이어에 residual이 들어갈지 진단
        resid_layers = []
        for L_idx, layer in enumerate(info['layers']):
            if layer['W_shape'][0] == layer['W_shape'][1]:
                # is_final 체크: 출력층은 residual 적용 안됨
                is_shared_final = (L_idx == info['shared_end_idx'] - 1)  # shared 마지막은 다음 head로 분기, 잠재 final 아님
                is_value_final = (L_idx == info['value_end_idx'] - 1)
                is_adv_final = (L_idx == len(info['layers']) - 1)
                if not (is_value_final or is_adv_final):
                    resid_layers.append(f"L{L_idx}({layer['W_shape'][0]}×{layer['W_shape'][1]})")
        if resid_layers:
            print(f"[Init] use_residual=True → skip 적용 레이어: {', '.join(resid_layers)}")
        else:
            print(f"[Init] use_residual=True 이지만 same-dim hidden layer 없음 → 효과 없음 (네트워크 구조 점검 필요)")
    if cfg.target_update_mode == 'soft':
        print(f"[Init] target_update: SOFT (τ={cfg.tau_srrhuif})")
    else:
        print(f"[Init] target_update: HARD (period={cfg.target_update_period} 호라이즌 업데이트마다 복사)")
    analyze_initial_network(theta, info, env, cfg, normalizer)
    
    # Filter state 의미:
    #   - FV + 'information': filter_state = S_Y (information sqrt, [n_x, n_x] lower-tri)
    #   - FV + 'covariance':  filter_state = P (full covariance, [n_x, n_x] symmetric)
    #   - node/layer:         filter_state = [S_Y_per_layer, ...] (list of dicts)
    if is_fv:
        filter_state = None
    else:
        filter_state = [None] * info['num_filter_layers']
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device, cfg)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)
    batch_hist = deque(maxlen=cfg.N_horizon)
    
    logger = LivePlotter(method_title, cfg.max_episodes, cfg.param_str,
                         filter_form=cfg.filter_form)
    
    steps_done = 0
    train_start_time = time.time()
    update_times = []
    
    prev_ep_delta = None
    prev_buf_saturated = False
    theta_ep_start = theta.squeeze().clone()

    s, _ = env.reset(seed=env_seed)
    
    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=env_seed + ep)
        buffer.set_current_episode(ep)
        
        ep_r, ep_l, ep_var, ep_k_gain, ep_start = 0, [], [], [], time.time()
        ep_q0, ep_q1 = [], []
        ep_i_mean, ep_i_max = [], []
        ep_avg_P = []  # [v6] dbg['avg_P'] 모아서 LivePlotter에 동적 P 추적
        
        last_h_k_traj, last_h_p_traj, last_h_ht_traj = [], [], []
        last_h_resid_traj, last_h_innov_decomp, last_h_cos_traj = [], [], []
        last_h_layer_ht, last_h_layer_delta = [], []
        last_h_layer_resid_max = []
        last_h_layer_cond, last_h_layer_ymax = [], []
        last_h_gain_traj, last_h_pos_traj, last_h_maxz_traj = [], [], []
        last_h_layer_gain = []  # [probe] per-h × per-layer mean_gain dict 리스트
        last_h_layer_spread, last_h_layer_amp, last_h_layer_spos = [], [], []  # [sigma-spread] per-h × per-layer
        last_ep_cos = None
        
        ep_cond_collect, ep_ymax_collect, ep_argmax_flips = {}, {}, []
        theta_ep_start = theta.squeeze().clone()
        
        for t in range(cfg.max_steps):
            steps_done += 1
            if steps_done <= cfg.warmup_step:
                eps = 1.0
                sp['current_q_std'] = cfg.q_init
                sp['current_r_std'] = cfg.r_init
            else:
                active_steps = steps_done - cfg.warmup_step
                decay_factor = np.exp(-active_steps / cfg.eps_decay_steps)
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * decay_factor
                # Q, R를 eps와 동일한 지수감쇠로 q_init→q_end, r_init→r_end 스케줄
                sp['current_q_std'] = cfg.q_end + (cfg.q_init - cfg.q_end) * decay_factor
                sp['current_r_std'] = cfg.r_end + (cfg.r_init - cfg.r_end) * decay_factor
            
            with torch.no_grad():
                s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                s_t = s_t_buffer
                if normalizer: s_t = normalizer.normalize(s_t)
                q_vals = forward_single(theta.squeeze(), info, s_t).squeeze()
                ep_q0.append(q_vals[0].item())
                ep_q1.append(q_vals[1].item())

            a = env.action_space.sample() if np.random.rand() < eps else q_vals.argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if steps_done > cfg.warmup_step and buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)
                
                batch_hist.append(batch)

                # ─────────────────────────────────────────────────────────
                # [v9+] Adam warm-up: 윈도우가 아직 안 찼고 옵션 켜져 있으면
                # 매 update 이벤트마다 Adam 한 스텝 (필터는 아직 안 돌아감).
                # target net soft/hard 업데이트는 필터 경로와 동일 규칙 적용.
                # ─────────────────────────────────────────────────────────
                if cfg.use_adam_warmup and len(batch_hist) < cfg.N_horizon:
                    adam_opt.zero_grad(set_to_none=True)
                    loss_adam = compute_adam_td_loss(theta_param, theta_target, batch, sp, cfg)
                    loss_adam.backward()
                    adam_opt.step()
                    with torch.no_grad():
                        theta.data.copy_(theta_param.data.view(-1, 1))
                    adam_steps_taken += 1

                    # target net update (필터 경로와 동일 규칙)
                    param_update_count += 1
                    if cfg.target_update_mode == 'soft':
                        theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                    else:  # 'hard'
                        if param_update_count % cfg.target_update_period == 0:
                            theta_target = theta.clone()

                    # PER priority 업데이트 (해당 batch만)
                    if cfg.use_per:
                        idx_per, td_per = _compute_per_priorities(
                            theta, theta_target, [batch], sp, cfg, normalizer
                        )
                        if idx_per is not None:
                            buffer.update_priorities(idx_per, td_per)

                    # Loss 기록 (필터 horizon-loss와 동일 슬롯에 push)
                    ep_l.append(float(loss_adam.detach().item()))

                if len(batch_hist) == cfg.N_horizon:
                    h_k_traj, h_p_traj, h_ht_traj = [], [], []
                    h_resid_traj, h_resid_in_innov_traj, h_ht_theta_traj = [], [], []
                    h_innov_traj, h_cos_traj = [], []
                    h_layer_ht, h_layer_delta, h_layer_cond, h_layer_ymax = [], [], [], []
                    h_layer_resid_max = []
                    h_gain_traj, h_pos_traj, h_maxz_traj = [], [], []  # [probe] per-h 집계
                    h_layer_gain = []                                  # [probe] per-h × per-layer
                    h_layer_spread, h_layer_amp, h_layer_spos = [], [], []  # [sigma-spread] per-h × per-layer
                    prev_h_delta = None
                    # [probe] Per-h activation regime 게이팅: 플래그 + cadence + warmup
                    #   매 h forward가 추가되므로 반드시 게이팅 (throughput 보호).
                    do_act_regime = (cfg.diag_act_regime
                                     and ep >= cfg.act_regime_warmup
                                     and (ep % max(cfg.act_regime_every, 1) == 0))
                    # [sigma-spread] 게이팅: step 내부에서 시그마 forward 1회 추가되므로 cadence 필수.
                    #   sp 통해 step 함수로 전달 (step은 ep를 모름).
                    do_sigma_spread = (cfg.diag_sigma_spread
                                       and ep >= cfg.sigma_spread_warmup
                                       and (ep % max(cfg.sigma_spread_every, 1) == 0))
                    sp['_do_sigma_spread'] = do_sigma_spread
                    # [v5] q_next_caches 사전계산 제거 — 매 horizon step 내부에서 계산
                    
                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            s_flip = batch_hist[0]['s'].t()
                            if normalizer: s_flip = normalizer.normalize(s_flip)
                            argmax_before = forward_single(theta.squeeze(), info, s_flip).argmax(dim=0)
                    
                    # ─────────────────────────────────────────────────────────
                    # [v7] Error-state 모드: 호라이즌 진입 전 1회 setup
                    #      - θ_anchor 결정 (frozen during horizon)
                    #      - Y_cache 일괄 계산 (ddqn_argmax 정책에 따라)
                    #      - Twin-Q면 min(Q1, Q2) Y 공유, 두 ctx 생성
                    # ─────────────────────────────────────────────────────────
                    horizon_ctx_2 = None
                    filter_state_es_2 = None
                    if cfg.state_form == 'error':
                        if cfg.use_twin:
                            # ★ Twin-Q: Y_min 계산 후 두 context (각자 anchor, 공유 Y)
                            Y_twin = compute_twin_y_cache(
                                theta, theta_target, theta_target_2, list(batch_hist), sp, cfg)
                            horizon_ctx = init_error_horizon(
                                theta, theta_target, list(batch_hist), sp, cfg, f_cache,
                                Y_cache_external=Y_twin)
                            horizon_ctx_2 = init_error_horizon(
                                theta_2, theta_target_2, list(batch_hist), sp, cfg, f_cache,
                                Y_cache_external=Y_twin)
                        else:
                            horizon_ctx = init_error_horizon(
                                theta, theta_target, list(batch_hist), sp, cfg, f_cache)
                        filter_state_es = None
                    
                    loop_count = cfg.N_horizon
                    
                    for h in range(loop_count):
                        theta_before_h = theta.squeeze().clone()
                        
                        # ── 메인 필터 (θ_1) ──
                        if cfg.state_form == 'error':
                            if is_rhukf and is_fv:
                                theta, filter_state_es, l_val, t_var, t_k_gain, dbg = rhukf_step_fv_error(
                                    filter_state_es, horizon_ctx, batch_hist[h], h, sp, cfg, f_cache)
                            elif is_rhukf:  # node/layer + cov + error
                                theta, filter_state_es, l_val, t_var, t_k_gain, dbg = rhukf_step_error(
                                    filter_state_es, horizon_ctx, batch_hist[h], h, sp, cfg, f_cache)
                            elif is_fv:  # FV + info + error
                                theta, filter_state_es, l_val, t_var, t_k_gain, dbg = srrhuif_step_fv_error(
                                    filter_state_es, horizon_ctx, batch_hist[h], h, sp, cfg, f_cache)
                            else:  # node/layer + info + error
                                theta, filter_state_es, l_val, t_var, t_k_gain, dbg = srrhuif_step_error(
                                    filter_state_es, horizon_ctx, batch_hist[h], h, sp, cfg, f_cache)
                        elif is_rhukf and is_fv:
                            theta, filter_state, l_val, t_var, t_k_gain, dbg = rhukf_step_fv(
                                theta, theta_target, filter_state, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache)
                        elif is_rhukf:  # node/layer + cov + absolute
                            theta, filter_state, l_val, t_var, t_k_gain, dbg = rhukf_step(
                                theta, theta_target, filter_state, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache)
                        elif is_fv:
                            theta, filter_state, l_val, t_var, t_k_gain, dbg = srrhuif_step_fv(
                                theta, theta_target, filter_state, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache)
                        else:
                            theta, filter_state, l_val, t_var, t_k_gain, dbg = srrhuif_step(
                                theta, theta_target, filter_state, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache)
                        
                        # ── Twin 필터 (θ_2) ── (동일 Y_twin으로 병렬 업데이트)
                        if cfg.use_twin:
                            if is_rhukf and is_fv:
                                theta_2, filter_state_es_2, _, _, _, _ = rhukf_step_fv_error(
                                    filter_state_es_2, horizon_ctx_2, batch_hist[h], h, sp, cfg, f_cache)
                            elif is_rhukf:
                                theta_2, filter_state_es_2, _, _, _, _ = rhukf_step_error(
                                    filter_state_es_2, horizon_ctx_2, batch_hist[h], h, sp, cfg, f_cache)
                            elif is_fv:
                                theta_2, filter_state_es_2, _, _, _, _ = srrhuif_step_fv_error(
                                    filter_state_es_2, horizon_ctx_2, batch_hist[h], h, sp, cfg, f_cache)
                            else:
                                theta_2, filter_state_es_2, _, _, _, _ = srrhuif_step_error(
                                    filter_state_es_2, horizon_ctx_2, batch_hist[h], h, sp, cfg, f_cache)
                        
                        h_delta = theta.squeeze() - theta_before_h
                        if prev_h_delta is not None:
                            d_norm, p_norm = torch.norm(h_delta), torch.norm(prev_h_delta)
                            cos = F.cosine_similarity(h_delta.unsqueeze(0), prev_h_delta.unsqueeze(0)).item() if (d_norm > 1e-8 and p_norm > 1e-8) else 0.0
                            h_cos_traj.append(cos)
                        prev_h_delta = h_delta.clone()
                        
                        ep_l.append(l_val); ep_var.append(t_var); ep_k_gain.append(t_k_gain)
                        ep_i_mean.append(dbg['innov_mean']); ep_i_max.append(dbg['innov_max'])
                        ep_avg_P.append(dbg['avg_P'])  # [v6] dynamic P tracking
                        
                        h_k_traj.append(t_k_gain); h_p_traj.append(dbg['avg_P']); h_ht_traj.append(dbg['ht_norm'])
                        h_resid_traj.append(dbg['resid_norm']); h_resid_in_innov_traj.append(dbg['resid_in_innov'])
                        h_ht_theta_traj.append(dbg['ht_theta_in_innov']); h_innov_traj.append(dbg['innov_norm'])
                        h_layer_ht.append(dbg['per_layer_ht']); h_layer_delta.append(dbg['per_layer_delta'])
                        h_layer_resid_max.append(dbg['per_layer_resid_max'])
                        h_layer_cond.append(dbg['per_layer_cond']); h_layer_ymax.append(dbg['per_layer_ymax'])

                        # [probe] Per-h activation regime: 이 fold의 operating point(theta_before_h)에서
                        #   batch_hist[h]['s']를 흘려 pre-activation regime + effective gain 측정.
                        if do_act_regime:
                            s_reg = batch_hist[h]['s']
                            if normalizer: s_reg = normalizer.normalize(s_reg)
                            reg = compute_act_regime(theta_before_h, info, s_reg, cfg.activation_fn)
                            tot = reg['__total__']
                            h_gain_traj.append(tot['mean_gain'])
                            h_pos_traj.append(tot['frac_pos'])
                            h_maxz_traj.append(tot['max_abs_z'])
                            h_layer_gain.append({l: reg[l]['mean_gain']
                                                 for l in reg if l != '__total__'})

                        # [sigma-spread] step이 dbg에 실어준 시그마 클라우드 레이어별 통계 수집.
                        if 'sigma_spread' in dbg:
                            ss = dbg['sigma_spread']
                            h_layer_spread.append({l: ss[l]['spread'] for l in ss if l != '__total__'})
                            h_layer_amp.append({l: ss[l]['amp'] for l in ss if l != '__total__'})
                            h_layer_spos.append({l: ss[l]['frac_pos'] for l in ss if l != '__total__'})

                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            argmax_after = forward_single(theta.squeeze(), info, s_flip).argmax(dim=0)
                            ep_argmax_flips.append((argmax_before != argmax_after).float().mean().item())
                    
                    if h_layer_cond:
                        for k, v in h_layer_cond[-1].items(): ep_cond_collect.setdefault(k, []).append(v)
                        for k, v in h_layer_ymax[-1].items(): ep_ymax_collect.setdefault(k, []).append(v)

                    # [v7+] Target update: soft 또는 hard (Twin이면 양쪽 갱신)
                    param_update_count += 1
                    if cfg.target_update_mode == 'soft':
                        theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                        if cfg.use_twin:
                            theta_target_2 = (1.0 - cfg.tau_srrhuif) * theta_target_2 + cfg.tau_srrhuif * theta_2
                    else:  # 'hard'
                        if param_update_count % cfg.target_update_period == 0:
                            theta_target = theta.clone()
                            if cfg.use_twin:
                                theta_target_2 = theta_2.clone()
                    
                    # [v9+] PER priority update — horizon 종료 후 최신 theta로 TD 재계산
                    if cfg.use_per:
                        idx_per, td_per = _compute_per_priorities(
                            theta, theta_target, list(batch_hist), sp, cfg, normalizer
                        )
                        if idx_per is not None:
                            buffer.update_priorities(idx_per, td_per)
                    
                    last_h_k_traj, last_h_p_traj, last_h_ht_traj = h_k_traj, h_p_traj, h_ht_traj
                    last_h_resid_traj, last_h_cos_traj = h_resid_traj, h_cos_traj
                    last_h_innov_decomp = list(zip(h_resid_in_innov_traj, h_ht_theta_traj, h_innov_traj))
                    last_h_layer_ht, last_h_layer_delta = h_layer_ht, h_layer_delta
                    last_h_layer_resid_max = h_layer_resid_max
                    last_h_layer_cond, last_h_layer_ymax = h_layer_cond, h_layer_ymax
                    if do_act_regime:  # 프로브가 돈 horizon에서만 갱신 (off 에피소드엔 직전 값 유지)
                        last_h_gain_traj, last_h_pos_traj, last_h_maxz_traj = h_gain_traj, h_pos_traj, h_maxz_traj
                        last_h_layer_gain = h_layer_gain
                    if do_sigma_spread and h_layer_spread:
                        last_h_layer_spread, last_h_layer_amp, last_h_layer_spos = h_layer_spread, h_layer_amp, h_layer_spos

                update_times.append(time.perf_counter() - update_start)

            if done or trunc: break

        # [video] 지정 에피소드마다 현재 θ로 greedy rollout을 백그라운드 mp4 녹화
        maybe_record_video(theta, info, cfg, ep)

        avg_l = np.mean(ep_l) if ep_l else 0
        avg_v = np.mean(ep_var) if ep_var else 0
        avg_k = np.mean(ep_k_gain) if ep_k_gain else 0 
        avg_q0 = np.mean(ep_q0) if ep_q0 else 0
        avg_q1 = np.mean(ep_q1) if ep_q1 else 0
        avg_i_mean = np.mean(ep_i_mean) if ep_i_mean else 0
        max_i_max = np.max(ep_i_max) if ep_i_max else 0
        avg_P_ep = np.mean(ep_avg_P) if ep_avg_P else cfg.p_init  # [v6] dynamic avg P
        
        logger.add(ep_r, avg_l, avg_P_ep, avg_v, avg_k, avg_q0, avg_q1)
        theta_norms = compute_layer_theta_norms(theta, info)
        null_ratio, null_abs, signal_abs = compute_advantage_null_ratio(theta, info)
        
        eff_rank_val, stable_rank_val = -1.0, -1.0
        if cfg.diag_eff_rank and buffer.current_size >= 128:
            with torch.no_grad():
                diag_batch = buffer.sample_batch(min(256, buffer.current_size))
                s_diag = normalizer.normalize(diag_batch['s'].t()) if normalizer else diag_batch['s'].t()
                _, shared_out = forward_single_with_shared(theta.squeeze(), info, s_diag)
                eff_rank_val, stable_rank_val = compute_effective_rank(shared_out)
        
        avg_cond_dict = {k: float(np.mean(v)) for k, v in ep_cond_collect.items()}
        avg_ymax_dict = {k: float(np.mean(v)) for k, v in ep_ymax_collect.items()}
        avg_argmax_flip = float(np.mean(ep_argmax_flips)) if ep_argmax_flips else 0.0
        
        ref_q = compute_ref_q_values(theta, info, normalizer, cfg.device) if cfg.diag_ref_states else None
        logger.add_diagnostics(avg_cond_dict, avg_ymax_dict, theta_norms, null_ratio, eff_rank_val, stable_rank_val, avg_argmax_flip, ref_q)
        
        buf_info = compute_buffer_diversity(buffer) if cfg.diag_buffer else None
        logger.add_buffer_diag(buf_info, ep)
        just_saturated = buf_info is not None and buf_info['is_saturated'] and not prev_buf_saturated
        if just_saturated: prev_buf_saturated = True

        # [v9+] Activation health (hidden 레이어 포화/죽은 뉴런)
        act_health = None
        if cfg.diag_act_health and buffer.current_size >= 32:
            n_act = min(cfg.act_health_n_sample, buffer.current_size)
            with torch.no_grad():
                idx_ah = torch.randperm(buffer.current_size, device=buffer.device)[:n_act]
                s_ah = buffer.S[idx_ah]
                s_ah = normalizer.normalize(s_ah) if normalizer else s_ah
                act_health = compute_activation_health(
                    theta.squeeze(), info, s_ah, cfg.activation_fn,
                    sat_thresh=cfg.act_health_sat_thresh,
                    dead_thresh=cfg.act_health_dead_thresh,
                )
        
        ep_delta = theta.squeeze() - theta_ep_start
        ep_delta_norm = torch.norm(ep_delta).item()
        if prev_ep_delta is not None and ep_delta_norm > 1e-8 and torch.norm(prev_ep_delta) > 1e-8:
            last_ep_cos = F.cosine_similarity(ep_delta.unsqueeze(0), prev_ep_delta.unsqueeze(0)).item()
        else: last_ep_cos = None
        prev_ep_delta = ep_delta.clone()
        target_drift = torch.norm(theta_target.squeeze() - theta.squeeze()).item()

        if ep % cfg.plot_interval == 0: logger.refresh()
        
        if ep % cfg.log_interval == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            sat_marker = " 🔔BUF_SATURATED" if just_saturated else ""
            
            prefix_tag = "[RHUKF]" if is_rhukf else "[SRRHUIF]"
            
            print(f"{prefix_tag} Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} | eps: {eps:.2f} | Buf: {buffer.current_size}/{cfg.buffer_size}{sat_marker} "
                  f"| Loss: {avg_l:.4f} | T_Var: {avg_v:.4f} | Q_std: {sp.get('current_q_std', cfg.q_init):.1e} | R_std: {sp.get('current_r_std', cfg.r_init):.1e} | P_avg: {avg_P_ep:.4f} (P0={eff_prior:.2f}) | K_Gain: {avg_k:.4f} "
                  f"| Q(0): {avg_q0:.2f} | Q(1): {avg_q1:.2f} | FB(chol/qr): {FALLBACK_COUNTS['chol_1e5']}/{FALLBACK_COUNTS['tria_qr']} | Time: {time.time()-ep_start:.2f}s")

            # ── [분석 레이어] 핵심 원인 진단(VERDICT) + verbosity gating ──
            _fb_delta = (FALLBACK_COUNTS['chol_1e5'] - _PREV_FB['chol_1e5']) \
                      + (FALLBACK_COUNTS['tria_qr'] - _PREV_FB['tria_qr'])
            _eff_ref = (cfg.shared_layers[-1] * 0.3) if cfg.shared_layers else None
            _diag_data = {
                'gain_h': last_h_gain_traj, 'amp_layer_h': last_h_layer_amp,
                'ht_layer_h': last_h_layer_ht, 'delta_layer_h': last_h_layer_delta,
                'cond_layer_h': last_h_layer_cond, 'p_traj': last_h_p_traj,
                'k_traj': last_h_k_traj, 'innov_decomp': last_h_innov_decomp,
                'dead_ratio': (act_health['__total__']['dead_ratio'] if act_health else None),
                'eff_rank': eff_rank_val, 'eff_rank_ref': _eff_ref,
                'argmax_flip': avg_argmax_flip, 'max_innov': max_i_max, 'fb_delta': _fb_delta,
            }
            _verdicts, _culprit, _trend = build_log_diagnosis(_diag_data, cfg)
            if _verdicts:
                print(f"        ⚑ VERDICT: " + " | ".join(_verdicts[:2]))
                if _culprit: print(f"        culprit: {_culprit}")
            else:
                print(f"        ⚑ VERDICT: OK")
            print(f"        trend: {_trend}")
            _PREV_FB['chol_1e5'], _PREV_FB['tria_qr'] = FALLBACK_COUNTS['chol_1e5'], FALLBACK_COUNTS['tria_qr']

            verbose = (cfg.diag_log_mode == 'always') or (cfg.diag_log_mode == 'auto' and len(_verdicts) > 0)
            dprint = file_print if verbose else (lambda *a, **k: None)

            dprint(f"          └─▶ Innov (Mean / Max): [{avg_i_mean:.4f} / {max_i_max:.4f}]")

            if verbose and last_h_k_traj:
                fmt = lambda traj: "[" + ", ".join([f"{v:.4f}" for v in traj]) + "]"
                fmt_e = lambda traj: "[" + ", ".join([f"{v:.2e}" for v in traj]) + "]"
                fmt2 = lambda traj: "[" + ", ".join([f"{v:+.3f}" for v in traj]) + "]"
                file_print(f"          └─▶ K_Gain/h:  {fmt(last_h_k_traj)}")
                file_print(f"          └─▶ P_avg/h:   {fmt_e(last_h_p_traj)}")
                if last_h_innov_decomp:
                    file_print(f"          └─▶ |z-ẑ|/h:   {fmt([d[0] for d in last_h_innov_decomp])}")
                    file_print(f"          └─▶ |H^Tθ|/h:  {fmt([d[1] for d in last_h_innov_decomp])}")
                    file_print(f"          └─▶ |innov|/h: {fmt([d[2] for d in last_h_innov_decomp])}")
                if last_h_cos_traj:
                    file_print(f"          └─▶ cos(δ)/h:  {fmt2(last_h_cos_traj)}")
                # [probe] Per-h activation regime: fold 따라 증가하면 runaway 시그니처
                if last_h_gain_traj:
                    fmt_pct = lambda traj: "[" + ", ".join([f"{100*v:.1f}%" for v in traj]) + "]"
                    file_print(f"          └─▶ gain/h:    {fmt(last_h_gain_traj)}")
                    file_print(f"          └─▶ pos%/h:    {fmt_pct(last_h_pos_traj)}")
                    file_print(f"          └─▶ maxz/h:    {fmt(last_h_maxz_traj)}")
                    if last_h_layer_gain:
                        g_labels = sorted(last_h_layer_gain[0].keys())
                        for h_idx in range(len(last_h_layer_gain)):
                            file_print(f"          ├─▶ gain   h={h_idx}:  " + " ".join([f"{l}={last_h_layer_gain[h_idx][l]:.3f}" for l in g_labels]))
                ep_cos_str = f"{last_ep_cos:+.3f}" if last_ep_cos is not None else "N/A"
                file_print(f"          └─▶ ep_cos: {ep_cos_str} | θ-target drift: {target_drift:.4f} | ep_Δθ: {ep_delta_norm:.4f}")
                
                if last_h_layer_ht:
                    file_print(f"          ══ [Tier 1] Layer-wise Diagnostics ══")
                    labels = sorted(last_h_layer_ht[0].keys())
                    for h_idx in range(len(last_h_layer_ht)):
                        file_print(f"          ├─▶ ||H^T|| h={h_idx}:  " + " ".join([f"{l}={last_h_layer_ht[h_idx][l]:.2f}" for l in labels]))
                    for h_idx in range(len(last_h_layer_resid_max)):
                        file_print(f"          ├─▶ ResMax  h={h_idx}:  " + " ".join([f"{l}={last_h_layer_resid_max[h_idx][l]:.2f}" for l in labels]))
                    for h_idx in range(len(last_h_layer_delta)):
                        file_print(f"          ├─▶ ||Δθ||  h={h_idx}:  " + " ".join([f"{l}={last_h_layer_delta[h_idx][l]:.4f}" for l in labels]))
                    max_ht_per_layer = {l: max(last_h_layer_ht[h][l] for h in range(len(last_h_layer_ht))) for l in labels}
                    dominant = max(max_ht_per_layer, key=max_ht_per_layer.get)
                    file_print(f"          └─▶ Dominant layer: {dominant} (max||H^T||={max_ht_per_layer[dominant]:.1f})")

                # [sigma-spread] 세로=horizon, 가로=layer. amp>1이 fold 따라 커지면 runaway.
                if last_h_layer_spread:
                    ss_labels = sorted(last_h_layer_spread[0].keys())
                    file_print(f"          ══ [Tier 2] Sigma-Spread Activation ══")
                    for h_idx in range(len(last_h_layer_spread)):
                        file_print(f"          ├─▶ spread h={h_idx}:  " + " ".join([f"{l}={last_h_layer_spread[h_idx][l]:.3f}" for l in ss_labels]))
                    for h_idx in range(len(last_h_layer_amp)):
                        file_print(f"          ├─▶ amp    h={h_idx}:  " + " ".join([f"{l}={last_h_layer_amp[h_idx][l]:.3f}" for l in ss_labels]))
                    for h_idx in range(len(last_h_layer_spos)):
                        file_print(f"          ├─▶ pos%   h={h_idx}:  " + " ".join([f"{l}={100*last_h_layer_spos[h_idx][l]:.0f}%" for l in ss_labels]))
                    # 레이어별 amp가 horizon 동안 1을 넘는 fold가 있으면 증폭(runaway) 신호
                    amp_max_per_layer = {l: max(last_h_layer_amp[h][l] for h in range(len(last_h_layer_amp))) for l in ss_labels}
                    hot = max(amp_max_per_layer, key=amp_max_per_layer.get)
                    file_print(f"          └─▶ Max amp layer: {hot} (peak amp={amp_max_per_layer[hot]:.3f}; >1=증폭)")

            dprint(f"          ══ DIAGNOSTICS ══")
            if verbose and last_h_layer_cond:
                labels = sorted(last_h_layer_cond[0].keys())
                for h_idx in range(len(last_h_layer_cond)):
                    file_print(f"          ├─▶ cond(Y)h={h_idx}: " + " ".join([f"{l}={last_h_layer_cond[h_idx][l]:.1e}" for l in labels]))
                for h_idx in range(len(last_h_layer_ymax)):
                    file_print(f"          ├─▶ Y_max  h={h_idx}: " + " ".join([f"{l}={last_h_layer_ymax[h_idx][l]:.1e}" for l in labels]))
            
            labels = sorted(theta_norms.keys())
            dprint(f"          ├─▶ ||θ|| per layer:   " + " ".join([f"{l}={theta_norms[l]:.3f}" for l in labels]))
            dprint(f"          ├─▶ Adv/Q null/signal: ratio={null_ratio:.4f} (null={null_abs:.4f}, signal={signal_abs:.4f})")

            # [v9+] Activation health (포화 / 죽은 뉴런)
            if verbose and act_health is not None:
                tot = act_health['__total__']
                file_print(
                    f"          ├─▶ Act health ({cfg.activation_fn}): "
                    f"sat={tot['n_sat']}/{tot['n_units']} ({100*tot['sat_ratio']:.1f}%)  "
                    f"dead={tot['n_dead']}/{tot['n_units']} ({100*tot['dead_ratio']:.1f}%)"
                )
                ah_labels = sorted([k for k in act_health.keys() if k != '__total__'])
                if ah_labels:
                    file_print(
                        "          ├─▶ Act per layer:      "
                        + " ".join([
                            f"{l}[sat={act_health[l]['n_sat']}/{act_health[l]['n_units']},"
                            f"dead={act_health[l]['n_dead']}/{act_health[l]['n_units']},"
                            f"|a|={act_health[l]['mean_abs']:.2f},fire={act_health[l]['fire_rate']:.2f}]"
                            for l in ah_labels
                        ])
                    )
                    file_print(
                        "          ├─▶ Pre-act stats:      "
                        + " ".join([
                            f"{l}[μ={act_health[l]['pre_mean']:+.2f},σ={act_health[l]['pre_std']:.2f}]"
                            for l in ah_labels
                        ])
                    )
            if verbose and eff_rank_val > 0:
                file_print(f"          ├─▶ Shared rank:       eff={eff_rank_val:.1f}/{cfg.shared_layers[-1] if cfg.shared_layers else 'N/A'}, stable={stable_rank_val:.2f}")
            dprint(f"          ├─▶ Argmax flip rate:  {avg_argmax_flip:.4f} (updates={len(ep_argmax_flips)})")
            if verbose and buf_info is not None:
                sat_str = "YES" if buf_info['is_saturated'] else "no"
                file_print(f"          ├─▶ Buffer diag:       fill={buf_info['fill_ratio']:.3f}({sat_str}) state_std={buf_info['state_std']:.4f} state_range={buf_info['state_range']:.3f}")
                file_print(f"          ├─▶ Buffer samples:    done_ratio={buf_info['done_ratio']:.4f} r_std={buf_info['reward_std']:.4f} r_mean={buf_info['reward_mean']:.4f}")
                file_print(f"          ├─▶ Buffer age:        ep[{buf_info['age_min']}..{buf_info['age_max']}] range={buf_info['age_range']} std={buf_info['age_std']:.2f}")
            if verbose and ref_q:
                file_print(f"          └─▶ Ref states:        " + " ".join([f"{name}:ΔQ={ref_q[name]['dq']:+.4f}(a={ref_q[name]['argmax']})" for name in REF_NAMES]))

    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
    env.close()
    logger.refresh()
    logger.save_diagnostic_plots()
    
    try:
        plot_cartpole_state_landscape(theta, info, cfg, normalizer, method_title, cfg.param_str)
    except Exception as e: print(f"[경고] 지형도 생성 중 오류 발생: {e}")
        
    logger.close()


# =========================================================================
# 11b. Pure Adam DDQN training (compare mode)
# =========================================================================
def train_adam():
    """순수 Adam DDQN 학습 루프 (filter 없음).
    Compare 용 — train_mode='adam'일 때 호출.
    측정식은 cfg.measurement_mode를 그대로 사용 (q_target / pure_reward).
    Filter-specific 진단은 모두 생략, 공통 진단(theta norms, act health 등)은 유지.
    """
    net_seed = cfg.network_seed if cfg.network_seed is not None else cfg.seed
    env_seed = cfg.env_seed if cfg.env_seed is not None else cfg.seed
    set_all_seeds(net_seed)
    apply_tf32_config(cfg)  # cfg가 코드에서 바뀐 경우에도 반영 (idempotent)
    env = gym.make(cfg.env_name, **build_env_kwargs(cfg))
    env.action_space.seed(net_seed)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)

    method_title = f"{'D3QN' if cfg.use_dueling else 'DDQN'} + ADAM"

    print(f"\n{'='*60}")
    print(f"  Pure Adam {method_title} v9.0 (compare mode)")
    print(f"  loss form: q_target (semi-gradient TD, always — pure_reward는 Adam 비호환)")
    print(f"  lr={cfg.adam_lr:g} | huber_c={cfg.huber_c} | batch={cfg.batch_size}")
    if cfg.measurement_mode == 'pure_reward':
        print(f"  NOTE: cfg.measurement_mode='pure_reward' 설정돼 있으나 무시하고 q_target 사용.")
    print(f"  Params: {info['total_params']} | Output Dir: {cfg.outdir}")
    print(f"  Seeds: network={net_seed}, env={env_seed}")
    print(f"{'='*60}\n")

    normalizer = InputNormalizer(cfg.device, cfg.obs_scale) if (cfg.use_input_norm and cfg.obs_scale) else None
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size,
          'normalizer': normalizer, 'device': cfg.device, 'cfg': cfg}

    theta = initialize_theta(info, cfg.device, cfg).view(-1, 1)
    theta_init = theta.clone()
    sp['theta_init'] = theta_init
    theta_target = theta.clone()

    theta_param = nn.Parameter(theta.squeeze().clone().detach(), requires_grad=True)
    adam_opt = torch.optim.Adam([theta_param], lr=cfg.adam_lr)

    analyze_initial_network(theta, info, env, cfg, normalizer)

    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device, cfg)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)

    logger = LivePlotter(method_title, cfg.max_episodes, cfg.param_str,
                         filter_form='covariance')  # plotter는 dummy로 채움

    steps_done = 0
    param_update_count = 0
    train_start_time = time.time()
    update_times = []
    prev_buf_saturated = False
    prev_ep_delta = None

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=env_seed + ep)
        buffer.set_current_episode(ep)

        ep_r, ep_l, ep_start = 0, [], time.time()
        ep_q0, ep_q1 = [], []
        theta_ep_start = theta.squeeze().clone()

        for t in range(cfg.max_steps):
            steps_done += 1
            if steps_done <= cfg.warmup_step:
                eps = 1.0
            else:
                active_steps = steps_done - cfg.warmup_step
                decay_factor = np.exp(-active_steps / cfg.eps_decay_steps)
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * decay_factor

            with torch.no_grad():
                s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                s_t = s_t_buffer
                if normalizer: s_t = normalizer.normalize(s_t)
                q_vals = forward_single(theta.squeeze(), info, s_t).squeeze()
                ep_q0.append(q_vals[0].item())
                ep_q1.append(q_vals[1].item())

            a = env.action_space.sample() if np.random.rand() < eps else q_vals.argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if steps_done > cfg.warmup_step and buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                t_upd = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)

                adam_opt.zero_grad(set_to_none=True)
                loss = compute_adam_td_loss(theta_param, theta_target, batch, sp, cfg)
                loss.backward()
                adam_opt.step()
                with torch.no_grad():
                    theta.data.copy_(theta_param.data.view(-1, 1))
                ep_l.append(float(loss.detach().item()))

                # target net update
                param_update_count += 1
                if cfg.target_update_mode == 'soft':
                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                else:
                    if param_update_count % cfg.target_update_period == 0:
                        theta_target = theta.clone()

                # PER priority update
                if cfg.use_per:
                    idx_per, td_per = _compute_per_priorities(
                        theta, theta_target, [batch], sp, cfg, normalizer
                    )
                    if idx_per is not None:
                        buffer.update_priorities(idx_per, td_per)

                update_times.append(time.perf_counter() - t_upd)

            if done or trunc: break

        # [video] 지정 에피소드마다 현재 θ로 greedy rollout을 백그라운드 mp4 녹화
        maybe_record_video(theta, info, cfg, ep)

        avg_l = np.mean(ep_l) if ep_l else 0.0
        avg_q0 = np.mean(ep_q0) if ep_q0 else 0.0
        avg_q1 = np.mean(ep_q1) if ep_q1 else 0.0

        # filter-specific은 0으로 (LivePlotter 시그니처 유지)
        logger.add(ep_r, avg_l, 0.0, 0.0, 0.0, avg_q0, avg_q1)

        # 공통 진단들
        theta_norms = compute_layer_theta_norms(theta, info)
        null_ratio, null_abs, signal_abs = compute_advantage_null_ratio(theta, info)

        eff_rank_val, stable_rank_val = -1.0, -1.0
        if cfg.diag_eff_rank and buffer.current_size >= 128:
            with torch.no_grad():
                diag_batch = buffer.sample_batch(min(256, buffer.current_size))
                s_diag = normalizer.normalize(diag_batch['s'].t()) if normalizer else diag_batch['s'].t()
                _, shared_out = forward_single_with_shared(theta.squeeze(), info, s_diag)
                eff_rank_val, stable_rank_val = compute_effective_rank(shared_out)

        ref_q = compute_ref_q_values(theta, info, normalizer, cfg.device) if cfg.diag_ref_states else None
        logger.add_diagnostics({}, {}, theta_norms, null_ratio, eff_rank_val, stable_rank_val, 0.0, ref_q)

        buf_info = compute_buffer_diversity(buffer) if cfg.diag_buffer else None
        logger.add_buffer_diag(buf_info, ep)
        just_saturated = buf_info is not None and buf_info['is_saturated'] and not prev_buf_saturated
        if just_saturated: prev_buf_saturated = True

        # Activation health
        act_health = None
        if cfg.diag_act_health and buffer.current_size >= 32:
            n_act = min(cfg.act_health_n_sample, buffer.current_size)
            with torch.no_grad():
                idx_ah = torch.randperm(buffer.current_size, device=buffer.device)[:n_act]
                s_ah = buffer.S[idx_ah]
                s_ah = normalizer.normalize(s_ah) if normalizer else s_ah
                act_health = compute_activation_health(
                    theta.squeeze(), info, s_ah, cfg.activation_fn,
                    sat_thresh=cfg.act_health_sat_thresh,
                    dead_thresh=cfg.act_health_dead_thresh,
                )

        ep_delta = theta.squeeze() - theta_ep_start
        ep_delta_norm = torch.norm(ep_delta).item()
        if prev_ep_delta is not None and ep_delta_norm > 1e-8 and torch.norm(prev_ep_delta) > 1e-8:
            last_ep_cos = F.cosine_similarity(ep_delta.unsqueeze(0), prev_ep_delta.unsqueeze(0)).item()
        else:
            last_ep_cos = None
        prev_ep_delta = ep_delta.clone()
        target_drift = torch.norm(theta_target.squeeze() - theta.squeeze()).item()

        if ep % cfg.plot_interval == 0: logger.refresh()

        if ep % cfg.log_interval == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            sat_marker = " 🔔BUF_SATURATED" if just_saturated else ""

            print(f"[ADAM] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} | eps: {eps:.2f} "
                  f"| Buf: {buffer.current_size}/{cfg.buffer_size}{sat_marker} "
                  f"| Loss: {avg_l:.4f} | lr: {cfg.adam_lr:.1e} "
                  f"| Q(0): {avg_q0:.2f} | Q(1): {avg_q1:.2f} "
                  f"| Updates: {param_update_count} | Time: {time.time()-ep_start:.2f}s")

            ep_cos_str = f"{last_ep_cos:+.3f}" if last_ep_cos is not None else "N/A"
            file_print(f"          └─▶ ep_cos: {ep_cos_str} | θ-target drift: {target_drift:.4f} | ep_Δθ: {ep_delta_norm:.4f}")

            file_print(f"          ══ DIAGNOSTICS ══")
            labels = sorted(theta_norms.keys())
            file_print(f"          ├─▶ ||θ|| per layer:   " + " ".join([f"{l}={theta_norms[l]:.3f}" for l in labels]))
            file_print(f"          ├─▶ Adv/Q null/signal: ratio={null_ratio:.4f} (null={null_abs:.4f}, signal={signal_abs:.4f})")

            if act_health is not None:
                tot = act_health['__total__']
                file_print(
                    f"          ├─▶ Act health ({cfg.activation_fn}): "
                    f"sat={tot['n_sat']}/{tot['n_units']} ({100*tot['sat_ratio']:.1f}%)  "
                    f"dead={tot['n_dead']}/{tot['n_units']} ({100*tot['dead_ratio']:.1f}%)"
                )
                ah_labels = sorted([k for k in act_health.keys() if k != '__total__'])
                if ah_labels:
                    file_print(
                        "          ├─▶ Act per layer:      "
                        + " ".join([
                            f"{l}[sat={act_health[l]['n_sat']}/{act_health[l]['n_units']},"
                            f"dead={act_health[l]['n_dead']}/{act_health[l]['n_units']},"
                            f"|a|={act_health[l]['mean_abs']:.2f},fire={act_health[l]['fire_rate']:.2f}]"
                            for l in ah_labels
                        ])
                    )
                    file_print(
                        "          ├─▶ Pre-act stats:      "
                        + " ".join([
                            f"{l}[μ={act_health[l]['pre_mean']:+.2f},σ={act_health[l]['pre_std']:.2f}]"
                            for l in ah_labels
                        ])
                    )

            if eff_rank_val > 0:
                file_print(f"          ├─▶ Shared rank:       eff={eff_rank_val:.1f}/{cfg.shared_layers[-1] if cfg.shared_layers else 'N/A'}, stable={stable_rank_val:.2f}")
            if buf_info is not None:
                sat_str = "YES" if buf_info['is_saturated'] else "no"
                file_print(f"          ├─▶ Buffer diag:       fill={buf_info['fill_ratio']:.3f}({sat_str}) state_std={buf_info['state_std']:.4f} state_range={buf_info['state_range']:.3f}")
                file_print(f"          ├─▶ Buffer samples:    done_ratio={buf_info['done_ratio']:.4f} r_std={buf_info['reward_std']:.4f} r_mean={buf_info['reward_mean']:.4f}")
                file_print(f"          ├─▶ Buffer age:        ep[{buf_info['age_min']}..{buf_info['age_max']}] range={buf_info['age_range']} std={buf_info['age_std']:.2f}")
            if ref_q:
                file_print(f"          └─▶ Ref states:        " + " ".join([f"{name}:ΔQ={ref_q[name]['dq']:+.4f}(a={ref_q[name]['argmax']})" for name in REF_NAMES]))

    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0
    env.close()
    logger.refresh()
    logger.save_diagnostic_plots()

    try:
        plot_cartpole_state_landscape(theta, info, cfg, normalizer, method_title, cfg.param_str)
    except Exception as e:
        print(f"[경고] 지형도 생성 중 오류 발생: {e}")

    logger.close()


if __name__ == "__main__":
    if cfg.save_file_log: setup_file_logging(os.path.join(cfg.outdir, "training_log.txt"))
    try:
        if cfg.train_mode == 'adam':
            train_adam()
        else:
            train_srrhuif()
    finally:
        finalize_videos()
        if cfg.save_file_log: close_file_logging()