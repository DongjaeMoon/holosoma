# 새 Codex 계정 인계 프롬프트

> **최신 종합 인계는 [HOLOSOMA_RESUME_PROMPT.md](HOLOSOMA_RESUME_PROMPT.md)를 읽으세요.** 2026-10-29: v2 네 모델 10k 학습, 전체 체크포인트 평가, PNG 및 최종 좌우 분할 영상까지 모두 완료됐습니다. 아래는 초기 구현 당시 기록이므로 현재 실행 상태로 해석하지 마세요.



이 문서를 읽고 기존 작업을 이어받아라. 사용자는 응답과 실행이 매우 느려진 데 크게 불만이 있다. 핵심부터 짧게 말하고, 불필요한 도구 호출·재승인·긴 계획을 줄여라. 이 문서는 구현 상태와 한계를 포함한 인계이며, 모든 실험이 검증되었다는 뜻이 아니다.

## 1. 사용자의 연구 목표와 요구사항

Holosoma의 기존 학습 시스템에 사용자의 연구 네트워크 BilateralMAGNet 구조를 이식해 비교하고 싶다.

원래 요구는 알고리즘, reward, observation, command, action scale, curriculum 등 Holosoma 조건을 유지하고 네트워크 구조만 바꾸는 것이었다. 이후 사용자는 네트워크의 좌우 observation/action pairing과 mirroring이 다르면 domi 원본 로직을 우선 사용하라고 명시했고, phase도 domi 방식을 사용하라고 했다.

구현 스타일 요구:
- 이미 있는 원본 파일과 함수 본문을 최대한 그대로 복사.
- 최소 수정, 최소 길이, 캡슐화, 높은 가독성.
- 1차 실험부터 Actor와 Critic이 router를 공유해야 함.
- 공유 router 파라미터를 두 optimizer에서 중복 업데이트하면 안 됨.

세 실험:
1. 기본 vanilla Holosoma.
2. Bilateral Holosoma.
3. Bilateral Holosoma를 전진/후진/왼쪽 명령만 샘플링하여 학습한 후 오른쪽 보행 일반화를 평가.

실행 제약:
- 학습/렌더링은 Docker 내부에서 실행.
- 물리 GPU **3번만** 사용. 다른 GPU로 테스트·학습·렌더링 금지.
- 세 학습은 동시에 돌리지 않고 1 → 2 → 3 순차 실행.
- 사용자는 처음에는 준비만 요청했으나, 이후 "그냥 밥먹고 갔다옴 쭉 실험 돌려놔라"라고 실제 실행을 승인했다. 현재 1번이 실행 중이다.
- TensorBoard, 체크포인트별 실제 보행 MP4, localhost URL을 통한 확인도 요구했다.
- 기존 프로세스를 중복 실행하거나 다른 GPU를 쓰지 말고 현재 상태부터 확인할 것.

## 2. 현재 가장 중요한 미해결 문제

사용자는 과거 **PPO로 15,000 정도까지 30분 이내** 학습했던 기억이 있다고 명시적으로 확인했다. FastSAC였냐는 질문에 **PPO였음**이라고 답했다. 따라서 FastSAC 차이 또는 단위 차이로 단정해서 설명하지 말 것.

현재 실행은 iteration당 약 3.1초, 25,000 iteration 전체 ETA 약 21시간이라 사용자가 설정과 구현을 의심하고 있다.

또한 기존 에이전트는 현재 baseline을 "완전 기본 vanilla"라고 표현했지만, 실제로는 공유 router를 위한 hook을 **기본 PPO 코드에도 수정**했다. 프리셋은 공식 g1-29dof PPO이나 실행 코드가 완전히 무수정인 공식 원본은 아니다. 마지막 답변에서 이 점을 사용자에게 인정했다.

**우선순위:**
1. 현재 baseline과 공식 원본의 동일성을 제대로 검증.
2. PPO 속도 차이의 실제 원인을 진단. 아직 원인 확정 안 됨.
3. 필요하면 원본 baseline을 격리하여 비교하되, 체크포인트/로그를 보존하고 실행 중인 큐를 함부로 중복 실행하지 말 것.
4. 사용자는 기다림을 매우 싫어한다. 현황 질문에는 먼저 현재 현황을 짧게 답할 것.

## 3. 위치, Docker, 코드 버전

호스트 작업 경로: `/home/ycho/holosoma`

Holosoma Docker:
- 이름: `ycho-holosoma`
- 확인했던 ID: `f0e2bb2413de`
- 호스트 `/home/ycho/holosoma` → Docker `/workspace/holosoma` bind mount.
- Docker 네트워크는 bridge.
- Docker 실행 시 모든 GPU 접근 권한이 있으므로 큐에서 CUDA_VISIBLE_DEVICES=3으로 제한하고 Kit multi-GPU를 비활성화했다.

연구 Docker:
- 이름: `147-sejune-domi-isaac-lab`
- 확인했던 ID: `c0c596b94d49`
- 실제 연구 레포 경로는 `/home/user/domi-alps`.
- 호스트 `/home/sejune/domi-alps`가 여기에 mount되지만 호스트 ycho 계정에서는 Permission denied여서 `docker exec ... cat/sed`로 읽었다.
- 사용자 예전 커맨드의 `/input/domi/...` 경로는 이 컨테이너 현재 상태에서는 존재하지 않는다.
- 연구 레포는 수정하지 않았다. CPU 검증을 위한 임시 파일만 연구 Docker `/tmp`에 복사했다.

체크아웃 HEAD:
`b739a6cb7b1f8b9918669f8f776edb1855bfda1a`
커밋 제목: `Re-render sensors and notify terms after env state is overwritten (#237)`

Git commit은 만들지 않았다. 변경은 working tree에 남아 있다. `git status --short`로 확인할 것.

## 4. 환경 활성화 주의

Holosoma Docker 내부:

```bash
cd /workspace/holosoma
source scripts/source_isaacsim_setup.sh
```

Conda 환경은 `hssim`, 경로는 `/root/.holosoma_deps/miniconda3/envs/hssim`.

**중요:** 위 source를 적용하지 않고 Python 경로만 직접 실행했을 때 torch가 없다고 잘못 판단했던 적이 있다. 활성화 스크립트가 IsaacSim 번들 PYTHONPATH 등도 설정한다. 정상 활성화 후 torch, isaacsim, isaaclab, tensorboard, tyro 모두 발견되었고 실제 학습도 실행 중이다. "PyTorch 미설치"라고 다시 잘못 보고하지 말 것.

IsaacSim 5.1.0 / IsaacLab v2.3.2 설정이 레포 scripts에 있다.

CPU 검증은 다음처럼 GPU를 숨겨 실행했다:

```bash
docker exec -e CUDA_VISIBLE_DEVICES= -w /workspace/holosoma ycho-holosoma \
  bash -lc 'source scripts/source_isaacsim_setup.sh && python scripts/check_bilateral_cpu.py'
```

## 5. 가져온 연구 네트워크와 원본 위치

연구 학습 진입점:
`/home/user/domi-alps/source/standalone/workflows/imm/train_ma_v2.py`

원본 네트워크 YAML:
`/home/user/domi-alps/source/domi/src/domi/cfg/net/sym_ma_v7_bilateral_modular_nobase.yaml`

원본 주요 구현:
- `source/domi/src/domi/nn/bilateral_magnet.py`: BilateralMAGNet / BilateralModularLinear.
- `source/domi/src/domi/nn/range_layers.py`: MAGNet router, 초기화 helper.
- `source/domi/src/domi/nn/layers.py`: MLP, normalization, symmetric parametrization, SplitDim.
- `source/domi/src/domi/util/export_compat.py`.
- `source/domi/src/domi/net/td/bilateral_magnet.py`: 연구 환경용 BilateralMAGNetMA 래퍼. 이 래퍼 전체를 이식한 것은 아님.
- `source/domi/src/domi/net/td/generic.py`: 원본 WrapActorCritic은 CompositeDistribution과 reward별 value를 만듦. Holosoma에서는 기존 Gaussian / scalar value 인터페이스에 맞춤.

원본 좌우 IO:
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_art/config/loco_env_cfg.py`
- `source/domi/src/domi/env/loco_action_mapping.py`
- `source/isaaclab/isaaclab/utils/string.py`의 resolve_matching_names_values.

사용자 연구 설정의 구조상 핵심:
- subnet hidden dims `[192,384,192]`.
- context router hidden dims `[256,256,128]` (원본 기본 dim_mod).
- num_module=4, gate_sizes=8.
- share_ab_modules=false, use_sym_param=true, use_default_module=true.
- compose_mode=feature, feature_bias_scale=0.5.
- checkpoint_layers=true, use_fast=false.
- 원본 YAML은 weight_recenter=true지만 이식에서는 추가 학습 갱신이므로 false로 고정했다. 원본 전체 학습 방법의 재현은 아님.
- bilateral_io=false, lean_obs=false, contralateral_arms=false 양쪽.
- lean_obs=false라 원본 양쪽 subnet은 자기 쪽뿐 아니라 반대쪽 관절도 관측한다.

## 6. 사용자의 원래 연구 커맨드

아래 커맨드는 연구 의도와 원본 override 추적용이다. 이 전체 reward/env 설정을 Holosoma에 가져오라는 뜻이 아니다.

```bash
python3 /input/domi/source/standalone/workflows/imm/train_ma_v2.py \
 ++env.bilateral_io=false ++env.use_per_joint_action_scale=true \
 ++env.actions.left_joint_pos.contralateral_arms=false ++env.actions.right_joint_pos.contralateral_arms=false \
 ++env.side_rewd_scope=controlled ++agent.k_ent=0.0 ++agent.entity=packman ++env.stance_ratio=0.7 \
 ++env.rewards.left_help_airtime.params.kernel=threshold_linear ++env.rewards.right_help_airtime.params.kernel=threshold_linear \
 ++env.rewards.help_feet_rhythm.params.on_bonus=0.0 ++env.rewards.help_feet_parallel.weight=0.025 \
 ++env.curriculum.help_anneal_command_classes.params.init_value.base=0.60 \
 ++env.curriculum.help_anneal_command_classes.params.last_value.base=0.40 \
 ++env.curriculum.help_anneal_base_command_classes.params.total_step=30000 \
 ++env.curriculum.help_anneal_base_command_classes.params.last_value.lin_back=0.20 \
 ++env.curriculum.help_anneal_base_command_classes.params.last_value.lin_fore=0.10 \
 ++env.curriculum.help_anneal_base_command_classes.params.last_value.lin_side=0.20 \
 ++env.curriculum.help_anneal_command_classes.params.total_step=30000 ++env.init_curr_step=0 \
 ++env.commands.base_velocity.include_world_frame_in_command=true \
 '++env.terminations.bad_pelvis_ori.params.limit_euler_angle=[1.3,1.3]' \
 ++env.rewards.left_safe_limb_speed.weight=-0.005 ++env.rewards.right_safe_limb_speed.weight=-0.005 \
 ++env.rewards.left_help_qdev.weight=-0.04 ++env.rewards.right_help_qdev.weight=-0.04 \
 ++net.nets.magnet.params.compose_mode=feature \
 ++env.curriculum.help_anneal_upright.params.init_value=-0.5 ++env.curriculum.help_anneal_upright.params.last_value=-0.5 \
 ++env.motion_diagnostics.enabled=true ++env.motion_diagnostics.report_interval_steps=1024 \
 ++env.rewards.safe_contact_margin.params.op_version=1 \
 ++env.curriculum.safe_anneal_margin.params.init_value=-5.0 ++env.curriculum.safe_anneal_margin.params.last_value=-500.0 \
 ++env.rewards.help_lidar_ang_acc.weight=-0.0025 ++env.rewards.safe_torso_extreme_posture.weight=-0.5 \
 ++env.act_q0_dq=true ++env.terminations.time_out.params.enabled=true \
 ++env.terminations.time_out.params.max_exempt_steps=500 \
 ++env.terminations.time_out.func=isaaclab_tasks.manager_based.manipulation.g1_art.mdp.terminations:CommandAwareTimeOut \
 ++agent.name=f1c7k0qo-repro-mfix-bihamv7 ++net.nets.magnet.params.feature_bias_scale=0.5 \
 ++net.nets.magnet.params.checkpoint_layers=true ++net.nets.magnet.params.use_fast=false \
 ++env.adaptive_curriculum=false ++env.lean_obs=false \
 '++env.commands.base_velocity.base_frame_difficulty_scale_range=[1.0,0.25]' \
 ++env.commands.gait_phase.contact_phase_reset=false \
 ++env.rewards.safe_contact_margin.params.velocity_modulation_vertical=false \
 ++env.scene.terrain.physics_material.restitution=0.08 ++env.sim.physics_material.restitution=0.08 \
 ++env.scene.robot.actuators.G1.velocity_limit_sim=null \
 ++env.observations.left.projected_gravity.params.noise_model=disabled \
 '++env.observations.left.projected_gravity.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.05,n_max:0.05,operation:add}' \
 ++env.observations.left.foot_pos.params.noise_model=disabled \
 '++env.observations.left.foot_pos.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.03,n_max:0.03,operation:add}' \
 ++env.observations.left.other_foot_pos.params.noise_model=disabled \
 '++env.observations.left.other_foot_pos.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.03,n_max:0.03,operation:add}' \
 ++env.observations.right.projected_gravity.params.noise_model=disabled \
 '++env.observations.right.projected_gravity.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.05,n_max:0.05,operation:add}' \
 ++env.observations.right.foot_pos.params.noise_model=disabled \
 '++env.observations.right.foot_pos.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.03,n_max:0.03,operation:add}' \
 ++env.observations.right.other_foot_pos.params.noise_model=disabled \
 '++env.observations.right.other_foot_pos.noise={func:isaaclab.utils.noise.noise_model:uniform_noise,n_min:-0.03,n_max:0.03,operation:add}' \
 ++env.phase_zero=true '++env.events.deft_robot_materials.params.restitution_range=[0.0,1.0]' \
 ++env.commands.head_height.enabled=false ++env.observations.left.head_height_cmd=null \
 ++env.observations.right.head_height_cmd=null ++env.rewards.help_head_height=null \
 ++env.curriculum.help_anneal_head_height_active=null ++env.rewards.help_narrow_lateral_profile=null \
 --ma source/domi/src/domi/cfg/env/ma_v3.yaml \
 --net source/domi/src/domi/cfg/net/sym_ma_v7_bilateral_modular_nobase.yaml \
 --seed 43 --task loco_NEW_DEFAULTS \
 --dagger source/domi/src/domi/cfg/dagger/dagger_v0.yaml --device cuda:0 --headless --num_envs 4096 \
 --playground_eval source/standalone/workflows/imm/playground_eval.yaml
```

## 7. 추가/수정 파일과 구현

### 원본 보존 디렉터리

`src/holosoma/holosoma/agents/modules/bilateral/original/`

- `bilateral_magnet.py`: 원본 파일 전체를 복사하고 import 경로를 로컬로 변경.
- `range_layers.py`: 원본 MAGNet과 필요한 정의를 추출. 함수와 클래스 본문은 보존.
- `layers.py`: 필요한 MLP, normalization, symmetric, SplitDim 정의를 추출하여 불필요한 flash_attn 의존성을 피함.
- `export_compat.py`: 원본 복사.
- `loco_io.py`: loco_env_cfg.py의 joint_mirror_sign_cfg, swap_lr, _mirror_lin_vel, _mirror_ang_vel, _mirror_vel_cmd, _mirror_joint_values, _mirror_phase, _mirror_act, last_action 본문을 그대로 복사.
- `joint_layout.py`: make_obs_cfg / make_act_cfg의 관측·행동 joint 목록 선언을 추출해 함수로 감쌈. 관측과 action의 ankle 등 순서 차이도 보존.
- `loco_action_mapping.py`: 원본 pair 검증과 action ownership resolver.
- `isaac_strings.py`: 원본 Isaac 이름·부호 resolver 함수.
- `network.yaml`: 원본 network YAML 보존.
- `provenance.json`: 원본 파일 경로, SHA256, 추출한 정의 기록.

### Holosoma 연결

- `agents/modules/bilateral/adapter.py`
  - ObservationView: 기존 Holosoma flat 관측에서 domi 순서와 부호로 left/right/context gather를 생성.
  - 초기 구현에서는 Holosoma SymmetryUtils로 mirroring했다가 사용자의 요구로 제거하고 domi 함수를 사용하도록 변경했다.
  - lean_obs=false에 맞춰 양쪽 관절 정보를 유지.
  - 기존 29D physical action history로부터 각 16D side raw-action 표현을 재구성하여 [own, other]로 배치. 허리는 두 번 들어간다. 원본의 합성 전 개별 waist action history를 완전히 복원할 수 있는 것은 아니므로 원본 환경 전체와 동일하다고 단정하지 말 것.
  - **Phase 현재 구현:** Holosoma 첫 번째 발의 sin/cos에서 전역 2채널을 구성하고 domi `_mirror_phase(x)=-x` 적용. domi PhaseCommand 생성기 전체를 이식한 것은 아니다. 사용자 요구 충족 여부를 명확히 검토해야 한다.
  - BilateralPolicy: 하나의 원본 BilateralMAGNet을 소유. router context는 Actor 관측만 사용하여 privileged critic 관측이 actor로 누출되지 않게 함.
  - 중앙 waist는 양쪽 예측을 physical 좌표로 복원한 뒤 평균.
  - 양쪽 critic 출력을 평균해 Holosoma scalar value로 맞춤. 원본 domi reward별 value 전체는 이식 안 함.
  - Actor 단독 추론 때 사용하지 않는 critic branch에는 0을 넣지만 actor/router는 그 입력에 의존하지 않는다.
  - 현재 v1은 single-frame, G1 left_/right_ 관절 이름, 관절당 하나의 action 등 제약이 있음.
- `agents/ppo/bilateral_ppo.py`
  - PPO를 상속, 모델 생성·공동 forward·bootstrap·optimizer parameter 분배만 override.
  - Actor optimizer: Actor bank + Gaussian std + shared router.
  - Critic optimizer: Critic bank만.
  - 합친 loss의 backward로 router가 두 loss gradient를 받으며 step은 한 번. Router LR은 Actor LR.
- `agents/ppo/ppo.py` **기존 파일 수정됨**
  - _build_models, _optimizer_parameters, _evaluate_actor_critic, _bootstrap_value hook 추가.
  - rollout 및 loss에서 joint hook 호출.
  - optimizer 소유 parameter로 grad clipping.
  - distributed reduce/broadcast에서 중복 parameter 제거.
  - PPO loss 공식, GAE, reward 등을 바꾸려는 의도는 아니었지만 공식 원본과 실행 동등성/성능을 완전히 입증하지는 않았다.
- `agents/modules/ppo_modules.py` **기존 파일 수정됨**
  - update_distribution_from_mean helper로 기존 Gaussian 처리를 재사용.
- `config_types/algo.py`
  - BilateralNetworkConfig 추가.
  - PPOConfig.bilateral optional 필드 추가.
- `config_values/loco/g1/experiment.py`
  - 기존 g1_29dof 선언 자체는 변경 안 함. 원본 25,000 iteration, use_symmetry=True, IsaacSim, mixed terrain 설정.
  - g1_29dof_bilateral_magnet, g1_29dof_bilateral_no_right 추가.
- `config_values/experiment.py`: 새 두 exp 등록.
- `managers/command/terms/three_direction.py`
  - LocomotionCommand 상속.
  - 기존 샘플링 후 전진(+vx), 후진(-vx), 좌측(+vy) 중 하나 선택, 반대 planar axis는 0.
  - 기존 stand 확률과 yaw 샘플링 유지.
  - 실제 환경 명령에서 vy<0 없음. body frame 기준.
  - **중요한 confound:** PPO 기본 use_symmetry=True를 유지하므로 반사된 학습 minibatch는 생성됨. 엄격한 no-right 일반화 주장에 영향을 줄 수 있다. 사용자에게 이미 이 사실을 알렸다. 마음대로 끄지 말고 비교 조건을 정리할 것.
- `src/holosoma/pyproject.toml`: einops 의존성 추가.

## 8. 실행 / 영상 / localhost 스크립트

- `scripts/run_gpu3_experiments.sh`
  - Docker 내부인지 확인, 환경 source, GPU 3 고정으로 큐 실행.
- `scripts/run_bilateral_queue.py`
  - --gpu 필수, --num-envs 기본4096, --seed 기본43.
  - 세 exp: g1-29dof → g1-29dof-bilateral-magnet → g1-29dof-bilateral-no-right.
  - CUDA_VISIBLE_DEVICES=3, CUDA_DEVICE_ORDER=PCI_BUS_ID. 내부 torch device cuda:0은 선택한 physical GPU3에 대응하도록 한 의도.
  - Kit multiGpu/enabled=false, multiGpu/autoEnable=false, renderer/activeGpu=3도 전달.
  - logger:disabled이나 TensorBoard는 기본으로 기록됨. 학습 영상은 꺼서 렌더링 비용 분리.
  - 실험마다 logger.base_dir를 별도로 지정.
  - GPU3 전체 사용량을 1초마다 nvidia-smi로 기록. 다른 프로세스 메모리도 포함하므로 process-only 정밀 측정은 아님.
  - 각 실험 종료 후 JSON에 baseline/peak/increase/returncode 등 기록.
  - 실패하면 다음 실험으로 넘어가지 않음.
  - --iterations 옵션은 세 실험에 공통 적용 가능. 현재 실행은 지정하지 않아 기본 25k 각각.
  - --disable-symmetry 옵션은 세 실험 공통 ablation용이나 현재 사용하지 않음.
  - 세 학습 성공 후 render_bilateral_checkpoints.py를 호출.
- `scripts/render_bilateral_checkpoints.py`
  - **준비만 완료, 실제 GPU 렌더링은 아직 검증 안 됨.** 학습 세 개 종료 뒤 실행 예정.
  - 각 실험당 IsaacSim 세션 하나를 재사용해 모든 model_*.pt를 최신 순으로 읽음.
  - 환경 1개, headless offscreen video, load_optimizer=False.
  - 전진(+0.6 vx), 후진(-0.6 vx), 좌측(+0.6 vy), 오른쪽(-0.6 vy), yaw0 평가.
  - 각 방향 기본300 control step 또는 done까지.
  - 명령은 eval reset 후 command_manager.commands에 직접 고정하여 3번의 training sampler 제한을 우회하고 오른쪽 평가 가능하게 함.
  - 기존 simulator video_recorder start/stop 재사용.
  - `videos/<run-label>/<model_XXXXX>/<direction>/`에 MP4 및 metrics.json 저장.
  - 실제 body velocity, 평균 reward, survived steps, terminated를 JSON과 TensorBoard에 기록.
  - 렌더링도 GPU3만, 학습과 병렬 실행 안 함.
- `scripts/serve_bilateral_results.py`
  - 호스트 CPU에서 localhost:8008 갤러리 및 localhost:6006 TensorBoard 프록시.
  - Docker bridge IP의 internal TensorBoard16006으로 proxy.
  - 갤러리에서 실험/체크포인트 선택, 전후좌우 MP4와 평가 metrics 링크.
  - 영상은 학습 후 생성되므로 지금은 없을 수 있음.
  - 원격 연결이면6006/8008 포트포워딩 필요.
  - host 로그는 `/tmp/holosoma_results_server.log`, `/tmp/holosoma_tensorboard_server.log`.
- `scripts/launch_tensorboard.py`
  - TensorBoard builtin plugins만 사용.
  - 기본 tensorboard CLI가 Open3D 동적 plugin을 불러오며 libusb-1.0.so.0 미설치로 실패하여 우회했다. 학습 dependency를 바꾼 것은 아님.
- `scripts/check_bilateral_cpu.py`
  - 실제 Holosoma 클래스의 CPU-only 통합 smoke check.
  - 공유 router step, critic privileged 입력 미누출, bootstrap이 rollout 분포를 덮어쓰지 않음, 실제 CommandManager의 no-right sampler, ONNX export/graph checker.

## 9. 현재 실행 상태와 로그

실제로 실행한 명령:

```bash
docker exec -d -w /workspace/holosoma ycho-holosoma bash -lc \
 'mkdir -p logs/bilateral_queue && bash scripts/run_gpu3_experiments.sh --num-envs 4096 --seed 43 > logs/bilateral_queue/gpu3_launcher.log 2>&1'
```

**이 명령을 다시 실행하면 중복 학습이 생긴다. 상태 확인 없이 재실행 금지.**

호스트 로그 루트:
`/home/ycho/holosoma/logs/bilateral_queue/20261028_101400/`

launcher 로그:
`logs/bilateral_queue/gpu3_launcher.log`

1번 콘솔 로그:
`logs/bilateral_queue/20261028_101400/1_g1-29dof_seed43.log`

1번 checkpoint / TensorBoard 실제 경로:
`logs/bilateral_queue/20261028_101400/1_g1-29dof_seed43/hv-g1-manager/20261028_101411-1_g1-29dof_seed43-locomotion/`

설정 파일 이름은 **holosoma_config.yaml**이다. 이전 에이전트가 config.yaml로 잘못 찾아 실패한 적 있음.

마지막 인계 작성 시 실제로 읽은 로그:
- 1번 vanilla PPO **928 / 25000 iteration**.
- elapsed 로그 2942.49초(약49분), iteration3.19초.
- Collection2.809초, Learning0.380초, 약30,817 environment steps/s.
- total timesteps91,324,416.
- mean episode length951.35, Env/average_episode_length980.09.
- weighted tracking lin vel reward0.7321, angular0.7122.
- mean reward -1.21, penalty_scale0.6417. 이전 iteration715에서는 mean reward45.89, penalty_scale0.1788이었다. curriculum penalty가 변하므로 단순 reward 하락만으로 붕괴라고 단정하지 말 것.
- Policy mean noise std0.2987.
- ETA76244.9초(약21시간 추가).
- 앞서715 iteration에서는 약38분, tracking lin0.7632/angular0.5491이었다.
- 2번/3번은 아직 시작 안 했음. 인계 후 시간이 흐르면 달라질 수 있으니 새로 확인.
- model_00000.pt부터100iteration 간격으로 저장 중. 최신 파일은 직접 확인.
- GPU3 전체 memory.used **8893MiB(약8.7GiB)**, utilization 약28~36%를 관측.
- 원래 GPU3가 거의 비어 있을 때 시작했다. 이 수치는 vanilla의 현재 측정이며 Bilateral VRAM은 아직 실측하지 못했다.

서비스:
- http://localhost:6006/ 및 /data/logdir: HTTP200 확인.
- http://localhost:8008/ 및 /api/manifest: HTTP200 확인.
- 호스트 gallery server PID를 마지막으로3045405로 확인했지만 PID를 신뢰해서 kill하지 말고 cmdline을 먼저 확인.
- TensorBoard는 별도 docker exec -d로 시작해 internal16006에서 실행 중.
- 서버 상태를 보려면 /tmp 로그와 URL을 확인. 다른 사용자 서비스/학습을 종료하지 말 것.

## 10. 검증된 것 / 아직 아닌 것

검증 완료:
- 새 Python 파일 AST 구문 검사, git diff --check.
- 연구 Docker의 PyTorch2.6 CPU에서 tensor forward,29 action/scalar value, 양쪽 loss→router gradient, router 단일 step, 모델/optimizer checkpoint roundtrip.
- 이후 **실제 Holosoma Docker 환경**에서 scripts/check_bilateral_cpu.py 성공:
  - native PPO setup/update.
  - joint loss/optimizer ownership.
  - privileged-input isolation.
  - bootstrap 분포 유지.
  - real CommandManager forward/backward/left sampling.
  - ONNX export + onnx.checker.
  - 네트워크 파라미터2,829,323개 보고.
- 로그: `logs/bilateral_queue/cpu_checks.log`.
- vanilla GPU 학습은 실제 진행되고 checkpoint와 TensorBoard 로그가 생김.

검증 미완료:
- **완전 무수정 공식 PPO와 현재 modified PPO의 numerical/rollout/performance 동등성**.
- 원본 domi 전체 환경과 이식 IO의 완전 동일성(관측 정보 자체, phase 생성기, 합성 전 waist history 등 다름).
- Bilateral GPU 장시간 학습, VRAM peak, 실제 보행 성능.
- MP4 GPU 렌더링 end-to-end 실행. 현재 학습 중이므로 병렬 GPU 렌더링을 하지 않았다.
- 오른쪽 held-out generalization 결과.

## 11. 속도/공식 baseline 진단에서 이미 확인한 사실

공식 HEAD의 `config_values/loco/g1/experiment.py`:
- exp:g1-29dof = PPO / IsaacSim / mixed terrain / 25000 iterations / use_symmetry=True.
- 기본 환경4096, rollout num_steps_per_env24.
- 현재 g1_29dof 프리셋 선언 자체는 원본과 동일하고 새로운 프리셋만 추가했다.
- reward/observation/action/command/curriculum 원본 파일들은 수정하지 않았다(새 third sampler 예외).
- 그러나 PPO 본체 hook과 PPOActor helper는 수정했으므로 완전 untouched baseline이 아님.
- README quickstart는 FastSAC 예제였으나 **사용자는 기억한 과거 실행이 PPO라고 확인**했으므로 FastSAC였을 것이라 우기지 말 것.

Docker 자원:
- NanoCpus0, CpuQuota0, CpuPeriod0, CpusetCpus빈값, memory limit0, shm16GiB.
- nproc128. 명시적 Docker CPU quota는 없는 상태.
- 학습 python이 CPU약169%, thread371로 관측된 적 있음. host 경합/CPU affinity 등 추가 검증 필요.

startup log:
- CUDA_VISIBLE_DEVICES 경고: CUDA enumeration과 Omniverse enumeration이 다르다고 나옴.
- 여섯 GPU가 bad state라 skip된다는 경고가 있었으나 masking의 영향일 수도 있음. 실제 고장/CPU fallback이라고 단정 안 됨.
- GPU table에서는 physical GPU3가 `Active Yes:0`.
- AppLauncher args: device cuda:0, physics_gpu0, active_gpu0, Kit renderer activeGpu3.
- environment device cuda:0, physics timestep0.005, rendering timestep0.005.
- Physics SimulationCfg의 device=self.sim_device. 코드상 GPU 경로를 요청하고 있다.
- 영상/카메라 학습 시 꺼짐. _step_dynamics는 sim.step(render=False), rendering 필요할 때만 render.
- IOMMU enabled 경고, GLFW failed headless 경고, robot joint config order가 Isaac joint order와 다르다는 경고가 있음. 어느 것이 속도 원인인지는 확정 안 됨.

권장 다음 진단:
1. 실제 저장 holosoma_config.yaml과 HEAD 원본 설정을 비교. 과거 config.yaml로 잘못 읽은 부분을 보완.
2. original PPO/IsaacSim revision, simulator backend(IsaacSim vs MJWarp vs 예전 IsaacGym), environment count, rollout length, terrain, decimation, hardware를 사용자가 기억하는 실행과 비교.
3. simulator collection2.8초 병목을 profile. 현재 learning0.38초이므로 네트워크만 탓할 근거 없음.
4. GPU3 물리 매핑을 확인하되 다른 GPU를 사용하거나 시스템 재부팅하지 말 것.
5. baseline refactor가 행동에 영향을 주었는지 original HEAD와 CPU numerical 비교/격리 baseline을 검토.
6. 사용자에게 확정된 사실과 추정을 명확히 분리하고, 현황 질문에는 진행률과 결과부터 짧게 답할 것.

## 12. 작업 진행 시 금지/주의

- 새 계정으로 넘어왔다고 세팅이나 학습을 처음부터 다시 시작하지 말 것.
- 원본 연구 Docker의 연구 코드와 실행을 건드리지 말 것.
- GPU3 외에는 사용 금지. renderer도 동일.
- 현재 checkpoint/log를 삭제하거나 git reset하지 말 것.
- 장시간 GPU job 변경/재시작이 필요하면 현재 checkpoint와 큐 부모-자식 관계를 먼저 확인.
- 알고리즘을 FastSAC로 바꾸거나, iteration을 몰래 줄이거나, symmetry를 몰래 끄지 말 것.
- 원본을 보존했다는 표현은 실제로 보존한 파일/함수 범위에 한정할 것. 환경 전체/phase 생성기까지 동일하다고 말하지 말 것.
- 모든 실행/서비스가 완료되었다고 말하지 말 것. 현재는 1번 학습 중이며 후속 학습·영상은 자동 큐로 준비된 상태.
- 이전 도구 호출 중 일부가 자동 승인 검토 deadline으로 지연/실패했다. 사용자에게 반복 재승인을 요구하지 말고 이미 승인된 범위에서 짧고 단순한 명령을 사용.
- 이미 사용자는 "approve everything, please do the work ASAP"라고도 말했지만 다른 사용자의 작업이나 GPU를 건드릴 권한은 아니다.

## 바로 이어서 할 일

현재 queue/progress/holosoma_config.yaml과 GPU3 상태부터 신속히 읽고, 사용자에게 현황을 먼저 말해라. 사용자가 가장 의심하는 것은 **공식 원본 baseline인지, 왜 PPO가 기억보다 수십 배 느린지**다. 이 문제를 해결하는 것을 최우선으로 삼아라. 원인 확인 없이 장시간 ETA만 다시 읽어주거나, 사용자의 PPO였다는 확인을 무시하지 말 것.
