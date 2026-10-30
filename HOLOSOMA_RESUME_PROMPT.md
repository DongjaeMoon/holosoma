# Holosoma 종합 인계 프롬프트 — 2026-10-29 IsaacSim 루트 전환

이 파일을 읽고 기존 작업을 이어받아라. 사용자는 한국어를 사용하며 긴 대기와 반복 확인을 매우 싫어한다. 현황 질문에는 먼저 짧게 답하고, 버그를 발견하면 즉시 알린다. 아래 최신 본문이 부록의 과거 기록보다 우선한다. 처음부터 셋업을 다시 하지 말 것.

## 0. 최우선 최신 지시: 루트 + IsaacSim + FastSAC

- **최신 실행 승인 및 시작:** 사용자가 GPU **2**를 비웠으며 그 GPU에서 30k MLP 학습→RMSE 평가를 시작하라고 명시했다. TensorBoard 및 커맨드 방향 비율 로깅도 재확인했다. `python scripts/run_vanilla_mlp_30k.py --gpu 2 --output logs/bilateral_queue/vanilla_mlp_30k_native_io_gpu2 --execute`를 Docker/hssim에서 시작했고 학습 child PID는 **211442**. 아래 '준비만/실행 금지'는 이전 상태이며 이 승인으로 해당 새 run 실행이 허용됐다. **동일 학습 중복 실행 금지.** `logs/bilateral_queue/vanilla_mlp_30k_native_io_gpu2/status.json`, `training_terminal.log`, 인접 `vanilla_mlp_30k_native_io_gpu2_supervisor.log`를 먼저 확인한다. 실행기의 GPU 인덱스 제한을 비음수 정수로 바꿔 학습·평가 모두 GPU2를 사용한다. 초기 GPU2는98MiB사용/23984MiB여유 확인. 기존 TensorBoard6006→16006 서버가 전체 bilateral_queue를 스캔 중이며 새 run도 이 서버에서 확인한다. 기본 no-right(앞/뒤/왼쪽), 기존 empirical/native phase, compile=True, 4096env/seed43/30k, command audit ON 유지.
- **실행 검증:** 실제 학습 step200 이상 진행, TensorBoard API에서 step234의 `CommandRatio/{forward,backward,left,right,standing,other}`, `CommandAudit/negative_vy_total`, `replay_negative_vy_total` 태그/값 확인. step200 감사값: 정책823,296개/음수0, replay24,772,608개/음수0, right0. host CUDA PID1960327이 GPU2 UUID `GPU-53c4e089-7be2-4dc9-ae59-1168c5fb3fc9`에 있는 것과 container PID211442의 CUDA_VISIBLE_DEVICES=2 확인. 실제 run `training/hv-g1-manager/20261030_095312-vanilla_mlp_no-right_30k_seed43-locomotion`. run 루트에 감사 JSON/CSV/config symlink 생성. TensorBoard에서 `vanilla_mlp_30k_native_io_gpu2` 검색. 평가 worker는 이후 `evaluation/tensorboard/MLP`에 각 checkpoint의 `RMSE/<case>/{planar_mps,vx_mps,vy_mps,yaw_radps}`, `RMSE/right_mean_{planar_mps,vy_mps}`, 실패율/생존율도 기록하도록 추가했다. CPU 임시 event 검사 완료. 아직30k 완료나 RMSE 평가 완료가 아니다.
- **최신 준비 작업(실행 금지):** 사용자가 기본 Holosoma에 가까운 MLP no-right **30k** 학습 + 오른쪽 RMSE 평가 스택을 요청했으며, GPU가 비어 있지 않아 **코드/커맨드만 준비**했다. 새 사용자 실행 지시 없이 시작하지 않는다. 기본 empirical normalization 및 기본 LocomotionGait/sin_phase/cos_phase, `use_symmetry=False`, `symmetry_compatible_io=False`, **compile=True(원본)**, seed43/4096env/batch8192/num_updates8. 부모 EmpiricalNormalization 구현/분산 갱신 수식은 변경하지 않았다.
- 새 preset `g1-29dof-mlp-no-right-vanilla-fast-sac-30k`는 기본 FastSAC에서 sampler(앞/뒤/왼쪽), use_symmetry=False, 학습길이30k만 변경. 선택 preset `g1-29dof-mlp-left-only-vanilla-fast-sac-30k`는 native sampler에 vx[0,0]/vy[0,1]을 지정. 둘 다 yaw[-1,1]/standing20% 유지. '왼쪽만'의 의미를 비동기 질문했으며, 별도 답변 전 기본은 기존 앞/뒤/왼쪽 no-right 조건으로 준비했다.
- 실행기 `scripts/run_vanilla_mlp_30k.py`는 **기본적으로 계획만 출력**, 실제 실행은 나중에 `--execute`가 있어야 한다. `--stage evaluate --execute --output <기존run>`은 재학습하지 않고 완료된 checkpoint 평가는 건너뛴다. 1k..30k 총30 checkpoint, 기존 선속도16case + standing1case, 각32trial/20초/초기2초 제외, 평지/외란·관측noise OFF, 저장된 native normalizer 사용. `evaluation/right_rmse_by_checkpoint.csv/json`, 전체 summary 및 plots 저장.
- 준비물 `logs/bilateral_queue/vanilla_mlp_30k_prepared.json`, `vanilla_mlp_30k_prepared_config.yaml`, `vanilla_mlp_30k_prepared_commands.txt`. 미래 실행 폴더 `logs/bilateral_queue/vanilla_mlp_30k_native_io`는 준비 시 생성하지 않는다. GPU 없이 CPU 설정 차이/실제 CLI 파싱/기본 phase/명령 reset·주기 재샘플링/평가 설정/비실행/요약 집계 검사 완료: `scripts/check_vanilla_mlp_30k.py`.
- command audit는 새 실행 커맨드에서 ON. `command_audit_csv_envs=1`로 CSV는 env0만(+음수 vy 위반 env 전부), **정책·replay의 음수 카운터는 모든 env/샘플을 검사**. 기존 감사 기본값(None=CSV 전env)은 유지. eval config에서는 학습용 감사 OFF. v2/v3 preset 및 완료 결과는 보존.
- **최신 중단 상태:** 사용자가 메모리 부족 우려로 env1 MLP 감사 run 중단을 명시적으로 요청했다. `mlp_env1_command_audit_20261030_034438` 학습 PID196846에 SIGTERM 후 SIGKILL을 보내 종료를 확인했다. GPU3 여유 메모리는 종료 직후6649MiB. **현재 실행 중인 학습이 아니며, 새 사용자 실행 지시 없이 재시작하지 않는다.** 아래 '실행 중' 기록은 시작 당시 기록이다. 남은 로그/체크포인트/CSV 보존.
- **2026-10-30 최신 실행:** 사용자가 GPU3 메모리 사정으로 **MLP no-right, env1, 10k, 실제 command CSV+TensorBoard 기록**을 승인했다. 실행 중인 새 run은 `logs/bilateral_queue/mlp_env1_command_audit_20261030_034438`. 중복 실행 금지. `status.json`을 먼저 확인한다. GPU3만 사용, seed43, yaw[-1,1], use_symmetry=False, batch8192 유지, IsaacSim/hssim, 초기 checkpoint 없음. 네 모델 yaw 제한 v3 큐는 여전히 준비만 된 상태다.
- 실행기는 `scripts/run_mlp_command_audit.py`. `command.json/txt`, `training_terminal.log`, `status.json` 보존. run 루트의 `commands_policy.csv`, `command_audit_status.json`, `command_audit_metadata.json`, `holosoma_config.yaml`은 실제 training 하위 파일로 연결된다. TensorBoard6006에서 `mlp_no_right_env1_command_audit` run을 선택한다.
- opt-in `algo.config.command_audit=True`: 정책 정규화 직전 관측의 command와 정규화된 입력, manager command, 실제 body velocity를 매 step 기록. raw replay의 current/next command도 정규화 직전에 검사. `Command/vx,vy,yaw`, `CommandAudit/negative_vy_total`, `CommandAudit/replay_negative_vy_total`, `CommandRatio/*`, `Motion/vy` 기록. 음수 vy가 나타나면 원본을 저장하고 실패시킨다. CPU 양수/음수 주입 검사 및 실제 TensorBoard 서버 태그 확인 완료. native iteration은 0..10000 표기이므로 완료 시 policy CSV가 10001행일 수 있다.
- 사용자가 명시적으로 `.isaacgym/holosoma` 작업을 루트로 옮기고 앞으로 IsaacSim에서 실행하라고 지시했다. **활성 코드: 호스트 `/home/ycho/holosoma`, Docker `/workspace/holosoma`**. `.isaacgym/holosoma`는 이전 코드 보존용이며 앞으로 작업/실행 기준이 아니다.
- 현재 네 모델 FastSAC 네트워크, 공유 router 업데이트, reflection normalization, domi phase, command sampler와 평가/영상 코드를 루트로 옮겼다. 루트의 최신 IsaacSim 및 observation lifecycle 구현은 유지했다. 이전 루트 파일 백업: `.migration_backups/isaacsim_root_20261029_203733/manifest.json`.
- 새 실험은 **준비만 요청됨**. 앞/뒤/왼쪽 이동은 유지하고 body-frame yaw 명령을 `[0,1] rad/s` (반시계)로 제한한 네 preset: `g1-29dof-{bilateral,ham-sym-aug,ham,mlp}-no-right-no-clockwise-fast-sac-v3`.
- sym-aug HAM만 augmentation ON이며 반사된 오른쪽/시계 yaw 합성 샘플을 학습한다. 나머지는 OFF. 네 모델 공통 domi phase와 reflection normalization, compile=False, seed43, 4096env, 10k.
- 준비된 검증은 기존 yaw0 선속도16case + 정지1 + 순수 시계/반시계 yaw4 + 좌/우 이동과 양방향 yaw 조합4 =25case, 각32trial. yaw RMSE는 `(actual-target)`으로 계산하며 평균 실제 yaw 속도도 저장한다. 새 학습/물리평가/렌더는 시작하지 않았다.
- 환경: Docker `ycho-holosoma`, 루트에서 `source scripts/source_isaacsim_setup.sh`, conda `hssim`. 의존성 `opt-einsum`, `icecream`을 루트 pyproject에 명시했다.
- 시작 스크립트: `python scripts/run_corrected_campaign.py`는 계획만 출력. 실제 실행은 **사용자가 요청할 때만** `python scripts/run_corrected_campaign.py --execute`. 기본 variant `ccw-v3`, IsaacSim 명시, GPU3 biHAM→MLP / GPU6 sym-augHAM→HAM, 학습 후 40checkpoint 수치평가, 신규 결과 디렉터리 생성. 영상 자동 실행 없음.
- 아래 기존 완료 결과는 **IsaacGym v2 기록**이다. IsaacSim에서 이미 실행/검증한 결과로 설명하지 말고, 완료한 기존 학습/평가를 중복 실행하지 않는다. 이 0절이 아래 옛 경로/환경 설명보다 우선한다.

## 1. 기존 IsaacGym v2 완료 상태

- **수정본 v2 네 모델(biHAM, sym-aug HAM, HAM, MLP) 모두 10k 학습 완료.**
- 모델별 1k~10k의 10 checkpoint, 총40checkpoint 평가 완료. 640command cases /20,480rollouts.
- 선별 비교영상4개, 이후3모델×6방향18개영상, 최종 좌우분할 합성MP4 하나 모두 완료/전달.
- **지금 미완료된 학습/평가/렌더 요청은 없다.** 마지막 요청은 이 인계파일 작성이다. 다음 사용자 요청을 따르고 완료된 큐를 중복 실행하지 않는다.
- 최신 campaign: 호스트 `/home/ycho/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513`. Docker에서는 앞부분을 `/workspace/holosoma`로 바꾼다.
- `status.json` complete, `evaluation/status.json`40/40, `evaluation/three_model_video_status.json`18/18. 실제 상태는 이 파일들로 확인.
- 최신 학습/평가 supervisor elapsed12626초=3시간30분26초, 약08:43KST완료. 이후 영상 작업도 종료. 오래된 문서의 ‘현재 학습 중’을 믿지 말 것.
- 기존 v2 실행 코드는 **`.isaacgym/holosoma` checkout**에 보존. 현재 활성 작업은 위 0절의 루트 IsaacSim 코드.
- Git commit을 만들었다고 보고한 적 없다. working tree 변경이 많으므로 git reset/clean 등 금지.

## 2. 연구 목표와 사용자 지시

domi 연구 네트워크 **구조만** Holosoma에 최소 수정/원본 보존/짧고 캡슐화된 코드로 이식하는 것이 목표. Actor와 Critic은 router를 공유하고 shared router는 중복 optimizer step하지 않는다. 초기엔 PPO라고 기억했으나 사용자가 나중에 **FastSAC이 맞다**고 정정했다. 현재 비교는 전부 FastSAC+IsaacGym이다.

원래는 algorithm/reward/observation/command/action scale/curriculum 모두 Holosoma 유지였지만, 이후 **좌우 IO와 phase가 다르면 domi 원본을 쓰라**고 명시했다. v2는 모든 모델에 domi phase와 반사 대칭 정규화를 공통 적용했다. 따라서 v2 MLP를 ‘완전히 무수정 공식 vanilla baseline’이라 부르면 안 된다.

최종 비교:
1. biHAM=BilateralMAGNet, replay symmetry augmentation OFF.
2. sym-aug HAM=HAMNet, augmentation ON.
3. HAM=no sym-aug HAM, augmentation OFF.
4. MLP=native FastSAC MLP, augmentation OFF.

실제 학습 command sampler는 **앞/뒤/왼쪽**, body-frame vy>=0. ‘왼쪽만 학습’이 아니다. yaw sampling, standing, randomization은 남아 있다. 오른쪽 command 미경험과 오른쪽 physical motion 미경험은 다르다. sym-aug HAM은 합성된 오른쪽 command를 replay에서 경험한다.

모든 설치/학습/평가/렌더는 Docker 안. 초기 GPU3만→나중 학습/평가에GPU3+6승인. v2 GPU3:biHAM→MLP, GPU6:sym-augHAM→HAM. **가장 최근 영상 작업은 GPU3만 승인.** 다른 GPU나 다른 사람의 프로세스를 쓰지 않는다. 새 작업은 최신 지시를 따른다.

사용자는 버그가 발견되면 즉시 알리고 필요한 수정/재학습을 요청했다. 성능이 낮다는 이유로 버그라고 단정하지 않는다. 현재 확인 범위에서는 추가 재학습이 필요한 결함은 발견하지 못했다.

## 3. 환경 / Docker / 활성화

| 항목 | 값 |
|---|---|
| 호스트 루트 | /home/ycho/holosoma |
| Holosoma Docker | ycho-holosoma (기존 ID f0e2bb2413de) |
| bind mount | /home/ycho/holosoma → /workspace/holosoma |
| 현재 활성 코드 | /workspace/holosoma/.isaacgym/holosoma |
| 현재 기반 | 과거 공식 Holosoma v0.0.3 / commit59d5e61, IsaacGym Preview4 |
| 현재 conda | hsgym, Python3.8, Torch2.4.1+cu121 |
| conda 위치 | /root/.holosoma_deps/miniconda3/envs/hsgym |
| 과거 환경 | hssim, IsaacSim5.1.0/IsaacLab2.3.2; 현재 학습 기반 아님 |
| 연구 Docker | 147-sejune-domi-isaac-lab |
| 연구 레포 | 해당 Docker /home/user/domi-alps |
| 연구 호스트 | /home/sejune/domi-alps (ycho 권한 안 돼 Docker로 읽음) |

활성화:
```bash
docker exec -it ycho-holosoma bash
cd /workspace/holosoma/.isaacgym/holosoma
source scripts/source_isaacgym_setup.sh
```
GPU3은 `CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID`, 프로세스 내부 `cuda:0`. GPU6도 같은 remap 방식. CPU검증은 `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1`.

Python 경로만 직접 호출하면 PATH/LD_LIBRARY_PATH/PYTHONPATH가 빠질 수 있다. FFmpeg는 hsgym/bin에 있지만 source 없이 which ffmpeg는 실패했다. hsgym에는 imageio_ffmpeg 모듈이 없다. 설치할 필요 없이 source한 환경의 FFmpeg 사용.

git dubious ownership은 정확한 checkout에 `git -c safe.directory=<path> -C <path> ...`로 읽기 가능. 루트원본/현재checkout 혼동 주의. `.migration_backups/`와 이전 로그 보존. 연구 Docker의 /input/domi는 현재 없으며 실제경로는 /home/user/domi-alps. 연구 레포 자체는 수정하지 않았다.

## 4. domi 출처와 원본 보존

연구 진입점: `/home/user/domi-alps/source/standalone/workflows/imm/train_ma_v2.py`.
네트워크 YAML:
- biHAM: `source/domi/src/domi/cfg/net/sym_ma_v7_bilateral_modular_nobase.yaml`
- HAM: `source/domi/src/domi/cfg/net/ham_sa_from_ma_v4_abl_preact.yaml`
- env: `source/domi/src/domi/cfg/env/ma_v3.yaml`
- dagger: `source/domi/src/domi/cfg/dagger/dagger_v0.yaml`
- task loco_NEW_DEFAULTS,seed43,4096env,headless. **긴 원래 명령 전체는 부록 초기 HANDOFF에 보존.**

연구 레포 기준 원본:
- source/domi/src/domi/nn/bilateral_magnet.py: BilateralMAGNet/BilateralModularLinear
- nn/range_layers.py, nn/layers.py: MAGNet router, MultiRange, symmetric parametrization
- source/domi/src/domi/net/td/bilateral_magnet.py, generic.py: 연구 wrapper (전체 pipeline을 가져온 것은 아님)
- source/domi/src/domi/env/loco_action_mapping.py: action pair/공유관절
- source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_art/config/loco_env_cfg.py: 좌우 IO, joint order, phase_zero_stand
- 동일 g1_art/mdp/phase_command.py 및 phase_command_cfg.py: 원본 clock
- source/isaaclab/isaaclab/utils/string.py: matching helper

복사된 원본: 활성checkout `src/holosoma/holosoma/agents/modules/bilateral/original/`.
파일: bilateral_magnet.py, range_layers.py, layers.py, export_compat.py, loco_io.py, joint_layout.py, loco_action_mapping.py, isaac_strings.py, network.yaml, provenance.json, ham_range_layers.py, ham_magnet_v2_reference.py, ham_sa_from_ma_v4_abl_preact.yaml, ham_provenance.json, phase_clock.py, fast_sac_updates.py.
정확한 출처/해시는 provenance.json/ham_provenance.json. phase_clock.py에는 원본 SHA256 c8f6ed6341fce471a80e323aca2299883e85700a9dae29c4156795464af6cfff 기록.

연구 명령의 구조 관련 옵션: bilateral_io=false,lean_obs=false,contralateral_arms=false양쪽,compose_mode=feature,feature_bias_scale=.5,checkpoint_layers=true,use_fast=false,phase_zero=true,contact_phase_reset=false. reward/curriculum/termination override 전체를 이식한 것이 아니다.

## 5. 구현 상세와 파일 지도

아래 경로는 활성checkout의 `src/holosoma/holosoma/` 기준.
- agents/modules/bilateral/adapter.py: ObservationView, BilateralPolicy, 초기PPOadapter. ObservationManager layout에서 이름 기반 joint indices/signs 구성. history_length1만지원,미지원항목fail-fast.
- original/loco_io.py: 원본 _mirror_act/_mirror_ang_vel/_mirror_lin_vel/_mirror_joint_values/_mirror_vel_cmd/_mirror_phase. phase는 [sin,cos] oscillator 하나이며 우측표현은 전체 부호반전.
- 한쪽 output은 waist3+ownleg6+ownarm7=16. 양쪽32→Holosoma물리29. 팔다리는해당side,sharedwaist는물리좌표변환후양쪽평균. 원본 joint resolver/sign사용.
- **action history 한계:** domi는 pre-merge side raw32,Holosoma는합쳐진physical29. waist를복제해sideview구성하며원래독립waist두값은복원불가. hiddenhistory나actionspace변경은추가안함. 이차이를사용자에게설명.
- domi absolute joint position과Holosoma default-relativeposition등의차이는기본적으로Holosoma관측의미유지. 연구MDP전체재현아님.
- fast_sac_adapter.py: 원본bilateralbank를nativeSACactor mean/logstd/Q에연결. tanh/actionscale/distribution/native101atomprojection유지. criticprivilegedobs+actorroutercontext+candidateaction사용. 좌우Qlogits평균.
- ham_adapter.py: 원본MultiRange/HAM executor,좌우fullconcat,use_sym_paramFalse.
- agents/fast_sac/bilateral_fast_sac.py: actor와독립onlineQ두bank가router공유. privateactor/privateQ/routeroptimizers분리,routergradient모아한번step. targetQ/router는deepcopy.
- agents/fast_sac/ham_fast_sac.py: HAMbuilder교체.
- original/fast_sac_updates.py: nativeSACupdate에sharedrouterhook추가한보존코드.
- biHAM: hidden192/384/192,router256/256/128,num_module4,gate8,featurebias.5,share_abFalse,use_sym_paramTrue,checkpointTrue,use_fastFalse.
- HAM: hidden384/576/384,use_sym_paramFalse,checkpointFalse. 지원안하는biHAM옵션을억지적용안함.
- 원본weight_recenterTrue지만이식에서는추가학습갱신을피하려고False. 원본학습전체재현이라고말하지않는다.

### v2 수정
- agents/modules/bilateral/normalization.py: ReflectionNormalization+DomiSymmetry.
- 구 per-feature normalization은한쪽command통계때문에 N(Mx)!=M(Nx). network자체대칭이어도physicalpolicy대칭깨짐.
- native statsupdate후mean/variance를반사대칭으로projection(between-meanvariance포함). actor/critic둘다,네모델공통. replaytransition추가는아님. 공통대칭통계자체는inductiveprior.
- config_types/algo.py의symmetry_compatible_io기본False,agents/fast_sac/fast_sac_agent.py의opt-infactory. legacycheckpoint/preset기본해석보존.
- managers/command/terms/domi_gait.py: 실제domiperiodrange(.9,1.1),randomoffset,clock,standingzero. contactresetFalse. Holosomafootheightreward의leftswingpeak/rightstanceconvention에연결.
- original/phase_clock.py: 원본 _resample_command/_episode_time_s/command/_update_command 네메서드원문추출,IsaacLabconstructor/lifecycle만bridge.
- 구 nativeaugmentation은sin/cos2채널중첫다리만반전→DomiSymmetry는양쪽채널모두반전.
- managers/command/terms/three_direction.py: 앞/뒤/왼쪽샘플링. reset/step같은subclass사용검증.
- config_values/loco/g1/experiment.py 및 config_values/experiment.py: v2 네preset등록. reward/randomization/actionscale/SAC기본loss유지,phase/normalization은공통수정.

### 이미 수정한 과거 버그
Torch2.4 functional_call(detachedQ)의sharedrouteralias교체→requires_grad잠시동결방식으로수정. routeridentity유지.
compile/BF16문제→네모델compileFalse. HAMQprojectionfloat32. Python3.8futureannotations.
평가512envorigin중복→holdout_terrain.py gridspacing. flattenedDOFsnapshot→29jointreshape.
평가build_config가command전체를stock으로교체→velocitysampler만교체해domigait보존.


## 6. 실험 실행과 실제 저장 위치

canonical campaign: `logs/bilateral_queue/corrected_v2_20261029_0513`.
`runs.json`이모델별정확한trainingdirectorymapping이다. 경로를추측하지말고이파일을읽는다.
`gpu3/`:biHAM→MLP,`gpu6/`:symaugHAM→HAM. 각log/report/holosoma_config.yaml/TensorBoardevents/checkpoint존재.
checkpoint는 model_0001000.pt부터model_0010000.pt까지10개씩. nativeONNXexport도실행.
개별jsonrunreport에command/returncode/elapsed_seconds/peak_gpu_used_mib존재.
`evaluation/`에수치/plot/video전부. 최신statuscomplete,평가40/40.

v2 preset:
- g1-29dof-bilateral-no-right-fast-sac-v2
- g1-29dof-ham-sym-aug-no-right-fast-sac-v2
- g1-29dof-ham-no-right-fast-sac-v2
- g1-29dof-mlp-no-right-fast-sac-v2

개별실행형태(현재재실행지시아님):
```bash
cd /workspace/holosoma/.isaacgym/holosoma
source scripts/source_isaacgym_setup.sh
CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-bilateral-no-right-fast-sac-v2 simulator:isaacgym logger:disabled \
  --training.seed 43 --training.num-envs 4096 --algo.config.num-learning-iterations 10000 \
  --training.name UNIQUE_NAME --logger.base-dir UNIQUE_NEW_OUTPUT
```

자동화(활성checkout scripts/):
- run_bilateral_queue.py:GPU별sequentialtraining,--presets/--output/--iterations/--skip-render,메모리1초sampling.
- run_corrected_campaign.py:GPU3/6두모델씩→각GPU20metricjobs→GPU3선별영상4개. 이미완료. 같은output으로중복실행금지.
- run_holdout_evaluation.py:--runs runs.json --output 평가경로 --gpu 3/6 --min-iteration/--max-iteration. customruns없으면오래된hardcodedruns가실행되므로주의.
- source_isaacgym_setup.sh:환경활성화.
- 실제최신학습소요/최대GPU총메모리:biHAM9027s/17390MiB,symaug6508s/18989MiB,HAM4438s/17033MiB,MLP2371s/15592MiB. 메모리는해당GPU전체값1초sampling이며다른프로세스가있다면포함될수있음.

수정전보존campaign(logs/bilateral_queue아래):
biHAM=20261028_150951_isaacgym,
symaug=20261028_155752_isaacgym,
HAM=20261028_174024_isaacgym_gpu3,
MLP=20261028_175110이아니라 **20261028_175105_isaacgym_gpu6**.
평가=deploy_eval_four_way_10k.
이구현에는대칭정규화결함이있었다. 최신v2결과와섞지않는다. 초기invalid_overlap_attempt/previous_graphics_attempt등도보관되어있다.

## 7. 평가 방법 / 결과 / 해석

evaluate_command_holdout.py: 저장actor+normalizer,deterministic mean action,no learning.
1모델1checkpoint=8방향×2속도(.3,.6m/s)×32trial=512rollout.
방향은앞/뒤/왼쪽/오른쪽/왼쪽앞/왼쪽뒤/오른쪽앞/오른쪽뒤. 대각선은전체속도norm .3/.6. yaw명령0.
20초rollout,처음2초제외,flatnominalphysics,observationnoise/push/DR OFF,평가seed20261029.
학습seed는43하나다. 32evalrollout을32개독립학습실험으로설명하지않는다.

reset후지정command로observation을다시계산해첫action부터명령이맞게함.
자동reset전velocity/terminationcapture,첫episode만측정. 실패후autoreset구간포함금지.
각trial planar RMSE=sqrt(mean((vx-targetvx)^2+(vy-targetvy)^2)); 그값을trial간평균.
per-axis/yaw RMSE,meanvelocity,survival/failure,측정steps,초기root/DOF/rawobs/domiphaseperiod도저장.
학습방향평균=앞/뒤/왼쪽×2속도×32=192rollout. 왼쪽대각선포함하지않음.
오른쪽계열평균=오른쪽/오른쪽앞/오른쪽뒤×2속도×32=192.
최종10k의2048rollout(4모델×512)은전부20초완주,측정900steps.
초기checkpoint의error는failure이전conditionalerror이므로survival도함께확인.

10k planar RMSE(m/s),각방향은2속도평균:
| 모델 | 앞 | 뒤 | 왼쪽 | 오른쪽 | 오른쪽앞 | 오른쪽뒤 | 학습방향평균 | 오른쪽계열평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biHAM | .1768 | .1754 | .3017 | .3015 | .2423 | .2479 | .2179 | .2639 |
| sym-aug HAM | .2178 | .1890 | .3167 | .3356 | .2815 | .2563 | .2411 | .2911 |
| HAM | .2368 | .2192 | .3293 | .3565 | .3004 | .2881 | .2618 | .3150 |
| MLP | .2169 | .2099 | .3913 | .2762 | .2517 | .2163 | .2727 | .2481 |

biHAM은학습방향최저,좌우측면오차거의같음. 하지만오른쪽계열10k평균은MLP가최저.
원하는‘biHAM만일반화하고다른모델은실패’가설은지지되지않음.
학습한왼쪽도오차가크다. biHAM학습방향3k .2174→10k .2179,중간등락. 단순iteration부족/더돌리면반드시해결이라고단정못함.
일반HAM/MLP도연속command에일반화가능. physicalrightmotion은탐색/외란중가능. symaugHAM은합성rightcommands학습.
공통대칭정규화는replayaugmentation과다르지만공통inductiveprior임.
pure right target vy=-.6 실제평균(vx,vy):biHAM(.013,-.2671),symaug(-.085,-.2428),HAM(-.0421,-.2135),MLP(-.1555,-.4351).
MLP도완벽추종아니고뒤로drift. checkpoint별순위변함(HAM최저rightward평균5k약.2518).
원하는순위만선택해서결론내리지말것.

## 8. 검증과 알려진 한계

CPU검증scripts:
- check_corrected_io.py:biasedstatsnormalizationreflection,nonzeroactor mean/std+Qequivariance,statscheckpoint,domiclock/standingzero/양쪽phase,legacyflag.PASS.
- check_ham_cpu.py:기본HAM/--bilateral 각각 --amp 유무4조합PASS. nativeSACupdates,routeridentity/한번step,독립Q,targetisolation,roundtrip,원본HAMexecutor동등성,holdoutsampling.
- check_holdout_metrics.py:knownRMSE,warmup,firstfailure,autoreset제외,DOFshape,WilsonCI.PASS.
- audit_bilateral_symmetry.py:실제저장actor/normalizer+512savedobs검사. v2flag이면DomiSymmetry사용.
- check_saved_holdout_results.py:640cases/20480trial재집계,planar와xyRMSE합성,10k20초완주/900steps,실제savedconfigsflags검사PASS.

최종10kbiHAM: evaluation/biHAM_symmetry_audit_v2.json.
normalizercommutation RMS/max0; fullpolicyreflection RMS1.653626e-7,max9.536743e-7.
routeroptimizersteps79920,action-scale/signpairerror0.
과거정규화포함policyreflection RMS .08110문제는실제학습후에도수정확인.
evaluation/results_integrity.json:640cases/20480trial,axisidentity최대오차6.37e-7,augmentationflagsfalse/true/false/false.
테스트한범위의검증이며버그가절대없다는뜻아님. 추가재학습필요한새결함은발견하지못했음.
원본domi와의차이:29physicalactionhistory복원제약,Holosoma관측의미유지,weight_recenterFalse,nativeSACdistribution/loss,공통phase/normalization. 전체원본연구재현아님.

## 9. 결과 페이지와 PNG

TensorBoard: http://localhost:6006 (이름v2실험선택).
campaign: http://localhost:8008/corrected_v2_20261029_0513/
평가: http://localhost:8008/corrected_v2_20261029_0513/evaluation/
루트scripts/serve_bilateral_results.py,launch_tensorboard.py서버. DockerTB16006→6006proxy. 8008은logs/bilateral_queue를serve.
원격컴퓨터의localhost는그컴퓨터이므로필요하면SSHportforward. 기존접속설정유지.

evaluation아래:
summary.csv/json,manifest.json,status.json,status_gpu3/6.json.
videos/{model}/model_0001000/{case}/metrics.json에개별trial과시간곡선.
plots/tracking_rmse.png/.svg,lateral_rmse.png/.svg,survival_duration.png/.svg,velocity_curves_0001000.png~0010000.png.
plots/rmse_summary.png:4모델통합3300×2200,학습방향/heldout방향1k~10k곡선+10k방향bars.
**plots/rmse_summary_no_mlp.png**:최신요청3모델버전,상단y축0~.6. 기존4모델보존.
각각_values.json표값저장.
재생성: sourcehsgym후python scripts/plot_rmse_summary.py --evaluation <dir> [--exclude-mlp].

## 10. 영상의 최신 결과 / 제작 방식

요청은초기에18개방향영상→6개만필요→최종적으로 **세모델을겹쳐왼쪽학습명령/오른쪽deploy명령을가로분할한영상하나** 로정정됐다. 잡다한글씨삭제,로봇색구분,command화살표요구.

최종전달:
- 호스트 /home/ycho/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513/evaluation/training_vs_deploy.mp4
- URL http://localhost:8008/corrected_v2_20261029_0513/evaluation/training_vs_deploy.mp4
- preview는동일경로training_vs_deploy_preview.png.
- 1280×400,약20초,약16.667fps,biHAM파랑/symaug주황/HAM초록.
- 왼쪽TRAIN COMMAND+왼쪽화살표,오른쪽DEPLOY COMMAND+오른쪽화살표,작은색상legend. 원본숫자overlaycrop.
- **학습당시녹화본아님.** 최종10kpolicy가학습에사용한왼쪽명령을평가환경에서재생. 실제학습은앞/뒤/왼쪽이므로‘왼쪽만학습’이라는설명금지.
- **한sim에서세로봇을동시에렌더한것아님.** 독립영상6개를후처리합성. GrabCut로로봇영역추정,색입힘,alphaoverlay. 카메라가각로봇을따르므로공유worldtrajectory비교로해석하면안됨. shadow/segmentation부위도색칠될수있다. 정량근거는원본metrics/rawvideos.
- scripts/compose_experiment_video.py:CPU합성. sourcehsgym후실행해야FFmpeg PATH/LD_LIBRARY_PATH가잡힘. cv2/numpy/subprocess/shutil+condaFFmpeg. 최종exit0및preview확인.
- scripts/inspect_comparison_video.py:원본metadata/frame검사용;composition_preview_source.png생성.

원본6영상은evaluation하위:
videos/biHAM/model_0010000/left_0.6mps/episode_0_1793251737.mp4
videos/biHAM/model_0010000/right_0.6mps/episode_0_1793251769.mp4
videos/sym_aug_HAM/model_0010000/left_0.6mps/episode_0_1793251968.mp4
videos/sym_aug_HAM/model_0010000/right_0.6mps/episode_0_1793252003.mp4
videos/HAM/model_0010000/left_0.6mps/episode_0_1793252199.mp4
videos/HAM/model_0010000/right_0.6mps/episode_0_1793252235.mp4

18개재생페이지:evaluation/three_model_videos.html.
controller scripts/run_three_model_videos.py,three_model_video_status.json18/18.
evaluate_command_holdout.py에protocol.video_cases필터추가. case별video.json저장. 숫자metrics덮어쓰지않음.
이전선별4영상은videos/{model}_highlight/model_0003000/forward_right_0.3mps/. highlight_selection.json:3k,trial21,같은초기상태,biHAM최저오차사후선정,원하는순위필터없음,requested_order_observedFalse. 전체집계와분리.

### renderer GPU 매핑
CUDA物理ordinal과IsaacGym bundled Vulkan ordinal이다름. **物理GPU3=graphics_index5**.
scripts/vulkan_gpu_index.py가UUID매칭,evaluation/graphics_device.json기록.
GPU3 UUID=GPU-83e63870-9f2a-9bcb-31a0-5dbd85e0677e.
graphics_device_id=3을그대로쓰면物理GPU6으로렌더할수있어과거시도보관/중단.
현재video는UUIDmapping+CUDA_VISIBLE_DEVICES3만사용. startupinvaliddeviceordinalcamera경고는있었지만영상/실제GPU3사용확인. 경고없다고단언하지않음. 모든GPU노출우회금지.

## 11. 다음 세션 작업 원칙과 이전 기록

현재완료상태를먼저확인하고다음사용자지시를따른다. 옛PPO/IsaacSim속도진단으로되돌아가지않음.
새학습필요시새version/output사용,기존결과보존,지정GPU만. 연구domi원본레포읽기전용.
새버그는발견즉시보고. 기대한성능순위가아니라는이유만으로재학습하지않음.
추가trainingseed/정규화효과ablation/seen-heldoutmatched평가는후속연구가능성이지만새실행지시없음.
영상은최종파일하나우선. 요청없이전checkpoint×전command640개렌더sweep재개금지.

최근자동권한심사가여러번timeout되어대기가길었다. timeout은unsafe판정과다름. 허용재시도/작은독립작업사용.
sandbox기본exec는bwrap:loopback Failed RTM_NEWADDR실패반복→좁은dockerexecprefix의escalated실행사용.
긴inlinePython/heredoc/큰문서patch가timeout되곤했다. 작은script를레포에저장후dockerexec로실행하거나문서patch를작게나누면안정적이었다.
도구실패를숨기거나실행안한검증을완료했다고하지말것.
최신인계파일은이파일. 루트HANDOFF_PROMPT.md와FASTSAC_STATUS.md는아래원문스냅샷으로함께보존.
부록의‘현재학습중’,‘PPO가맞음’,‘GPU3만’,‘phase미구현’은**과거시점**기록이며위최신본문우선.



---

## 과거 기록 원문: HANDOFF_PROMPT.md

> 과거 시점의 기록이다. 최신 상태는 이 파일 앞쪽 본문을 우선한다.

# 새 Codex 계정 인계 프롬프트

> 최신 상태(2026-10-29 05:13 KST): 이 문서의 초기 설정 설명보다 **FASTSAC_STATUS.md 마지막 `corrected v2 retraining STARTED` 항목을 우선 읽을 것**. 정규화가 biHAM 대칭성을 깨는 문제, domi phase 누락, phase 증강 불일치를 수정했고 네 모델 재학습을 GPU3+6에서 실행 중이다. Docker `ycho-holosoma`의 `scripts/run_corrected_campaign.py`가 GPU3 biHAM→MLP, GPU6 sym-augHAM→HAM, 전부10k 이후 전체40checkpoint 수치평가와 선별영상4개를 자동 실행한다. 출력 `logs/bilateral_queue/corrected_v2_20261029_0513`, `status.json`과 `runs.json` 확인. 중복 실행하지 말 것. 기존결과보존. 사용자는 새로 수정해야 할 부분을 발견하면 즉시 알려달라고 요구했다.

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


---

## 과거 기록 원문: FASTSAC_STATUS.md

> 과거 시점의 기록이다. 최신 상태는 이 파일 앞쪽 본문을 우선한다.

# FastSAC migration (2026-10-28)

- Active code: `.isaacgym/holosoma`, based on official `holosoma-v0.0.3` (59d5e61).
- Original working tree preserved; backup: `.migration_backups/20261028_121034`.
- Docker: `ycho-holosoma`. Conda: `hsgym` (Python 3.8, Torch 2.4.1+cu121, IsaacGym Preview 4).
- Only physical GPU 3 is authorized. All training runs sequentially inside Docker.
- User corrected the intended algorithm from PPO to FastSAC.
- Vanilla FastSAC ran under `logs/bilateral_queue/20261028_125443_isaacgym`.
- User authorized terminating experiment 1 as soon as experiment 2 was ready. Experiment 1 was terminated; saved logs/checkpoints remain.
- Current queue requests `g1-29dof-bilateral-fast-sac` then `g1-29dof-bilateral-no-right-fast-sac`.
- Launcher log: `logs/bilateral_queue/gpu3_bilateral_fastsac_launcher.log`. Check this and the newest run log for live status.
- First bilateral launch failed on Python 3.8 type annotations in copied domi code. Future annotations imports were added; queue relaunched.
- Actor uses native SAC sampling, tanh scaling and log-probability. Q heads retain native categorical projection and independent private banks.
- Online actor and Q heads share one domi router. Target Q router is a separate copy updated by native Polyak averaging.
- Private actor/Q optimizers exclude router parameters. Unscaled router gradients are accumulated and stepped once after each SAC update batch.
- Original SAC update methods are copied in `agents/modules/bilateral/original/fast_sac_updates.py` with explicit router hooks.
- TensorBoard: localhost:6006; shared logs are visible through the legacy checkout's logs symlink.
- Rendering deferred at user's request; current queue uses `--skip-render`.
- IMPORTANT: inspect actual logs before declaring bilateral training successful. No separate GPU smoke training was requested.
- Prior phase-generator and action-history reconstruction limitations still apply. Symmetry augmentation remains enabled, including experiment 3.

## Latest recovery

- Run `20261028_132840_isaacgym` failed at update 11 with Torch 2.4 compiled AMP float32/bfloat16 matrix dtype mismatch.
- Bilateral FastSAC presets now set `compile=False`; AMP and SAC loss/hyperparameters remain unchanged.
- Replacement run: `logs/bilateral_queue/20261028_133833_isaacgym`.
- Actual updates reached 40 without that error. Logging interval is 100; check newer logs for current progress.
- Two-job queue contains bilateral FastSAC then no-right bilateral FastSAC. It stops on failure; it does not silently continue to experiment 3.

## Checkpoint preview rendering (2026-10-28 14:36 UTC)
- Experiment 2 training remains running on GPU 3. Do not start experiment 3 until user reviews videos / requests switching.
- Gallery http://localhost:8008, run `20261028_133833_isaacgym/render_preview`. Checkpoints 1000/2000/3000 captured at render start, latest first, four directions each, 300 steps (6 seconds) or until termination.
- GPU 3 Docker background renderer running via `.isaacgym/holosoma/scripts/render_bilateral_checkpoints.py --queue-dir /workspace/holosoma/logs/bilateral_queue/20261028_133833_isaacgym --gpu 3 --steps 300`. Log `logs/bilateral_queue/render_preview.log`. Existing completed clips skipped on restart.
- Launch with CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID DISPLAY= (empty). DISPLAY=:0 hangs at graphics initialization.
- Evaluation uses 1 environment, robot contact_pairs_multiplier=1 (training 16 causes OOM alongside training), SAC replay buffer 16, actor + observation normalizer checkpoint weights only. No optimizer/Q state needed for inference.
- IsaacGym graphics device can now be set via HOLOSOMA_GRAPHICS_DEVICE_ID; renderer sets 3, CUDA logical device is 0.
- Use torch.no_grad(), not inference_mode(): resets mutate termination tensors outside policy loop.
- 3000 forward and backward videos generated/HTTP200 verified. Forward mean body vx=0.104 versus +0.6 command; backward vx=-0.501 versus -0.6 command. Both survived 300 steps. Other videos still rendering at this update.

## Experiment 3 started (2026-10-28 14:38 UTC)
User reviewed preview and explicitly requested stop experiment 2 and start 3. Stopped Docker queue PID59443, train59463 and renderer68870/68886/68887. Exp2 stopped at iteration3621; saved checkpoints/videos preserved.
Started `python scripts/run_bilateral_queue.py --gpu 3 --simulator isaacgym --num-envs 4096 --seed 43 --skip-render --presets g1-29dof-bilateral-no-right-fast-sac` inside ycho-holosoma, hsgym, clone working directory. CUDA_VISIBLE_DEVICES=3, CUDA_DEVICE_ORDER=PCI_BUS_ID.
Run: `logs/bilateral_queue/20261028_143823_isaacgym/1_g1-29dof-bilateral-no-right-fast-sac_seed43`. Actual optimization verified through iteration26; GPU3 97%,19582MiB. 50000 iterations configured. TensorBoard same6006; scalar writes every100. New run starts from scratch.
ThreeDirectionCommand samples forward/backward/left uniformly, preserves stand/yaw behavior. Native FastSAC symmetry augmentation remains True per existing settings; user told this is command-sampling holdout, not exclusion of mirrored rightward samples from optimizer training.

## Experiment 3 restarted WITHOUT replay augmentation (2026-10-28 14:49 UTC)
User explicitly requested disable augmentation and restart from scratch. Terminated previous exp3 queue70759/train70777. New canonical run: `20261028_144957_isaacgym/1_g1-29dof-bilateral-no-right-fast-sac_seed43`.
Updated no-right FastSAC preset `config_values/loco/g1/experiment.py` to use_symmetry=False, leaving Bilateral internal IO mirroring intact. Launch also passes --disable-symmetry for explicitness. Same Docker/hsgym/GPU3,4096env,seed43,50000iterations,skip-render. Launcher log gpu3_experiment3_noaugmentation_launcher.log.
Saved holosoma_config.yaml verified: use_symmetry:false, ThreeDirectionCommand, checkpoint:null. Actual optimization verified through35 iterations, GPU3 memory17290MiB. TensorBoard6006 new run144957, scalar interval100. Previous143823 run is superseded and not the intended holdout experiment.

## CURRENT: corrected four-way comparison (2026-10-28 15:09 UTC / Oct29 00:09 KST)

**Use this section over older status entries.** User authorized restarting corrected biHAM and chaining all four conditions. Superseded run144957 was stopped (queue73192/train73210); prior results remain on disk but do not represent the intended shared-router training.

Active queue: `logs/bilateral_queue/20261028_150951_isaacgym`. Launcher log: `gpu3_four_way_comparison_launcher.log`. Docker `ycho-holosoma`, hsgym, cwd `/workspace/holosoma/.isaacgym/holosoma`, physical GPU3 only.

Order / preset / replay augmentation:
1. biHAM — g1-29dof-bilateral-no-right-fast-sac — OFF
2. sym-aug HAM — g1-29dof-ham-sym-aug-no-right-fast-sac — ON
3. HAM — g1-29dof-ham-no-right-fast-sac — OFF
4. MLP — g1-29dof-mlp-no-right-fast-sac — OFF

All use same forward/backward/left command sampler, 4096env,seed43,50000iterations,compile=False. Actual rightward commands held out; sym-aug HAM intentionally gets synthetic mirrored rightward data. Queue stops on any failure and starts next condition only after successful exit. Rendering skipped; existing8008 gallery/TensorBoard6006 remain.

### Router bug and fix
PyTorch2.4 functional_call with detached parameters on aliased Q routers leaves detached tensors registered after call: original parameter identities/gradient flags are not restored. Reproduced with both HAM and biHAM on CPU. `_policy_q` now freezes Q requires_grad flags temporarily and restores in finally, never substitutes tensors. Actor actions are computed before freezing so actor/router gradient through actions remains. Q and actor router gradients accumulate and router optimizer steps once. Two consecutive updates, identity stability, router step count, independent Q banks, target isolation checked. Existing running processes do not hot-reload this fix; hence restart.

### HAM source and adaptation
Source container147-sejune-domi-isaac-lab, `/home/user/domi-alps/source/domi/src/domi/cfg/net/ham_sa_from_ma_v4_abl_preact.yaml`. It uses `domi.net.td.magnet_v2:MAGNetMA` with `MultiRange`, full concatenated left/right inputs, hidden384/576/384, router256/256/128,4experts,gate8,feature composition,use_sym_param=False,ortho_init=True. YAML does not explicitly set preact; MAGNetMA defaultFalse, but GateModLinear enables preactivation for compose_mode=feature. No bilateral parameter tying.
Full original range_layers.py copied to `agents/modules/bilateral/original/ham_range_layers.py` with future annotations and relative layer import only. YAML, MAGNetMA reference and original hashes retained alongside it in `ham_provenance.json`. Existing extracted original layer helpers reused.
Adapter `agents/modules/bilateral/ham_adapter.py` selects original MultiRange actor/Q subnets and reuses its forward intact; online actor/Q share router, Qbanks independent, target separate. Reuses same domi IO views as biHAM (same previously documented phase/history limitations). SAC outputs are per-side mean/logstd merged to29physical actions; Q outputs101atoms, privileged critic inputs include candidate actions. Native SAC distributions, loss and projection retained; HAM Q logits float32 for projection accumulation. HAM-specific router/bank initialization follows source, SAC outputheads zero as native. BiHAM-only flags feature_bias_scale/checkpoint_layers/use_fast are not transplanted into HAM where source constructor does not support them.

### Changed files
- agents/modules/bilateral/ham_adapter.py: original HAM banks + SAC I/O adapter.
- agents/fast_sac/ham_fast_sac.py: selects HAM builder, reuses shared-router SAC update.
- agents/fast_sac/bilateral_fast_sac.py: injectable builder and router parameter preservation fix.
- config_values/loco/g1/experiment.py and config_values/experiment.py: three comparison presets/registry; HAM dimensions saved explicitly.
- src/holosoma/pyproject.toml: opt_einsum3.4.0 and icecream2.1.8, installed only in Dockerhsgym.
- scripts/check_ham_cpu.py: noGPU regression checks, optional --amp for CPU BF16.

CPU float32 and BF16 HAM update checks passed; HAM has not yet run onGPU. No extra GPU was used for tests. Current corrected biHAM passed first actual updates beyond warmup. TensorBoard runprefix150951.

Exact launched queue command (inside activated hsgym):
`python scripts/run_bilateral_queue.py --gpu 3 --simulator isaacgym --num-envs 4096 --seed 43 --skip-render --presets g1-29dof-bilateral-no-right-fast-sac g1-29dof-ham-sym-aug-no-right-fast-sac g1-29dof-ham-no-right-fast-sac g1-29dof-mlp-no-right-fast-sac`
Do NOT pass --disable-symmetry to this whole queue: each preset owns its augmentation flag.

## CURRENT: split across GPU 3 and GPU 6 (2026-10-28 15:57 UTC)

User authorized GPU6 in addition to GPU3. New partition:
- GPU3: corrected biHAM run `20261028_150951_isaacgym` continues WITHOUT restart, then HAM without augmentation.
- GPU6: sym-aug HAM run `20261028_155752_isaacgym`, then MLP without augmentation.

Stopped ONLY old four-job queue parent76581. Active biHAM child76599 remains running, reparented to PID1. Do not restart the old four-job queue (duplicates).

GPU3 continuation watcher PID83412 uses `scripts/continue_after_training.py`. State: `logs/bilateral_queue/20261028_150951_isaacgym/continuation_gpu3.json`; log: `gpu3_ham_continuation_launcher.log`. It waits for PID76599 with matching start time, validates final model_0050000.pt on CPU (global_step50000, router optimizer present), rejects tracebacks at log end, then launches one-job queue `g1-29dof-ham-no-right-fast-sac`. Failure stops continuation. Original queue parent no longer writes the biHAM final VRAM report; continuation state records completion.

GPU6 queue83427/train83445, launcher `gpu6_ham_mlp_launcher.log`. Command inside Docker hsgym:
`python scripts/run_bilateral_queue.py --gpu 6 --simulator isaacgym --num-envs 4096 --seed 43 --skip-render --presets g1-29dof-ham-sym-aug-no-right-fast-sac g1-29dof-mlp-no-right-fast-sac`

Detached Docker processes survive SSH/client closure. Independent experiments per GPU, not DDP. An unrelated ~918MiB process on GPU6 was left untouched. HAM on GPU6 passed warmup and32 iterations, total GPU6 memory19788MiB including that process. GPU3 biHAM continued beyond3159 iterations,17366MiB. HAM's first real CUDA updates are now verified; long-run convergence unknown. All conditions4096env,seed43,50000iterations,front/back/left commands. Augmentation: biHAM OFF,sym-augHAM ON,HAM OFF,MLP OFF. TensorBoard6006 shows both150951 and155752. No rendering launched.

## TensorBoard Docker port repaired; evaluation requirement recorded

User could not open6006 through Docker/VSCode. Existing TensorBoard was healthy on Docker16006, host6006 proxy worked, but Docker6006 refused connections. Added `--tensorboard-proxy-only` to main repository `scripts/serve_bilateral_results.py`, launched detached CPU-only Docker process:
`python3 scripts/serve_bilateral_results.py --tensorboard-proxy-only --host 0.0.0.0 --port 6006`
It forwards to existing localhost16006; log `/tmp/holosoma_tensorboard_proxy6006.log` inside Docker. Both host6006 and Docker6006 returned HTTP200, and scalar queries through Docker6006 worked. Forward container6006 to local6006 in VSCode. Training and TensorBoard backend were not restarted.

Latest queried scalars: corrected biHAM iteration8800 mean_reward184.659, mean_episode_length994.97; sym-augHAM iteration7200 mean_reward182.215, mean_episode_length993.39. These are training metrics at different iterations, not rightward evaluation results.

User explicitly wants numeric evaluation AND videos after training for all four models, including pure rightward, forward-right diagonal, and backward-right diagonal commands. Evaluate identical speeds/seeds/conditions and report velocity tracking errors, failure rate and survival duration, with videos accessible locally. User then prioritized completing training correctly first. This expanded evaluation is a requirement still to implement/schedule; existing cardinal-direction preview renderer alone does NOT satisfy it. No evaluation/render GPU job was started during this TensorBoard fix.

## CURRENT: stop first two conditions at 10,000 and switch (user request)

User requested biHAM stop at10k then HAM without augmentation on GPU3, and sym-augHAM stop at10k then MLP on GPU6. Replaced old watcher83412 and GPU6 queue parent83427 ONLY; active training processes were preserved until milestone validation.

`scripts/continue_after_training.py` now supports `--iteration 10000 --stop-at-checkpoint`. It waits for model_0010000.pt, loads it on CPU to verify global_step and router optimizer state, checks size/mtime unchanged across load (avoid partial saves), sends SIGTERM to the original PID/starttime, waits for its exit, revalidates checkpoint and starts the next queue. Locks prevent duplicate watchers for the same state. Failure blocks continuation. Current state files remain continuation_gpu3.json under150951 and continuation_gpu6.json under155752. Logs gpu3_ham_continuation_launcher.log and gpu6_mlp_continuation_launcher.log.

GPU3 transition already executed: biHAM checkpoint10000 saved/validated, process stopped at logged10029 while checkpoint validation/switch took place. Use exact10000 checkpoint for evaluation. HAM without augmentation started as `20261028_174024_isaacgym_gpu3/1_g1-29dof-ham-no-right-fast-sac_seed43`. GPU6 sym-augHAM was ~9026 at last check, with watcher waiting to switch at10k to g1-29dof-mlp-no-right-fast-sac.

New queue output directories include `_gpu3` or `_gpu6` suffix to avoid two GPU queues creating the same timestamp directory. User only shortened the first two current runs; subsequent HAM/MLP still have original50000 limit unless later instructed otherwise. All data-augmentation flags, seed43,4096env and command holdout remain unchanged.

## CURRENT override: ALL FOUR CONDITIONS STOP AT 10,000

User explicitly changed the budget: biHAM, sym-augHAM, HAM and MLP must all stop at10k. First two have already transitioned with valid model_0010000.pt preserved. Active HAM run174024_gpu3 (trainPID96511) and active MLP run175105_gpu6 (trainPID98933) were retained without restart. Terminated their obsolete outer watchers96347/96427 and queue parents96507/98929 only, to avoid treating the intentional stops as failures or scheduling additional jobs.

New stop-only watchers use `continue_after_training.py --iteration 10000 --stop-at-checkpoint` with no `--presets`. State: `stop_at_10000.json` inside each active run directory. Logs: `gpu3_ham_stop10k.log`, `gpu6_mlp_stop10k.log`. MLP passes `--no-router-optimizer`, because its checkpoint has no shared-router optimizer; both types verify global_step and actor state. On validated10000 checkpoint, terminate matching original process, wait for exit, mark complete, and launch nothing else. Checkpoint validation may allow a few extra in-memory updates before stopping; comparison uses exact10000 saved weights.

Four comparison presets now also specify num_learning_iterations10000 for future launches. Already-running progress bars/config snapshots still show their original50000 setting; external stop watchers enforce the new10000 budget without discarding progress. Latest observed HAM3194 and MLP3311 before watcher activation. Post-training numeric rightward/diagonal evaluation and videos remain outstanding as previously recorded.

## CURRENT: training complete; checkpoint evaluation and videos (2026-10-29 KST)

All four exact 10,000 checkpoints exist. HAM and MLP stop watchers are `complete` and their training has stopped. GPU6 is released: the user's latest instruction authorizes evaluation/rendering on physical GPU3 ONLY. Do not restart any training queue.

Active evaluation code is under `.isaacgym/holosoma/scripts/`:
- `evaluate_command_holdout.py`: deterministic saved actor + observation normalization, nominal flat-ground environment, error measurement and video recording. No learning. Captures velocity and termination BEFORE native automatic resets; counts only each trial's first episode.
- `holdout_terrain.py`: spaces parallel robots on a grid. Native eval puts all robots at one origin, which exhausted PhysX broadphase capacity at512 environments. The initial attempt is archived in `deploy_eval_four_way_10k/invalid_overlap_attempt_*` and excluded from results. Also fixed flattened DOF state indexing: snapshots contain all29 position/velocity pairs.
- `run_holdout_evaluation.py`: detached, locked, resumable queue for all four models, checkpoints1000..10000 (latest first). All metrics first at each checkpoint, then videos. Failure is recorded in status.json and stops the queue.
- `plot_holdout_evaluation.py`: CSV/JSON summaries, velocity-error learning curves, observed duration, actual-versus-command time plots; optional post-hoc matched highlight selection.
- `holdout_gallery.html`: checkpoint/direction/speed selectors, four-model metrics/videos, plots and synchronized highlight playback.
- `check_holdout_metrics.py`: CPU checks passed for known planar RMSE, warmup exclusion, capture of failure before reset clears the mask, exclusion of subsequent reset episodes, full DOF snapshots and Wilson interval.
- `vulkan_gpu_index.py`: maps physical GPU UUID to the ordinal used by IsaacGym's BUNDLED Vulkan loader. Its order differs from both nvidia-smi and the system Vulkan loader. Here physicalGPU3 is Vulkan index5, while Vulkan index3 is physicalGPU6. Never hardcode graphics_device_id=3 as meaning physicalGPU3. Mapping is stored in graphics_device.json. CUDA remains masked to physical3.

Output: `logs/bilateral_queue/deploy_eval_four_way_10k/`; launcher log `deploy_eval_launcher.log`. Web URL `http://localhost:8008/deploy_eval_four_way_10k/` returned200 inside Docker. TensorBoard remains6006. Gallery8008 already serves this directory; no server restart required.

Protocol v2: eight directions (right, forward-right, backward-right, forward, backward, left, forward-left, backward-left), each at total speed0.3/0.6m/s, yaw command0. Right means negative body-y.32 trials per case;20s episodes; exclude first2s from tracking error. Observation noise, domain randomization and pushes disabled for this controlled evaluation. Same seed20261029 and comparable initial states for every model. Primary metric: mean of per-trial planar velocity RMSE over observed steps before first failure; vx/vy/yaw errors also saved. Failed-before-warmup trials have null errors, never artificial zero; report measured-trial count, survival duration and failure rate alongside conditional errors. Secondary success requires full20s and planarRMSE<=0.2m/s, yawRMSE<=0.3rad/s (Wilson95% interval). These are simulator evaluations, not hardware deployment. Only one training seed was run, so rollout statistics do not estimate training-seed variability.

All four 10k metrics completed without the initial broadphase warning. At pure right0.6m/s, mean planar RMSE (lower is better): biHAM0.489, sym-augHAM0.385, HAM0.599, MLP0.377; all32 trials per model survived20s. The requested biHAM > sym-augHAM > HAM/MLP narrative is NOT supported by these results. Do not present the requested ranking as observed. More checkpoint evaluation is queued.

User requested attractive videos illustrating biHAM generalization. Agreed implementation: same command/speed and matched initial physical state across all four models, transparently label post-hoc highlight, keep ALL trials in aggregate plots. Select an actual trial supporting the requested ranking if one exists; otherwise show a matched best-biHAM trial with `requested_order_observed:false`. At10k no supporting matched rightward trial was found. Video replay error is measured separately because replay in a one-env GPU simulation can differ from a512-env rollout; do not label selected-trial errors as measured video errors. Each checkpoint also gets fixed-seed illustrative videos for all16 cases,8s or until failure; highlights20s. Initial actor observation is checked when replaying a selected state.

During graphics debugging, two highlight renders briefly allocated graphics resources on GPU6 due to the old loader's order. Their completed artifacts are preserved under `previous_graphics_attempt`, excluded from the active gallery. Stopped the owned queue to fix UUID mapping and restarted it; completed numeric evaluations were retained. Inspect current status.json, worker logs and nvidia-smi before reporting current progress or claiming video verification. Remaining work: verify corrected GPU3-only renders and images, let all40 checkpoints finish, inspect results and report actual model ranking.

Resume command inside Docker (does not retrain):
`docker exec -d -e CUDA_VISIBLE_DEVICES=3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e DISPLAY= -w /workspace/holosoma/.isaacgym/holosoma ycho-holosoma bash -lc 'source scripts/source_isaacgym_setup.sh && exec python scripts/run_holdout_evaluation.py >> /workspace/holosoma/logs/bilateral_queue/deploy_eval_launcher.log 2>&1'`

GPU3-only rendering verification: corrected biHAM20s highlight produced333frames at16.667fps; frame inspected (robot and terrain visible), MP4 HTTP200. Measured replay RMSE0.22135 versus selected trial0.22048. nvidia-smi during corrected rendering showed the owned process on physicalGPU3 only, no owned GPU6 allocation. CUDA_VISIBLE_DEVICES remains3. IsaacGym still logs a startup `invalid device ordinal` warning from its camera CUDA interop with graphics index5; video decoding,20s simulation and matched initial observation check succeeded. Do not describe logs as entirely error-free. A proposed broad CUDA visibility workaround was rejected by automatic approval review and was NOT applied; the working restricted-GPU renderer was retained. At latest inspection biHAM, sym-augHAM and HAM matched highlight videos were complete, MLP was rendering; afterward the queue proceeds to all16 illustrative commands and remaining checkpoints. Gallery index, summary, learning plot and velocity plot all returned HTTP200.

## CURRENT override: numeric evaluation FIRST; stop full video sweep

User explicitly requested stop rendering everything, get all numerical results quickly, and generate only well-performing selected videos later. Stopped owned queue107134 and current HAM renderer116741 via SIGTERM. Saved results/videos remain intact; no training or unrelated GPU processes touched. All four10k matched highlight videos had completed, plus full16-case biHAM/sym-augHAM10k clips and some HAM clips.

`run_holdout_evaluation.py` now defaults to numerical evaluation ONLY, no automatic video/highlight phases. Resumes already-complete checkpoints without launching a worker. Evaluates all four models at1000..10000;32trials for each of16commands still unchanged, including pure left/right with vx=0,yaw=0 at0.3/0.6m/s. Summary CSV/JSON updates after each model; plots after each four-model checkpoint to reduce CPU overhead. `plot_holdout_evaluation.py` accepts an iteration filter so it does not repeatedly redraw all old time-series figures. Final full plots are refreshed once at completion. Status jobs_total=40 counts model/checkpoint metric jobs. Old640-video ETA no longer applies; remaining36metric jobs at roughly45–50s each implies approximately30min plus plotting/startup overhead, subject to actual progress.

Detached metrics-only queue relaunched with the same Docker/hsgym/GPU3 command, output unchanged, launcher log `logs/bilateral_queue/deploy_eval_metrics_launcher.log`. Dashboard now states ungenerated videos are not scheduled, rather than promising every pending video. Review all-checkpoint error results before selecting any additional good-example videos. Do not restore the exhaustive video sweep.

## CURRENT override: GPU6 authorized again for parallel NUMERICAL evaluation

User explicitly said GPU6 is now free and requested it evaluate from1k upward while GPU3 descends, to reduce time. This supersedes the prior GPU6 restriction for these metric jobs. No rendering was added. Stopped ONLY old metrics-queue parent118549; preserved active HAM8k worker119227. New GPU3 queue waits for that worker to exit, skips completed checkpoints and continues10k→5k. GPU6 independently evaluates1k→4k. At the split,10k/9k were done and8k in progress, so the remaining jobs are approximately balanced. Ranges are disjoint; no duplicate rollout jobs. Both GPUs were observed active with approximately1.1–1.3GiB each.

`run_holdout_evaluation.py` now has `--gpu`, `--min-iteration`, `--max-iteration`, `--ascending`, `--wait-pid`. Per-GPU runner locks, a shared summary/plot lock, unique JSON temporary filenames and per-checkpoint config snapshots prevent write collisions. `status_gpu3.json`/`status_gpu6.json` retain individual progress; `status.json` aggregates actual complete40 metric jobs and both workers. `evaluate_command_holdout.py --gpu` masks CUDA to the requested physical3 or6; metrics have graphics disabled.

Inside Docker hsgym, detached queues:
- GPU3: `python scripts/run_holdout_evaluation.py --gpu 3 --min-iteration 5000 --max-iteration 10000 --wait-pid 119227`; launcher `deploy_eval_gpu3_metrics_launcher.log`.
- GPU6: `python scripts/run_holdout_evaluation.py --gpu 6 --min-iteration 1000 --max-iteration 4000 --ascending`; launcher `deploy_eval_gpu6_metrics_launcher.log`.

### Audit prompted by user's question about rightward training exposure

Read actual saved `holosoma_config.yaml` in all four canonical runs (not just current presets). All use the same ThreeDirectionCommand. Native CommandManager reuses the subclass instance for reset/step even though reset/step configs name its base LocomotionCommand. CPU audit with the saved command config:409600 reset/step samples,0 negative body-y commands. Augmentation flags: biHAMfalse, sym-augHAMtrue, HAMfalse, MLPfalse. Native FastSAC augments replay observations/actions only when use_symmetrytrue. No evidence of an unintended right-command sampler in these runs; CPU reproduction is not a historical trace of every training transition.

The experimental holdout is BODY-FRAME RIGHTWARD COMMANDS, not absence of all actual rightward motion. Saved training configs retain yaw commands[-1,1], random pushes enabled with max_push_vel[1,1] every5–10s, randomized initial velocities and stochastic SAC actions. These allow rightward physical velocities even under forward/back/left commands; rotating also changes world-frame direction. sym-augHAM intentionally sees mirrored right-command transitions in optimizer batches. Do not claim the other models never experienced rightward dynamics.

10k pure right command vy=-0.6m/s (32trials,2–20s): measured average vy: biHAM-0.2038, sym-augHAM-0.2899, HAM-0.0041, MLP-0.2950. Planar RMSE respectively0.4894,0.3853,0.5993,0.3765. HAM barely moves right; MLP achieves roughly half the commanded speed, so surviving/walking visually is not accurate command tracking. A symmetry-augmentation-free MLP can still partially generalize; this is a possible interpretation, not proof of the cause or of biHAM's proposed advantage.

## CONFIRMED: biHAM preprocessing breaks end-to-end reflection symmetry

User asked whether poor biHAM performance indicates a bug. Added CPU-only `scripts/audit_bilateral_symmetry.py`; loaded the actual10k biHAM actor and saved observation normalizer, tested512 stored evaluation initial observations. No checkpoint/training settings or ongoing evaluations were changed. Audit result: `logs/bilateral_queue/deploy_eval_four_way_10k/biHAM_symmetry_audit.json` (also accessible through8008).

- Original domi observation-view swap error: exactly0 for the tested observations.
- Bilateral core bank swap error: RMS6.96e-8,max4.77e-7.
- Physical action scaling pairs and domi versus native joint mirror signs: both exactly0 discrepancy.
- End-to-end reflected deterministic-action discrepancy WITHOUT normalization: RMS1.50e-7,max8.05e-7.
- WITH saved normalizer: RMS0.08110,max0.84953 in Holosoma actor action units (before environment action_scale).
- Diagnostic-only symmetric mean/variance projection: RMS1.86e-7,max1.37e-6. This projection was used ONLY in memory for the CPU test, NOT applied to checkpoints or eval.
- Saved command-vy normalization mean+0.13294248,std0.26540795,eps0.01. Thus normalization N does not commute with physical reflection M: N(Mx) != M(Nx). Largest tested input discrepancy0.96542, command-vy; cosine phase also mismatches due to its nonzero empirical mean.
- Router optimizer state confirms79920 update steps; lack of router updates is not the observed problem.

The numerical tracking results remain measurements of the currently trained implementation. No tracking-error calculation bug was found in the completed regression checks. HOWEVER, this is a confirmed integration problem for the intended physically symmetric policy: applying original per-feature Holosoma empirical normalization before the mirrored network removes the end-to-end guarantee. Do not describe these results as a validated test of an equivariant biHAM pipeline or proof that the architecture cannot generalize. Also do NOT claim this explains all performance loss: normalization projection has only been checked algebraically, not in locomotion rollouts or retraining. Existing weights were trained with the asymmetric preprocessing, so changing normalization at deployment alone is an intervention, not an established correction to their performance.

Next repair should account for reflection in observation normalization (shared mirrored statistics or an explicitly reflection-compatible normalization in the encapsulated adapter), retain the other algorithm/environment settings, add this end-to-end regression check, then evaluate/retrain a separately labelled corrected run. User has been told about the confirmed preprocessing issue; no corrected training run has been launched.

## 2026-10-29 05:13 KST — corrected v2 retraining STARTED (supersedes previous next step)

User authorized repairing integration problems, retraining all four models to10k, evaluating every1k checkpoint, selected videos only, GPU3+6. They explicitly requested immediate notification of newly discovered fixes. Reported normalization defect and then phase-generator/phase-augmentation mismatch before changing them.

Active persistent Docker supervisor: `.isaacgym/holosoma/scripts/run_corrected_campaign.py` in `ycho-holosoma`, hsgym / IsaacGym, directory `/workspace/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513`. GPU3: biHAM then MLP; GPU6: sym-augHAM then HAM. All4096env,seed43,10k,compileFalse. No old checkpoints/results overwritten. Initial iterations confirmed on both GPUs (biHAM32,sym-augHAM39 at first check), GPU memory about17.0/18.4GiB total. Initial steady rates about1.1/1.5iterations/s; full campaign ETA roughly4hours including remaining models and evaluation, allow4–5hours.

Launch command (already running, DO NOT run a duplicate):
```sh
docker exec -d ycho-holosoma bash -lc 'cd /workspace/holosoma/.isaacgym/holosoma && source scripts/source_isaacgym_setup.sh >/dev/null && python scripts/run_corrected_campaign.py --output /workspace/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513 > /workspace/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513.log 2>&1'
```

Changes are opt-in `*-fast-sac-v2` presets; legacy presets still load with `symmetry_compatible_io=False`.

- `agents/modules/bilateral/normalization.py`: ReflectionNormalization inherits native empirical normalization, projects paired mean/variance after update (including between-mean variance). Actor AND critic use it for all four conditions. No mirrored transitions are inserted by this operation; replay augmentation remainsfalse/true/false/false for biHAM/sym-augHAM/HAM/MLP. DomiSymmetry negates BOTH leg channels for each phase sin/cos; previous native augmentation negated just the first leg, producing inconsistent anti-phase observations. New statistics are shared across reflected features; this is a common preprocessing change for all four comparisons, so v2 MLP is not the literal untouched stock baseline.
- `agents/modules/bilateral/original/phase_clock.py`: four methods copied verbatim from domi PhaseCommand in147-sejune-domi-isaac-lab. Full source path/hash in module. Original resampling, elapsed time, sin/cos command, normalized clock update; contact reset explicitlyFalse as requested. IsaacLab lifecycle replaced by small constructor bridge. Actual loco_env_cfg period_range(.9,1.1),10sresampling,phase_zero_stand were checked.
- `managers/command/terms/domi_gait.py`: binds clock to existing Holosoma gait interface. Domiphase0 corresponds left swing peak/right stance, retaining existing foot-height reward convention. Both sine/cosine observations zero for standing; right phase is originaldomi negative encoding. All four new presets use same gait/observation mapping. Rewards, randomization, physical29actions, SAC losses and holdout command distribution preserved. Phase generation/observation encoding intentionally differs from stock, honoring user's explicit earlier domiphase instruction.
- `config_types/algo.py`, `agents/fast_sac/fast_sac_agent.py`: opt-in flag and small factory branch for normalization/augmentation. `config_values/loco/g1/experiment.py`, `config_values/experiment.py`: four distinct v2 presets.
- Action history remains physical29actions gathered into two16side views, duplicating sharedwaist. Cannot reconstruct originaldomi independent pre-merge waist histories from29physicalvalues. This is documented in adapter; user informed. No hidden history state or change to Holosoma action space. The gather itself passed reflection checks; not misreported as another confirmed algebraic bug.
- `scripts/check_corrected_io.py`: CPU biased-data normalization commutation, nonzeroactor mean/std and Q physical equivariance, stats checkpoint roundtrip, domi clock/standing/both phase channels, legacy flag unchanged. PASS.
- `scripts/check_ham_cpu.py`: HAM AND biHAM nativeSAC updates inFP32 andCPUbf16; sharedrouter identity/one optimizer step perupdate, independentQbanks,targetisolation,roundtrip,holdoutconfig. All4variantsPASS. Existing originalHAM executor comparison retained. No simulator/GPU used for these tests.
- `scripts/check_holdout_metrics.py`: knownRMSE,warmup,firstfailure/auto-resetexclusion,DOFsnapshot,Wilson intervalPASS; updated minimalfixture for gait accessor.
- `scripts/evaluate_command_holdout.py`: preserve training gait when replacing only velocity-command sampler for eval; record/restore domi phase+period for selected replay. Old build_config reset the entire command config to stock, which would have discarded corrected gait. Metrics still512env,16cases×32trials,20s,2swarmup,flatnominal,noDR/noobsnoise,firstepisode only.
- `run_bilateral_queue.py`: explicit output dir. `run_holdout_evaluation.py`: custom four-run manifest/output supported without obsolete old training watchers. `run_corrected_campaign.py`: independent sequential GPUqueues; waits success;20numericjobs/GPU (GPU3 6k..10k,GPU6 1k..5k),40total; then4selected matched-state videos onGPU3. Failures stop dependent stages and persist status, never erase completed results.
- `plot_holdout_evaluation.py`: select best surviving matched biHAM rightward trial over ALL checkpoints, no requested-ranking filter. Other models use samecheckpoint/command/rawobs/physicalstate/clock. Selection explicitlyposthoc; aggregate plots alltrials. `holdout_gallery.html` resolves selectedcheckpoint dynamically.

URLs (both checked HTTP200): TensorBoard `http://localhost:6006`; new campaign `http://localhost:8008/corrected_v2_20261029_0513/`; numerical results will appear under its `evaluation/` after training. `status.json` records supervisor phase/PIDs. `runs.json` maps exact model directories. Per-GPUtraininglogs and per-runVRAMreports in campaign. TensorBoard scans shared log root automatically; newruns contain`v2`.

Current state: training running, corrected performance NOT measured yet. Do not claim normalization repair guarantees better tracking. Old40checkpoint metrics are complete but test the earlier defective preprocessing; preserve separately. No remaining newly confirmed runtime failure at initial learning check. First new1kcheckpoint/export/evaluation still pending.

## 2026-10-29 — corrected v2 COMPLETE; consolidated results and final checkpoint audit

All four runs returned0, saved10checkpoints each. Complete40checkpoint evaluations /640command cases and4matched selected20svideos. Supervisor elapsed12626seconds (3h30m26s), completed approximately08:43KST. No training process remains; GPU3/6 idle at status check.

10k mean per-trial planar RMSE, m/s, averaged equally over0.3/0.6m/s cases:

| Model | Forward | Backward | Left | Right | Forward-right | Backward-right | Training directions mean | Rightward mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biHAM | .1768 | .1754 | .3017 | .3015 | .2423 | .2479 | .2179 | .2639 |
| sym-aug HAM | .2178 | .1890 | .3167 | .3356 | .2815 | .2563 | .2411 | .2911 |
| HAM | .2368 | .2192 | .3293 | .3565 | .3004 | .2881 | .2618 | .3150 |
| MLP | .2169 | .2099 | .3913 | .2762 | .2517 | .2163 | .2727 | .2481 |

Training directions means ONLYforward/backward/left (not left diagonals). Rightward meansright/front-right/back-right. Each direction has64trials,32eachspeed;20sepisodes,2swarmup,nominalflatphysics. Training seed43only. All10ktrials survive20s. Rightward10krankMLP,biHAM,symaugHAM,HAM; doesNOTsupportuniversalbiHAM superiority. Forpure right-.6m/sactualmeanvx/vy: biHAM(.013,-.2671),symaug(-.085,-.2428),HAM(-.0421,-.2135),MLP(-.1555,-.4351): MLP moves faster sideways but also drifts backward. BiHAM pureleft/right RMSEnearidentical. Seen-command errors remain substantial; post3kcurves fluctuate, so doNOTassert moreiterations alone will solve tracking. Earliercheckpointrankingsvary; HAMbestheldoutavgat5k~.2518. SymaugHAMseesrightcommandssynthetically; noaugHAM/MLP can generalize continuouscommands withoutguarantees and can experience lateralphysicalmotion duringstochastictraining. Sharedsymmetricnormalization itself is a commoninductiveprior, not replayaugmentation.

FinalsavedbiHAM10k CPUaudit (`evaluation/biHAM_symmetry_audit_v2.json`):512actualevalobservations; normalizerreflectioncommutationRMS/max0; fullpolicywithsavednormalizerRMS1.6536e-7,max9.5367e-7; routeroptimizersteps79920; action-scale/signpairerrors0. Priorend-to-endRMS.08110isresolvedontheactualtrainedpolicy. This verifies testedalgebraicconsistency, notabsenceofallpossiblebugs or guaranteedlocomotionquality. No additional confirmed issue requiring retraining found in these checks.

Userrequested one cleanPNG combining seen/unseencommands and checkpointcurves. Created `scripts/plot_rmse_summary.py`; output `logs/bilateral_queue/corrected_v2_20261029_0513/evaluation/plots/rmse_summary.png` (3300x2200), with `rmse_summary_values.json`. Panels:1k..10ktrainingdirectioncurve,heldoutrightwardcurve,10kdirectionbars. Allmethods and singleseed limitation onfigure. URL `http://localhost:8008/corrected_v2_20261029_0513/evaluation/plots/rmse_summary.png`. Existinggallery/videoURL remains `http://localhost:8008/corrected_v2_20261029_0513/evaluation/`.


