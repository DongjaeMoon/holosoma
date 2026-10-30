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
