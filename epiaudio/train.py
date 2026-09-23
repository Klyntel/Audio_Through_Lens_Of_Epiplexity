""" Runn Epiplexity requential Natural Data Experiment with Audio

TODO: Implement this

- Add a dataset preparer for audio like in epiplexity/picodo/dataset
    - Mirror the format used there as much as possible to allow us to compare to previous work and use the existing model arch
- Implement sweep feature
    - see epiplexity/picodo/sweeps/reuential.yml, we can make our own grid search algorithm surely
"""

# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false
# this code is mostly as is from the og work
# we swap env between jax and pytorch, therefore I feel like this is a non issue. 

import os
from typing import Any
import jax
import jax.numpy as jnp
import optax
# So we still want to pull from picodo from as much as we can
import picodo.data as data
import epiplexity.picodo.utils as utils
from picodo.train import (
    _is_gcs_path,
    _gcs_path_exists,
    _download_gcs_to_temp,
    get_scheduler,
    _compute_mup_scales,
    compute_loss_and_grads,
    apply_grads,
    distill_loss_and_grads,
    apply_student_grads,
    # eval_step,
    eval_features,
    downstream_eval_step,
    downstream_ft_eval,
    _upload_temp_to_gcs,
    loss_fn
)
import epiaudio.model as model_lib
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
import pickle
from math import prod
import copy
import tempfile
from comet_ml import Experiment

""" Thank claude.

the jitted eval loop unloops too much into memory creating a 360gb 
memory use. Claude suggests moving the jit into a helper function
to cut down on jit unrolling

Its possible this was caused by a newer vrs of jax or me running
this on a laptop for testing. I want to keep that in mind as we
go forward. 
"""
def eval_step(model, dataset):        
  x, _, _ = data.get_in_out(dataset[0])
  features = get_features_jit(model, x) 
  eval_loss = 0.0
  for batch in dataset:
    eval_loss += loss_fn(model, batch)  
  return {'eval_loss': eval_loss / len(dataset), 'features': features}

@nnx.jit
def get_features_jit(model, x):
  return model.get_features(x)


def train_and_evaluate(cfg: DictConfig):
  """ Train_and_evaluate function mostly as is from epiplexity but with
      wandb swapped for comet-ml logging
  """
  save_target = cfg.save
  save_path = None
  if isinstance(save_target, str):
    save_path = save_target
  elif save_target:
    save_path = f"gs://wandb_model_checkpoints_us_central2/{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T}/trained.pkl"
    print(f"config.save set to True; defaulting to {save_path}")

  resume_path = cfg.resume_from
  if not resume_path:
    resume_path = None
  elif resume_path == 'default':
    if cfg.T_ckpt is not None:
      if cfg.T <= 0:
        cfg.T = cfg.T_ckpt
      resume_path = f"gs://wandb_model_checkpoints_us_central2/{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T_ckpt}/trained.pkl"
    else:
      resume_path = f"gs://wandb_model_checkpoints_us_central2/{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T}/trained.pkl"

  # Base training controls
  train_teacher_base = cfg.train_teacher
  train_student_base = cfg.train_student
  assert train_teacher_base or train_student_base, 'At least one of train_teacher or train_student must be True'
  enforce_max_kl = train_teacher_base and train_student_base
  if enforce_max_kl:
    max_kl = float(cfg.max_kl)
    if max_kl <= 0:
      raise ValueError('max_kl must be a positive number')
  else:
    max_kl = float('inf')

  # No batch-size slowdown; teacher batch size is the configured batch
  accumulation_steps = int(cfg.A) if cfg.A is not None else 1
  if accumulation_steps < 1:
    raise ValueError('cfg.A must be at least 1 for gradient accumulation')
  if int(cfg.B) % accumulation_steps != 0:
    raise ValueError(f'cfg.B ({cfg.B}) must be divisible by cfg.A ({cfg.A})')
  teacher_batch_size = int(cfg.B)
  teacher_microbatch_size = teacher_batch_size // accumulation_steps
  ft_batch_size = int(cfg.B_ft) if getattr(cfg, 'B_ft', None) is not None else teacher_batch_size

  # datasets
  get_batch_train_teacher, ds_train_size = data.make_ds_loader(cfg.ds_path, 'train', cfg.model.L, teacher_microbatch_size)
  try: # either test or val
    get_batch_test, ds_test_size = data.make_ds_loader(cfg.ds_path, 'test', cfg.model.L, cfg.B)
  except Exception as _:
    get_batch_test, ds_test_size = data.make_ds_loader(cfg.ds_path, 'val', cfg.model.L, cfg.B)
  print(f'Train: {ds_train_size:.2g} tokens')
  print(f'Test: {ds_test_size:.2g} tokens')

  if os.path.exists(f'{cfg.ds_path}/meta.pkl'):
    with open(f'{cfg.ds_path}/meta.pkl', 'rb') as f:
      meta = pickle.load(f)
      cfg.model.V = int(jnp.ceil(meta['vocab_size'] / 32)) * 32
  elif cfg.ds_path == 'open':
    cfg.model.V = 96
  elif cfg.ds_path == 'cifar5m':
    cfg.model.V = 256

  # model
  if cfg.model.P is not None:
    cfg.model.D = round(((cfg.model.P * 1e6 / cfg.model.N / 12) ** 0.5) / 64) * 64
    if cfg.model.D < 64:
       exit()
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
  model = model_lib.create_sharded_model(cfg.model, mesh, cfg.seed)
  student_model = None
  resume_local_path = resume_path
  resume_temp_path = None
  if resume_path is not None:
    if _is_gcs_path(resume_path):
      if not _gcs_path_exists(resume_path):
        raise FileNotFoundError(f"Checkpoint {resume_path} not found on GCS")
      resume_temp_path = _download_gcs_to_temp(resume_path)
      resume_local_path = resume_temp_path
    else:
      if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint {resume_path} not found")
    assert resume_local_path is not None
    try:
      with open(resume_local_path, 'rb') as f:
        resume_state = pickle.load(f)
    finally:
      if resume_temp_path is not None and os.path.exists(resume_temp_path):
        os.remove(resume_temp_path)
    resume_state = jax.tree_util.tree_map(jnp.asarray, resume_state)
    ref_state = nnx.state(model)
    pspecs = nnx.get_partition_spec(ref_state)
    with mesh:
      sharded_state = jax.lax.with_sharding_constraint(resume_state, pspecs)
      nnx.update(model, sharded_state)
    print(f"Loaded checkpoint from {resume_path}")
  num_params = sum(prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(model)))
  print(f'Number of parameters: {num_params:.2g}')

   # possibly auto-set T from fitted scale/exponent
  assert cfg.T is not None or (cfg.scale is not None and cfg.exponent is not None), \
      "Either T or fitted scale/exponent must be provided."
  if cfg.T is None:
      C = 1e15 * ((num_params / cfg.scale) ** cfg.exponent)
      cfg.T = round(C / (6 * num_params))
  cfg.T = cfg.T or ds_train_size
  cfg.T_eval = cfg.T_eval or ds_test_size


  for key, value in cfg.items():
    print(f'{key}: {value}')
  train_tokens_per_step = teacher_batch_size * cfg.model.L
  eval_tokens_per_step = cfg.B * cfg.model.L
  num_train_steps = cfg.T // train_tokens_per_step
  num_test_steps = cfg.T_eval // eval_tokens_per_step

  data_sharding = NamedSharding(mesh, P('data'))
  with mesh: 
    ds_test = [jax.device_put(get_batch_test(i), data_sharding) for i in range(num_test_steps)]
  if cfg.downstream_ds_path is not None:
    get_batch_train_downstream, ds_train_size_downstream = data.make_downstream_ds_loader(cfg.downstream_ds_path, 'train', cfg.model.L, ft_batch_size)
    get_batch_test_downstream, ds_test_size_downstream = data.make_downstream_ds_loader(cfg.downstream_ds_path, 'test', cfg.model.L, ft_batch_size)
    cfg.T_eval_downstream = cfg.T_eval_downstream or ds_test_size_downstream
    downstream_tokens_per_step = ft_batch_size * cfg.model.L
    num_test_steps_downstream = cfg.T_eval_downstream // downstream_tokens_per_step
    with mesh: 
      ds_test_downstream = [jax.device_put(get_batch_test_downstream(i), data_sharding) for i in range(num_test_steps_downstream)]
    cfg.T_downstream = cfg.T_downstream or ds_train_size_downstream
    num_train_steps_downstream = cfg.T_downstream // downstream_tokens_per_step
    with mesh: 
      ds_train_downstream = [jax.device_put(get_batch_train_downstream(i), data_sharding) for i in range(num_train_steps_downstream)]
    meta_downstream = pickle.load(open(f'{cfg.downstream_ds_path}/meta.pkl', 'rb'))
    # assert meta_downstream['ctx_len'] == cfg.model.L, f'Downstream context length {meta_downstream["ctx_len"]} does not match model context length {cfg.model.L}'
    assert meta_downstream['vocab_size'] <= cfg.model.V, f'Downstream vocab size {meta_downstream["vocab_size"]} does not match pre-trained vocab size {cfg.model.V}'
    print(f'Downstream train: {ds_train_size_downstream:.2g} tokens')
    print(f'Downstream test: {ds_test_size_downstream:.2g} tokens')
    if cfg.downstream_ds_path == 'fen2cp':
      cfg.reinit_readout = True
      print('Reinitializing readout for fen2cp')

  warmup_steps = cfg.opt.warmup_tokens // train_tokens_per_step
  schedule_fn = get_scheduler(cfg.opt.schedule, cfg.opt.decay_frac, warmup_steps, num_train_steps)

  teacher_mup_scales = _compute_mup_scales(cfg.model, model, cfg.opt)

  # Set up EMA models and decay if requested
  # Semantics: if ema <= 0 => OFF; if 0 < ema <= 1 => fraction of total teacher steps;
  # if ema > 1 => treat as window size (number of steps to average over).
  teacher_ema_cfg = cfg.teacher_ema
  student_ema_cfg = cfg.student_ema
  def _ema_decay(val: float):
    v = float(val) if val is not None else 0.0
    if v <= 0.0:
      return None
    if v > 1.0:
      window = int(round(v))
    else:
      window = int(round(v * num_train_steps))
    window = max(1, window)
    # Approximate windowed average: decay ~ 1 - 1/window
    return 1.0 - 1.0 / window

  teacher_ema_decay = _ema_decay(teacher_ema_cfg)
  student_ema_decay = _ema_decay(student_ema_cfg)
  teacher_ema_model = None
  student_ema_model = None

  student_cfg = None
  student_mup_scales = None
  if train_student_base:
    student_cfg = copy.deepcopy(cfg.model)
    if cfg.model.P_student is not None:
      if cfg.model.P_student >= cfg.model.P:
        print(f"Skipping run with P_student={cfg.model.P_student} >= P={cfg.model.P}")
        exit()
      student_cfg.P = cfg.model.P_student
      student_cfg.D = round(((student_cfg.P * 1e6 / student_cfg.N / 12) ** 0.5) / 64) * 64
      if student_cfg.D < 64:
        raise ValueError('Student model D must be at least 64')
    student_model = model_lib.create_sharded_model(student_cfg, mesh, cfg.seed)
    student_mup_scales = _compute_mup_scales(student_cfg, student_model, cfg.opt)
  # Create EMA model containers after models are constructed
  if teacher_ema_decay is not None:
    teacher_ema_model = copy.deepcopy(model)
  if train_student_base and student_ema_decay is not None:
    student_ema_model = copy.deepcopy(student_model)
  # No LR/beta adjustments from distillation controls; use configured values
  teacher_lr = cfg.opt.lr
  teacher_b2 = cfg.opt.b2

  def _scale_by_runtime_factor():
    def init_fn(_):
      return ()
    def update_fn(updates, state, params=None, *, lr_scale=1.0, **_):
      del params
      return jax.tree.map(lambda u: u * lr_scale, updates), state
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

  def _maybe_support_extra_args(transform, enable: bool):
    if not enable:
      return transform
    return optax.with_extra_args_support(transform)

  def build_tx(base_lr, base_b2, mup_scales, use_schedule: bool = True, external_lr: bool = False):
    needs_support = external_lr
    scale_by_opt = optax.scale_by_adam(b1=cfg.opt.b1, b2=base_b2, eps=1e-20)
    scale_by_opt = _maybe_support_extra_args(scale_by_opt, needs_support)
    mup_transform = optax.GradientTransformation(
        lambda _: None,
        lambda u, s, _: (jax.tree.map(lambda x, sc: x * sc, u, mup_scales), s),
    )
    mup_transform = _maybe_support_extra_args(mup_transform, needs_support)
    transforms = [scale_by_opt, mup_transform]
    weight_decay = float(getattr(cfg.opt, 'wd', 0.0) or 0.0)
    if weight_decay != 0.0:
      transforms.append(_maybe_support_extra_args(optax.add_decoupled_weight_decay(weight_decay), needs_support))
    if use_schedule:
      transforms.append(_maybe_support_extra_args(optax.scale_by_schedule(schedule_fn), needs_support))
    if external_lr:
      transforms.append(_scale_by_runtime_factor())
    transforms.append(_maybe_support_extra_args(optax.scale(-1.0 * base_lr), needs_support))
    return optax.chain(*transforms)

  teacher_tx = build_tx(teacher_lr, teacher_b2, teacher_mup_scales, use_schedule=True)
  teacher_optimizer = nnx.ModelAndOptimizer(model, teacher_tx, wrt=nnx.Param)
  if train_student_base:
    student_tx = build_tx(cfg.opt.lr, cfg.opt.b2, student_mup_scales, use_schedule=False, external_lr=True)
    student_optimizer = nnx.ModelAndOptimizer(student_model, student_tx, wrt=nnx.Param)

  # Helper to update EMA state in place
  @nnx.jit
  def _ema_update(ema_state, src_state, decay):
    return jax.tree.map(lambda e, s: decay * e + (1.0 - decay) * s, ema_state, src_state)

  def _update_ema_model(ema_model, src_model, decay):
    if ema_model is None or decay is None:
      return
    ema_state = nnx.state(ema_model)
    src_state = nnx.state(src_model)
    new_ema = _ema_update(ema_state, src_state, decay)
    nnx.update(ema_model, new_ema)

  # start wandb
  if cfg.wandb_project is not None and os.environ["COMET_ML_API"] is not None:
    config = utils.flatten_dict(cfg)
    config['num_params'] = num_params
    config = {k.split('/')[-1]: v for k, v in config.items() if isinstance(k, str)}
    
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API"],
        project_name=cfg.wandb_project,
        workspace="epi-audio"
    )

    # When launched from a sweep, each run is given an explicit name so it can
    # be identified in the Comet UI; the sweep name is attached as a tag so all
    # runs in the same sweep can be grouped/compared together.
    run_name = OmegaConf.select(cfg, "run_name", default=None)
    if run_name:
        experiment.set_name(run_name)
    sweep_name = OmegaConf.select(cfg, "sweep_name", default=None)
    if sweep_name:
        experiment.add_tag(sweep_name)
    if OmegaConf.select(cfg, "tag", default=None):
        experiment.add_tag(cfg.tag)

    experiment.log_parameters(config)

  # training loop
  # pending_train_metrics = None
  pending_eval_metrics = None
  tau = 0
  pbar = tqdm(total=num_train_steps)
  
  eval_steps = set(
      [int(i) for i in jnp.linspace(warmup_steps, num_train_steps-1, cfg.num_evals)] +
      [int(i) for i in jnp.geomspace(1, num_train_steps, 50)]
  )
  downstream_eval_steps = set([int(i) for i in jnp.linspace(warmup_steps, num_train_steps-1, cfg.num_evals)][::max(1, cfg.num_evals//cfg.num_evals_downstream)][1:])


  train_loss_sum = 0
  elapsed = 0
  KX = 0
  # Running prequential epiplexity estimate K(M); updated at every eval step and
  # returned at the end so callers (e.g. a sweep) can rank runs by epiplexity.
  epiplexity_estimate = None
  prev_features = None
  pending_feature_metrics = None
  # Track token counts separately when gating is enabled
  teacher_tokens_seen = 0
  student_tokens_seen = 0
  last_distill_kl = 0
  # Requential-coding epiplexity estimate K(M)_req. Only defined when a student
  # is distilled (it accumulates teacher->student KL); stays None for
  # teacher-only runs, where the prequential K(M) is the epiplexity estimate.
  current_km_req = None
  if train_student_base:
    distill_rng = jax.random.PRNGKey(cfg.seed + 1)
    cumulative_distill_kl = 0.0
    current_km_req = 0.0
    if cfg.B_student is not None:
      student_batch_size = cfg.B_student
    else:
      student_batch_size = cfg.B
    if student_batch_size % accumulation_steps != 0:
      raise ValueError(f'cfg.B_student ({student_batch_size}) must be divisible by cfg.A ({accumulation_steps})')
    student_microbatch_size = student_batch_size // accumulation_steps
    distill_tokens_per_step = student_batch_size * cfg.model.L
    @nnx.jit
    def generate_teacher_samples(model_for_gen, rng_for_gen):
      return model_lib.generate(
          model_for_gen,
          student_batch_size,
          cfg.model.L,
          rng_for_gen,
          bos_token=0,
          temperature=1.0,
          data_sharding=data_sharding,
          use_kv_cache=True,
      )
  with mesh:
    step = 0
    student_step = 0
    teacher_step = 0
    train_teacher = train_teacher_base
    train_student = train_student_base
    if not (train_teacher_base or train_student_base):
      raise ValueError('At least one of train_teacher or train_student must be True')
    while teacher_step < num_train_steps:
      # training iteration (may include student-only updates)
      schedule_scale = schedule_fn(teacher_step)
      lr_t = teacher_lr * schedule_scale
      train_metrics = {}
      teacher_train_loss = None
      # Determine training mode
      train_teacher = train_teacher_base
      train_student = train_student_base
      if enforce_max_kl and (last_distill_kl > max_kl):
        # Pause teacher updates when the KL exceeds the configured cap.
        train_teacher = False
      step += 1
      if not train_teacher_base:
        # if teacher is frozen, this sets teacher_step == student_step == step
        # otherwise we would block evals
        teacher_step += 1
        pbar.update(1)
      elif train_teacher:
        accum_grads = None
        accum_loss = 0.0
        for accum_idx in range(accumulation_steps):
          micro_idx = teacher_step * accumulation_steps + accum_idx
          batch_teacher = jax.device_put(get_batch_train_teacher(micro_idx), data_sharding)
          micro_loss, micro_grads = compute_loss_and_grads(teacher_optimizer.model, batch_teacher)
          accum_loss += micro_loss
          if accum_grads is None:
            accum_grads = micro_grads
          else:
            accum_grads = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, micro_grads)
        mean_loss = accum_loss / accumulation_steps
        mean_grads = jax.tree_util.tree_map(lambda g: g / accumulation_steps, accum_grads)
        apply_grads(teacher_optimizer, mean_grads)
        teacher_train_loss = mean_loss
        train_metrics = {'train_loss': mean_loss}
        teacher_tokens_seen += train_tokens_per_step
        teacher_step += 1
        pbar.update(1)
        tau += lr_t
        # EMA update for teacher
        _update_ema_model(teacher_ema_model, teacher_optimizer.model, teacher_ema_decay)      # Logging and compute accounting use teacher tokens
      compute_spent = 6 * teacher_tokens_seen * num_params
      train_metrics |= {'tokens': teacher_tokens_seen, 'compute': compute_spent, 'tau': tau, 'lr': lr_t}
      if teacher_train_loss is not None:
        KX += teacher_train_loss * train_tokens_per_step / 1e6 / jnp.log(2)
        train_loss_sum += teacher_train_loss
        elapsed += 1
      step_log: dict[str, Any] = {
          'step': step,
          'teacher_step': teacher_step,
      }
      if train_student:
        # Choose teacher params for generation (EMA if enabled)
        # Use the raw teacher weights for distillation/gating. EMA can lag heavily (e.g., large windows),
        # which would make the student chase a stale teacher and inflate the measured KL.
        teacher_for_gen = teacher_ema_model if teacher_ema_decay is not None else teacher_optimizer.model
        synthetic_batch, teacher_logits, distill_rng = generate_teacher_samples(
            teacher_for_gen,
            distill_rng,
        )
        synthetic_mask = jnp.ones_like(synthetic_batch, dtype=jnp.bool_)
        # Distill with gradient accumulation over the student batch
        accum_grads_student = None
        accum_ce = 0.0
        accum_kl = 0.0
        for accum_idx in range(accumulation_steps):
          start = accum_idx * student_microbatch_size
          end = start + student_microbatch_size
          micro_batch = (
              synthetic_batch[start:end],
              synthetic_mask[start:end],
          )
          micro_teacher_logits = teacher_logits[start:end]
          ce_loss, kl_loss, micro_grads = distill_loss_and_grads(
              student_optimizer.model,
              micro_batch,
              micro_teacher_logits,
          )
          accum_ce += ce_loss
          accum_kl += kl_loss
          if accum_grads_student is None:
            accum_grads_student = micro_grads
          else:
            accum_grads_student = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads_student, micro_grads)
        mean_ce = accum_ce / accumulation_steps
        mean_kl = accum_kl / accumulation_steps
        mean_student_grads = jax.tree_util.tree_map(lambda g: g / accumulation_steps, accum_grads_student)
        apply_student_grads(student_optimizer, mean_student_grads, lr_scale=schedule_scale)
        distill_metrics = {'distill_ce': mean_ce, 'distill_kl': mean_kl}
        student_step += 1
        train_metrics |= distill_metrics
        distill_kl_value = float(distill_metrics['distill_kl'])
        last_distill_kl = distill_kl_value
        cumulative_distill_kl += distill_kl_value
        student_tokens_seen += distill_tokens_per_step
        current_km_req = cumulative_distill_kl * distill_tokens_per_step / 1e6 / jnp.log(2)
        train_metrics['K(M)_req'] = jnp.asarray(current_km_req, dtype=jnp.float32)
        step_log.update({
            'student_step': student_step,
            'distill_ce': float(distill_metrics['distill_ce']),
            'distill_kl': distill_kl_value,
            'K(M)_req': current_km_req,
        })
        # EMA update for student
        _update_ema_model(student_ema_model, student_optimizer.model, student_ema_decay)
      # Log separate token streams
      step_log['teacher_tokens'] = int(teacher_tokens_seen)
      if train_student_base:
        step_log['student_tokens'] = int(student_tokens_seen)
      if train_teacher_base and train_student_base:
        tokens = int(teacher_tokens_seen)
      else:
        tokens = max(teacher_tokens_seen, student_tokens_seen)
      step_log['tokens'] = tokens
      if cfg.wandb_project is not None:
        experiment.log_metrics(step_log, step=student_step,)
        # TODO add more paramters for experiment
        # wandb.log(step_log)
      
      # async logging
      if pending_eval_metrics is not None:
        if cfg.wandb_project is not None: 
            experiment.log_metrics(pending_eval_metrics, step=student_step,) 
            #wandb.log(pending_eval_metrics)
        pending_eval_metrics = None

      # eval step at linearly spaced intervals (in teacher-step space)
      if (teacher_step in eval_steps) or (teacher_step == num_train_steps):
        # Use EMA models for evaluation when enabled
        eval_teacher_model = teacher_ema_model if teacher_ema_decay is not None else teacher_optimizer.model
        pending_eval_metrics = eval_step(eval_teacher_model, ds_test)
        features = pending_eval_metrics.pop('features')
        # Log both raw and EMA teacher eval losses when EMA is enabled
        if teacher_ema_decay is not None:
          # Current pending_eval_metrics['eval_loss'] is EMA teacher's loss
          ema_teacher_loss = pending_eval_metrics['eval_loss']
          raw_teacher_metrics = eval_step(teacher_optimizer.model, ds_test)
          raw_teacher_metrics.pop('features')
          pending_eval_metrics['ema_teacher_eval_loss'] = ema_teacher_loss
          pending_eval_metrics['teacher_eval_loss'] = raw_teacher_metrics['eval_loss']
        else:
          # No EMA: treat current eval_loss as teacher_eval_loss
          pending_eval_metrics['teacher_eval_loss'] = pending_eval_metrics['eval_loss']
        if prev_features is not None:
          pending_feature_metrics = eval_features(features, prev_features)
        else:
          pending_feature_metrics = {'h': jnp.sqrt(jnp.mean(features**2))}
        pending_eval_metrics |= {
          'tokens': tokens, 'compute': compute_spent, 'tau': tau, 'lr': lr_t, 
          'student_step': student_step, 'teacher_step': teacher_step, 'step': step,
          'student_tokens': student_tokens_seen, 'teacher_tokens': teacher_tokens_seen,
        }
        pending_eval_metrics |= pending_feature_metrics
        prev_features = features
        if elapsed > 0:
          L = train_loss_sum / elapsed
          train_loss_sum = 0
          elapsed = 0
          pending_eval_metrics["train_loss"] = L
          pending_eval_metrics["K(X|M)"] = L * (teacher_tokens_seen / 1e6) / jnp.log(2)
          pending_eval_metrics["K(M)"] = KX - pending_eval_metrics["K(X|M)"]
          epiplexity_estimate = float(pending_eval_metrics["K(M)"])
        else:
          train_loss_sum = 0
          elapsed = 0
        # pending_eval_metrics["K(X)"] = jnp.trapezoid(jnp.array(Ls), jnp.array(Ts)) / jnp.log(2) # Gbits
        pending_eval_metrics["K(X)"] = KX
        # Log the requential-coding estimate alongside the prequential K(M)
        # whenever a student is being distilled (None for teacher-only runs).
        if current_km_req is not None:
          pending_eval_metrics["K(M)_req"] = jnp.asarray(current_km_req, dtype=jnp.float32)
        if train_student_base:
          if student_ema_decay is not None:
            # Log both raw and EMA student eval losses
            student_eval_metrics_raw = eval_step(student_optimizer.model, ds_test)
            student_eval_metrics_raw.pop('features')
            pending_eval_metrics['student_eval_loss'] = student_eval_metrics_raw['eval_loss']
            student_eval_metrics_ema = eval_step(student_ema_model, ds_test)
            student_eval_metrics_ema.pop('features')
            pending_eval_metrics['ema_student_eval_loss'] = student_eval_metrics_ema['eval_loss']
          else:
            student_eval_metrics = eval_step(student_optimizer.model, ds_test)
            student_eval_metrics.pop('features')
            pending_eval_metrics['student_eval_loss'] = student_eval_metrics['eval_loss']
          # Expose current gating mode and KL status for debugging
          pending_eval_metrics['train_mode'] = (
              'both' if (train_teacher and train_student)
              else 'teacher_only' if train_teacher else 'student_only'
          )
          if last_distill_kl is not None:
            pending_eval_metrics['kl_current'] = last_distill_kl
            pending_eval_metrics['max_kl'] = float(max_kl)
            pending_eval_metrics['max_kl_enforced'] = bool(enforce_max_kl)
        if cfg.downstream_ds_path is not None and teacher_step in downstream_eval_steps:
          downstream_metrics = {}
          eval_teacher_for_downstream = teacher_ema_model if teacher_ema_decay is not None else teacher_optimizer.model
          downstream_metrics |= downstream_eval_step(eval_teacher_for_downstream, ds_test_downstream)
          downstream_metrics |= downstream_ft_eval(
              mesh,
              eval_teacher_for_downstream,
              ds_train_downstream,
              ds_test_downstream,
              data_sharding,
              teacher_mup_scales,
              cfg
          )
          if train_student_base:
            eval_student_for_downstream = student_ema_model if student_ema_decay is not None else student_optimizer.model
            student_down_metrics = downstream_eval_step(eval_student_for_downstream, ds_test_downstream)
            downstream_metrics |= {f'student_{k}': v for k, v in student_down_metrics.items()}
            student_ft_metrics = downstream_ft_eval(
                mesh,
                eval_student_for_downstream,
                ds_train_downstream,
                ds_test_downstream,
                data_sharding,
                student_mup_scales,
                cfg
            )
            downstream_metrics |= {f'student_{k}': v for k, v in student_ft_metrics.items()}
          pending_eval_metrics |= downstream_metrics
    if cfg.wandb_project is not None and experiment is not None and pending_eval_metrics is not None:
      experiment.log_metrics(pending_eval_metrics, step=student_step)
      experiment.end()
    #   wandb.log(pending_eval_metrics)
    #   wandb.finish()
  if save_path is not None:
    final_state = jax.tree_util.tree_map(jax.device_get, nnx.state(teacher_optimizer.model))
    is_gcs_target = _is_gcs_path(save_path)
    if is_gcs_target:
      tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(save_path)[-1] or '.pkl', delete=False)
      tmp.close()
      local_target = tmp.name
    else:
      local_target = save_path
      directory = os.path.dirname(local_target)
      if directory:
        os.makedirs(directory, exist_ok=True)
    try:
      with open(local_target, 'wb') as f:
        pickle.dump(final_state, f)
      if is_gcs_target:
        _upload_temp_to_gcs(local_target, save_path)
    finally:
      if is_gcs_target and os.path.exists(local_target):
        os.remove(local_target)
    print(f"Saved final checkpoint to {save_path}")

  # Final prequential epiplexity estimate K(M) for this run (None if no eval ran).
  return epiplexity_estimate