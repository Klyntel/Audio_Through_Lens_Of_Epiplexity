""" Updated model.py for picodo/model.py

Due to jax update, need to type self.blocks as nnx.list
This file only changes that line to work with latest jax vrs. 
"""

# pyright: reportMissingImports=false

import jax
from flax import nnx
from jax.sharding import Mesh
from omegaconf.dictconfig import DictConfig

from epiplexity.picodo.model import TransformerDecoder, TransformerBlock, fsdp_init, generate as generate


class ModernTransformerDecoder(TransformerDecoder):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    # added nnx.list from train.py
    self.blocks = nnx.List([TransformerBlock(cfg, rngs) for _ in range(cfg.N)])
    self.out_ln = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.readout = nnx.Linear(in_features=cfg.D, out_features=cfg.V, use_bias=False, kernel_init=fsdp_init('readout', cfg), dtype=cfg.dtype, rngs=rngs)


def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
  """Updated created_sharded_model so it replicates behavior in train.py

  But with our ModernTransformerDecoder
  """
  @nnx.jit
  def initialize_sharded_model():
    # below is only different line!
    model = ModernTransformerDecoder(c, rngs=nnx.Rngs(seed))
    # above is only diffrent line!
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

  with mesh:
    model = initialize_sharded_model()

  return model