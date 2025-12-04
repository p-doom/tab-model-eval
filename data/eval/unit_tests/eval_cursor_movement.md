# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py
```

# User
<stdout>
     1  from typing import Dict
     2
     3  import jax
     4  import jax.numpy as jnp
     5  import flax.nnx as nnx
     6
     7  from jasmine.utils.nn import AxialTransformer
     8
     9
    10  class DynamicsMaskGIT(nnx.Module):
    11      """
    12      MaskGIT dynamics model
    13
    14      Dimension keys:
    15          B: batch size
    16          T: sequence length
    17          N: number of patches per frame
    18          L: latent dimension
    19          V: vocabulary size (number of latents)
    20      """
    21
    22      def __init__(
    23          self,
    24          model_dimension: int,
    25          ffn_dimension: int,
    26          num_latents: int,
    27          latent_action_dim: int,
    28          num_blocks: int,
    29          num_heads: int,
    30          dropout: float,
    31          mask_limit: float,
    32          param_dtype: jnp.dtype,
    33          dtype: jnp.dtype,
    34          use_flash_attention: bool,
    35          rngs: nnx.Rngs,
    36          decode: bool,
    37      ):
    38          self.model_dim = model_dimension
    39          self.ffn_dim = ffn_dimension
    40          self.num_latents = num_latents
    41          self.latent_action_dim = latent_action_dim
    42          self.num_blocks = num_blocks
    43          self.num_heads = num_heads
    44          self.dropout = dropout
    45          self.mask_limit = mask_limit
    46          self.param_dtype = param_dtype
    47          self.dtype = dtype
    48          self.use_flash_attention = use_flash_attention
    49          self.decode = decode
    50          self.transformer = AxialTransformer(
    51              self.model_dim,
    52              self.model_dim,
    53              self.ffn_dim,
    54              self.num_latents,
    55              self.num_blocks,
    56              self.num_heads,
    57              self.dropout,
    58              self.param_dtype,
    59              self.dtype,
    60              decode=self.decode,
    61              use_flash_attention=self.use_flash_attention,
    62              spatial_causal=False,
    63              temporal_causal=True,
    64              rngs=rngs,
    65          )
    66          self.patch_embed = nnx.Embed(self.num_latents, self.model_dim, rngs=rngs)
    67          self.mask_token = nnx.Param(
    68              nnx.initializers.lecun_uniform()(rngs.params(), (1, 1, 1, self.model_dim))
    69          )
    70          self.action_up = nnx.Linear(
    71              self.latent_action_dim,
    72              self.model_dim,
    73              param_dtype=self.param_dtype,
    74              dtype=self.dtype,
    75              rngs=rngs,
    76          )
    77
    78      def __call__(
    79          self,
    80          batch: Dict[str, jax.Array],
    81      ) -> tuple[jax.Array, jax.Array]:
    82          # --- Mask videos ---
    83          video_tokens_BTN = batch["video_tokens"]
    84          latent_actions_BTm11L = batch["latent_actions"]
    85          vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
    86
    87          batch_size = vid_embed_BTNM.shape[0]
    88          _rng_prob, *_rngs_mask = jax.random.split(batch["mask_rng"], batch_size + 1)
    89          mask_prob = jax.random.uniform(
    90              _rng_prob, shape=(batch_size,), minval=self.mask_limit
    91          )
    92          per_sample_shape = vid_embed_BTNM.shape[1:-1]
    93          mask = jax.vmap(
    94              lambda rng, prob: jax.random.bernoulli(rng, prob, per_sample_shape),
    95              in_axes=(0, 0),
    96          )(jnp.asarray(_rngs_mask), mask_prob)
    97          mask = mask.at[:, 0].set(False)
    98          vid_embed_BTNM = jnp.where(
    99              jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed_BTNM
   100          )
   101
   102          # --- Predict transition ---
   103          act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
   104          padded_act_embed_BT1M = jnp.pad(
   105              act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
   106          )
   107          vid_embed_BTNp1M = jnp.concatenate(
   108              [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
   109          )
   110          logits_BTNp1V = self.transformer(vid_embed_BTNp1M)
   111          logits_BTNV = logits_BTNp1V[:, :, 1:]
   112          return logits_BTNV, mask
   113
   114
   115  class DynamicsCausal(nnx.Module):
   116      """Causal dynamics model"""
   117
   118      def __init__(
   119          self,
   120          model_dimension: int,
   121          ffn_dimension: int,
   122          num_latents: int,
   123          latent_action_dim: int,
   124          num_blocks: int,
   125          num_heads: int,
   126          dropout: float,
   127          param_dtype: jnp.dtype,
   128          dtype: jnp.dtype,
   129          use_flash_attention: bool,
   130          rngs: nnx.Rngs,
   131          decode: bool,
   132      ):
   133          self.model_dim = model_dim
   134          self.ffn_dim = ffn_dim
   135          self.num_latents = num_latents
   136          self.latent_action_dim = latent_action_dim
   137          self.num_blocks = num_blocks
   138          self.num_heads = num_heads
   139          self.dropout = dropout
   140          self.param_dtype = param_dtype
   141          self.dtype = dtype
   142          self.use_flash_attention = use_flash_attention
   143          self.decode = decode
   144          self.transformer = AxialTransformer(
   145              self.model_dim,
   146              self.model_dim,
   147              self.ffn_dim,
   148              self.num_latents,
   149              self.num_blocks,
   150              self.num_heads,
   151              self.dropout,
   152              self.param_dtype,
   153              self.dtype,
   154              use_flash_attention=self.use_flash_attention,
   155              spatial_causal=True,
   156              temporal_causal=True,
   157              rngs=rngs,
   158              decode=self.decode,
   159          )
   160          self.patch_embed = nnx.Embed(self.num_latents, self.model_dim, rngs=rngs)
   161          self.action_up = nnx.Linear(
   162              self.latent_action_dim,
   163              self.model_dim,
   164              param_dtype=self.param_dtype,
   165              dtype=self.dtype,
   166              rngs=rngs,
   167          )
   168
   169      def __call__(
   170          self,
   171          batch: Dict[str, jax.Array],
   172      ) -> tuple[jax.Array, jax.Array]:
   173          video_tokens_BTN = batch["video_tokens"]
   174          latent_actions_BTm11L = batch["latent_actions"]
   175          vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
   176          act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
   177          padded_act_embed_BT1M = jnp.pad(
   178              act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
   179          )
   180          vid_embed_BTNp1M = jnp.concatenate(
   181              [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
   182          )
   183          logits_BTNp1V = self.transformer(vid_embed_BTNp1M)
   184          logits_BTNV = logits_BTNp1V[:, :, :-1]
   185          return logits_BTNV, jnp.ones_like(video_tokens_BTN)
   186
   187
   188  class DynamicsDiffusion(nnx.Module):
   189      """Diffusion transformer dynamics model"""
   190
   191      def __init__(
   192          self,
   193          model_dimension: int,
   194          ffn_dimension: int,
   195          latent_patch_dim: int,
   196          latent_action_dim: int,
   197          num_blocks: int,
   198          num_heads: int,
   199          denoise_steps: int,
   200          dropout: float,
   201          param_dtype: jnp.dtype,
   202          dtype: jnp.dtype,
   203          use_flash_attention: bool,
   204          rngs: nnx.Rngs,
   205          decode: bool,
   206      ):
   207          self.model_dim = model_dimension
   208          self.ffn_dim = ffn_dimension
   209          self.latent_patch_dim = latent_patch_dim
   210          self.latent_action_dim = latent_action_dim
   211          self.num_blocks = num_blocks
   212          self.num_heads = num_heads
   213          self.dropout = dropout
   214          self.param_dtype = param_dtype
   215          self.dtype = dtype
   216          self.use_flash_attention = use_flash_attention
   217          self.denoise_steps = denoise_steps
   218          self.decode = decode
   219          self.diffusion_transformer = AxialTransformer(
   220              self.latent_patch_dim,
   221              self.model_dim,
   222              self.ffn_dim,
   223              self.latent_patch_dim,
   224              self.num_blocks,
   225              self.num_heads,
   226              self.dropout,
   227              self.param_dtype,
   228              self.dtype,
   229              use_flash_attention=self.use_flash_attention,
   230              spatial_causal=False,
   231              temporal_causal=True,
   232              rngs=rngs,
   233              decode=self.decode,
   234          )
   235          self.action_up = nnx.Linear(
   236              self.latent_action_dim,
   237              self.latent_patch_dim,
   238              param_dtype=self.param_dtype,
   239              dtype=self.dtype,
   240              rngs=rngs,
   241          )
   242          self.timestep_embed = nnx.Embed(
   243              num_embeddings=self.denoise_steps,
   244              features=self.latent_patch_dim,
   245              param_dtype=self.param_dtype,
   246              dtype=self.dtype,
   247              rngs=rngs,
   248          )
   249
   250      def __call__(
   251          self,
   252          batch: Dict[str, jax.Array],
   253      ) -> tuple[jax.Array, jax.Array]:
   254          # Code adapted from https://github.com/kvfrans/shortcut-models/blob/main/baselines/targets_naive.py
   255          _rng_time, _rng_noise = jax.random.split(batch["rng"])
   256          latents_BTNL = batch["token_latents"]
   257          latent_actions_BTm11L = batch["latent_actions"]
   258          B, T, N, L = latents_BTNL.shape
   259
   260          # --- Add noise to latents ---
   261          denoise_step_BT = jax.random.randint(
   262              _rng_time, (B, T), minval=0, maxval=self.denoise_steps
   263          )
   264          denoise_step_embed_BT1L = self.timestep_embed(denoise_step_BT).reshape(
   265              B, T, 1, self.latent_patch_dim
   266          )
   267          denoise_t_BT = denoise_step_BT / self.denoise_steps
   268          denoise_t_BT11 = denoise_t_BT[:, :, jnp.newaxis, jnp.newaxis]
   269          noise_BTNL = jax.random.normal(_rng_noise, (B, T, N, L))
   270          noised_latents_BTNL = (
   271              1
   272              - (1 - 1e-5) * denoise_t_BT11  # we adopt 1e-5 from kvfrans/shortcut-models
   273          ) * noise_BTNL + denoise_t_BT11 * latents_BTNL
   274
   275          # --- Process actions ---
   276          act_embed_BTm11L = self.action_up(latent_actions_BTm11L)
   277          padded_act_embed_BT1L = jnp.pad(
   278              act_embed_BTm11L, ((0, 0), (1, 0), (0, 0), (0, 0))
   279          )
   280
   281          # --- Call the diffusion transformer ---
   282          inputs_BTNp2L = jnp.concatenate(
   283              [padded_act_embed_BT1L, denoise_step_embed_BT1L, noised_latents_BTNL],
   284              axis=2,
   285          )
   286          outputs_BTNp2L = self.diffusion_transformer(
   287              inputs_BTNp2L,
   288          )
   289          pred_latents_BTNL = outputs_BTNp2L[:, :, 2:]
   290          return pred_latents_BTNL, denoise_t_BT
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '19,30p'
```

# User
<stdout>
    19          V: vocabulary size (number of latents)
    20      """
    21
    22      def __init__(
    23          self,
    24          model_dimension: int,
    25          ffn_dimension: int,
    26          num_latents: int,
    27          latent_action_dim: int,
    28          num_blocks: int,
    29          num_heads: int,
    30          dropout: float,
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '24,24c\\t\tmodel_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '25,25c\\t\tffn_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '34,45p'
```

# User
<stdout>
    34          use_flash_attention: bool,
    35          rngs: nnx.Rngs,
    36          decode: bool,
    37      ):
    38          self.model_dim = model_dimension
    39          self.ffn_dim = ffn_dimension
    40          self.num_latents = num_latents
    41          self.latent_action_dim = latent_action_dim
    42          self.num_blocks = num_blocks
    43          self.num_heads = num_heads
    44          self.dropout = dropout
    45          self.mask_limit = mask_limit
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '38,38c\\t\tself.model_dim = model_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '39,39c\\t\tself.ffn_dim = ffn_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '115,126p'
```

# User
<stdout>
   115  class DynamicsCausal(nnx.Module):
   116      """Causal dynamics model"""
   117
   118      def __init__(
   119          self,
   120          model_dimension: int,
   121          ffn_dimension: int,
   122          num_latents: int,
   123          latent_action_dim: int,
   124          num_blocks: int,
   125          num_heads: int,
   126          dropout: float,
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '120,120c\\t\tmodel_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '121,121c\\t\tffn_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '128,139p'
```

# User
<stdout>
   128          dtype: jnp.dtype,
   129          use_flash_attention: bool,
   130          rngs: nnx.Rngs,
   131          decode: bool,
   132      ):
   133                  self.model_dim = model_dim
   134                  self.ffn_dim = ffn_dim
   135          self.num_latents = num_latents
   136          self.latent_action_dim = latent_action_dim
   137          self.num_blocks = num_blocks
   138          self.num_heads = num_heads
   139          self.dropout = dropout
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '133,133c\\t\tself.model_dim = model_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '134,134c\\t\tself.ffn_dim = ffn_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '188,199p'
```


# User
<stdout>
   188  class DynamicsDiffusion(nnx.Module):
   189      """Diffusion transformer dynamics model"""
   190
   191      def __init__(
   192          self,
   193          model_dimension: int,
   194          ffn_dimension: int,
   195          latent_patch_dim: int,
   196          latent_action_dim: int,
   197          num_blocks: int,
   198          num_heads: int,
   199          denoise_steps: int,
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '193,193c\\t\tmodel_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '194,194c\\t\tffn_dim: int,' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <EVAL>
```bash
cat -n jasmine/models/dynamics.py | sed -n '202,213p'
```

# User
<stdout>
   202          dtype: jnp.dtype,
   203          use_flash_attention: bool,
   204          rngs: nnx.Rngs,
   205          decode: bool,
   206      ):
   207          self.model_dim = model_dimension
   208          self.ffn_dim = ffn_dimension
   209          self.latent_patch_dim = latent_patch_dim
   210          self.latent_action_dim = latent_action_dim
   211          self.num_blocks = num_blocks
   212          self.num_heads = num_heads
   213          self.dropout = dropout
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '133,133c\\t\tself.model_dim = model_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '134,134c\\t\tself.ffn_dim = ffn_dim' jasmine/models/dynamics.py
```

# User
<stdout>
</stdout>




