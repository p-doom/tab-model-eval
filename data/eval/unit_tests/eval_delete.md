# Assistant <NO_EVAL>
```bash
cat -n jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
     1	from dataclasses import dataclass
     2	import time
     3	import os
     4	import optax
     5	
     6	import dm_pix as pix
     7	import einops
     8	import jax
     9	import jax.numpy as jnp
    10	import numpy as np
    11	import orbax.checkpoint as ocp
    12	from PIL import Image, ImageDraw
    13	import tyro
    14	from flax import nnx
    15	
    16	from jasmine.models.genie import GenieDiffusion
    17	from jasmine.utils.dataloader import get_dataloader
    18	
    19	
    20	@dataclass
    21	class Args:
    22	    # Experiment
    23	    seed: int = 0
    24	    seq_len: int = 16
    25	    image_channels: int = 3
    26	    image_height: int = 90
    27	    image_width: int = 160
    28	    data_dir: str = "data/coinrun_episodes"
    29	    checkpoint: str = ""
    30	    print_action_indices: bool = True
    31	    output_dir: str = "gifs/"
    32	    # Sampling
    33	    batch_size: int = 1
    34	    start_frame: int = 1
    35	    diffusion_denoise_steps: int = 64
    36	    diffusion_corrupt_context_factor: float = 0.1
    37	    # Tokenizer checkpoint
    38	    tokenizer_dim: int = 512
    39	    tokenizer_ffn_dim: int = 2048
    40	    latent_patch_dim: int = 32
    41	    num_patch_latents: int = 1024
    42	    patch_size: int = 16
    43	    tokenizer_num_blocks: int = 4
    44	    tokenizer_num_heads: int = 8
    45	    # LAM checkpoint
    46	    lam_dim: int = 512
    47	    lam_ffn_dim: int = 2048
    48	    latent_action_dim: int = 32
    49	    num_actions: int = 6
    50	    lam_patch_size: int = 16
    51	    lam_num_blocks: int = 4
    52	    lam_num_heads: int = 8
    53	    use_gt_actions: bool = False
    54	    # Dynamics checkpoint
    55	    dyna_dim: int = 512
    56	    dyna_ffn_dim: int = 2048
    57	    dyna_num_blocks: int = 6
    58	    dyna_num_heads: int = 8
    59	    param_dtype = jnp.float32
    60	    dtype = jnp.bfloat16
    61	    use_flash_attention: bool = True
    62	
    63	
    64	args = tyro.cli(Args)
    65	
    66	if __name__ == "__main__":
    67	    """
    68	    Dimension keys:
    69	        B: batch size
    70	        T: number of input (conditioning) frames
    71	        N: number of patches per frame
    72	        S: sequence length
    73	        H: height
    74	        W: width
    75	        E: B * (S - 1)
    76	    """
    77	    jax.distributed.initialize()
    78	
    79	    rng = jax.random.key(args.seed)
    80	
    81	    # --- Load Genie checkpoint ---
    82	    rngs = nnx.Rngs(rng)
    83	    genie = GenieDiffusion(
    84	        # Tokenizer
    85	        in_dim=args.image_channels,
    86	        tokenizer_dim=args.tokenizer_dim,
    87	        tokenizer_ffn_dim=args.tokenizer_ffn_dim,
    88	        latent_patch_dim=args.latent_patch_dim,
    89	        num_patch_latents=args.num_patch_latents,
    90	        patch_size=args.patch_size,
    91	        tokenizer_num_blocks=args.tokenizer_num_blocks,
    92	        tokenizer_num_heads=args.tokenizer_num_heads,
    93	        # LAM
    94	        lam_dim=args.lam_dim,
    95	        lam_ffn_dim=args.lam_ffn_dim,
    96	        latent_action_dim=args.latent_action_dim,
    97	        num_actions=args.num_actions,
    98	        lam_patch_size=args.lam_patch_size,
    99	        lam_num_blocks=args.lam_num_blocks,
   100	        lam_num_heads=args.lam_num_heads,
   101	        lam_co_train=False,
   102	        use_gt_actions=args.use_gt_actions,
   103	        # Dynamics
   104	        dyna_dim=args.dyna_dim,
   105	        dyna_ffn_dim=args.dyna_ffn_dim,
   106	        dyna_num_blocks=args.dyna_num_blocks,
   107	        dyna_num_heads=args.dyna_num_heads,
   108	        param_dtype=args.param_dtype,
   109	        dtype=args.dtype,
   110	        use_flash_attention=args.use_flash_attention,
   111	        diffusion_denoise_steps=args.diffusion_denoise_steps,
   112	        # FIXME (f.srambical): implement spatiotemporal KV caching and set decode=True
   113	        decode=False,
   114	        rngs=rngs,
   115	    )
   116	
   117	    # Need to delete lam decoder for checkpoint loading
   118	    if not args.use_gt_actions:
   119	        assert genie.lam is not None
   120	        del genie.lam.decoder
   121	
   122	    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
   123	    handler_registry.add(
   124	        "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
   125	    )
   126	    handler_registry.add(
   127	        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
   128	    )
   129	    checkpoint_options = ocp.CheckpointManagerOptions(
   130	        step_format_fixed_length=6,
   131	    )
   132	    checkpoint_manager = ocp.CheckpointManager(
   133	        args.checkpoint,
   134	        options=checkpoint_options,
   135	        handler_registry=handler_registry,
   136	    )
   137	
   138	    dummy_tx = optax.adamw(
   139	        learning_rate=optax.linear_schedule(0.0001, 0.0001, 10000),
   140	        b1=0.9,
   141	        b2=0.9,
   142	        weight_decay=1e-4,
   143	        mu_dtype=args.dtype,
   144	    )
   145	    dummy_optimizer = nnx.ModelAndOptimizer(genie, dummy_tx)
   146	
   147	    abstract_optimizer = nnx.eval_shape(lambda: dummy_optimizer)
   148	    abstract_optimizer_state = nnx.state(abstract_optimizer)
   149	    restored = checkpoint_manager.restore(
   150	        checkpoint_manager.latest_step(),
   151	        args=ocp.args.Composite(
   152	            model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
   153	        ),
   154	    )
   155	    restored_optimizer_state = restored["model_state"]
   156	    nnx.update(dummy_optimizer, restored_optimizer_state)
   157	
   158	    # --- Define sampling function ---
   159	    def _sampling_fn(model: GenieDiffusion, batch: dict) -> jax.Array:
   160	        """Runs Genie.sample with pre-defined generation hyper-parameters."""
   161	        frames = model.sample(
   162	            batch,
   163	            args.seq_len,
   164	            args.diffusion_denoise_steps,
   165	            args.diffusion_corrupt_context_factor,
   166	        )
   167	        return frames
   168	
   169	    # --- Define autoregressive sampling loop ---
   170	    def _autoreg_sample(
   171	        genie: GenieDiffusion, rng: jax.Array, batch: dict
   172	    ) -> jax.Array:
   173	        batch["videos"] = batch["videos"][:, : args.start_frame]
   174	        batch["rng"] = rng
   175	        generated_vid_BSHWC = _sampling_fn(genie, batch)
   176	        return generated_vid_BSHWC
   177	
   178	    # --- Get video + latent actions ---
   179	    array_record_files = [
   180	        os.path.join(args.data_dir, x)
   181	        for x in os.listdir(args.data_dir)
   182	        if x.endswith(".array_record")
   183	    ]
   184	    dataloader = get_dataloader(
   185	        array_record_files,
   186	        args.seq_len,
   187	        args.batch_size,
   188	        args.image_height,
   189	        args.image_width,
   190	        args.image_channels,
   191	        # We don't use workers in order to avoid grain shutdown issues (https://github.com/google/grain/issues/398)
   192	        num_workers=0,
   193	        prefetch_buffer_size=1,
   194	        seed=args.seed,
   195	    )
   196	    dataloader = iter(dataloader)
   197	    batch = next(dataloader)
   198	    gt_video = jnp.asarray(batch["videos"], dtype=jnp.float32) / 255.0
   199	    print(f"DEBUG: gt_video shape: {gt_video.shape}")
   200	    batch["videos"] = gt_video.astype(args.dtype)
   201	    # Get latent actions for all videos in the batch
   202	    action_batch_E = None
   203	    if not args.use_gt_actions:
   204	        action_batch_E = genie.vq_encode(batch, training=False)
   205	        batch["latent_actions"] = action_batch_E
   206	        print(f"DEBUG: action_batch_E shape: {action_batch_E.shape}")
   207	
   208	    # --- Sample + evaluate video ---
   209	    recon_video_BSHWC = _autoreg_sample(genie, rng, batch)
   210	    print(f"DEBUG: recon_video_BSHWC shape: {recon_video_BSHWC.shape}")
   211	    recon_video_BSHWC = recon_video_BSHWC.astype(jnp.float32)
   212	    print(f"DEBUG: recon_video_BSHWC shape: {recon_video_BSHWC.shape}")
   213	
   214	    gt = gt_video.clip(0, 1)[:, args.start_frame :]
   215	    print(f"DEBUG: gt shape: {gt.shape}")
   216	    recon = recon_video_BSHWC.clip(0, 1)[:, args.start_frame :]
   217	    print(f"DEBUG: recon shape: {recon.shape}")
   218	
   219	    ssim_vmap = jax.vmap(pix.ssim, in_axes=(0, 0))
   220	    psnr_vmap = jax.vmap(pix.psnr, in_axes=(0, 0))
   221	    ssim = jnp.asarray(ssim_vmap(gt, recon))
   222	    psnr = jnp.asarray(psnr_vmap(gt, recon))
   223	    per_frame_ssim = ssim.mean(0)
   224	    per_frame_psnr = psnr.mean(0)
   225	    avg_ssim = ssim.mean()
   226	    avg_psnr = psnr.mean()
   227	
   228	    print("Per-frame SSIM:\n", per_frame_ssim)
   229	    print("Per-frame PSNR:\n", per_frame_psnr)
   230	
   231	    print(f"SSIM: {avg_ssim}")
   232	    print(f"PSNR: {avg_psnr}")
   233	
   234	    # --- Construct video ---
   235	    true_videos = (gt_video * 255).astype(np.uint8)
   236	    pred_videos = (recon_video_BSHWC * 255).astype(np.uint8)
   237	    video_comparison = np.zeros((2, *recon_video_BSHWC.shape), dtype=np.uint8)
   238	    video_comparison[0] = true_videos[:, : args.seq_len]
   239	    video_comparison[1] = pred_videos
   240	    frames = einops.rearrange(video_comparison, "n b t h w c -> t (b h) (n w) c")
   241	
   242	    # --- Save video ---
   243	    imgs = [Image.fromarray(img) for img in frames]
   244	    # Write actions on each frame, on each row (i.e., for each video in the batch, on the GT row)
   245	    B = batch["videos"].shape[0]
   246	    if action_batch_E is not None:
   247	        action_batch_BSm11 = jnp.reshape(action_batch_E, (B, args.seq_len - 1, 1))
   248	    else:
   249	        action_batch_BSm11 = jnp.reshape(
   250	            batch["actions"][:, :-1], (B, args.seq_len - 1, 1)
   251	        )
   252	    for t, img in enumerate(imgs[1:]):
   253	        d = ImageDraw.Draw(img)
   254	        for row in range(B):
   255	            if args.print_action_indices:
   256	                action = action_batch_BSm11[row, t, 0]
   257	                y_offset = row * batch["videos"].shape[2] + 2
   258	                d.text((2, y_offset), f"{action}", fill=255)
   259	
   260	    os.makedirs(args.output_dir, exist_ok=True)
   261	    imgs[0].save(
   262	        os.path.join(args.output_dir, f"generation_{time.time()}.gif"),
   263	        save_all=True,
   264	        append_images=imgs[1:],
   265	        duration=250,
   266	        loop=0,
   267	    )
   268	    # Save predicted videos as PNG image with all frames (skipping first 4) for each item in batch
   269	    skip = 4
   270	    # pred_videos: (B, T, H, W, C)
   271	    B, T, H, W, C = pred_videos.shape
   272	
   273	    for i in range(B):
   274	        # Predicted
   275	        pred_strip = np.concatenate(
   276	            [pred_videos[i, t] for t in range(skip, args.seq_len)], axis=1
   277	        )  # resulting shape: (H, (T-skip)*W, C)
   278	        pred_img = Image.fromarray(pred_strip)
   279	
   280	        # Write actions on each frame for the predicted strip
   281	        if args.print_action_indices:
   282	            d = ImageDraw.Draw(pred_img)
   283	            for t in range(skip, args.seq_len):
   284	                if action_batch_E is not None:
   285	                    action = int(action_batch_BSm11[i, t - 1, 0])
   286	                else:
   287	                    action = int(batch["actions"][i, t - 1])
   288	                x_offset = (t - skip) * pred_videos.shape[3] + 2
   289	                y_offset = 2
   290	                d.text((x_offset, y_offset), f"{action}", fill=255)
   291	
   292	        pred_img.save(
   293	            os.path.join(
   294	                args.output_dir, f"generation-{args.seed}-sample{i}-diffusion.png"
   295	            )
   296	        )
   297	        # GT
   298	        gt_strip = np.concatenate(
   299	            [true_videos[i, t] for t in range(skip, args.seq_len)], axis=1
   300	        )  # resulting shape: (H, (T-skip)*W, C)
   301	        gt_img = Image.fromarray(gt_strip)
   302	        gt_img.save(
   303	            os.path.join(args.output_dir, f"generation-{args.seed}-sample{i}-gt.png")
   304	        )
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/baselines/diffusion/sample_diffusion.py | sed -n '195,206p'
```

# User
<stdout>
   195      )
   196      dataloader = iter(dataloader)
   197      batch = next(dataloader)
   198      gt_video = jnp.asarray(batch["videos"], dtype=jnp.float32) / 255.0
   199      print(f"DEBUG: gt_video shape: {gt_video.shape}")
   200      batch["videos"] = gt_video.astype(args.dtype)
   201      # Get latent actions for all videos in the batch
   202      action_batch_E = None
   203      if not args.use_gt_actions:
   204          action_batch_E = genie.vq_encode(batch, training=False)
   205          batch["latent_actions"] = action_batch_E
   206          print(f"DEBUG: action_batch_E shape: {action_batch_E.shape}")
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '199,199d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '205,205d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
cat -n jasmine/baselines/diffusion/sample_diffusion.py | sed -n '205,216p'
```

# User
<stdout>
   205
   206      # --- Sample + evaluate video ---
   207      recon_video_BSHWC = _autoreg_sample(genie, rng, batch)
   208      print(f"DEBUG: recon_video_BSHWC shape: {recon_video_BSHWC.shape}")
   209      recon_video_BSHWC = recon_video_BSHWC.astype(jnp.float32)
   210      print(f"DEBUG: recon_video_BSHWC shape: {recon_video_BSHWC.shape}")
   211
   212      gt = gt_video.clip(0, 1)[:, args.start_frame :]
   213      print(f"DEBUG: gt shape: {gt.shape}")
   214      recon = recon_video_BSHWC.clip(0, 1)[:, args.start_frame :]
   215      print(f"DEBUG: recon shape: {recon.shape}")
   216
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '208,208d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '209,209d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>


# Assistant <NO_EVAL>
```bash
sed -i '211,211d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>


# Assistant <EVAL>
```bash
sed -i '212,212d' jasmine/baselines/diffusion/sample_diffusion.py
```

# User
<stdout>
</stdout>
