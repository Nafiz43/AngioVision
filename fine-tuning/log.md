[INFO] device = cuda
[INFO] pooling: default=mean, frame=mean, sequence=mean
[INFO] Run dir: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32
[INFO] Loss CSV: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/21_4_16_32_loss.csv
[INFO] Rolling checkpoint path: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/last.pt
[INFO] Per-epoch checkpoints will be stored as: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/epoch_<N>.pt
[INFO] Report variant summary by accession:
       Accessions with >=1 report : 1899
       Min reports per accession  : 5
       Max reports per accession  : 5
       Mean reports per accession : 5.00
[INFO] Total params:      366,597,633
[INFO] Trainable params:  177,418,497
[INFO] Frozen params:     189,179,136
[INFO] ViT trainable encoder blocks: [9, 10, 11] / total=12
[INFO] ViT embeddings trainable: False
[INFO] ViT final layernorm trainable: True
[INFO] Text encoder trainable: False
[INFO] vision_proj trainable: True
[INFO] text_proj trainable: True
[INFO] temporal_mode: sinusoidal
[INFO] temporal_on_frames: True (scale=0.75)
[INFO] temporal_on_sequences: True (scale=0.5)
[INFO] enable_generation: True
[INFO] decoder model: gpt2
[INFO] decoder trainable: True
[INFO] generation_visual_proj trainable: True
[INFO] ViT gradient checkpointing enabled.
[INFO] Optimizer parameter groups:
       vision backbone: 21,855,744 params, lr=1e-05
       projection/head: 1,574,913 params, lr=0.0001
       gen vis head:    1,181,184 params, lr=0.0001
       decoder:         152,806,656 params, lr=5e-05
[INFO] AMP enabled.
[INFO] Loss CSV initialized at: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/21_4_16_32_loss.csv
[INFO] Gradient accumulation: 4 steps (effective batch ~= batch_size * grad_accum)
[INFO] Saved rolling checkpoint: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/last.pt
[INFO] Saved epoch checkpoint:   /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/epoch_1.pt
[INFO] Saved rolling checkpoint: /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/last.pt
[INFO] Saved epoch checkpoint:   /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/21_4_16_32/epoch_2.pt
