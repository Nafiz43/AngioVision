The core problem: Training a large vision-language model runs out of GPU memory (OOM = Out of Memory). These tweaks attack that problem from several different angles.

1. Mixed Precision Training (AMP)
Normally, numbers in a neural network are stored as 32-bit floats. AMP lets the model use 16-bit floats for most operations instead. This roughly halves the memory used to store activations during training, with almost no loss in accuracy. A "GradScaler" is added alongside it to prevent the smaller number format from causing numerical instability.
2. Gradient Accumulation
Instead of feeding, say, 16 images to the GPU at once (a batch of 16), you feed 2 images at a time, 8 times in a row, and only update the model weights after all 8 steps. The model "sees" the same amount of data, but the GPU only has to hold 2 images worth of memory at any moment.
3. Gradient Checkpointing for the Vision Encoder (ViT)
During training, the model normally saves all intermediate calculations so it can use them during the backward pass. Checkpointing says: "don't save those — just recompute them when needed." It's a deliberate trade of extra computation time for significantly less memory.
4. Streaming Image Loading
Previously, all video frames were loaded into memory at once as a big list of images. Now, only a small chunk of frames is loaded at a time, processed, and then immediately discarded. Think of it like reading a book a page at a time instead of photocopying the whole thing first.
5. Chunked CLIP Loss
The contrastive loss used in CLIP-style training requires comparing every image in a batch against every other image, which creates a matrix of size Batch × Batch. For a large batch, this matrix gets huge. The chunked approach computes the loss in smaller pieces so that full matrix never has to exist in memory all at once.
6. Skipping Gradient Saving for Frozen Layers
If a part of the model (e.g., the vision tower) is frozen and won't be updated, there's no reason to save the gradients for it. Wrapping those layers in no_grad() tells PyTorch not to bother, saving memory proportional to how many layers are frozen.
7. Fixed Input Image Size
If video frames happen to be high resolution, they can silently cause a memory spike. Setting a fixed input size (e.g., 224×224) ensures no frame ever accidentally comes in oversized and blows the memory budget.
8. Emptying the Cache Each Step
PyTorch holds onto freed GPU memory in a cache to reuse it later. Occasionally, this cached-but-fragmented memory can prevent new allocations from fitting, even when technically enough memory is free. Clearing the cache each step is a "last resort" fix for those edge cases — it adds a tiny overhead but can prevent crashes when you're right at the memory limit.