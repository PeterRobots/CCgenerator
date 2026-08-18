def unload_model(model, low_resources=False):
    import gc
    import torch
    if low_resources:
        gc.collect(); torch.cuda.empty_cache(); del model
