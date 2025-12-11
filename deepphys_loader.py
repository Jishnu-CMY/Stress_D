# deepphys_loader.py
import torch
import os

def load_deepphys(weights_path, device=None):
    """
    Loads DeepPhys model class from the file where it's defined (user already has DeepPhys code).
    Assumes DeepPhys class is in the PYTHON path (e.g., copy the class into deepphys_arch.py
    or paste it into this file above load_deepphys).
    Returns: model (eval, on device), device
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # If DeepPhys class is defined in this project scope, import it.
    # Try common places
    try:
        # try local import (if you saved the class as deepphys_arch.py)
        from deepphys_arch import DeepPhys  # <-- ensure this file exists or move the class here
    except Exception:
        # Maybe the class is pasted above in this file; try to import from current scope
        try:
            from __main__ import DeepPhys  # unlikely in typical runs
        except Exception as e:
            raise RuntimeError("Could not import DeepPhys class. Place the class definition in deepphys_arch.py "
                               "or ensure DeepPhys is importable. Original error: " + str(e))

    # instantiate model with default image size 36 (match checkpoint training)
    model = DeepPhys(in_channels=3, img_size=72)

    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    # load checkpoint and handle "module." prefixes and common wrapper keys
    ckpt = torch.load(weights_path, map_location="cpu")
    # ckpt might be a plain state_dict, or a dict with key 'state_dict' or 'model'
    if isinstance(ckpt, dict):
        # try common locations
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    # fix keys that start with "module."
    try:
        from collections import OrderedDict
        new_state = OrderedDict()
        # if state appears to be nested mapping of params
        if isinstance(state, dict):
            for k, v in state.items():
                name = k
                if name.startswith("module."):
                    name = name[len("module."):]
                new_state[name] = v
            model.load_state_dict(new_state, strict=False)
        else:
            # state is weird; try direct load
            model.load_state_dict(state)
    except Exception as e:
        # raise informative error that prints top-level keys to debug
        raise RuntimeError(f"Failed to load checkpoint into DeepPhys model: {e}. "
                           f"Inspect checkpoint keys with: python -c \"import torch; ck=torch.load('{weights_path}', map_location='cpu'); print(type(ck)); print(list(ck.keys())[:50])\"")

    model.to(device)
    model.eval()
    return model, device
