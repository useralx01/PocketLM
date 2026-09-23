# Public source privacy manifest

This repository is a clean source snapshot. It intentionally excludes:

- model weights and tensor artifacts (`models/`, GGUF, safetensors, PyTorch and NumPy tensors)
- compiled native binaries and downloaded runtimes
- runtime state, model registries/inventory, caches, generated proof files, logs, PIDs and screenshots
- prompts, chats, sessions, user profiles, credentials, API tokens and `.env` files
- local absolute paths, machine/user identifiers, private engineering journals and work documents

The included native C++ sources can be built locally with `tools/build_native.py`.
Fixtures that require tensor files are not included in this public snapshot. Ignore
rules are a convenience, not a security boundary; review the Git index before pushing.
