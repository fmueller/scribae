# Third-Party License Audit

_Generated: 2026-02-21T11:12:12.996841Z (UTC)._

## Scope

- Dependency source: `uv.lock` closure for `scribae` runtime + `translation` extra + `dev` group on current Linux/Python environment.
- License source: installed package metadata (`License-Expression`, `License`, and Trove license classifiers).
- Project license: Apache-2.0 (`pyproject.toml`).

## Headline result

- No strong-copyleft license was detected in resolvable metadata as a mandatory dependency.
- A few packages use MPL-2.0 (file-level copyleft), which is generally compatible with Apache-2.0 when their terms are preserved.
- 25 packages from the selected lock closure were not installed in this environment and require manual verification before release.

## Items needing attention

- MPL-2.0 packages: certifi, pathspec, tqdm.
- Unknown/empty license metadata packages: 1 (see table).
- Missing (not installed) packages: cuda-bindings, cuda-pathfinder, jinja2, markupsafe, mpmath, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-cufile-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-cusparselt-cu12, nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvshmem-cu12, nvidia-nvtx-cu12, setuptools, sympy, torch, triton.

## Compliance checklist for Scribae

- Keep your own `LICENSE` (Apache-2.0) in source distribution and wheel (already configured via `license-files = ["LICENSE"]`).
- If distributing bundled binaries/containers with dependencies included, include a third-party notices file and retain upstream license texts/notices for bundled packages.
- If distributing only source/wheel that declares dependencies (not vendored code), package managers normally deliver dependency license files in each dependency distribution.
- Before release, run this audit in a fully synced environment (including `torch` path) and resolve all `UNKNOWN`/`missing` entries.

## Dependency license table

| Package | Version | Declared license | Apache-2.0 compatibility |
|---|---:|---|---|
| annotated-types | 0.7.0 | License :: OSI Approved :: MIT License | Compatible |
| anyio | 4.12.1 | MIT | Compatible |
| build | 1.4.0 | MIT | Compatible |
| certifi | 2026.1.4 | MPL-2.0 | Compatible (file-level copyleft obligations) |
| cffi | 2.0.0 | MIT | Compatible |
| cfgv | 3.5.0 | MIT | Compatible |
| charset-normalizer | 3.4.4 | MIT | Compatible |
| click | 8.3.1 | BSD-3-Clause | Compatible |
| colorama | 0.4.6 | License :: OSI Approved :: BSD License | Compatible |
| cryptography | 46.0.3 | Apache-2.0 OR BSD-3-Clause | Compatible |
| distlib | 0.4.0 | PSF-2.0 | Compatible |
| docutils | 0.22.4 | License :: Public Domain; License :: OSI Approved :: BSD License; License :: OSI Approved :: GNU General Public License (GPL) | Compatible |
| Faker | 40.1.2 | MIT License | Compatible |
| filelock | 3.20.3 | Unlicense | Compatible |
| fsspec | 2026.1.0 | BSD-3-Clause | Compatible |
| genai-prices | 0.0.51 | MIT | Compatible |
| griffe | 1.15.0 | ISC | Compatible |
| h11 | 0.16.0 | MIT | Compatible |
| hf-xet | 1.2.0 | Apache-2.0 | Compatible |
| httpcore | 1.0.9 | BSD-3-Clause | Compatible |
| httpx | 0.28.1 | BSD-3-Clause | Compatible |
| huggingface-hub | 0.36.0 | Apache | Compatible |
| id | 1.5.0 | License :: OSI Approved :: Apache Software License | Compatible |
| identify | 2.6.16 | MIT | Compatible |
| idna | 3.11 | BSD-3-Clause | Compatible |
| importlib_metadata | 8.7.1 | Apache-2.0 | Compatible |
| iniconfig | 2.3.0 | MIT | Compatible |
| jaraco.classes | 3.4.0 | License :: OSI Approved :: MIT License | Compatible |
| jaraco.context | 6.1.0 | MIT | Compatible |
| jaraco.functools | 4.4.0 | MIT | Compatible |
| jeepney | 0.9.0 | MIT | Compatible |
| joblib | 1.5.3 | BSD-3-Clause | Compatible |
| keyring | 25.7.0 | MIT | Compatible |
| librt | 0.7.8 | MIT | Compatible |
| lingua-language-detector | 2.1.1 | Apache-2.0 | Compatible |
| logfire-api | 4.19.0 | MIT | Compatible |
| markdown-it-py | 4.0.0 | License :: OSI Approved :: MIT License | Compatible |
| mdurl | 0.1.2 | License :: OSI Approved :: MIT License | Compatible |
| more-itertools | 10.8.0 | MIT | Compatible |
| mypy | 1.19.1 | MIT | Compatible |
| mypy_extensions | 1.1.0 | MIT | Compatible |
| nh3 | 0.3.2 | MIT | Compatible |
| nodeenv | 1.10.0 | BSD | Compatible |
| numpy | 2.4.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Compatible |
| opentelemetry-api | 1.39.1 | Apache-2.0 | Compatible |
| packaging | 25.0 | License :: OSI Approved :: Apache Software License; License :: OSI Approved :: BSD License | Compatible |
| pathspec | 1.0.3 | License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0) | Compatible (file-level copyleft obligations) |
| platformdirs | 4.5.1 | MIT | Compatible |
| pluggy | 1.6.0 | MIT | Compatible |
| pre_commit | 4.5.1 | MIT | Compatible |
| pycparser | 3.0 | BSD-3-Clause | Compatible |
| pydantic | 2.12.5 | MIT | Compatible |
| pydantic-ai | 1.44.0 | MIT | Compatible |
| pydantic-ai-slim | 1.44.0 | MIT | Compatible |
| pydantic_core | 2.41.5 | MIT | Compatible |
| pydantic-graph | 1.44.0 | MIT | Compatible |
| Pygments | 2.19.2 | BSD-2-Clause | Compatible |
| pyproject_hooks | 1.2.0 | License :: OSI Approved :: MIT License | Compatible |
| pytest | 9.0.2 | MIT | Compatible |
| python-frontmatter | 1.1.0 | MIT | Compatible |
| PyYAML | 6.0.3 | MIT | Compatible |
| readme_renderer | 44.0 | Apache License, Version 2.0 | Compatible |
| regex | 2026.1.15 | Apache-2.0 AND CNRI-Python | Compatible |
| requests | 2.32.5 | Apache-2.0 | Compatible |
| requests-toolbelt | 1.0.0 | Apache 2.0 | Compatible |
| rfc3986 | 2.0.0 | Apache 2.0 | Compatible |
| rich | 14.2.0 | MIT | Compatible |
| ruff | 0.14.13 | License :: OSI Approved :: MIT License | Compatible |
| sacremoses | 0.1.1 | License :: OSI Approved :: MIT License | Compatible |
| safetensors | 0.7.0 | License :: OSI Approved :: Apache Software License | Compatible |
| SecretStorage | 3.5.0 | BSD-3-Clause | Compatible |
| sentencepiece | 0.2.1 | UNKNOWN | Needs manual review |
| shellingham | 1.5.4 | ISC License | Compatible |
| tokenizers | 0.22.2 | License :: OSI Approved :: Apache Software License | Compatible |
| tqdm | 4.67.1 | MPL-2.0 AND MIT | Compatible (file-level copyleft obligations) |
| transformers | 4.57.6 | Apache 2.0 License | Compatible |
| twine | 6.2.0 | Apache-2.0 | Compatible |
| typer | 0.21.1 | MIT | Compatible |
| typing_extensions | 4.15.0 | PSF-2.0 | Compatible |
| typing-inspection | 0.4.2 | MIT | Compatible |
| urllib3 | 2.6.3 | MIT | Compatible |
| virtualenv | 20.36.1 | MIT | Compatible |
| zipp | 3.23.0 | MIT | Compatible |
