# Deep Learning from Scratch

This is an agglomeration of various "from scratch" implementations, with some minor modifications as necessary.

Goals:
1. Maintain compatibility with original implementations
2. Add additional comments following YouTube videos from original authors
3. Make minor API modifications to mirror Huggingface where possible (e.g. training loops)
4. Connect the various components together, where possible

This is currently a work in progress.

## Setup

- Install conda / pip environment
  - Same process if updating conda packages (see below for just pip package updates)
  ```bash
  cd env
  conda env update --file conda.yaml --prune
  conda activate dl_scratch

  # NOTE: To see progress for pip install, comment out `-pip:` section in conda.yaml and use:
  # pip install -r requirements.txt
  ```

- Update pip packages only
  ```bash
  cd env
  conda activate dl_scratch
  pip install -r requirements.txt
  ```

- Set PYTHONPATH
  - Add to ~/.zshrc or ~/.bashrc
  ```bash
  # Add location of top-level github dir, e.g. `deep-learning-from-scratch`
  # We use this for absolute imports like `from dl_scratch.micrograd.engine import ...`
  export PYTHONPATH=~/deep-learning-from-scratch
  ```

## Micrograd

- From [GitHub: karpathy/micrograd](https://github.com/karpathy/micrograd)
  - Minor updates: add typing, add comments, more tests
  - Minor refactoring:
    - Move `trace_graph.ipynb` -> `micrograd/utils.py`
    - Move `demo.ipynb` -> `./notebooks/micrograd_svm.ipynb` (with minor changes)
    - Add `micrograd_simple.ipynb` following notebooks from YouTube walkthrough below
- Walkthrough: [YouTube (Andrej Karpathy): The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=VyGDLSMWqAeSy8jg)
  - Notebooks: [GitHub (karpathy/nn-zero-to-hero/lectures/micrograd)](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)

### Micrograd: Demo

- See [./notebooks/micrograd_simple.ipynb](./notebooks/micrograd_simple.ipynb)
- See [./notebooks/micrograd_svm.ipynb](./notebooks/micrograd_svm.ipynb)

### Micrograd: Tests

```console
❯ pytest dl_scratch/test/test_micrograd.py
======================================= test session starts =======================================
platform linux -- Python 3.10.15, pytest-8.3.4, pluggy-1.5.0
rootdir: ~/dl_from_scratch
configfile: pyproject.toml
collected 4 items

dl_scratch/test/test_micrograd.py ....  [100%]
======================================== 4 passed in 2.36s ========================================
```