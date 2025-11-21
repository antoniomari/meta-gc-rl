To check models meta-learned from scratch
```^TEST-antmaze-ddpgbc-(F|R)|fomaml_max_grad```

To check comparison of losses on reptile with different merging_eps
```(FT-TEST-antmaze-ddpgbc-(R)-(200)-1-(mc0.2-lr0.0003-ilr3e-05|notarget|mc0.2-m.*))|(FT-TEST-antmaze-ddpgbc-PT)```

Fixed bug in implementation (was not evaluating using correct parameter depending on how `select` was called)
- Updated Reptile to be RX (Reptile fixed)
- Updated Fomaml to be FX (Fomaml fixed)
```^FT-TEST-antmaze-ddpgbc-(R|RX|PT)```


After bugfix of implementation (JAX differentiation wrt parameters), started to run experiments on 1-inner-step meta learning + no-meta learning objective on both models trained from scratch on test-goals and model fine-tuned from a pretrained policy.

The hyperparameter settings used:
- For no-meta learning:
    - `lr = [3e-03, 3e-04, 3e-05]`
- For meta-learning (both `fomaml` and `reptile`):
    - inner-steps and num-meta-batches fixed to 1
    - for `reptile`, merging eps=1
    - `(inner_lr, lr) = [(3e-04, 3e-04), (3e-05, 3e-04), (3e-05, 3e-05)]`




Big hyperparam sweep.
We want to assess the performance of our algorithms on TEST-Goals settings first and then on ALL-Goals
