Our code will generate `results.pth` file inside the `training.save_dir` for local evaluation.

To locally evaluate your results on Waymo validation set, please run the `waymo_eval.py` script as follow:

```bash
python waymo_eval.py --root-path ${SAVE_DIR}
```
