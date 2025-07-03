1. Install (if you don't already have the project's environment)
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.3.1 torchdata>=0.9.0 datasets==3.3.2
```

2. Run the script to see the total number of steps in an epoch (should be 64)
```
python script.py
```

3. Run with interrupt at step 124 (close to the end of epoch 1)
```
python script.py --resume --interrupt-step=124
```
After this you will see a `checkpoints/on_exception.ckpt` file.

4. Run with resume and you will see that only two steps are run for all following epochs.
```
python script.py --resume
```


Even simpler repro is:

```
PERSISTENT_WORKERS=True python resume.py
```

```
PERSISTENT_WORKERS=False python resume.py
```


