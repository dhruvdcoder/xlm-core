A lightning checkpoint contains the model weights, optimizer state, and other training artifacts. You can extract the model weights from a checkpoint using the `xlm job_type=extract_checkpoint` script.

!!! example 
```bash
xlm "job_type=extract_checkpoint" \
"job_name=owt_flexmdm" \
"experiment=owt_flexmdm" \
+post_training=default \
+post_training.checkpoint_path=location/of/checkpoints/34-400000.ckpt \
+post_training.model_state_dict_path=output/folder/model_state_dict.pth
```