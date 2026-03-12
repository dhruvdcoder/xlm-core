# Insertion Language Model

## Training

To train the model on OpenWebText-1024 split, you can use the following command:
```bash
xlm job_type=train job_name=owt_ilm experiment=owt_ilm \
per_device_batch_size=32 \
trainer_strategy=ddp \
trainer.devices=8 \
trainer.num_nodes=1 \
++trainer.precision=bf16-mixed \
compile=true
```

## Download a checkpoint



## Cite
If you use this model in your research, please cite the original paper along with xLM.

```
@misc{patel2025insertionlanguagemodelssequence,
      title={Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions}, 
      author={Dhruvesh Patel and Aishwarya Sahoo and Avinash Amballa and Tahira Naseem and Tim G. J. Rudner and Andrew McCallum},
      year={2025},
      eprint={2505.05755},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.05755}, 
}
```

