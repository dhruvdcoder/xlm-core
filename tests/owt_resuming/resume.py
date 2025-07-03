# !/usr/bin/env python3
# torch==2.5.1
# datasets==3.3.2
# torchdata>=0.9.0
import datasets
import pprint
from torchdata.stateful_dataloader import StatefulDataLoader

import os

PERSISTENT_WORKERS = (
    os.environ.get("PERSISTENT_WORKERS", "True").lower() == "true"
)
print(f"PERSISTENT_WORKERS: {PERSISTENT_WORKERS}")

# PERSISTENT_WORKERS = True  # Incorrect resume


# ds = datasets.load_from_disk("dataset").to_iterable_dataset(num_shards=4)
def generator():
    for i in range(128):
        yield {"x": i}


ds = datasets.Dataset.from_generator(
    generator, features=datasets.Features({"x": datasets.Value("int32")})
).to_iterable_dataset(num_shards=4)

dl = StatefulDataLoader(
    ds, batch_size=2, num_workers=2, persistent_workers=PERSISTENT_WORKERS
)
global_step = 0
epoch = 0
ds_state_dict = None
state_dict = None
resumed = False
while True:
    if epoch >= 3:
        break
    if state_dict is not None:
        dl = StatefulDataLoader(
            ds,
            batch_size=2,
            num_workers=2,
            persistent_workers=PERSISTENT_WORKERS,
        )
        dl.load_state_dict(state_dict)
        state_dict = None
        ds_state_dict = None
        resumed = True
        print("resumed")
    for i, batch in enumerate(dl):
        print(f"epoch: {epoch}, global_step: {global_step}, batch: {batch}")
        global_step += 1  # consume datapoint
        # simulate error
        if global_step == 124 and not resumed:
            ds_state_dict = ds.state_dict()
            state_dict = dl.state_dict()
            print("checkpoint")
            print("ds_state_dict")
            pprint.pprint(ds_state_dict)
            print("dl_state_dict")
            pprint.pprint(state_dict)
            break

    if state_dict is None:
        ds.set_epoch(epoch)
        epoch += 1
