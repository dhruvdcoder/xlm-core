import torch
from torch.utils.data import Dataset


class LengthSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Dataset,
        shuffle: bool = True,
        max_length_diff: int = 5,
    ):
        # Sort data by length
        self.lengths = [self.get_length(ex) for ex in data_source]
        self.sorted_indices = sorted(
            range(len(data_source)),
            key=lambda i: self.lengths[i],
            reverse=True,
        )
        self.sorted_lengths = [self.lengths[i] for i in self.sorted_indices]
        self.max_length_diff = max_length_diff
        self.shuffle = shuffle
        if self.shuffle:
            assert (
                max_length_diff >= 1
            ), "max_length_diff should be at least 1 if shuffle is True"
        self.boundaries = self._get_boundaries()

    @staticmethod
    def get_length(ex):
        return len(ex["input_ids"])  # Adjust this based on your data structure

    def _get_boundaries(self):

        num_points = len(self.sorted_indices)
        start = 0
        boundaries = []
        while start < num_points:
            # Initialize the end of the current group
            end = start

            # Expand the group as long as the length difference is within the allowed range
            while (
                end < num_points - 1  # Ensure we don't go out of bounds
                and (
                    self.sorted_lengths[end + 1] - self.sorted_lengths[start]
                    <= self.max_length_diff
                )
            ):
                end += 1

            # Add the end index of this group to the boundaries
            boundaries.append(end)

            # Start the next group from the next index
            start = end + 1

        # Add the final boundary (total number of points)
        boundaries.append(num_points)
        return boundaries

    def __iter__(self):
        sorted_indices = torch.tensor(self.sorted_indices)
        if self.shuffle:
            generator = torch.Generator().manual_seed(
                torch.empty((), dtype=torch.int64).random_().item()
            )
            for i in range(len(self.boundaries) - 1):
                start = self.boundaries[i]
                end = self.boundaries[i + 1]
                shuffled = (
                    start + torch.randperm(end - start, generator=generator)
                ).tolist()
                sorted_indices[start:end] = sorted_indices[shuffled]
        return iter(sorted_indices.tolist())

    def __len__(self):
        return len(self.sorted_indices)
