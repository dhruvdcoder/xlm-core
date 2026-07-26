"""Unit tests for ``xlm.utils.nn`` helpers."""

from xlm.utils.nn import pad_truncate_list


class TestPadTruncateList:
    def test_truncation_returns_zero_num_padded_left(self):
        padded, num_padded = pad_truncate_list(
            [1, 2, 3, 4, 5],
            max_len=3,
            pad_token=0,
            pad_left=True,
            return_num_padded=True,
        )
        assert padded == [3, 4, 5]
        assert num_padded == 0

    def test_truncation_returns_zero_num_padded_right(self):
        padded, num_padded = pad_truncate_list(
            [1, 2, 3, 4, 5],
            max_len=3,
            pad_token=0,
            pad_left=False,
            return_num_padded=True,
        )
        assert padded == [1, 2, 3]
        assert num_padded == 0

    def test_padding_reports_positive_num_padded(self):
        padded, num_padded = pad_truncate_list(
            [1, 2],
            max_len=5,
            pad_token=0,
            pad_left=True,
            return_num_padded=True,
        )
        assert padded == [0, 0, 0, 1, 2]
        assert num_padded == 3
