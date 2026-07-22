from __future__ import annotations

import pytest

from verification.remote_nccl_alltoallv_worker import (
    expected_receive_values,
    split_count,
    split_plan,
)


@pytest.mark.parametrize(
    ("rank", "sparse", "expected_send", "expected_recv"),
    [
        (0, False, [1, 2], [1, 3]),
        (1, False, [3, 1], [2, 1]),
        (0, True, [2, 0], [2, 1]),
        (1, True, [1, 2], [0, 2]),
    ],
)
def test_split_plan_is_transpose_consistent(
    rank: int,
    sparse: bool,
    expected_send: list[int],
    expected_recv: list[int],
) -> None:
    assert split_plan(rank, 2, sparse=sparse) == (
        expected_send,
        expected_recv,
    )
    for peer in range(2):
        assert expected_send[peer] == split_count(rank, peer, sparse=sparse)
        assert expected_recv[peer] == split_count(peer, rank, sparse=sparse)


def test_expected_receive_values_follow_sender_order() -> None:
    assert expected_receive_values(0, 2, sparse=False) == [0.0, 1000.0, 1001.0, 1002.0]
    assert expected_receive_values(1, 2, sparse=False) == [100.0, 101.0, 1100.0]
    assert expected_receive_values(0, 2, sparse=True) == [0.0, 1.0, 1000.0]
    assert expected_receive_values(1, 2, sparse=True) == [1100.0, 1101.0]
