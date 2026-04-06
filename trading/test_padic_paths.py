import unittest

from futures.padic_paths import (
    action_to_digit,
    agreement_depth,
    decode_actions,
    digit_to_action,
    pow3_upto,
    prefix_bucket,
    push_digit,
)


def _encode_actions(actions: list[int]) -> int:
    pow3 = pow3_upto(len(actions))
    path_id = 0
    for depth, action in enumerate(actions):
        path_id = push_digit(path_id, depth, action_to_digit(action), pow3=pow3)
    return path_id


def _lcp_depth(a: list[int], b: list[int], max_depth: int) -> int:
    out = 0
    for i in range(min(len(a), len(b), max_depth)):
        if a[i] != b[i]:
            break
        out += 1
    return out


class TestPadicPaths(unittest.TestCase):
    def test_action_digit_roundtrip(self):
        for action in (-1, 0, 1):
            self.assertEqual(digit_to_action(action_to_digit(action)), action)

    def test_agreement_depth_matches_list_lcp(self):
        a = [-1, 0, 1, 1, 0]
        b = [-1, 0, 1, -1, 0]
        id_a = _encode_actions(a)
        id_b = _encode_actions(b)
        for max_depth in range(0, 8):
            self.assertEqual(agreement_depth(id_a, id_b, max_depth=max_depth), _lcp_depth(a, b, max_depth))

    def test_agreement_depth_identical_is_max_depth(self):
        a = [1, 0, -1, 0]
        id_a = _encode_actions(a)
        self.assertEqual(agreement_depth(id_a, id_a, max_depth=0), 0)
        self.assertEqual(agreement_depth(id_a, id_a, max_depth=4), 4)
        self.assertEqual(agreement_depth(id_a, id_a, max_depth=10), 10)

    def test_decode_actions_recovers_prefix(self):
        actions = [-1, 0, 1, 0, 1]
        path_id = _encode_actions(actions)
        self.assertEqual(decode_actions(path_id, depth=len(actions)), actions)

    def test_prefix_bucket_groups_by_common_prefix(self):
        # common prefix depth 3
        a = [-1, 0, 1, 1]
        b = [-1, 0, 1, -1]
        id_a = _encode_actions(a)
        id_b = _encode_actions(b)
        pow3 = pow3_upto(8)
        self.assertEqual(prefix_bucket(id_a, depth=3, pow3=pow3), prefix_bucket(id_b, depth=3, pow3=pow3))
        self.assertNotEqual(prefix_bucket(id_a, depth=4, pow3=pow3), prefix_bucket(id_b, depth=4, pow3=pow3))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            push_digit(0, -1, 0)
        with self.assertRaises(ValueError):
            push_digit(0, 0, 3)
        with self.assertRaises(ValueError):
            action_to_digit(2)
        with self.assertRaises(ValueError):
            digit_to_action(9)


if __name__ == "__main__":
    unittest.main()

