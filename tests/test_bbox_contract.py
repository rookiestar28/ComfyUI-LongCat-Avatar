import unittest

from LongCat_Video.bbox_contract import parse_person_boxes


class BBoxContractTests(unittest.TestCase):
    def test_blank_boxes_are_optional(self):
        self.assertIsNone(parse_person_boxes(""))
        self.assertIsNone(parse_person_boxes(None))

    def test_requires_two_person_boxes(self):
        with self.assertRaisesRegex(ValueError, "requires at least person1 and person2"):
            parse_person_boxes("[1, 2, 3, 4]")

    def test_parses_required_person_boxes(self):
        self.assertEqual(
            parse_person_boxes("[1, 2, 3, 4], [5, 6, 7, 8]"),
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )

    def test_parses_flat_others_boxes(self):
        self.assertEqual(
            parse_person_boxes("[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]"),
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        )

    def test_parses_nested_others_boxes(self):
        self.assertEqual(
            parse_person_boxes("[1, 2, 3, 4], [5, 6, 7, 8], [[9, 10, 11, 12], [13, 14, 15, 16]]"),
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        )

    def test_appends_extra_other_arguments(self):
        self.assertEqual(
            parse_person_boxes("[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]"),
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        )

    def test_rejects_non_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            parse_person_boxes("[1, 2, 3, 'x'], [5, 6, 7, 8]")

    def test_rejects_bad_other_length(self):
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            parse_person_boxes("[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]")

    def test_rejects_malformed_text(self):
        with self.assertRaisesRegex(ValueError, "p_box must be formatted"):
            parse_person_boxes("[1, 2, 3, 4], nope")


if __name__ == "__main__":
    unittest.main()
