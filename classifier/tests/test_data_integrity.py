import unittest

from steering_classificator.train_pytorch import LABELS, metrics, validate_datasets
from steering_classificator.preprocessing import preprocess_equation


def datasets():
    train = [{"equation": f"y'=x^{i+2}", "label": label, "split": "train", "source_file": "class_train.xlsx"}
             for i, label in enumerate(LABELS)]
    valid = [{"equation": f"y'=x^{i+5}", "label": label, "split": "valid", "source_file": "class_train.xlsx"}
             for i, label in enumerate(LABELS)]
    test = [{"equation": f"y'=x^{i+8}", "label": label} for i, label in enumerate(LABELS)]
    return train + valid, test


class DataIntegrityTests(unittest.TestCase):
    def test_nested_boxed_input_matches_plain_routing_input(self):
        self.assertEqual(preprocess_equation(r"\boxed{y^{\prime}=x^{2}}"),
                         preprocess_equation(r"y^{\prime}=x^{2}"))
    def test_clean_separate_data_passes(self):
        data, test = datasets()
        validate_datasets(data, test, [])

    def test_test_leakage_with_spacing_change_fails(self):
        data, test = datasets()
        test[0]["equation"] = " y' = x^2 "
        with self.assertRaisesRegex(ValueError, "test equation"):
            validate_datasets(data, test, [])

    def test_duplicates_across_train_and_validation_fail(self):
        data, test = datasets()
        data[3]["equation"] = data[0]["equation"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_datasets(data, test, [])

    def test_original_test_equation_removed_from_current_test_still_fails(self):
        data, test = datasets()
        with self.assertRaisesRegex(ValueError, "test equation"):
            validate_datasets(data, test, [{"equation": data[0]["equation"]}])

    def test_test_source_filename_fails_even_for_new_equation(self):
        data, test = datasets()
        data[0]["source_file"] = "test_polinom.xlsx"
        with self.assertRaisesRegex(ValueError, "test source"):
            validate_datasets(data, test, [])

    def test_evaluation_confusion_matrix_orientation(self):
        report = metrics([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["per_class"]["polynomial"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
