import unittest

from LongCat_Video.scheduler_contract import (
    OFFICIAL_AVATAR_V15_SCHEDULER,
    OFFICIAL_AVATAR_V15_STEPS,
    scheduler_audit_summary,
    validate_avatar_scheduler_contract,
)


class SchedulerContractTests(unittest.TestCase):
    def test_official_scheduler_contract_accepts_bounded_avatar_v15_path(self):
        contract = validate_avatar_scheduler_contract(
            scheduler_name=OFFICIAL_AVATAR_V15_SCHEDULER,
            steps=12,
        )

        self.assertEqual(contract.scheduler_name, OFFICIAL_AVATAR_V15_SCHEDULER)
        self.assertEqual(contract.steps, 12)
        self.assertEqual(contract.status, "official_bounded")

    def test_scheduler_contract_defaults_to_official_fixed_path(self):
        contract = validate_avatar_scheduler_contract()

        self.assertEqual(contract.scheduler_name, OFFICIAL_AVATAR_V15_SCHEDULER)
        self.assertEqual(contract.steps, OFFICIAL_AVATAR_V15_STEPS)

    def test_scheduler_contract_rejects_unverified_reference_scheduler(self):
        with self.assertRaisesRegex(ValueError, "not adopted"):
            validate_avatar_scheduler_contract(scheduler_name="longcat_distill_euler")

    def test_scheduler_contract_rejects_out_of_range_step_count(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            validate_avatar_scheduler_contract(steps=51)

    def test_scheduler_audit_summary_documents_reference_scheduler_boundary(self):
        summary = scheduler_audit_summary()

        self.assertIn("default 8 steps", summary)
        self.assertIn("1-50", summary)
        self.assertIn("longcat_distill_euler", summary)
        self.assertIn("not adopted", summary)


if __name__ == "__main__":
    unittest.main()
