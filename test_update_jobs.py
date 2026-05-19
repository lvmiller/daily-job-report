import io
import unittest
from contextlib import redirect_stdout

from google.genai import errors as genai_errors

import update_jobs


class QuotaHandlingTests(unittest.TestCase):
    def test_detects_genai_resource_exhausted_error(self):
        error = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
            None,
        )

        self.assertTrue(update_jobs.is_quota_exhausted(error))

    def test_detects_too_many_requests_text(self):
        error = Exception("429 TooManyRequests: retry after the rate limit resets")

        self.assertTrue(update_jobs.is_quota_exhausted(error))

    def test_does_not_treat_authentication_errors_as_quota(self):
        error = genai_errors.ClientError(
            401,
            {
                "error": {
                    "code": 401,
                    "message": "API key not valid.",
                    "status": "UNAUTHENTICATED",
                }
            },
            None,
        )

        self.assertFalse(update_jobs.is_quota_exhausted(error))

    def test_quota_handler_reports_preservation_and_exits_successfully(self):
        error = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
            None,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = update_jobs.handle_generation_exception(
                "search execution",
                error,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Preserving existing jobs.json and jobs_history.json",
            output.getvalue(),
        )

    def test_non_quota_handler_fails(self):
        error = RuntimeError("response was not valid JSON")

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = update_jobs.handle_generation_exception(
                "structured JSON extraction",
                error,
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Exception triggered during structured JSON extraction",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
