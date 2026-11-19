"""Use the historical inference coordinator with the zero-shot source freeze."""
import shared_pipeline
from shared_pipeline import validate_stage, verify_freeze


if __name__ == "__main__":
    shared_pipeline.CORE_FILES.extend(["shared_pipeline.py", "historical_prepare.py", "verify_resume.py",
                                      "requirements.txt", "requirements-data.txt", "requirements-analysis.txt"])
    shared_pipeline.main()
