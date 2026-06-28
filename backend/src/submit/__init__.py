"""Kaggle NeuroGolf submission automation package."""

from submit.auth import AuthError, ensure_credentials
from submit.history import record
from submit.kaggle_api import (
    COMPETITION,
    KaggleCLIError,
    SubmissionNameError,
    confirm_submission,
    list_submissions,
    poll,
)
from submit.kaggle_api import submit as kaggle_submit
from submit.packager import (
    SUBMISSION_NAME,
    PackagingError,
    build_submission_zip,
    collect_onnx_files,
)
from submit.validator import TaskValidation, ValidationError, validate_onnx_files

__all__ = [
    "COMPETITION",
    "SUBMISSION_NAME",
    "AuthError",
    "KaggleCLIError",
    "PackagingError",
    "SubmissionNameError",
    "TaskValidation",
    "ValidationError",
    "build_submission_zip",
    "collect_onnx_files",
    "confirm_submission",
    "ensure_credentials",
    "kaggle_submit",
    "list_submissions",
    "poll",
    "record",
    "validate_onnx_files",
]
