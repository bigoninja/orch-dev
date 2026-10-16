class OrchError(Exception):
    """Base for all orch errors."""


class ConfigError(OrchError): ...


class PlanValidationError(OrchError):
    def __init__(self, issues: list[str]):
        super().__init__("; ".join(issues))
        self.issues = issues


class ModelUnavailableError(OrchError):
    """The pinned model failed (missing/auth/provider error). Never substitute."""


class AdapterTimeout(OrchError): ...


class GitError(OrchError): ...


class LockHeldError(OrchError): ...


class Escalation(OrchError):
    def __init__(self, kind: str, message: str, task_id: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.task_id = task_id
