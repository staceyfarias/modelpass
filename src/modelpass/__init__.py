"""modelpass -- pluggable access to the AI subscriptions you already pay for.

modelpass drives the agent runtimes bundled with existing AI subscriptions (Claude
Agent SDK, Codex SDK) under the user's own login, instead of paying metered
model-API rates for personal development work. It never extracts or replays
subscription tokens; it drives the vendor's own runtime, which is the pattern the
vendors document and support (D1).

Quick shape::

    import modelpass

    bridge = modelpass.Bridge()
    for event in bridge.chat(connection="claude-sub", message="hello"):
        if isinstance(event, modelpass.TextDeltaEvent):
            print(event.text, end="")

A fresh install has no AI connectivity at all. An API key sitting in the
environment is not a connection and never becomes one: configuration is the
informed-consent step (D2).

Schema-bound output is one keyword away, using each runtime's own native
mechanism (D13)::

    for event in bridge.chat(
        connection="claude-sub",
        message="...",
        schema=MY_SCHEMA,
    ):
        if isinstance(event, modelpass.StructuredOutputEvent):
            result = event.data

The contract each entry point holds itself to, the capability matrix and the
dated verification record behind it are in ``docs/api-and-runtimes.md``.
"""

from __future__ import annotations

from .adapters.base import Adapter, RunRequest, SessionHandle, SessionRequest
from .bridge import Answer, Bridge, ValidationReport, chat, default_bridge
from .capabilities import (
    DEFAULT_REGISTRY,
    STATIC_TABLE,
    VERIFY_CELLS,
    Capability,
    CapabilityRegistry,
    Support,
    VerifiedCapabilities,
    VerifyReport,
    runtime_auth_modes,
)
from .connections import (
    ALLOWED_ENV_PASSTHROUGH,
    DEFAULT_GROUP,
    Account,
    AccountBinding,
    Connection,
    CredentialKind,
    CredentialRef,
    Group,
    GroupSelection,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from .errors import (
    AdapterFailed,
    AdapterNotImplemented,
    AuthModeMismatch,
    CapabilityNotSupported,
    ConfigError,
    ConnectionDisabled,
    CredentialRefIsSecret,
    DuplicateConnection,
    GroupUnavailable,
    GuardStop,
    InvalidConnection,
    InvalidGuards,
    InvalidSchema,
    InvalidSession,
    InvalidTool,
    NoSuchConnection,
    NoSuchGroup,
    NoSuchSecret,
    PreflightFailed,
    QuotaExhausted,
    RunTimedOut,
    RuntimeGated,
    RuntimeNotAvailable,
    SecretStillReferenced,
    SessionBusy,
    SessionClosed,
    SessionNotFound,
    StructuredOutputRejected,
    SubpassError,
    UnsafeLaunch,
    VendorRunFailed,
)
from .guards import GuardTracker
from .manage import (
    AddResult,
    ConnectionManager,
    ConnectionPlan,
    Credential,
    EnabledResult,
    GroupsResult,
    RemoveResult,
    RenameResult,
)
from .preflight import (
    FORBIDDEN_LAUNCH_ARGS,
    SCRUB_RULES,
    AccountProfile,
    CacheEligibility,
    Directive,
    PreflightPlan,
    Receipt,
    ScrubRule,
    check_launch_args,
    env_names_to_scrub,
    passthrough_env_names,
    plan_launch,
    scrub_env,
)
from .prompt_cache import (
    PROMPT_CACHE_TTLS,
    VENDOR_DEFAULT,
    PromptCacheDisposition,
    PromptCachePlan,
    plan_prompt_cache,
)
from .retry import RetryVerdict, classify_error, classify_terminal
from .runlog import RunLog, RunRecord, run_log_for
from .runtimes import Runtime
from .sampling_rules import SamplingPlan, SamplingRules, plan_sampling, rules_for
from .schema import (
    normalize_schema,
    openai_strict_issues,
    to_openai_strict,
    validate_instance,
)
from .secrets import SecretPermissions, SecretStore
from .sessions import ChatSession, Session, WorkerSession, scratch_project_folder
from .store import AccountStore, ConnectionStore, StoreSettings
from .tools import ToolDef, normalize_tools
from .types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    AuthMode,
    CacheControl,
    ContentBlock,
    FailoverEvent,
    GuardStopEvent,
    GuardWarningEvent,
    Message,
    ReceiptEvent,
    Retryable,
    Role,
    Sampling,
    SessionInfo,
    SessionKind,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    ThinkingEvent,
    Timeout,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UsageScope,
    VendorEvent,
)

__version__ = "0.2.1"

__all__ = [
    "ALLOWED_ENV_PASSTHROUGH",
    "CALLER_TOOL_SERVER",
    "DEFAULT_GROUP",
    "DEFAULT_REGISTRY",
    "FORBIDDEN_LAUNCH_ARGS",
    "PROMPT_CACHE_TTLS",
    "SCRUB_RULES",
    "STATIC_TABLE",
    "VENDOR_DEFAULT",
    "VERIFY_CELLS",
    "Account",
    "AccountBinding",
    "AccountProfile",
    "AccountStore",
    "Adapter",
    "AdapterFailed",
    "AdapterNotImplemented",
    "AddResult",
    "AgentEvent",
    "Answer",
    "AuthMode",
    "AuthModeMismatch",
    "Bridge",
    "CacheControl",
    "CacheEligibility",
    "Capability",
    "CapabilityNotSupported",
    "CapabilityRegistry",
    "ChatSession",
    "ConfigError",
    "Connection",
    "ConnectionDisabled",
    "ConnectionManager",
    "ConnectionPlan",
    "ConnectionStore",
    "ContentBlock",
    "Credential",
    "CredentialKind",
    "CredentialRef",
    "CredentialRefIsSecret",
    "Directive",
    "DuplicateConnection",
    "EnabledResult",
    "FailoverEvent",
    "Group",
    "GroupSelection",
    "GroupUnavailable",
    "GroupsResult",
    "GuardStop",
    "GuardStopEvent",
    "GuardTracker",
    "GuardWarningEvent",
    "Guards",
    "InvalidConnection",
    "InvalidGuards",
    "InvalidSchema",
    "InvalidSession",
    "InvalidTool",
    "Message",
    "NoSuchConnection",
    "NoSuchGroup",
    "NoSuchSecret",
    "PreflightFailed",
    "PreflightPlan",
    "PromptCacheDisposition",
    "PromptCachePlan",
    "QuotaAction",
    "QuotaExhausted",
    "QuotaPolicy",
    "Receipt",
    "ReceiptEvent",
    "RemoveResult",
    "RenameResult",
    "RetryVerdict",
    "Retryable",
    "Role",
    "RunLog",
    "RunRecord",
    "RunRequest",
    "RunTimedOut",
    "Runtime",
    "RuntimeGated",
    "RuntimeNotAvailable",
    "Sampling",
    "SamplingPlan",
    "SamplingRules",
    "ScrubRule",
    "SecretPermissions",
    "SecretStillReferenced",
    "SecretStore",
    "Session",
    "SessionBusy",
    "SessionClosed",
    "SessionHandle",
    "SessionInfo",
    "SessionKind",
    "SessionNotFound",
    "SessionRequest",
    "StoreSettings",
    "StructuredOutputEvent",
    "StructuredOutputRejected",
    "SubpassError",
    "Support",
    "TerminalEvent",
    "TerminalStatus",
    "TextBlock",
    "TextDeltaEvent",
    "ThinkingEvent",
    "Timeout",
    "TokenUsage",
    "ToolCallEvent",
    "ToolDef",
    "ToolResultEvent",
    "UnsafeLaunch",
    "UsageEvent",
    "UsageScope",
    "ValidationReport",
    "VendorEvent",
    "VendorRunFailed",
    "VerifiedCapabilities",
    "VerifyReport",
    "WorkerSession",
    "__version__",
    "chat",
    "check_launch_args",
    "classify_error",
    "classify_terminal",
    "default_bridge",
    "env_names_to_scrub",
    "normalize_schema",
    "normalize_tools",
    "openai_strict_issues",
    "passthrough_env_names",
    "plan_launch",
    "plan_prompt_cache",
    "plan_sampling",
    "rules_for",
    "run_log_for",
    "runtime_auth_modes",
    "scratch_project_folder",
    "scrub_env",
    "to_openai_strict",
    "validate_instance",
]
