"""Strict JSON contracts shared by CLI, tool execution and LLM providers."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Selector(StrictModel):
    assessment_id: str
    country: str | None = Field(default=None, description="ISO alpha-3 country code, e.g. FJI for Fiji, IDN for Indonesia; canonical full names also accepted")
    scope: Literal["all_modeled", "excluding_Australia", "Pacific_islands"] | None = None
    period: Literal["2026", "2027", "2026–2027"] = "2027"
    metric: Literal["revenue_loss_2025usd_mean", "p_revenue_loss", "var95_2025usd", "es95_2025usd"] = (
        "revenue_loss_2025usd_mean"
    )


class AssessmentArgs(StrictModel):
    assessment_id: str


class EmptyArgs(StrictModel):
    pass


class SearchArgs(AssessmentArgs):
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)


class CompareArgs(Selector):
    earlier_assessment_id: str


class ScenarioSpec(StrictModel):
    extension: Literal["ar2", "neutral_decay", "la_nina_rebound"] = "ar2"
    climate_dependence: Literal["empirical", "independent", "comonotonic"] = "empirical"
    residual_dependence: Literal["shared_year", "independent"] = "shared_year"
    onset: Literal["2026-05-01", "2026-06-01"] | None = None
    fiscal_transmission: Literal["estimated", "zero"] = "estimated"


class ScenarioArgs(AssessmentArgs):
    specification: ScenarioSpec


class ScenarioCompareArgs(Selector):
    scenario_ids: list[str] = Field(min_length=2, max_length=4)


class AssessmentDateArgs(AssessmentArgs):
    field: Literal["as_of", "forecast_issue", "forecast_issue_date", "official_last_center_month", "latest_observed_center"]


class ScenarioParameterArgs(AssessmentArgs):
    scenario_id: str = "central"
    parameter: Literal["fiscal_prior_scale", "agriculture_prior_scale"] = "fiscal_prior_scale"


class ConvertArgs(StrictModel):
    evidence_id: str
    unit: Literal["2025 USD", "million 2025 USD", "billion 2025 USD", "probability", "percent"]


class Claim(StrictModel):
    kind: Literal["numeric", "metadata", "descriptive", "interpretation"]
    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)
    value: float | str | None = None
    unit: str | None = None


class Findings(StrictModel):
    title: str = Field(max_length=200)
    claims: list[Claim] = Field(min_length=1, max_length=30)
    limitations: list[str] = Field(min_length=1, max_length=20)


class Limits(StrictModel):
    max_turns: int = Field(default=12, ge=1, le=30)
    max_tools: int = Field(default=24, ge=1, le=60)
    max_scenarios: int = Field(default=3, ge=0, le=3)
    max_repairs: int = Field(default=2, ge=0, le=3)
    spend_ceiling_usd: float = Field(default=0, ge=0)
    input_usd_per_million: float = Field(default=0, ge=0)
    output_usd_per_million: float = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)
