from pydantic import BaseModel, Field
from typing import Literal, Optional


class CreditApplication(BaseModel):
    """Input schema for credit risk prediction."""
    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: float = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt to income ratio")
    loan_history_count: int = Field(..., description="Number of previous loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (0 or 1)")
    loan_amount: Optional[float] = Field(default=10000, description="Loan amount")
    interest_rate: Optional[float] = Field(default=10.0, description="Interest rate")
    num_credit_lines: Optional[int] = Field(default=2, description="Number of credit lines")
    delinquency_2yrs: Optional[int] = Field(default=0, description="Delinquencies in last 2 years")
    loan_purpose: Optional[Literal["business", "debt_consolidation", "home_improvement", "major_purchase", "other"]] = Field(default="other", description="Loan purpose")


class PredictionResult(BaseModel):
    """Output schema for prediction results."""
    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: Literal["low", "medium", "high"] = Field(..., description="Risk classification band")


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band from probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict_default(application: CreditApplication) -> PredictionResult:
    """Make a credit default prediction."""
    from src.model import get_model, get_scaler, get_feature_names
    
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()
    
    loan_purpose = application.loan_purpose or "other"
    
    features = {
        "credit_score": application.credit_score,
        "annual_income": application.income,
        "debt_to_income": application.debt_to_income,
        "employment_years": application.employment_years,
        "loan_amount": application.loan_amount,
        "interest_rate": application.interest_rate,
        "verified_income": application.verified_income,
        "num_credit_lines": application.num_credit_lines,
        "delinquency_2yrs": application.delinquency_2yrs,
        "loan_purpose_business": 1 if loan_purpose == "business" else 0,
        "loan_purpose_debt_consolidation": 1 if loan_purpose == "debt_consolidation" else 0,
        "loan_purpose_home_improvement": 1 if loan_purpose == "home_improvement" else 0,
        "loan_purpose_major_purchase": 1 if loan_purpose == "major_purchase" else 0,
        "loan_purpose_other": 1 if loan_purpose == "other" else 0,
        "home_ownership_MORTGAGE": 1 if application.home_ownership == "mortgage" else 0,
        "home_ownership_OTHER": 1 if application.home_ownership == "other" else 0,
        "home_ownership_OWN": 1 if application.home_ownership == "own" else 0,
        "home_ownership_RENT": 1 if application.home_ownership == "rent" else 0,
    }
    
    X = [[features[name] for name in feature_names]]
    X_scaled = scaler.transform(X)
    
    prob_default = float(model.predict_proba(X_scaled)[0, 1])
    
    return PredictionResult(
        approved=prob_default < 0.35,
        default_probability=round(prob_default, 4),
        risk_band=get_risk_band(prob_default)
    )