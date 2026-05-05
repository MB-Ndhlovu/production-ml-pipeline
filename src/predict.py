from pydantic import BaseModel, Field
from typing import Literal, Optional

class PredictionInput(BaseModel):
    """Input schema for credit scoring prediction."""
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of previous loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        default="other", description="Home ownership status"
    )
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")
    loan_amount: Optional[float] = Field(default=10000, description="Requested loan amount")
    interest_rate: Optional[float] = Field(default=0.12, description="Interest rate")
    num_credit_lines: Optional[int] = Field(default=2, description="Number of credit lines")
    delinquency_2yrs: Optional[int] = Field(default=0, description="Delinquencies in last 2 years")
    loan_purpose: Optional[Literal["business", "debt_consolidation", "home_improvement", "major_purchase", "other"]] = Field(
        default="other", description="Loan purpose"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "income": 65000,
                "credit_score": 720,
                "employment_years": 5,
                "debt_to_income": 0.28,
                "loan_history_count": 2,
                "age": 34,
                "home_ownership": "rent",
                "verified_income": 1
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for credit scoring prediction."""
    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: Literal["low", "medium", "high"] = Field(..., description="Risk classification")

    class Config:
        json_schema_extra = {
            "example": {
                "approved": True,
                "default_probability": 0.12,
                "risk_band": "low"
            }
        }

def get_risk_band(probability: float) -> str:
    """Determine risk band based on default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"

def is_approved(probability: float, threshold: float = 0.35) -> bool:
    """Determine if application should be approved based on probability threshold."""
    return probability < threshold

def prepare_features(input_data: PredictionInput) -> dict:
    """Convert input to feature dictionary matching training format."""
    features = {
        "annual_income": input_data.income,
        "credit_score": input_data.credit_score,
        "employment_years": input_data.employment_years,
        "debt_to_income": input_data.debt_to_income,
        "loan_history_count": input_data.loan_history_count,
        "age": input_data.age,
        "verified_income": input_data.verified_income,
        "loan_amount": input_data.loan_amount,
        "interest_rate": input_data.interest_rate,
        "num_credit_lines": input_data.num_credit_lines,
        "delinquency_2yrs": input_data.delinquency_2yrs,
    }
    
    home = input_data.home_ownership.upper()
    features["home_ownership_MORTGAGE"] = 1 if home == "MORTGAGE" else 0
    features["home_ownership_OTHER"] = 1 if home == "OTHER" else 0
    features["home_ownership_OWN"] = 1 if home == "OWN" else 0
    features["home_ownership_RENT"] = 1 if home == "RENT" else 0
    
    purpose = input_data.loan_purpose
    features["loan_purpose_business"] = 1 if purpose == "business" else 0
    features["loan_purpose_debt_consolidation"] = 1 if purpose == "debt_consolidation" else 0
    features["loan_purpose_home_improvement"] = 1 if purpose == "home_improvement" else 0
    features["loan_purpose_major_purchase"] = 1 if purpose == "major_purchase" else 0
    features["loan_purpose_other"] = 1 if purpose == "other" else 0
    
    return features