from fastapi import APIRouter
from app.schemas.farm_schema import FarmInput

router = APIRouter(prefix="/farm", tags=["Farm"])

@router.post("/validate")
def validate_farm_input(data: FarmInput):
    return {"status": "valid", "received": data}
