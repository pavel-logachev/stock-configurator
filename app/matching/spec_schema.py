from typing import Any

from pydantic import BaseModel, Field


class NormalizedSpecItem(BaseModel):
    name: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=1)
    brand: str | None = None
    part_number: str | None = None
    required_attributes: dict[str, str] = Field(default_factory=dict)


class NormalizedSpec(BaseModel):
    items: list[NormalizedSpecItem] = Field(default_factory=list)


class StockSpecItem(BaseModel):
    item_type: str = Field(default="unknown", min_length=1)
    quantity: int = Field(default=1, ge=1)
    name: str | None = None
    category: str | None = None
    brand: str | None = None
    part_number: str | None = None
    server_qty: int | None = Field(default=None, ge=1)
    form_factor: str | None = None
    cpu_per_server: int | None = Field(default=None, ge=1)
    total_cpu_required: int | None = Field(default=None, ge=1)
    cpu_vendor_preference: str | None = None
    cpu_family_preference: str | None = None
    cpu_min_cores_per_cpu: int | None = Field(default=None, ge=1)
    cpu_generation_or_model_hint: str | None = None
    ram_gb_per_server: int | None = Field(default=None, ge=1)
    ram_type_preference: str | None = None
    storage_required: bool | None = None
    storage_type_preference: str | None = None
    storage_interface_preference: str | None = None
    storage_min_capacity: str | None = None
    psu_count_per_server: int | None = Field(default=None, ge=1)
    location: str | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)


class StockSpec(BaseModel):
    items: list[StockSpecItem] = Field(default_factory=list)
    shipment_city: str | None = None
    server_qty: int | None = Field(default=None, ge=1)
    form_factor: str | None = None
    cpu_per_server: int | None = Field(default=None, ge=1)
    total_cpu_required: int | None = Field(default=None, ge=1)
    cpu_vendor_preference: str | None = None
    cpu_family_preference: str | None = None
    cpu_min_cores_per_cpu: int | None = Field(default=None, ge=1)
    cpu_generation_or_model_hint: str | None = None
    ram_gb_per_server: int | None = Field(default=None, ge=1)
    ram_type_preference: str | None = None
    storage_required: bool | None = None
    storage_type_preference: str | None = None
    storage_interface_preference: str | None = None
    storage_min_capacity: str | None = None
    psu_count_per_server: int | None = Field(default=None, ge=1)
    location: str | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)
    source_text: str | None = None


class StockSpecExtractionResult(BaseModel):
    spec_json: StockSpec
    confirmation_text: str = ""
    unclear_points: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
