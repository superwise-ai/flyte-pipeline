import pandera as pa
from pandera.typing import Series


class TrainData(pa.SchemaModel):
    carat: Series[float]
    cut: Series[str]
    color: Series[str]
    clarity: Series[str]
    depth: Series[float]
    table: Series[float]
    x: Series[float]
    y: Series[float]
    z: Series[float]

    class Config:
        coerce = True


class TargetSerie(pa.SchemaModel):
    price: Series[int]

    class Config:
        coerce = True


class RawData(pa.SchemaModel):
    carat: Series[float]
    cut: Series[str]
    color: Series[str]
    clarity: Series[str]
    depth: Series[float]
    table: Series[float]
    price: Series[int]
    x: Series[float]
    y: Series[float]
    z: Series[float]

    class Config:
        coerce = True


class PredictionData(RawData):
    prediction: Series[int]
