import pathlib
import pandas as pd
import numpy as np
import pyarrow as pa


# Serializacja i deserializacja dla pathlib.Path
def path_serializer(obj):
    return str(obj)


def path_deserializer(data):
    return pathlib.Path(data)


# Serializacja i deserializacja dla pandas.Series
def series_serializer(obj):
    return obj.to_dict()


def series_deserializer(data):
    return pd.Series(data)


# Serializacja i deserializacja dla pandas.DataFrame
def dataframe_serializer(obj):
    return obj.to_dict(orient='split')


def dataframe_deserializer(data):
    return pd.DataFrame(**data)


# Serializacja i deserializacja dla numpy.ndarray
def ndarray_serializer(obj):
    return obj.tolist()


def ndarray_deserializer(data):
    return np.array(data)


# Tworzymy kontekst serializacji
context = pas.SerializationContext()

# Rejestrujemy typy z custom serializer/deserializer
context.register_type(pathlib.Path, 'PathType',
                      custom_serializer=path_serializer,
                      custom_deserializer=path_deserializer,
                      pickle=False)

context.register_type(pd.Series, 'SeriesType',
                      custom_serializer=series_serializer,
                      custom_deserializer=series_deserializer,
                      pickle=False)

context.register_type(pd.DataFrame, 'DataFrameType',
                      custom_serializer=dataframe_serializer,
                      custom_deserializer=dataframe_deserializer,
                      pickle=False)

context.register_type(np.ndarray, 'NumpyArrayType',
                      custom_serializer=ndarray_serializer,
                      custom_deserializer=ndarray_deserializer,
                      pickle=False)

# Testowanie serializacji
obj_path = pathlib.Path('/home/user')
obj_series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
obj_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
obj_array = np.array([[1, 2], [3, 4]])

for obj in [obj_path, obj_series, obj_df, obj_array]:
    serialized = context.serialize(obj)
    deserialized = context.deserialize(serialized)
    print(type(deserialized), deserialized)
