from typing import Any

import numpy as np
import pandas as pd
from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from django.db.models import JSONField
from django_pandas.managers import DataFrameManager


class NumpyJSONEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class NumpyJSONField(JSONField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('encoder', NumpyJSONEncoder)
        super().__init__(*args, **kwargs)


class PandasJSONField(JSONField):
    """
    JSONField, który akceptuje i zwraca pandas.DataFrame / pandas.Series.
    W bazie przechowuje JSON z markerem __pd__ i danymi w 'orient=split'.

    Parametry:
      pandas_type: 'auto' | 'dataframe' | 'series'
      auto_parse_dates: bool (hook pod ewentualne heurystyki konwersji ISO->datetime)
    """

    def __init__(self, *args, pandas_type: str = "auto", auto_parse_dates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if pandas_type not in ("auto", "dataframe", "series"):
            raise ValueError("pandas_type must be one of: 'auto', 'dataframe', 'series'")
        self.pandas_type = pandas_type
        self.auto_parse_dates = auto_parse_dates

    # ---------------------------
    # Prywatne metody pomocnicze
    # ---------------------------

    @staticmethod
    def _is_dataframe(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    @staticmethod
    def _is_series(obj: Any) -> bool:
        return isinstance(obj, pd.Series)

    @staticmethod
    def _datetime_to_iso(s: pd.Series) -> pd.Series:
        """Konwertuj datetime64[*] -> ISO8601 (string)."""
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64tz_dtype(s):
            # Jeśli ma strefę, sprowadź do UTC dla stabilności
            if pd.api.types.is_datetime64tz_dtype(s):
                return s.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        return s

    def _df_to_payload(self, df: pd.DataFrame) -> dict[str, Any]:
        df2 = df.copy()
        for col in df2.columns:
            df2[col] = self._datetime_to_iso(df2[col])
        df2 = df2.where(pd.notna(df2), None)

        payload: dict[str, Any] = {
            "__pd__": "DataFrame",
            "split": {
                "index": df2.index.tolist(),
                "columns": df2.columns.tolist(),
                "data": df2.values.tolist(),
            }
        }
        if df2.index.name is not None:
            payload["index_name"] = df2.index.name
        if isinstance(df2.columns, pd.MultiIndex):
            payload["columns_names"] = list(df2.columns.names)
        else:
            payload["columns_names"] = [getattr(df2.columns, "name", None)]
        return payload

    def _series_to_payload(self, s: pd.Series) -> dict[str, Any]:
        s2 = self._datetime_to_iso(s)
        s2 = s2.where(pd.notna(s2), None)
        return {
            "__pd__": "Series",
            "split": {"name": s2.name, "index": list(s2.index), "data": s2.tolist()},
        }

    def _payload_to_df(self, payload: dict[str, Any]) -> pd.DataFrame:
        split = payload["split"]
        df = pd.DataFrame(split["data"], index=split["index"], columns=split["columns"])
        if payload.get("index_name") is not None:
            df.index.name = payload["index_name"]
        col_names = payload.get("columns_names")
        if col_names:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns.names = col_names
            else:
                df.columns.name = col_names[0]
        # Ewentualne auto-parse dat można dodać tutaj, bazując na self.auto_parse_dates
        return df

    @staticmethod
    def _payload_to_series(payload: dict[str, Any]) -> pd.Series:
        split = payload["split"]
        return pd.Series(split["data"], index=split["index"], name=split.get("name"))

    # ---------------------------
    # Konwersje pola
    # ---------------------------

    def to_python(self, value: Any) -> Any:
        if self._is_dataframe(value) or self._is_series(value) or value is None:
            return value

        value = super().to_python(value)
        if isinstance(value, dict) and value.get("__pd__") == "DataFrame":
            return self._payload_to_df(value)
        if isinstance(value, dict) and value.get("__pd__") == "Series":
            return self._payload_to_series(value)
        return value

    def get_prep_value(self, value: Any) -> Any:
        if value is None:
            return super().get_prep_value(None)

        if self.pandas_type == "dataframe" and not self._is_dataframe(value):
            raise ValidationError("PandasJSONField (dataframe) wymaga pandas.DataFrame albo None.")
        if self.pandas_type == "series" and not self._is_series(value):
            raise ValidationError("PandasJSONField (series) wymaga pandas.Series albo None.")

        if self._is_dataframe(value):
            value = self._df_to_payload(value)
        elif self._is_series(value):
            value = self._series_to_payload(value)
        # else: zwykły JSON-owalny obiekt przejdzie dalej

        return super().get_prep_value(value)

    def from_db_value(self, value: Any, expression, connection) -> Any:
        value = super().from_db_value(value, expression, connection)
        if isinstance(value, dict) and value.get("__pd__") == "DataFrame":
            return self._payload_to_df(value)
        if isinstance(value, dict) and value.get("__pd__") == "Series":
            return self._payload_to_series(value)
        return value


class Feature(models.Model):
    class Meta:
        db_table = 'feature'
        unique_together = ('dataset', 'fid')

    # Nazwa zbioru danych np. seg04
    dataset = models.TextField(db_index=True)
    # Identyfikator pliku np. first_0003
    fid = models.TextField()
    # Grupa .np first
    group = models.TextField()
    # Numer w grupie np. 3
    nr = models.IntegerField()
    # Nazwa pliku np. first_0003.nii.gz
    filename = models.TextField()

    # ============================================================
    # Clinical
    # ============================================================
    # Wiek pacjenta
    wiek = models.IntegerField()
    # Występowanie niewydolności serca (0/1)
    niewydolnosc_serca = models.IntegerField()
    # Występowanie choroby niedokrwiennej serca (0/1) 
    choroba_niedokrwienna_serca = models.IntegerField()
    # Występowanie cukrzycy (0/1)
    cukrzyca = models.IntegerField()
    # Występowanie nadciśnienia tętniczego (0/1)
    nadcisnienie_tetnicze = models.IntegerField()
    # Występowanie zaburzeń lipidowych (0/1)
    zaburzenia_lipidowe = models.IntegerField()
    # Występowanie migotania przedsionków (0/1)
    af = models.IntegerField()
    # Płeć pacjenta 1 - kobieta, 0 - mężczyzna
    plec = models.IntegerField()
    # Etykieta 'lvm lub lvedv'
    label_or = models.IntegerField()
    # Etykieta dla objętości końcowo-rozkurczowej lewej komory
    label_lvedv = models.IntegerField()
    # Etykieta dla masy mięśnia lewej komory
    label_lvm = models.IntegerField()

    # ============================================================
    # BSI
    # ============================================================
    # Rozmiar organów w px - wszystko poza tłem, płuca i serce
    organs_area = models.IntegerField()
    # Rozmiar serca w px
    heart_area = models.IntegerField()
    # Szerokości serca w px, szerokości pasków 1-pikselowych, interpolowane do 50 wartości
    heart_widths_50 = NumpyJSONField()
    # Szerokości prawego płuca w px, szerokości pasków 1-pikselowych, interpolowane do 50 wartości
    right_lung_widths_50 = NumpyJSONField()
    # Szerokości lewego płuca w px, szerokości pasków 1-pikselowych, interpolowane do 50 wartości
    left_lung_widths_50 = NumpyJSONField()
    # Powierzchnie i powierzchnie znormalizowane do organs_area
    # C - powierzchnia koła w środku serca
    # L - powierzchnia lewej części serca obok koła
    # R - powierzchnia prawej części serca obok koła
    # U - powierzchnia nad kołem
    c_area = models.FloatField()
    norm_c_area = models.FloatField()
    u_area = models.FloatField()
    norm_u_area = models.FloatField()
    l_area = models.FloatField()
    norm_l_area = models.FloatField()
    r_area = models.FloatField()
    norm_r_area = models.FloatField()
    # Promień dużego środkowego koła
    center_radius = models.FloatField()
    # Stosunek szerokości serca do srednicy centralnego koła
    heart_width_diameter_ratio = models.FloatField()

    # ============================================================
    # LSI
    # ============================================================
    # Współczynnik nachylenia centralnej linii serca pod środkiem koła.
    slope = models.FloatField()
    # Kąt między dwiema ścieżkami (centralnej linii przy środku centralnego koła i linii przysrodku małego lewego koła)
    angle = models.FloatField()
    # Promień małego lewego koła
    left_radius = models.IntegerField()

    # ============================================================
    # BB
    # ============================================================
    # Szerokość serca w pikselach
    heart_width = models.FloatField()
    # Wysokość serca w pikselach
    heart_height = models.FloatField()
    # Współczynnik kształtu serca - szerokość do wysokości
    heart_ratio = models.FloatField()
    # Szerokość płuc w pikselach
    lungs_width = models.FloatField()
    # Wysokość płuc w pikselach
    lungs_height = models.FloatField()
    # Stosunek szerokości serca do szerokości płuc
    heart_lungs_ratio = models.FloatField()
    # Średnia wartość pikseli w siatce nałożonej na prostokąt serca
    heart_grid_5x5 = NumpyJSONField()
    heart_grid_6x6 = NumpyJSONField()

    # ============================================================
    # CURVE
    # ============================================================
    # Promień okregu dopasowanego do dolnej połowy krzywej między sercem a lewym płucem
    curve_radius = models.FloatField()
    # Kąty odcinków krzywej między sercem a lewym płucem
    curve_angles = NumpyJSONField()
    # Kąt nachylenia krzywej między prawym płucem a sercem
    right_slope = models.FloatField()

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f'{self.dataset}/{self.fid}'


class Data(models.Model):
    class Meta:
        db_table = 'data'
        unique_together = ('dataset', 'features', 'label')

    dataset = models.TextField(db_index=True)
    features = models.TextField(db_index=True)
    label = models.TextField(db_index=True)
    X = PandasJSONField()
    y = PandasJSONField()

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f'Dataset {self.dataset}/{self.features}/{self.label}'


class Split(models.Model):
    class Meta:
        db_table = 'split'
        unique_together = ('data', 'seed')

    data = models.ForeignKey(Data, on_delete=models.PROTECT, related_name='splits')
    seed = models.IntegerField()
    X_train = PandasJSONField()
    X_test = PandasJSONField()
    y_train = PandasJSONField()
    y_test = PandasJSONField()

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f'Split {self.data.dataset}/{self.data.features}/{self.data.label} {self.seed}'


class Correlation(models.Model):
    class Meta:
        db_table = 'correlation'
        unique_together = ('dataset', 'feature', 'label')

    dataset = models.TextField()
    label = models.TextField()
    feature = models.TextField()
    pearson = models.FloatField()
    spearman = models.FloatField()
    kendall = models.FloatField()

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f'{self.dataset}/{self.feature} ({self.pearson}, {self.spearman})'


class Classif(models.Model):
    class Meta:
        db_table = 'classif'
        unique_together = ('split', 'model')

    split = models.ForeignKey(Split, on_delete=models.PROTECT, related_name='classifs')

    best_params = NumpyJSONField()
    best_value = models.FloatField()
    y_test = NumpyJSONField()
    y_proba = NumpyJSONField()
    model = models.TextField(db_index=True)

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f'Classif {self.model} {self.best_value}'


class Results(models.Model):
    """
    Wynik pojedynczej próby (seed) dla danego modelu.
    Zawiera m.in. AUC, punkt Youdena i zinterpolowaną krzywą ROC do wspólnej siatki fpr_grid.
    """

    class Meta:
        db_table = 'result'

    classif = models.OneToOneField(Classif, on_delete=models.PROTECT, related_name='result')

    # metryki globalne
    roc_auc = models.FloatField()

    # Youden (dla pojedynczej próby)
    youden_j = models.FloatField()
    threshold_j = models.FloatField()
    # true positive rate, TPR=TP/P=TP/(TP+FN)
    tpr_j = models.FloatField()  # czułość (ang. sensitivity, recall) lub odsetek prawdziwie pozytywnych
    # true negative rate, TNR=TN/N=TN/(FP+TN)
    tnr_j = models.FloatField()  # swoistość (ang. specificity, SPC) lub odsetek prawdziwie negatywnych
    # accuracy ACC=(TP+TN)/(P+N)=(TP+TN)/(TP+FN+TN+FP)
    acc_j = models.FloatField()  # dokładność (ang. accuracy, ACC)
    # positive predictive value, PPV=TP/(TP+FP)
    ppv_j = models.FloatField()  # precyzja (ang. precision) lub dodatnia wartość predykcyjna
    # negative predictive value, NPV=TN/(TN+FN)
    npv_j = models.FloatField()  # ujemna wartość predykcyjna
    # false positive rate, FPR=FP/N=FP/(FP+TN)=1-TNR
    fpr_j = models.FloatField()  # odsetek fałszywie pozytywnyc
    # false negative rate, FNR=FN/P=FN/(TP+FN)=1-TPR
    fnr_j = models.FloatField()  # odsetek fałszywie negatywnych

    # krzywa ROC w siatce fpr_grid
    fpr_grid = models.JSONField()  # [float]
    tpr_interp = models.JSONField()  # [float] TPR(fpr_grid)

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f"{self.classif} | AUC={self.roc_auc:.3f}"


class FinalResult(models.Model):
    """
    Agregat po wielu obiektach Results (po seedach) dla danego modelu.
    Przechowuje średnią krzywą ROC (mean ± std) oraz średni punkt Youdena.
    """

    class Meta:
        db_table = 'final_result'
        unique_together = ('data', 'model')

    data = models.ForeignKey(Data, on_delete=models.PROTECT, related_name='final_results')
    model = models.TextField(db_index=True)

    n_runs = models.IntegerField()  # ile Results weszło do agregacji

    # Średnia krzywa ROC (na wspólnej siatce fpr_grid)
    fpr_grid = models.JSONField()  # [float]
    tpr_mean = models.JSONField()  # [float]
    tpr_std = models.JSONField()  # [float]

    # AUC
    auc_mean = models.FloatField()
    auc_std = models.FloatField()

    # Średni punkt Youdena (po współrzędnych), plus rozstępy
    thr_j_mean = models.FloatField()
    thr_j_std = models.FloatField()
    # ile Results miało skończony próg (roc_curve daje inf, gdy J=0 tylko w punkcie (0,0))
    n_thr_j = models.IntegerField(null=True)

    tpr_j_mean = models.FloatField()
    tpr_j_std = models.FloatField()
    tnr_j_mean = models.FloatField()
    tnr_j_std = models.FloatField()
    acc_j_mean = models.FloatField()
    acc_j_std = models.FloatField()
    ppv_j_mean = models.FloatField()
    ppv_j_std = models.FloatField()
    npv_j_mean = models.FloatField()
    npv_j_std = models.FloatField()
    fpr_j_mean = models.FloatField()
    fpr_j_std = models.FloatField()
    fnr_j_mean = models.FloatField()
    fnr_j_std = models.FloatField()

    objects = models.Manager()
    df_objects = DataFrameManager()

    def __str__(self):
        return f"FinalResult {self.model} | AUC≈{self.auc_mean:.3f}"


class CompareROC(models.Model):
    class Meta:
        db_table = 'compare_roc'
        unique_together = ('classif1', 'classif2')

    classif1 = models.ForeignKey(Classif, on_delete=models.PROTECT, related_name='compare_roc1')
    classif2 = models.ForeignKey(Classif, on_delete=models.PROTECT, related_name='compare_roc2')
    auc1 = models.FloatField()
    auc2 = models.FloatField()
    auc_diff = models.FloatField()
    auc_diff_se = models.FloatField()
    z = models.FloatField()
    p = models.FloatField()
    ci_diff_start = models.FloatField()
    ci_diff_end = models.FloatField()
    p_diff_is_significant = models.BooleanField()
    ci_diff_is_significant = models.BooleanField()

    objects = models.Manager()
    df_objects = DataFrameManager()
