import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics import roc_curve, auc, confusion_matrix

from cardio.helpers.compare_auc_delong_xu import delong_auc_diff_test_ci
from cardio.helpers.features_labels import get_features
from ._base import Command as BaseCommand
from ...helpers.data import flatten_data
from ...helpers.parallel import parallel
from ...models import Correlation, Classif, Results, FinalResult, Data, CompareROC, Feature

_logger = logging.getLogger('analyze')


class Command(BaseCommand):
    help = 'Analyze features and classification results'

    def add_arguments(self, parser):
        parser.add_argument('action', type=str, help='Action to perform')

    def handle(self, *args, **options):
        action = options['action']
        for dataset in settings.DATASETS:
            _logger.info(f'Dataset: {dataset}')
            self.choose_dataset(dataset)

            if action == 'features':
                params = []
                fields = Feature._meta.get_fields()
                cols = [f.name for f in fields if f.name not in {'id', 'dataset', 'fid', 'group', 'nr', 'filename'}]

                for label in settings.LABELS_COLUMN:
                    params.append((label, cols))
                parallel(analyze_features_worker, params)
            elif action == 'classif':
                classifs = (Classif.objects.filter(split__data__dataset=dataset,
                                                   split__data__label__in=settings.LABELS_COLUMN).only('id')
                            .values_list('id', flat=True))
                parallel(analyze_classif_worker, [(cid,) for cid in classifs])

                params = []
                for label in settings.LABELS_COLUMN:
                    for features in settings.FEATURES_COLUMNS:
                        data = Data.objects.get(dataset=dataset, features=features, label=label)
                        models = Classif.objects.filter(split__data=data).distinct('model').values_list('model',
                                                                                                        flat=True)
                        for model in models:
                            params.append((data.id, model))
                parallel(analyze_classif_final_worker, params)

                params = []
                for label in settings.LABELS_COLUMN:
                    clsfs = Classif.objects.filter(split__data__dataset=dataset, split__data__label=label)
                    for model in clsfs.distinct('model').values_list('model', flat=True):
                        params.append((dataset, label, model))
                parallel(analyze_roc_worker, params)
            elif action == 'compare_table':
                analyze_compare_table(dataset)
            elif action == 'age':
                analyze_age(dataset)
            else:
                raise ValueError(f'Unknown action: {action}')


def analyze_features_worker(label: str, cols: list[str]):
    data = get_features()
    _logger.info(f'Prepare data for label: {label}')
    data, feature_cols = flatten_data(data, cols)

    _logger.info(f'Calc Pearson')
    pearson = data[feature_cols].corrwith(data[label]).fillna(0)
    _logger.info(f'Calc Spearman')
    spearman = data[feature_cols].corr(method='spearman')[label].fillna(0)
    _logger.info(f'Calc Kendall')
    kendall = data[feature_cols].corr(method='kendall')[label].fillna(0)

    for (idx1, pearson_value), (idx2, spearman_value), (idx3, kendall_value) in zip(
            pearson.items(), spearman.items(), kendall.items()):
        assert idx1 == idx2 == idx3
        if idx1 in settings.LABELS_COLUMN:
            continue
        Correlation.objects.update_or_create(
            dataset=settings.DATASET_NAME,
            label=label,
            feature=idx1,
            defaults=dict(
                pearson=pearson_value,
                spearman=spearman_value,
                kendall=kendall_value,
            )
        )


def analyze_roc_worker(dataset: str, label: str, model: str):
    for feature1, feature2 in [
        ('ratio', 'ratio_clinical'),
        ('clinical', 'ratio_clinical'),
        ('ratio', 'ratio_clinical2'),
        ('clinical2', 'ratio_clinical2'),
    ]:
        clsfs = Classif.objects.filter(split__data__dataset=dataset, split__data__label=label, model=model)
        clfs1 = clsfs.filter(split__data__features=feature1).order_by('split__seed')
        clfs2 = clsfs.filter(split__data__features=feature2).order_by('split__seed')
        # print(label, model, clfs1.count(), clfs2.count())
        if clfs1.count() != clfs2.count():
            print('Invalid classifications {label}/{model} count {clfs1.count()} != {clfs2.count()}')
            continue
        for c1, c2 in zip(clfs1, clfs2):
            delong = delong_auc_diff_test_ci(np.array(c1.y_test), np.array(c1.y_proba), np.array(c2.y_proba))
            CompareROC.objects.update_or_create(
                classif1=c1,
                classif2=c2,
                defaults=dict(
                    ci_diff_start=delong['ci_diff'][0],
                    ci_diff_end=delong['ci_diff'][1],
                    auc1=delong['auc1'],
                    auc2=delong['auc2'],
                    auc_diff=delong['diff'],
                    auc_diff_se=delong['se_diff'],
                    z=delong['z'],
                    p=delong['p'],
                    p_diff_is_significant=delong['p'] < 0.05,
                    ci_diff_is_significant=delong['ci_diff'][0] * delong['ci_diff'][1] > 0
                )
            )


def analyze_classif_worker(classif_id):
    classif = Classif.objects.get(id=classif_id)
    fpr_grid = np.linspace(0, 1, 101)
    # Krzywa ROC dla tej próby (pełna lista progów)
    fpr, tpr, thresholds = roc_curve(classif.y_test, classif.y_proba, drop_intermediate=False)
    auc_val = auc(fpr, tpr)

    # Youden J i próg
    youden = tpr - fpr  # J = TPR - FPR
    j_idx = int(np.nanargmax(youden))
    thr_star = float(thresholds[j_idx])

    # Zgodnie z definicją roc_curve: pozytywna gdy score > threshold
    y_pred = (np.array(classif.y_proba) > thr_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(classif.y_test, y_pred, labels=[0, 1]).ravel()

    tpr_j = tp / (tp + fn) if (tp + fn) else 0.0
    tnr_j = tn / (fp + tn) if (fp + tn) else 0.0
    acc_j = (tp + tn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) else 0.0
    ppv_j = tp / (tp + fp) if (tp + fp) else 0.0
    npv_j = tn / (tn + fn) if (tn + fn) else 0.0
    fpr_j = fp / (fp + tn) if (fp + tn) else 0.0
    fnr_j = fn / (tp + fn) if (tp + fn) else 0.0

    # Interpolacja TPR na wspólną siatkę FPR do uśredniania
    tpr_interp = np.interp(fpr_grid, fpr, tpr).tolist()
    tpr_interp[0] = 0.0

    # Zapis do DB
    Results.objects.update_or_create(
        classif=classif,
        defaults=dict(
            roc_auc=float(auc_val),
            youden_j=float(youden[j_idx]),
            threshold_j=thr_star,
            fpr_grid=fpr_grid.tolist(),
            tpr_interp=tpr_interp,
            tpr_j=tpr_j,
            tnr_j=tnr_j,
            acc_j=acc_j,
            ppv_j=ppv_j,
            npv_j=npv_j,
            fpr_j=fpr_j,
            fnr_j=fnr_j,
        )
    )


def analyze_classif_final_worker(data_id: int, model: str):
    """
    Zbiera wszystkie Results dla danego dataset, liczy średnią krzywą ROC,
    średni punkt Youdena i AUC, następnie zapisuje/aktualizuje FinalResult.
    """
    data = Data.objects.get(id=data_id)
    qs = Results.objects.filter(classif__model=model, classif__split__data=data)
    if not qs.exists():
        raise ValueError(f"Brak wyników do agregacji dla podanych "
                         f"{data.dataset=} {data.features=} {data.label=} {model=}.")

    # weryfikacja spójności siatki
    fpr_grid = qs.first().fpr_grid
    tpr_stack = []
    auc_list = []
    thr_j_list = []
    tpr_j_list, tnr_j_list, acc_j_list, ppv_j_list, npv_j_list, fpr_j_list, fnr_j_list = ([], [], [], [],
                                                                                          [], [], [])

    for res in qs:
        tpr_stack.append(np.array(res.tpr_interp, dtype=float))
        thr_j_list.append(res.threshold_j)
        tpr_j_list.append(res.tpr_j)
        tnr_j_list.append(res.tnr_j)
        acc_j_list.append(res.acc_j)
        ppv_j_list.append(res.ppv_j)
        npv_j_list.append(res.npv_j)
        fpr_j_list.append(res.fpr_j)
        fnr_j_list.append(res.fnr_j)
        auc_list.append(res.roc_auc)

    tprs = np.vstack(tpr_stack)  # (n_runs, len_grid)
    tpr_mean = tprs.mean(axis=0)
    tpr_std = tprs.std(axis=0, ddof=0)

    # roc_curve starts with threshold inf; it is chosen when J > 0 is never reached (all predicted negative),
    # so the threshold statistics use finite thresholds only
    thr_j_finite = np.array(thr_j_list, dtype=float)
    thr_j_finite = thr_j_finite[np.isfinite(thr_j_finite)]
    n_thr_j = len(thr_j_finite)
    if n_thr_j < len(thr_j_list):
        _logger.info(f'{data.features} {data.label} {model}: {len(thr_j_list) - n_thr_j} of {len(thr_j_list)} '
                     f'runs without finite Youden threshold, excluded from threshold statistics')
    thr_j_mean = np.mean(thr_j_finite) if n_thr_j else np.nan
    thr_j_std = np.std(thr_j_finite, ddof=0) if n_thr_j else np.nan
    tpr_j_mean = np.mean(tpr_j_list)
    tpr_j_std = np.std(tpr_j_list, ddof=0)
    tnr_j_mean = np.mean(tnr_j_list)
    tnr_j_std = np.std(tnr_j_list, ddof=0)
    acc_j_mean = np.mean(acc_j_list)
    acc_j_std = np.std(acc_j_list, ddof=0)
    ppv_j_mean = np.mean(ppv_j_list)
    ppv_j_std = np.std(ppv_j_list, ddof=0)
    npv_j_mean = np.mean(npv_j_list)
    npv_j_std = np.std(npv_j_list, ddof=0)
    fpr_j_mean = np.mean(fpr_j_list)
    fpr_j_std = np.std(fpr_j_list, ddof=0)
    fnr_j_mean = np.mean(fnr_j_list)
    fnr_j_std = np.std(fnr_j_list, ddof=0)
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list, ddof=0)

    # zapisz/aktualizuj FinalResult (unikalne po experiment+model_type)
    fr, _created = FinalResult.objects.update_or_create(
        data=data,
        model=model,
        defaults=dict(
            n_runs=qs.count(),
            fpr_grid=fpr_grid,
            tpr_mean=tpr_mean.tolist(),
            tpr_std=tpr_std.tolist(),
            thr_j_mean=thr_j_mean,
            thr_j_std=thr_j_std,
            n_thr_j=n_thr_j,
            tpr_j_mean=tpr_j_mean,
            tpr_j_std=tpr_j_std,
            tnr_j_mean=tnr_j_mean,
            tnr_j_std=tnr_j_std,
            acc_j_mean=acc_j_mean,
            acc_j_std=acc_j_std,
            ppv_j_mean=ppv_j_mean,
            ppv_j_std=ppv_j_std,
            npv_j_mean=npv_j_mean,
            npv_j_std=npv_j_std,
            fpr_j_mean=fpr_j_mean,
            fpr_j_std=fpr_j_std,
            fnr_j_mean=fnr_j_mean,
            fnr_j_std=fnr_j_std,
            auc_mean=auc_mean,
            auc_std=auc_std,
        )
    )
    plot_mean_roc(fr)


def plot_mean_roc(fr: FinalResult, show_youden: bool = True):
    """
    Rysuje średnią krzywą ROC (mean ± std). Opcjonalnie zaznacza średni punkt Youdena
    (z „krzyżykami” błędów = odchylenia standardowe po foldach).

    """
    results_dir = settings.RESULTS_DIR / f'classify_{fr.data.features}' / fr.data.label
    results_dir.mkdir(parents=True, exist_ok=True)
    fpr = np.array(fr.fpr_grid, dtype=float)
    tpr_m = np.array(fr.tpr_mean, dtype=float)
    tpr_s = np.array(fr.tpr_std, dtype=float)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr_m, label=f"Średnia ROC (AUC≈{fr.auc_mean:.3f} ± {fr.auc_std:.3f})")
    plt.fill_between(fpr, tpr_m - tpr_s, tpr_m + tpr_s, alpha=0.2, label="±1 STD")

    if show_youden:
        plt.scatter([fr.fpr_j_mean], [fr.tpr_j_mean], s=50, label="Średni punkt Youdena")
        plt.errorbar(fr.fpr_j_mean, fr.tpr_j_mean,
                     xerr=fr.fpr_j_std, yerr=fr.tpr_j_std,
                     fmt='none', capsize=4)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Losowa")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"{fr.data.features}/{fr.data.label} - {fr.model}\n"
              f"Śr. próg Youdena: {fr.thr_j_mean:.3f} ± {fr.thr_j_std:.3f} | n={fr.n_runs}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'ROC_{fr.model}.png')
    plt.close(fig)


MODEL_NAMES = {
    'SVM': 'Support vector machine',
    'KNN': 'k-nearest neighbours',
    'MLP': 'Multi-layer perceptron',
    'Voting': 'Soft voting ensemble',
    'Stacking': 'Stacking ensemble',
}


def analyze_compare_table(dataset: str):
    """
    Podsumowanie testów DeLonga (CompareROC) po wszystkich podziałach dla każdej pary zestawów cech,
    etykiety i modelu. Zapis do RESULTS_DIR/compare_roc_table.xlsx (arkusz na parę cech i etykietę).
    W CompareROC: auc_diff = auc1 - auc2, więc wzrost AUC po dodaniu cech oznacza auc2 > auc1.
    """
    qs = CompareROC.objects.filter(classif1__split__data__dataset=dataset,
                                   classif1__split__data__label__in=settings.LABELS_COLUMN)
    df = qs.values('classif1__model', 'classif1__split__data__features', 'classif2__split__data__features',
                   'classif1__split__data__label', 'classif1__split__y_test', 'classif2__split__y_test',
                   'auc1', 'auc2', 'p')
    df = pd.DataFrame.from_records(df).rename(columns={
        'classif1__model': 'model',
        'classif1__split__data__features': 'features1',
        'classif2__split__data__features': 'features2',
        'classif1__split__data__label': 'label',
    })
    if df.empty:
        _logger.warning(f'No CompareROC rows for {dataset}')
        return

    # DeLong wymaga tych samych pacjentów w obu zbiorach testowych
    df['same_test_set'] = [
        sorted(pd.Series(y1).index.tolist()) == sorted(pd.Series(y2).index.tolist())
        for y1, y2 in zip(df.pop('classif1__split__y_test'), df.pop('classif2__split__y_test'))
    ]
    df['diff'] = df['auc2'] - df['auc1']
    df['improved'] = df['diff'] > 0
    df['significant'] = df['p'] < 0.05

    out_file = settings.RESULTS_DIR / 'compare_roc_table.xlsx'
    with pd.ExcelWriter(out_file) as writer:
        for (f1, f2, label), g in df.groupby(['features1', 'features2', 'label']):
            if not g['same_test_set'].all():
                _logger.warning(f'{f1} vs {f2} / {label}: {(~g["same_test_set"]).sum()} comparisons '
                                f'on different test sets, DeLong results are not valid')
            rows = []
            for model, m in g.groupby('model'):
                n = len(m)
                rows.append({
                    'Model': MODEL_NAMES.get(model, model),
                    f'AUC {f1}': f'{m["auc1"].mean():.3f} ± {m["auc1"].std(ddof=0):.3f}',
                    f'AUC {f2}': f'{m["auc2"].mean():.3f} ± {m["auc2"].std(ddof=0):.3f}',
                    'Difference': f'{m["diff"].mean():+.3f} ± {m["diff"].std(ddof=0):.3f}',
                    'Splits with AUC increase': f'{m["improved"].sum()}/{n} ({m["improved"].mean():.0%})',
                    'Splits with p < 0.05': f'{m["significant"].sum()}/{n} ({m["significant"].mean():.0%})',
                    'same_test_set': bool(m['same_test_set'].all()),
                    '_auc2': m['auc2'].mean(),
                })
            table = pd.DataFrame(rows).sort_values('_auc2', ascending=False).drop(columns='_auc2')
            sheet = f'{f1}-{f2}-{label.removeprefix("label_")}'[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
            _logger.info(f'{f1} vs {f2} / {label}\n{table.to_string(index=False)}')
    _logger.info(f'Saved {out_file}')


def analyze_age(dataset: str):
    df = Feature.df_objects.filter(dataset=dataset).to_dataframe()
    for label in settings.LABELS_COLUMN:
        results_dir = settings.RESULTS_DIR / f'wiek'
        results_dir.mkdir(parents=True, exist_ok=True)
        # Przygotowanie danych
        X = df[['wiek']].values  # Zmienna niezależna (wiek)
        y = df[label].values  # Zmienna zależna (choroba: 0/1)

        # Trenowanie modelu regresji logistycznej
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X, y)

        # Przewidywanie prawdopodobieństwa zachorowania
        y_prob = model.predict_proba(X)[:, 1]  # Prawdopodobieństwo klasy 1
        y_pred = model.predict(X)

        # Wydruk wyników regresji
        print("=" * 50)
        print("REGRESJA LOGISTYCZNA")
        print("=" * 50)
        print(f"Współczynnik (wiek): {model.coef_[0][0]:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        print(f"Dokładność: {model.score(X, y):.4f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y, y_pred))

        # Obliczanie krzywej ROC
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        # Znalezienie optymalnego progu (indeks Youdena)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]

        print("\n" + "=" * 50)
        print("ANALIZA KRZYWEJ ROC")
        print("=" * 50)
        print(f"AUC (Area Under Curve): {roc_auc:.4f}")
        print(f"Optymalny próg: {optimal_threshold:.4f}")
        print(f"Czułość (TPR) przy optymalnym progu: {tpr[optimal_idx]:.4f}")
        print(f"Swoistość (1-FPR) przy optymalnym progu: {1 - fpr[optimal_idx]:.4f}")

        # Wizualizacja
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Krzywa ROC
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        axes[0].scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                        label=f'Optimal threshold: {optimal_threshold:.3f}')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Krzywa ROC')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Przewidywane prawdopodobieństwa w zależności od wieku
        wiek_range = np.linspace(df['wiek'].min(), df['wiek'].max(), 100).reshape(-1, 1)
        prob_range = model.predict_proba(wiek_range)[:, 1]
        wiek_progu_direct = df['wiek'].iloc[optimal_idx]

        axes[1].scatter(df['wiek'], df['label_lvm'], alpha=0.5, label='Dane rzeczywiste')
        axes[1].plot(wiek_range, prob_range, color='red', lw=2, label='Przewidywane prawdopodobieństwo')
        axes[1].axvline(x=wiek_progu_direct, color='green', linestyle='--',
                        label=f'Optymalny próg: {wiek_progu_direct}')
        axes[1].set_xlabel('Wiek')
        axes[1].set_ylabel('Prawdopodobieństwo choroby')
        axes[1].set_title('Regresja logistyczna: Choroba vs Wiek')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / f'ROC_{label}.png')

        # Macierz pomyłek
        cm = confusion_matrix(y, y_pred)
        print("\n" + "=" * 50)
        print("MACIERZ POMYŁEK")
        print("=" * 50)
        print(cm)
