from django.conf import settings

from ._base import Command as BaseCommand
from ...models import FinalResult, Correlation, CompareROC


class Command(BaseCommand):
    help = 'Export results to XLSX'

    def handle(self, *args, **options):
        self.final_results()
        self.compare_roc()
        self.correlations()

    def final_results(self):
        self.stdout.write(self.style.SUCCESS('Exporting Final results'))
        final_results_df = FinalResult.df_objects.all().to_dataframe(fieldnames=(
            'data__dataset',
            'data__features',
            'data__label',
            'model',
            'n_runs',
            'auc_mean',
            'auc_std',
            'thr_j_mean',
            'thr_j_std',
            'n_thr_j',
            'tpr_j_mean',
            'tpr_j_std',
            'tnr_j_mean',
            'tnr_j_std',
            'acc_j_mean',
            'acc_j_std',
            'ppv_j_mean',
            'ppv_j_std',
            'npv_j_mean',
            'npv_j_std',
            'fpr_j_mean',
            'fpr_j_std',
            'fnr_j_mean',
            'fnr_j_std',
        ))
        final_results_df.rename(columns={
            'data__dataset': 'dataset',
            'data__features': 'features',
            'data__label': 'label',
        }, inplace=True)
        final_results_df.to_excel(settings.RESULTS_BASE_DIR / 'final_results.xlsx', index=False)

    def correlations(self):
        self.stdout.write(self.style.SUCCESS('Exporting Correlations'))
        corr_df = Correlation.df_objects.filter(dataset__in=['seg03', 'seg03.1']).to_dataframe()
        corr_df.to_excel(settings.RESULTS_BASE_DIR / 'correlations.xlsx', index=False)

        agg = (corr_df.groupby(['label', 'feature'], as_index=False).agg(
            pearson_mean=('pearson', 'mean'),
            pearson_std=('pearson', 'std'),
            spearman_mean=('spearman', 'mean'),
            spearman_std=('spearman', 'std'),
            kendall_mean=('kendall', 'mean'),
            kendall_std=('kendall', 'std'),
            n=('dataset', 'nunique')))  # liczba różnych datasetów na parę

        agg.to_excel(settings.RESULTS_BASE_DIR / 'correlations_mean.xlsx', index=False)

    def compare_roc(self):
        self.stdout.write(self.style.SUCCESS('Exporting ROC compare'))
        compare_df = CompareROC.df_objects.all().to_dataframe(fieldnames=(
            'classif1__split__data__dataset',
            'classif1__split__seed',
            'classif1__model',
            'classif1__split__data__features',
            'classif2__split__data__features',
            'classif2__split__data__label',
            'auc1',
            'auc2',
            'auc_diff',
            'auc_diff_se',
            'z',
            'p',
            'ci_diff_start',
            'ci_diff_end',
            'p_diff_is_significant',
            'ci_diff_is_significant',
        ))
        compare_df.rename(columns={
            'classif2__split__data__label': 'label',
            'classif1__split__data__dataset': 'dataset',
            'classif1__split__seed': 'seed',
            'classif1__model': 'model',
            'classif1__split__data__features': 'classif1_features',
            'classif2__split__data__features': 'classif2_features',
        }, inplace=True)
        compare_df.to_excel(settings.RESULTS_BASE_DIR / 'compare_roc.xlsx', index=False)

        agg = (compare_df.groupby(['dataset', 'model', 'classif1_features', 'classif2_features', 'label'], as_index=False)
        .agg(
            auc1_mean=('auc1', 'mean'),
            auc1_std=('auc1', 'std'),
            auc2_mean=('auc2', 'mean'),
            auc2_std=('auc2', 'std'),
            auc_diff_mean=('auc_diff', 'mean'),
            auc_diff_std=('auc_diff', 'std'),
            auc_diff_se_mean=('auc_diff_se', 'mean'),
            auc_diff_se_std=('auc_diff_se', 'std'),
            z_mean=('z', 'mean'),
            z_std=('z', 'std'),
            p_mean=('p', 'mean'),
            p_std=('p', 'std'),
            p_diff_is_significant_count=('p_diff_is_significant', 'sum'),
            p_diff_is_not_significant_count=('p_diff_is_significant', lambda x: (~x).sum()),
            ci_diff_is_significant_count=('ci_diff_is_significant', 'sum'),
            ci_diff_is_not_significant_count=('ci_diff_is_significant', lambda x: (~x).sum()),
        ))
        agg.to_excel(settings.RESULTS_BASE_DIR / 'compare_roc_mean.xlsx', index=False)
