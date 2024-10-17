import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline


class DataAnalysis(object):

    def __init__(self, csv_filepath: str, *args, **kwargs) -> None:
        self.title: str = csv_filepath.split('\\')[-1].split('/')[-1]
        self.df: pd.DataFrame = pd.read_csv(csv_filepath, *args, **kwargs)
        self.df_types: pd.DataFrame | None = None

    def show_basics(self) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, self.title, '=' * 5)
        print(pd.concat([df.head(2), df.tail(2)]).to_string())
        print('=' * 5, 'DESCRIBE', '=' * 5)
        print(df.describe().to_string())
        print('=' * 5, 'INFO', '=' * 5)
        df.info()
        self.df: pd.DataFrame = df

    def any_na(self) -> bool:
        df: pd.DataFrame = self.df
        any_na = 0 < df.isna().sum().sum()
        self.df: pd.DataFrame = df
        return any_na

    def handle_na(self, column: str, method: str) -> None:
        df: pd.DataFrame = self.df
        method = method.upper()
        if 'DROP' == method:
            df = df.dropna(subset=column, ignore_index=True)
        elif 'MODE' == method:
            df[column] = df[column].fillna(df[column].mode()[0])
        elif 'MEDIAN' == method:
            df[column] = df[column].fillna(df[column].median())
        elif 'MEAN' == method:
            df[column] = df[column].fillna(df[column].mean())
        elif 'BFILL' == method:
            df[column] = df[column].fillna(method='bfill')
        elif 'FFILL' == method:
            df[column] = df[column].fillna(method='ffill')
        self.df: pd.DataFrame = df

    def handle_duplicates(self) -> None:
        df: pd.DataFrame = self.df
        df = df.drop_duplicates(ignore_index=True)
        self.df: pd.DataFrame = df

    def analyse_columns(self) -> None:
        df: pd.DataFrame = self.df
        if self.df_types is None:
            n = len(df.columns)
            type_ls = [pd.NA] * n
            mode_ls = [pd.NA] * n
            median_ls = [pd.NA] * n
            mean_ls = [pd.NA] * n
            var_ls = [pd.NA] * n
            std_ls = [pd.NA] * n
            kurt_ls = [pd.NA] * n
            skew_ls = [pd.NA] * n
            for i, column in enumerate(df.columns):
                df_col = df[column]
                if pd.api.types.is_bool_dtype(df_col.dtype) or pd.api.types.is_object_dtype(df_col.dtype):
                    type_ls[i] = 'nominal'.upper()
                    mode_ls[i] = df_col.mode()[0]
                elif pd.api.types.is_datetime64_any_dtype(df_col.dtype) or df_col.nunique() / len(df_col) < 0.05:
                    type_ls[i] = 'ordinal'.upper()
                    mode_ls[i] = df_col.mode()[0]
                    median_ls[i] = df_col.median()
                else:
                    type_ls[i] = 'interval_or_ratio'.upper()
                    median_ls[i] = df_col.median()
                    mean_ls[i] = df_col.mean()
                    var_ls[i] = df_col.var()
                    std_ls[i] = df_col.std()
                    kurt_ls[i] = df_col.kurt()
                    skew_ls[i] = df_col.skew()
            self.df_types: pd.DataFrame = pd.DataFrame({
                'name': df.columns.values,
                'dtype': df.dtypes.values,
                'type': type_ls,
                'mode': mode_ls,
                'median': median_ls,
                'mean': mean_ls,
                'var': var_ls,
                'std': std_ls,
                'kurt': kurt_ls,
                'skew': skew_ls,
            }, index=df.columns)
        print('=' * 5, 'ANALYSE COLUMNS', '=' * 5)
        print(self.df_types.to_string())
        self.df: pd.DataFrame = df

    def sns_barplot(self, *args, **kwargs) -> None:
        df: pd.DataFrame = self.df
        plt.figure()
        sns.barplot(df, *args, **kwargs)
        plt.show()
        self.df: pd.DataFrame = df

    def sns_boxplot(self, *args, **kwargs) -> None:
        df: pd.DataFrame = self.df
        plt.figure()
        sns.boxplot(df, *args, **kwargs)
        plt.show()
        self.df: pd.DataFrame = df

    def sns_histplot(self, *args, **kwargs) -> None:
        df: pd.DataFrame = self.df
        plt.figure()
        sns.histplot(df, *args, **kwargs)
        plt.show()
        self.df: pd.DataFrame = df

    def sns_scatterplot(self, *args, **kwargs) -> None:
        df: pd.DataFrame = self.df
        plt.figure()
        sns.scatterplot(df, *args, **kwargs)
        plt.show()
        self.df: pd.DataFrame = df

    def sns_stripplot(self, *args, **kwargs) -> None:
        df: pd.DataFrame = self.df
        plt.figure()
        sns.stripplot(df, *args, **kwargs)
        plt.show()
        self.df: pd.DataFrame = df

    # def plt_show(self, *args, **kwargs) -> None:
    #     df: pd.DataFrame = self.df
    #     plt.show(*args, **kwargs)
    #     self.df: pd.DataFrame = df

    def check_normality(self, column: str) -> None:
        df: pd.DataFrame = self.df
        df_col = df[column]
        if len(df_col) <= 2000:
            print('=' * 5, 'Shapiro-Wilk Normality Test', '=' * 5)
            stat, p_value = stats.shapiro(df_col)
            print('Statistic:', stat)
            print('P-value:', p_value)
            if p_value < 0.05:
                print(f"'{column}' is not normally distributed (at the {5}% significance level).")
            else:
                print(f"'{column}' is normally distributed (at the {5}% significance level).")
        else:
            print('=' * 5, 'Anderson-Darling Normality Test', '=' * 5)
            anderson_result = stats.anderson(df_col, dist='norm')
            print('Statistic:', anderson_result.statistic)
            for sig_level, crit_value in zip(anderson_result.significance_level, anderson_result.critical_values):
                print(f"Significance level {sig_level}%: {crit_value}")
            if anderson_result.critical_values[2] < anderson_result.statistic:
                print(f"'{column}' is not normally distributed (at the {5}% significance level).")
            else:
                print(f"'{column}' is normally distributed (at the {5}% significance level).")
        sm.qqplot(df_col, line='s')
        plt.show()
        self.df: pd.DataFrame = df

    def anova(self, column: str, groupby: str) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'ANOVA', '=' * 5)
        stat, p_value = stats.f_oneway(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df: pd.DataFrame = df

    def kruskal_wallis(self, column: str, groupby: str) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'Kruskal-Wallis', '=' * 5)
        stat, p_value = stats.kruskal(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df: pd.DataFrame = df

    def t_test(self, column: str, groupby: str) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'T-Test', '=' * 5)
        stat, p_value = stats.ttest_ind(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df: pd.DataFrame = df

    def mann_whitney_u_test(self, column: str, groupby: str) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'Mann-Whitney U Test', '=' * 5)
        stat, p_value = stats.mannwhitneyu(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df: pd.DataFrame = df

    def chi_square_test(self, column: str, groupby: str) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'Chi-Square Test', '=' * 5)
        contingency_table = pd.crosstab(df[column], df[groupby])
        stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        print(f"dof: {dof}")
        print(f"expected: {expected}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df: pd.DataFrame = df

    def regression(self, column_x, column_y) -> None:
        df: pd.DataFrame = self.df
        print('=' * 5, 'Regression', '=' * 5)
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[column_x], df[column_y])
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value ** 2}")
        print(f"P-value: {p_value}")
        print(f"Standard error: {std_err}")
        plt.figure()
        sns.scatterplot(df[column_x], df[column_y])
        plt.plot(df[column_x], intercept + slope * df[column_x], 'r', label='Fitted line')
        plt.show()
        self.df: pd.DataFrame = df

    def show_text_columns(self) -> list[str]:
        df: pd.DataFrame = self.df
        columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
        print('=' * 5, 'Text Columns', '=' * 5)
        print(pd.DataFrame(
            {
                'Column Name': columns,
                'Average Entry Length': [df[col].apply(len).mean() for col in columns],
                'Unique Entries': [df[col].nunique() for col in columns],
            }
        ))
        self.df: pd.DataFrame = df
        return columns

    def vader_sentiment_analysis(self, column: str) -> None:
        df: pd.DataFrame = self.df
        vader_analyzer = SentimentIntensityAnalyzer()
        res = df[column].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
        print('=' * 5, 'Vader Sentiment Analysis', '=' * 5)
        print(
            res,
            res.apply(lambda x: 'positive' if x >= +0.05 else 'negative' if x <= -0.05 else 'neutral'),
            sep='\n',
        )
        self.df: pd.DataFrame = df

    def textblob_sentiment_analysis(self, column: str) -> None:
        df: pd.DataFrame = self.df
        res = df[column].apply(lambda x: TextBlob(x).sentiment)
        print('=' * 5, 'Text-Blob Sentiment Analysis', '=' * 5)
        print(
            res.apply(lambda x: x.polarity),
            res.apply(lambda x: 'positive' if x.polarity > 0 else 'negative' if x.polarity < 0 else 'neutral'),
            res.apply(lambda x: x.subjectivity),
            sep='\n',
        )
        self.df: pd.DataFrame = df

    def distilbert_sentiment_analysis(self, column: str) -> None:
        df: pd.DataFrame = self.df
        sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        res = df[column].apply(lambda x: sentiment_pipeline(x)[0])
        print('=' * 5, 'Distilbert Sentiment Analysis', '=' * 5)
        print(
            res.apply(lambda x: x['score']),
            res.apply(lambda x: int(x['label'].split(' ')[0])).apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral'),
            sep='\n',
        )
        self.df: pd.DataFrame = df


if __name__ == '__main__':
    csv_filepath = r'C:\rocky_d\code\scse\mini-project\Dataset-Group_1_Pandas\Netflix Movies and TV Shows\netflix_titles_cleaned.csv'
    analysis = DataAnalysis(csv_filepath)
    analysis.show_basics()
    analysis.analyse_columns()
