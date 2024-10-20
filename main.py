import time
from dataanalysis import DataAnalysis


def get_col(cols):
    prompt = f"Please select a column (column name or integer in [0, {len(cols) - 1}]) > "
    col_input = input(prompt)
    while not (col_input in cols or col_input.isdigit() and 0 <= int(col_input) < len(cols)):
        col_input = input(prompt)
    return col_input if col_input in cols else cols[int(col_input)]


def main():
    csv_filepath = input('Please enter a CSV file path > ')
    analysis = DataAnalysis(csv_filepath)
    print()
    analysis.show_basics()
    print()

    methods = ['DROP', 'MODE', 'MEDIAN', 'MEAN', 'BFILL', 'FFILL']
    prompt = f"Please select a method in {methods} (in any case of letters) > "
    while analysis.any_na():
        analysis.show_basics()
        print()
        print('Please handle missing values.')
        col = get_col(analysis.df.columns)
        choice = input(prompt)
        while not choice.upper() in methods:
            choice = input(prompt)
        analysis.handle_na(col, choice)
        print(f"Missing values in {col} have been handled.")
        print()

    analysis.show_basics()
    print()
    analysis.handle_duplicates()
    print('Duplicates have been handled.')
    print()

    while True:
        try:
            analysis.show_basics()
            print()
            analysis.analyse_columns()
            print()
            print('0. Exit.')
            print('1. Plot variable distribution.')
            print('2. Check normality.')
            print('3. Conduct ANOVA.')
            print('4. Conduct Kruskal-Wallis.')
            print('5. Conduct T-Test.')
            print('6. Conduct Mann-Whitney U Test.')
            print('7. Conduct Chi-Square Test.')
            print('8. Conduct Regression.')
            print('9. Conduct sentiment analysis.')
            print('10. Call eval().')
            print('11. Call exec().')
            print()
            choice = input('Please choose (integer in [0, 11]) > ')
            while not (choice.isdigit() and 0 <= int(choice) <= 11):
                choice = input('Please choose (integer in [0, 11]) > ')
            choice = int(choice)
            print()
            if 0 == choice:
                break
            elif 1 == choice:
                print('0. Go back.')
                print('1. Plot distribution.')
                print('2. Barplot.')
                print('3. Boxplot.')
                print('4. Histplot.')
                print('5. Scatterplot.')
                print('6. Stripplot.')
                print()
                choice = input('Please choose (integer in [0, 6]) > ')
                while not (choice.isdigit() and 0 <= int(choice) <= 6):
                    choice = input('Please choose (integer in [0, 6]) > ')
                choice = int(choice)
                print()
                if 0 == choice:
                    continue
                elif 1 == choice:
                    print('Column:')
                    col = get_col(analysis.df.columns)
                    analysis.plot_distribution(col)
                elif 2 == choice:
                    print('Column x:')
                    col_x = get_col(analysis.df.columns)
                    print('Column y:')
                    col_y = get_col(analysis.df.columns)
                    analysis.sns_barplot(x=col_x, y=col_y)
                elif 3 == choice:
                    print('Column x:')
                    col_x = get_col(analysis.df.columns)
                    print('Column y:')
                    col_y = get_col(analysis.df.columns)
                    analysis.sns_boxplot(x=col_x, y=col_y)
                elif 4 == choice:
                    print('Column:')
                    col = get_col(analysis.df.columns)
                    analysis.sns_histplot(x=col, kde=True)
                elif 5 == choice:
                    print('Column x:')
                    col_x = get_col(analysis.df.columns)
                    print('Column y:')
                    col_y = get_col(analysis.df.columns)
                    analysis.sns_scatterplot(x=col_x, y=col_y)
                elif 6 == choice:
                    print('Column x:')
                    col_x = get_col(analysis.df.columns)
                    print('Column y:')
                    col_y = get_col(analysis.df.columns)
                    analysis.sns_stripplot(x=col_x, y=col_y)
            elif 2 == choice:
                col = get_col(analysis.df.columns)
                analysis.check_normality(col)
            elif 3 == choice:
                print('Column:')
                col = get_col(analysis.df.columns)
                print('Group by:')
                groupby = get_col(analysis.df.columns)
                analysis.anova(col, groupby)
            elif 4 == choice:
                print('Column:')
                col = get_col(analysis.df.columns)
                print('Group by:')
                groupby = get_col(analysis.df.columns)
                analysis.kruskal_wallis(col, groupby)
            elif 5 == choice:
                print('Column:')
                col = get_col(analysis.df.columns)
                print('Group by:')
                groupby = get_col(analysis.df.columns)
                analysis.t_test(col, groupby)
            elif 6 == choice:
                print('Column:')
                col = get_col(analysis.df.columns)
                print('Group by:')
                groupby = get_col(analysis.df.columns)
                analysis.mann_whitney_u_test(col, groupby)
            elif 7 == choice:
                print('Column:')
                col = get_col(analysis.df.columns)
                print('Group by:')
                groupby = get_col(analysis.df.columns)
                analysis.chi_square_test(col, groupby)
            elif 8 == choice:
                print('Column x:')
                col_x = get_col(analysis.df.columns)
                print('Column y:')
                col_y = get_col(analysis.df.columns)
                analysis.regression(col_x, col_y)
            elif 9 == choice:
                print('Text column:')
                col = get_col(analysis.show_text_columns())
                print('0. Go back.')
                print('1. Vader.')
                print('2. Text-Blob.')
                print('3. Distilbert.')
                print()
                choice = input('Please choose (integer in [0, 3]) > ')
                while not (choice.isdigit() and 0 <= int(choice) <= 3):
                    choice = input('Please choose (integer in [0, 3]) > ')
                choice = int(choice)
                print()
                if 0 == choice:
                    continue
                elif 1 == choice:
                    analysis.vader_sentiment_analysis(col)
                elif 2 == choice:
                    analysis.textblob_sentiment_analysis(col)
                elif 3 == choice:
                    analysis.distilbert_sentiment_analysis(col)
            elif 10 == choice:
                df = analysis.df
                print(eval(input('Code ("df" is available) > ')))
                del df
            elif 11 == choice:
                df = analysis.df
                exec(input('Code ("df" is available) > '))
                del df
        except Exception as e:
            print('\033[91m', end='')
            print(f"Exception caught:")
            print(e)
            print('\033[0m', end='')
        print()
        time.sleep(1)
    print('Goodbye.')
    print()


if __name__ == '__main__':
    main()
