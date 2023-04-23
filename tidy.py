import pandas as pd

df = pd.read_csv("data/MES_012023.csv")
# print(df.head())
df_pivot = df.pivot_table(index=['Country', 'Balance', 'Product'], columns='Time',
                          values=['Value'], aggfunc='first')

df_pivot.columns = [col[-1][:-2] + "20" + col[-1][-2:] for col in df_pivot.columns.values]
df_pivot.columns = pd.to_datetime(df_pivot.columns, format='%b-%Y')
df_pivot = df_pivot[sorted(df_pivot.columns)]
df_pivot.columns = df_pivot.columns.strftime('%B-%Y')
df_pivot = df_pivot.reset_index()


df_electricity = df_pivot.groupby('Product').transform(lambda x: x.fillna(x.mean()))
df_electricity.insert(0, "Product", df_pivot["Product"])
df_electricity.insert(0, "Balance", df_pivot["Balance"])
df_electricity.insert(0, "Country", df_pivot["Country"])
df_electricity = df_electricity.drop(df_pivot[df_pivot['Country'] == 'Costa Rica'].index)
df_electricity = df_electricity.drop(df_pivot[df_pivot['Country'] == 'Malta'].index)
df_electricity.to_csv("data/TIDY_Monthly Electricity Statistics.csv")


df = pd.read_csv("data/GDP-INT-Export.csv")
df['GDP at purchasing power parities (Billion 2015$ PPP)'] = [country.strip() for country in df['GDP at purchasing power parities (Billion 2015$ PPP)']]
df = df.drop(df[~df['GDP at purchasing power parities (Billion 2015$ PPP)'].isin(df_electricity['Country'])].index)
cols_to_keep = [str(i) for i in range(2010,2023)]
cols_to_keep.insert(0,"GDP at purchasing power parities (Billion 2015$ PPP)")
df = df[df.columns.intersection(cols_to_keep)]
cols_to_keep[0]="Country"
df.columns = cols_to_keep
tidy_gdp = df
tidy_gdp.to_csv("data/TIDY_GDP.csv")



