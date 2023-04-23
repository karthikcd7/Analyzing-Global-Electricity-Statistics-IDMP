import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")

'''
Electricity Produced by Renewable Energy Resources each Year (2015-2022)
'''
df_renewable = df[(df['Product'] == 'Total Renewables (Geo, Solar, Wind, Other)')]
df_renewable_by_year = df_renewable.groupby(df_renewable.columns.str[-4:], axis=1).sum()
fig1, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(df_renewable_by_year.iloc[:-1].sum()[:8], '-o')
ax1.set_title('Total Electricity Produced by Renewable Energy each Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('GWh')
# plt.show()
fig1.savefig('output/renewableByYear.png')


'''
Electricity Produced by Non-Renewable Energy Resources each Year (2015-2022)
'''
df_non_renewable = df[df['Product'].isin(
    ['Coal, Peat and Manufactured Gases', 'Natural Gas', 'Oil and Petroleum Products', 'Nuclear',
     'Other Combustible Non-Renewables'])]
df_non_renewable_by_country = df_non_renewable.groupby("Country").sum(numeric_only=True)
df_non_renewable_by_year = df_non_renewable_by_country.groupby(
    df_non_renewable_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(df_non_renewable_by_year.iloc[:-1].sum()[:8], '-o')
ax2.set_title('Total Electricity Produced by Non-Renewable Energy each Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('GWh')
# plt.show()
fig2.savefig('output/non-renewableByYear.png')


'''
Electricity production by non-renewable fuel in a stacked bar plot
'''
df_non_renewable = df[df['Product'].isin(
    ['Coal, Peat and Manufactured Gases', 'Natural Gas', 'Oil and Petroleum Products', 'Nuclear',
     'Other Combustible Non-Renewables'])]
df_electricity_consumption = df_non_renewable.groupby("Product").sum(numeric_only=True)
df_electricity_consumption_by_year = df_electricity_consumption.groupby(
    df_electricity_consumption.columns.str[-4:], axis=1).sum(numeric_only=True)

df_electricity_consumption_by_year.drop(df_electricity_consumption_by_year.columns[-2:], axis=1, inplace=True)
df_electricity_consumption_by_year = pd.DataFrame(df_electricity_consumption_by_year)
df_electricity_consumption_by_year = df_electricity_consumption_by_year.T

fig3, ax3 = plt.subplots(figsize=(11, 5))
ax3 = df_electricity_consumption_by_year.plot(kind='bar', stacked=True, figsize=(10, 6))
ax3.legend(fontsize = 6)
ax3.set_title('Electricity Production by Non-Renewable Fuel Type')
ax3.set_xlabel('Year')
ax3.set_ylabel('Electricity Production (GWh)')
# plt.show()
fig3.savefig('output/nonRenElectricityProd.png')


'''
Electricity Produced by Non-Renewable Fuels each Year (2010 - 2022)
'''
df_coal_production = df[df['Product'].isin(['Coal, Peat and Manufactured Gases'])]
df_coal_production_by_country = df_coal_production.groupby("Country").sum(numeric_only=True)
df_coal_production_by_year = df_coal_production_by_country.groupby(
    df_coal_production_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

df_natural_gas = df[df['Product'].isin(['Natural Gas'])]
df_natural_gas_by_country = df_natural_gas.groupby("Country").sum(numeric_only=True)
df_natural_gas_by_year = df_natural_gas_by_country.groupby(
    df_natural_gas_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

df_fossil_fuel = df[df['Product'].isin(['Oil and Petroleum Products'])]
df_fossil_fuel_by_country = df_fossil_fuel.groupby("Country").sum(numeric_only=True)
df_fossil_fuel_by_year = df_fossil_fuel_by_country.groupby(
    df_fossil_fuel_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

fig4, ax4 = plt.subplots(figsize=(11, 5))
ax4.plot(df_coal_production_by_year.iloc[:-1].sum()[:13], '-o', label='Coal')
ax4.plot(df_natural_gas_by_year.iloc[:-1].sum()[:13], '-o', label='Natural Gas')
ax4.plot(df_fossil_fuel_by_year.iloc[:-1].sum()[:13], '-o', label='Fossil Fuels')
ax4.set_title('Total Electricity Produced by Non-Renewable Fuels each Year (2010- 2022)')
ax4.set_xlabel('Year')
ax4.set_ylabel('Electricity Produced (GWh)')
ax4.legend(bbox_to_anchor=(0.96, 1), fontsize = 6)
# plt.show()
fig4.savefig('output/nonRenewableProductionByYear.png')



'''
Electricity production by renewable fuel in a stacked bar plot
'''
df_non_renewable = df[df['Product'].isin(
    ['Wind', 'Solar', 'Hydro', 'Geothermal', 'Other Renewables', 'Combustible Renewables'])]
df_electricity_consumption = df_non_renewable.groupby("Product").sum(numeric_only=True)
df_electricity_consumption_by_year = df_electricity_consumption.groupby(
    df_electricity_consumption.columns.str[-4:], axis=1).sum(numeric_only=True)

df_electricity_consumption_by_year.drop(df_electricity_consumption_by_year.columns[-2:], axis=1, inplace=True)
df_electricity_consumption_by_year = pd.DataFrame(df_electricity_consumption_by_year)
df_electricity_consumption_by_year = df_electricity_consumption_by_year.T

fig5, ax5 = plt.subplots(figsize=(11, 5))
ax5 = df_electricity_consumption_by_year.plot(kind='bar', stacked=True, figsize=(10, 6))
ax5.legend(fontsize = 8)
ax5.set_title('Electricity Production by Renewable Fuel Type')
ax5.set_xlabel('Year')
ax5.set_ylabel('Electricity Production (GWh)')
# plt.show()
fig5.savefig('output/renElectricityProd.png')


'''
Electricity Produced by Renewable Fuels each Year (2010- 2022)
'''
df_wind_production = df[df['Product'].isin(['Wind'])]
df_wind_production_by_country = df_wind_production.groupby("Country").sum(numeric_only=True)
df_wind_production_by_year = df_wind_production_by_country.groupby(
    df_wind_production_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

df_hydro = df[df['Product'].isin(['Hydro'])]
df_hydro_by_country = df_hydro.groupby("Country").sum(numeric_only=True)
df_hydro_by_year = df_hydro_by_country.groupby(
    df_hydro_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

df_solar = df[df['Product'].isin(['Solar'])]
df_solar_by_country = df_solar.groupby("Country").sum(numeric_only=True)
df_solar_by_year = df_solar_by_country.groupby(
    df_solar_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

df_combustible = df[df['Product'].isin(['Combustible Renewables'])]
df_combustible_by_country = df_combustible.groupby("Country").sum(numeric_only=True)
df_combustible_by_year = df_combustible_by_country.groupby(
    df_combustible_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)

fig6, ax6 = plt.subplots(figsize=(11, 5))
ax6.plot(df_wind_production_by_year.iloc[:-1].sum()[:13], '-o', label='Wind')
ax6.plot(df_hydro_by_year.iloc[:-1].sum()[:13], '-o', label='Hydro')
ax6.plot(df_solar_by_year.iloc[:-1].sum()[:13], '-o', label='Solar')
ax6.plot(df_combustible_by_year.iloc[:-1].sum()[:13], '-o', label='Combustible Renewables')
ax6.set_title('Total Electricity Produced by Renewable Fuels each Year (2010- 2022)')
ax6.set_xlabel('Year')
ax6.set_ylabel('Electricity Produced (GWh)')
ax6.legend(bbox_to_anchor=(0.96, 1), fontsize = 6)
# plt.show()
fig6.savefig('output/renewableProductionByYear.png')


'''
Total Net Electricity Production each Year (2010- 2022)
'''
df = pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")
df = df.dropna()
df_net = df[(df['Product'] == 'Electricity') & (df['Balance'] == 'Net Electricity Production')]
df_net_by_country = df_net.groupby("Country").sum(numeric_only=True)
df_net_production_by_year = df_net_by_country.groupby(
    df_net_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)
df_net_production_by_year = df_net_production_by_year.drop(df_net_production_by_year.columns[-1], axis=1)
df_net_production_by_year.to_csv("monthlutable_1.csv")

fig7, ax7 = plt.subplots(figsize=(11, 5))

x = df_net_production_by_year.iloc[:-1].sum()[:13].index
y = df_net_production_by_year.iloc[:-1].sum()[:13].values/1e7

# plot the scatter plot with lines joining the points
ax7.plot(x, y, marker='o', linestyle='-', color='b')

ax7.set_title('Total Net Electricity Production each Year')
ax7.set_xlabel('Year')
ax7.set_ylabel('GWh (x10$^7$)')
# plt.show()
fig7.savefig('output/coal_naturalGasProductionByYear.png')



'''
Total Distribution losses each Year (2010- 2022)
'''
df = pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")
df=df.dropna()
df_loss = df[(df['Balance'] == 'Distribution Losses')]
df_loss_by_country = df_loss.groupby("Country").sum(numeric_only=True)
df_loss_by_year = df_loss_by_country.groupby(
    df_loss_by_country.columns.str[-4:], axis=1).sum(numeric_only=True)
df_loss_by_year = df_loss_by_year.drop(df_loss_by_year.columns[-1], axis=1)


fig8, ax8 = plt.subplots(figsize=(11, 5))
ax8.plot(df_loss_by_year.iloc[:-1].sum()[:13], '-o')
ax8.set_title(' Total distribution loss each year')
ax8.set_xlabel('Year')
ax8.set_ylabel('GWh')
# plt.show()
fig8.savefig('output/coal_naturalGasProductionByYear.png')


'''
GDP vs Electricity for APAC(2010- 2022)
'''
df_electricity = pd.read_csv('data/monthlutable_1.csv')
df_gdp = pd.read_csv('data/TIDY_GDP.csv')
APAC=["Australia", "India", "Japan","New Zealand"]

df_electricity = df_electricity[df_electricity['Country'].isin(APAC)]
df_gdp = df_gdp[df_gdp['Country'].isin(APAC)]
# Melt the DataFrames to convert them from wide to long format
df_electricity_melt = pd.melt(df_electricity, id_vars=['Country'], var_name='Year', value_name='Electricity')
df_gdp_melt = pd.melt(df_gdp, id_vars=['Country'], var_name='Year', value_name='GDP')

# Merge the two DataFrames on 'Country' and 'Year' columns
df_combined = pd.merge(df_electricity_melt, df_gdp_melt, on=['Country', 'Year'])

# Set the 'Country' and 'Year' columns as the index
df_combined.set_index(['Country', 'Year'], inplace=True)

# Plot the data
fig9, ax9 = plt.subplots()

for country in APAC:
    df = df_combined.loc[country]
    ax9.plot(df['GDP'], df['Electricity'], label=country, marker='o')

ax9.set_xlabel('GDP')
ax9.set_ylabel('Electricity')
ax9.legend()
ax9.set_title('Net Electricity Production vs. GDP for APAC Countries')

# plt.show()




'''
GDP vs Electricity for EMEA(2010- 2022)
'''

# Read the csv files
df_electricity = pd.read_csv('data/monthlutable_1.csv')
df_gdp = pd.read_csv('data/TIDY_GDP.csv')
Eastern_Europe=[ "Poland", "Serbia"]

Western_Europe=["Austria", "Belgium", "France", "Germany", "Netherlands"]

Northern_Europe=["Finland", "Iceland"]

Southern_Europe=["Croatia", "Cyprus", "Greece", "Italy", "Portugal"]

Central_Europe=["Hungary", "Ireland", "United Kingdom"]
countries=Eastern_Europe+Western_Europe+Northern_Europe+Southern_Europe

df_electricity = df_electricity[df_electricity['Country'].isin(countries)]
df_gdp = df_gdp[df_gdp['Country'].isin(countries)]
# Melt the DataFrames to convert them from wide to long format
df_electricity_melt = pd.melt(df_electricity, id_vars=['Country'], var_name='Year', value_name='Electricity')
df_gdp_melt = pd.melt(df_gdp, id_vars=['Country'], var_name='Year', value_name='GDP')

# Merge the two DataFrames on 'Country' and 'Year' columns
df_combined = pd.merge(df_electricity_melt, df_gdp_melt, on=['Country', 'Year'])

# Set the 'Country' and 'Year' columns as the index
df_combined.set_index(['Country', 'Year'], inplace=True)

# Plot the data
fig10, ax10 = plt.subplots()

for country in countries:
    df = df_combined.loc[country]
    ax10.plot(df['GDP'], df['Electricity'], label=country, marker='o')

ax10.set_xlabel('GDP')
ax10.set_ylabel('Electricity')

ax10.legend(fontsize=8)
ax10.set_title('Net Electricity Production vs. GDP for EMEA Countries')

# plt.show()
fig10.savefig("output/GDP_ELEC_EMEA_IMG.png")



'''
net electricity production Scatter plot for APAC countries(2015- 2022)
'''
df = pd.read_csv("data/monthlutable.csv")
APAC=["Australia", "India", "Japan", "New Zealand"]
df = df[df['Country'].isin(APAC)]
# set Country as the index
df = df.set_index("Country")

# create a Year column based on the column names
df = df.rename(columns=lambda x: int(x) if x.isnumeric() else x)
df = df.reset_index()
df = pd.melt(df, id_vars=["Country"], var_name="Year", value_name="Electricity")
df["Year"] = df["Year"].astype(int)

# create a colormap
cmap = cm.get_cmap("tab20")

# plot the data as a scatter plot with colors
fig11, ax11 = plt.subplots(figsize=(10, 6))
for i, (country, group) in enumerate(df.groupby("Country")):
    color = cmap(i % 20)
    rgba = color[:3] + (1,)  # Set alpha channel to 0.7 for a brighter point
    group.plot.scatter(x="Year", y="Electricity", color=rgba, ax=ax11, label=country)

# set the axis labels
ax11.set_xlabel("Year")
ax11.set_ylabel("Electricity")

# set the title
ax11.set_title("Net Electricity Production for each APAC country over the years")

# set the legend outside the plot
ax11.legend(bbox_to_anchor=(0.96, 1), loc="upper left")

plt.savefig("output/electricity_plot.png", dpi=300, bbox_inches="tight")
# plt.show()


'''
net electricity production Scatter plot for EMEA countries(2015- 2022)
'''

df = pd.read_csv("data/monthlutable.csv")
Eastern_Europe=[ "Poland","Serbia"]

Western_Europe=["Austria", "Belgium", "France", "Germany", "Netherlands"]

Northern_Europe=["Finland", "Iceland"]

Southern_Europe=["Greece","Cyprus" "Italy", "Portugal", "Croatia"]

Central_Europe=["Hungary", "Ireland", "United Kingdom"]
countries=Eastern_Europe+Western_Europe+Northern_Europe+Southern_Europe


df = df[df['Country'].isin(countries)]
df = df.set_index("Country")

# create a Year column based on the column names
df = df.rename(columns=lambda x: int(x) if x.isnumeric() else x)
df = df.reset_index()
df = pd.melt(df, id_vars=["Country"], var_name="Year", value_name="Electricity")
df["Year"] = df["Year"].astype(int)


cmap = cm.get_cmap("tab20")

fig12, ax12 = plt.subplots(figsize=(10, 6))
for i, (country, group) in enumerate(df.groupby("Country")):
    color = cmap(i % 20)
    rgba = color[:3] + (1,)
    group.plot.scatter(x="Year", y="Electricity", color=rgba, ax=ax12, label=country)

ax12.set_xlabel("Year")
ax12.set_ylabel("Electricity")
ax12.set_title("Net Electricity Production for each EMEA country over the years")
ax12.legend(fontsize=8)
ax12.legend(bbox_to_anchor=(1.15, 1), loc="upper right")
plt.savefig("output/electricity_plot.png", dpi=300, bbox_inches="tight")
# plt.show()

df1=pd.read_csv("data/TIDY_GDP.csv")
df2=pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")

df_renewable = df2[(df2['Product'] == 'Total Renewables (Geo, Solar, Wind, Other)')]
df_renewable_by_year = df_renewable.groupby(df_renewable.columns.str[-4:], axis=1).sum()
data = df_renewable_by_year.iloc[1:-1].sum()[:13]
data = data.to_frame()
data = data.reset_index()
data.columns = ["Year", "Electricity Production"]
growth_rate_electricity = ((data['Electricity Production'] - data['Electricity Production'].shift(1)) / data['Electricity Production'].shift(1)) * 100

data = []
for col in df1: 
    data.append(df1[col].sum())
data = data[2:]
growth_rate_gdp = []
for i in range(1, len(data)):
    growth_rate = ((data[i] - data[i-1]) / data[i-1]) * 100
    growth_rate_gdp.append(growth_rate)

growth_rate_electricity = list(growth_rate_electricity)
growth_rate_electricity = growth_rate_electricity[1:]
plt.figure(figsize=(10,6))
plt.plot(range(2011, 2023), growth_rate_gdp, label="GDP")
plt.plot(range(2011, 2023), growth_rate_electricity, label="Renewable Energy Resources")
plt.legend()
plt.xlabel('Year')
plt.ylabel('Growth rate (%)')
plt.title('Yearly growth rate of GDP vs Electricity produced using Renewable Energy Resouces')
# plt.show()


df_non_renewable = df2[df2['Product'].isin(
    ['Coal, Peat and Manufactured Gases', 'Natural Gas', 'Oil and Petroleum Products', 'Nuclear',
     'Other Combustible Non-Renewables'])]
df_non_renewable_by_year = df_non_renewable.groupby(df_non_renewable.columns.str[-4:], axis=1).sum()
data_non_rene = df_non_renewable_by_year.iloc[1:-1].sum()[:13]
data_non_rene = data_non_rene.to_frame()
data_non_rene = data_non_rene.reset_index()
data_non_rene.columns = ["Year", "Electricity Production"]
growth_rate_electricity_non_rene = ((data_non_rene['Electricity Production'] - data_non_rene['Electricity Production'].shift(1)) / data_non_rene['Electricity Production'].shift(1)) * 100

growth_rate_electricity_non_rene = list(growth_rate_electricity_non_rene)
growth_rate_electricity_non_rene = growth_rate_electricity_non_rene[1:]
plt.figure(figsize=(10,6))
plt.plot(range(2011, 2023), growth_rate_gdp, label="GDP")
plt.plot(range(2011, 2023), growth_rate_electricity_non_rene, label="Non Renewable Energy Resources")
plt.legend()
plt.xlabel('Year')
plt.ylabel('Growth rate (%)')
plt.title('Yearly growth rate of GDP vs Electricity produced using Non-Renewable Energy Resouces')

plt.show()



