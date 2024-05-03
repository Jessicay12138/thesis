
"""
12/12/2023
Calculations of real correlation between stocks and bonds
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# fp = 'Annual_Returns.csv'
fp = "Monthly_Real_Returns.csv"
# fp= "Daily_Log_Returns.csv"

df = pd.read_csv(fp)

bond_mean = 0.024355
equity_mean = 0.073668

# df["Code"] = np.where((df.Real_Annual_Bond<bond_mean) & (df.Real_Annual_Equity<equity_mean), "negative",0)
# df["Code"] = np.where((df.Real_Annual_Bond>bond_mean) & (df.Real_Annual_Equity>equity_mean), "positive", df["Code"] )
# df["Code"] = np.where(((df.Real_Annual_Bond<bond_mean) & (df.Real_Annual_Equity>equity_mean)) | ((df.Real_Annual_Bond>bond_mean) & (df.Real_Annual_Equity<equity_mean)), "opposite", df["Code"] )

# both_positive = df[df["Code"] == "positive"]
# both_negative = df[df["Code"] == "negative"]
# opposite = df[df["Code"] == "opposite"]

# both_positive.to_csv("both_positive.csv")
# both_negative.to_csv("both_negative.csv")
# opposite.to_csv("opposite.csv")

# # df.set_index('Date', inplace=True)
# # rolling_corr = df['Real_Monthly_Equity'].rolling(window=36).corr(df['Real_Monthly_Bond'].shift(1))
# # rolling_corr.to_csv("3_month_corr.csv")

date_column = 'Date'
bond = df['Real_Monthly_Bond']
equity = df['Real_Monthly_Equity']

df[date_column] = pd.to_datetime(df['Date'])
df['Year'] = df[date_column].dt.year
df['Month'] = df[date_column].dt.month
df.set_index('Date', inplace=True)
# print(df)
# rolling_corr = df['Real_Monthly_Equity'].rolling(window=1).corr(df['Real_Monthly_Bond'].shift(1))
# print(rolling_corr)
# plt.plot(df["Date"], df['Correlation'])
# plt.title("Rolling 3-Year Correlation by Month")
# plt.ylabel("Correlation")
# plt.xlabel("Year")
# plt.show()

correlations_by_year = {}
for year in df['Year'].unique():
    year_data = df[df['Year'] == year]
    correlation = year_data['Real_Monthly_Bond'].corr(year_data['Real_Monthly_Equity'])
    correlations_by_year[year] = correlation

correlations = pd.DataFrame(list(correlations_by_year.items()), columns=['Year', 'Correlation'])
correlations["Correlation_Values"]  = pd.Series(correlations_by_year.values()).rolling(2).mean()
plt.plot(correlations_by_year.keys(), correlations["Correlation_Values"]  )
plt.title("Rolling 2-Year Correlation")
plt.axhline(0, color='black', linestyle='--')
plt.ylabel("Correlation")
plt.xlabel("Year")
plt.show()

# correlations = pd.DataFrame(list(correlations_by_year.items()), columns=['Year', 'Correlation'])
correlations["Correlation"] = np.where((correlations["Correlation_Values"] < 0), "negative","positive")
# correlations["Correlation"] = np.where((correlations["Correlation"] > 0), "positive",0)

# print(correlations)
annual_fp = 'Annual_Returns.csv'
annual_df = pd.read_csv(annual_fp)
df = pd.merge(correlations, annual_df, left_on="Year", right_on="Date", how='inner')
# print(merged_df)
plt.plot(df["Year"], df["Correlation_Values"], label="Correlation"  )
plt.plot(df["Year"], df["Real_Annual_Equity"], color='navy', label='Equity')
plt.plot(df["Year"], df["Real_Annual_Bond"], color='mediumslateblue', label='Bond')
plt.legend()
plt.title("Rolling 2-Year Correlation")
plt.ylabel("Correlation")
plt.xlabel("Year")
plt.show()

both_positive = df[df["Correlation"] == "positive"]
both_negative = df[df["Correlation"] == "negative"]
# plt.scatter(both_positive['Real_Annual_Bond'][both_positive['Real_Annual_Bond']>0], both_positive['Real_Annual_Equity'][both_positive['Real_Annual_Equity']>0], color="red")

both_positive.to_csv("positive.csv")
both_negative.to_csv("negative.csv")

# df["Code"] = np.where((df.Real_Annual_Bond<bond_mean) & (df.Real_Annual_Equity<equity_mean), "negative",0)
# df["Code"] = np.where((df.Real_Annual_Bond>bond_mean) & (df.Real_Annual_Equity>equity_mean), "positive", df["Code"] )
# df["Code"] = np.where(((df.Real_Annual_Bond<bond_mean) & (df.Real_Annual_Equity>equity_mean)) | ((df.Real_Annual_Bond>bond_mean) & (df.Real_Annual_Equity<equity_mean)), "opposite", df["Code"] )

# both_positive = df[df["Code"] == "positive"]
# both_negative = df[df["Code"] == "negative"]
# opposite = df[df["Code"] == "opposite"]

# both_positive.to_csv("both_positive.csv")
# both_negative.to_csv("both_negative.csv")
# opposite.to_csv("opposite.csv")

num_positive = len(both_positive)
num_negative = len(both_negative)
# num_opposite = len(opposite)
total = num_positive + num_negative #+ num_opposite

print("Number of both above mean", num_positive)
print("Number of both below mean", num_negative)
# # print("Number of opposite directions", num_opposite)

print("Raw probabilities of positive", num_positive/total)
print("Raw probabilities of negative", num_negative/total)
# # print("Raw probabilities of opposite", num_opposite/total)

print("Overall Bond Mean", df["Real_Annual_Bond"].mean())
print("Positive Bond Mean", both_positive["Real_Annual_Bond"].mean())
print("Negative Bond Mean", both_negative["Real_Annual_Bond"].mean())
# # print("opposite Bond Mean", opposite["Real_Annual_Bond"].mean())

print("Overall Equity Mean", df["Real_Annual_Equity"].mean())
print("Positive Equity Mean", both_positive["Real_Annual_Equity"].mean())
print("Negative Equity Mean", both_negative["Real_Annual_Equity"].mean())
# # print("opposite Equity Mean", opposite["Real_Annual_Equity"].mean())

print("Overall Riskless Mean", df["Real_Annual_Riskless"].mean())
print("Positive Riskless Mean", both_positive["Real_Annual_Riskless"].mean())
print("Negative Riskless Mean", both_negative["Real_Annual_Riskless"].mean())

print("Overall Bond SD", df["Real_Annual_Bond"].std())
print("Positive Bond SD", both_positive["Real_Annual_Bond"].std())
print("Negative Bond SD", both_negative["Real_Annual_Bond"].std())
# # print("opposite Bond SD", opposite["Real_Annual_Bond"].std())

print("Overall Equity SD", df["Real_Annual_Equity"].std())
print("Positive Equity SD", both_positive["Real_Annual_Equity"].std())
print("Negative Equity SD", both_negative["Real_Annual_Equity"].std())
# # print("opposite Equity SD", opposite["Real_Annual_Equity"].std())

print("Overall Riskless SD", df["Real_Annual_Riskless"].std())
print("Positive Riskless SD", both_positive["Real_Annual_Riskless"].std())
print("Negative RisklessSD", both_negative["Real_Annual_Riskless"].std())

print("Overall correlation", df["Real_Annual_Bond"].corr(df["Real_Annual_Equity"]))
print("Positive correlation", both_positive["Real_Annual_Bond"].corr(both_positive["Real_Annual_Equity"]))
print("Negative correlation", both_negative["Real_Annual_Bond"].corr(both_negative["Real_Annual_Equity"]))
# print(opposite["Real_Annual_Bond"].corr(opposite["Real_Annual_Equity"]))

def markov_chain(correlation_list):
    positive_to_positive = 0
    positive_to_negative = 0
    positive_to_opposite = 0 
    negative_to_negative = 0
    negative_to_positive = 0
    negative_to_opposite = 0
    opposite_to_positive = 0
    opposite_to_negative = 0
    opposite_to_opposite = 0
    total = 0

    for i in range(1,len(correlation_list)):
        if correlation_list[i - 1] == "positive" and correlation_list[i] == "positive":
            positive_to_positive += 1
        elif correlation_list[i - 1] == "positive" and correlation_list[i] == "negative":
            positive_to_negative += 1
        elif correlation_list[i - 1] == "positive" and correlation_list[i] == "opposite":
            positive_to_opposite += 1
        elif correlation_list[i - 1] == "negative" and correlation_list[i] == "positive":
            negative_to_positive += 1
        elif correlation_list[i - 1] == "negative" and correlation_list[i] == "negative":
            negative_to_negative += 1
        elif correlation_list[i - 1] == "negative" and correlation_list[i] == "opposite":
            negative_to_opposite += 1
        elif correlation_list[i - 1] == "opposite" and correlation_list[i] == "positive":
            opposite_to_positive += 1
        elif correlation_list[i - 1] == "opposite" and correlation_list[i] == "negative":
            opposite_to_negative += 1
        elif correlation_list[i - 1] == "opposite" and correlation_list[i] == "opposite":
            opposite_to_opposite += 1
        total += 1
    
    prob_pos_pos = positive_to_positive / (positive_to_negative+positive_to_positive+positive_to_opposite)
    prob_pos_neg = positive_to_negative / (positive_to_negative+positive_to_positive+positive_to_opposite)
    prob_pos_opp = positive_to_opposite / (positive_to_negative+positive_to_positive+positive_to_opposite)
    print("Positive to positive, negative, opposite", prob_pos_pos, prob_pos_neg, prob_pos_opp)
    prob_neg_pos = negative_to_positive / (negative_to_positive+negative_to_negative+negative_to_opposite)
    prob_neg_neg = negative_to_negative /(negative_to_positive+negative_to_negative+negative_to_opposite)
    prob_neg_opp = negative_to_opposite /(negative_to_positive+negative_to_negative+negative_to_opposite)
    print("Negative to positive, negative, opposite", prob_neg_pos, prob_neg_neg, prob_neg_opp)
    # prob_opp_pos = opposite_to_positive / (opposite_to_positive+opposite_to_negative+opposite_to_opposite)
    # prob_opp_neg = opposite_to_negative / (opposite_to_positive+opposite_to_negative+opposite_to_opposite)
    # prob_opp_opp = opposite_to_opposite / (opposite_to_positive+opposite_to_negative+opposite_to_opposite)
    # print("Opposite to positive, negative, opposite", prob_opp_pos, prob_opp_neg, prob_opp_opp)
    print("total", total)
    return prob_pos_pos

print("Markov Chain:", markov_chain(df["Correlation"]))
# print(markov_chain(['negative','negative','negative','negative','negative','negative','negative','negative','positive','negative','positive','negative','positive','positive','negative','positive','positive','positive','negative','negative','positive','positive','positive','positive','positive','positive','positive','negative','positive','negative','positive','negative','positive','positive','positive','negative','positive','positive','negative','negative','negative','negative','negative','negative','positive','negative','negative','negative','negative','positive','negative','negative','negative','negative','positive','negative','negative','positive','positive','positive','negative','positive']))
