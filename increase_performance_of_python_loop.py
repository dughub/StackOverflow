# "How to the increase performance of a Python loop?"
# link here: https://stackoverflow.com/questions/66744864/how-to-the-increase-performance-of-a-python-loop

# Question:
# I have a DataFrame with almost 14 million rows. I am working with financial options data and ideally I need an
# interest rate (called risk-free rate) for each option according to it’s time to maturity. According to the
# literature I’m following, one way to do this is to get US Treasury Bonds interest rates and, for each option,
# check what is the Treasury Bond rate whose maturity is closest to the time to maturity of the option
# (in absolute terms). To achieve this I created a loop that will fill a Dataframe with those differences.
# My code is far from elegant and it is a bit messy because there are combinations of dates and maturities for which
# there are no rates. Hence the conditionals inside the loop. After the loop is done I can look at what is the
# maturity with the lowest absolute difference and choose the rate for that maturity. The script was taking so long to
# run that I added tqdm to have some kind of feedback of what is happening.
#
# I tried running the code. It will take days to complete and it is slowing down as the iterations increase
# (I know this from tqdm). At first I was adding rows to the differences DataFrame using DataFrame.loc. But as I
# thought that was the reason the code was slowing down over time, I switched to DataFrame.append. The code is still
# slow and slowing down over time.
#
# I searched for a way to increase performance and found this question: How to speed up python loop. Someone suggests
# using Cython but honestly I still consider myself a beginner to Python so from looking at the examples it doesn’t
# seem something trivial to do. Is that my best option? If it takes a lot of time to learn than I can also do what
# others do in the literature and just use the 3-month interest rate for all options. But I would prefer not to go
# there there. Maybe there are other (easy) answers to my problem, please let me know. I include a reproducible code
# example (although with only 2 rows of data):

# Problem Breakdown:
# Loops and if statements with bad DRY and redundancy, not leveraging parallelism of pandas / numpy

# Here is my refactored code:

import pandas as pd
import numpy as np
import time

# SET TIMING TEST PARAMETERS HERE
repetitions = 5
powers = [3, 4, 5, 6, 7, 8, ]  # 9, 10]
sample_sizes = [10 ** p for p in powers] + [14000000]
print(sample_sizes)

# Maturity periods, months and years
month_periods = np.array([1, 2, 3, 6, ], dtype=np.float64)
year_periods = np.array([1, 2, 3, 4, 5, 7, 10, 20, 30, ], dtype=np.float64)

# Create column names for maturities
maturity_cols = [f"month_{m:02.0f}" for m in month_periods] + [f"year_{y:02.0f}" for y in year_periods]

# Normalise months
month_periods = month_periods / 12

# Concatenate into single array
maturities = np.concatenate((month_periods, year_periods))

# Begin timing execution for different sample sizes
time_dict = {}

for n_records in sample_sizes:

    # Create some dummy data
    np.random.seed(seed=42)  # Seed PRN generator
    date_range = pd.date_range(start="2004-01-01", end="2021-01-30", freq='D')  # Dates to sample from

    dates = np.random.choice(date_range, size=n_records, replace=True)
    maturity_times = np.random.random(size=n_records)

    options = pd.DataFrame(list(zip(dates, maturity_times)), columns=['QuoteDate', 'Time_to_Maturity', ])

    times = []
    for t in range(repetitions):

        # Start timer
        t0 = time.time()

        # Create date masks
        after = options['QuoteDate'] >= pd.to_datetime("2008-01-01")
        before = options['QuoteDate'] <= pd.to_datetime("2015-01-01")

        # Combine date masks
        between = after & before

        # Flip date mask
        outside = np.logical_not(between)

        # Select data with masks
        df_outside = options[outside].copy()
        df_between = options[between].copy()

        # Smaller chunks
        df_a = df_between[df_between['Time_to_Maturity'] > 25].copy()
        df_b = df_between[df_between['Time_to_Maturity'] <= 3.5 / 12].copy()
        df_c = df_between[df_between['Time_to_Maturity'] <= 4.5 / 12].copy()
        df_d = df_between[
            (df_between['Time_to_Maturity'] >= 2 / 12) & (df_between['Time_to_Maturity'] <= 4.5 / 12)].copy()

        # For each maturity period, add difference column using different formula
        for i, col in enumerate(maturity_cols):
            # Add a line here for each subset / chunk of data which requires a different formula
            df_a[col] = ((maturities[i] - df_outside['Time_to_Maturity']) + 40).abs()
            df_b[col] = ((maturities[i] - df_outside['Time_to_Maturity']) / 2).abs()
            df_c[col] = (maturities[i] - df_outside['Time_to_Maturity'] + 1).abs()
            df_d[col] = (maturities[i] - df_outside['Time_to_Maturity'] * 0.8).abs()
            df_outside[col] = (maturities[i] - df_outside['Time_to_Maturity']).abs()

        # Concatenate dataframes back to one dataset
        frames = [df_outside, df_a, df_b, df_c, df_d, ]
        output = pd.concat(frames).dropna(how='any')

        # End timer
        t1 = time.time()
        times.append(t1 - t0)

    time_dict[n_records] = times

# Print times
print("| Records | Time (seconds) |")
print("|-|-|")
for k, v in time_dict.items():
    mn = np.mean(v)
    print(f"| {k:,.0f}\t|\t{mn:02.4f} |")
