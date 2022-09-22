import argparse
from Prediction import *
from DataSelector import *
import pandas as pd

# Usage: python gen_triplet.py --csv_dir ../datalist/utk/UTK_train_coral.csv --tau 6 --sample_per_age 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_dir', type=str)
    parser.add_argument('--tau', type=int)
    parser.add_argument('--sample_per_age', type=int)

    args = parser.parse_args()


    train_df = pd.read_csv(args.csv_dir)
    train_df = train_df.drop(["Unnamed: 0", "database"], axis=1)

    tau = args.tau

    min_age = train_df['age'].min()
    max_age = train_df['age'].max()
    sample_per_age = args.sample_per_age

    # Assume we have all the age in min-max range
    # only go to max_age instead of max_age + 1 because the denominator of p-rank will be zero
    for lb in tqdm(range(min_age, max_age)):
        ub = min(max_age, lb + tau)
        
        min_age_candidate = train_df[train_df['age'] == lb]
        max_age_candidate = train_df[train_df['age'] == ub]
        
        for age in range(lb, ub + 1):
            curr_age_candidate = train_df[train_df['age'] == age]
            if (age == min_age):
                lb_df = min_age_candidate.sample(n=sample_per_age*2)
                curr_df = lb_df.iloc[:int(len(lb_df)/2), :]
                lb_df = lb_df.iloc[int(len(lb_df)/2):, :]
                ub_df = max_age_candidate.sample(n=sample_per_age)
                all_df = lb_df.merge(curr_df, how='cross')
                all_df = all_df.merge(ub_df, how='cross')
            elif (age == max_age):
                lb_df = min_age_candidate.sample(n=sample_per_age)
                ub_df = max_age_candidate.sample(n=sample_per_age*2)
                curr_df = ub_df.iloc[:int(len(ub_df)/2), :]
                ub_df = ub_df.iloc[int(len(ub_df)/2):, :]

                temp_all_df = lb_df.merge(curr_df, how='cross')
                temp_all_df = temp_all_df.merge(ub_df, how='cross')

                all_df = pd.concat([all_df, temp_all_df], ignore_index=True)
            else:
                lb_df = min_age_candidate.sample(n=sample_per_age)
                curr_df = curr_age_candidate.sample(n=sample_per_age)
                ub_df = max_age_candidate.sample(n=sample_per_age)

                temp_all_df = lb_df.merge(curr_df, how='cross')
                temp_all_df = temp_all_df.merge(ub_df, how='cross')

                all_df = pd.concat([all_df, temp_all_df], ignore_index=True)

    all_df.columns = ['filename_lb', 'age_lb', 'filename_curr', 'age_curr', 'filename_ub', 'age_ub']
    all_df['p_rank'] = (all_df['age_curr'] - (all_df['age_ub'] + all_df['age_lb'])/2)/((all_df['age_ub'] - all_df['age_lb'])/2)

    # We are using crop and aligned face [".chip.jpg"]
    all_df['filename_lb'] = all_df['filename_lb'] + ".chip.jpg"
    all_df['filename_curr'] = all_df['filename_curr'] + ".chip.jpg"
    all_df['filename_ub'] = all_df['filename_ub'] + ".chip.jpg"

    all_df.to_csv('pregressor_train_data.csv', index=False)
    print(f'Complete! {len(all_df) was generated}')

