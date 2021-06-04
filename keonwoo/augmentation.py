import pandas as pd
import numpy as np
from tqdm import tqdm

# slide_ratio가 작을수록, max_seq_len작을수록 더 많이 나눔
def augmentation(df, max_seq_len, slide_ratio=0.2):
    dtype = dict(df.dtypes)
    dtype["aug_idx"] = np.dtype("int64")

    columns = list(df.columns)
    columns.append("aug_idx")

    new = pd.DataFrame(columns=columns).astype(dtype)
    group = df.sort_values(by=["userID", "Timestamp"], axis=0).groupby("userID")
    aug_idx = 0
    for i, (k, g) in enumerate(tqdm(group)):
        t_group = g.sort_values(by=["Timestamp"], axis=0).groupby(
            ["testId"], sort=False
        )
        slide_window = int(len(t_group) * sliding_ratio)
        if slide_window == 0:
            aug_data = t_group.apply(lambda x: x)
            aug_data["aug_idx"] = int(aug_idx)
            new = new.append(aug_data)
            aug_idx += 1
            continue
        if len(t_group) % slide_window == 0:
            aug_cnt = int(len(t_group) / slide_window)
        else:
            aug_cnt = int(len(t_group) // slide_window) + 1
        tmp = pd.DataFrame(columns=columns).astype(dtype)
        pass_last = False
        for j in range(aug_cnt):
            start_idx = j * slide_window
            aug_data = pd.DataFrame(columns=columns).astype(dtype)
            if pass_last:
                break
            for k, (key, t) in enumerate(t_group):
                if k >= start_idx:
                    if len(aug_data) + len(t) > max_seq:
                        break
                    t["aug_idx"] = int(aug_idx)
                    aug_data = aug_data.append(t)
                    if j == aug_cnt - 2 and len(aug_data) > (max_seq // 2):
                        tmp = tmp.append(t)
                    if k == len(t_group) - 1:
                        pass_last = True
            if len(aug_data) < max_seq // 2:
                tmp["aug_idx"] = int(aug_idx)
                aug_data = tmp.append(aug_data)
            new = new.append(aug_data)
            aug_idx += 1
    return new


def test_augmentation(df, max_seq_len):
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    dtype = dict(df_test.dtypes)

    columns = list(df_test.columns)

    new = pd.DataFrame(columns=columns).astype(dtype)

    group = df.groupby("userID")

    for k, g in group:
        if len(g) <= max_seq_len:
            new = new.append(g.iloc[:])
        elif (
            g.iloc[-max_seq_len].problem_number == 1
            and g.iloc[-max_seq_len].testId != g.iloc[-max_seq_len - 1].testId
        ):
            new = new.append(g.iloc[-max_seq_len:])
        elif (
            g.iloc[-max_seq_len + 1].problem_number == 1
            and g.iloc[-max_seq_len + 1].testId != g.iloc[-max_seq_len - 1 + 1].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 1 :])
        elif (
            g.iloc[-max_seq_len + 2].problem_number == 1
            and g.iloc[-max_seq_len + 2].testId != g.iloc[-max_seq_len - 1 + 2].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 2 :])
        elif (
            g.iloc[-max_seq_len + 3].problem_number == 1
            and g.iloc[-max_seq_len + 3].testId != g.iloc[-max_seq_len - 1 + 3].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 3 :])
        elif (
            g.iloc[-max_seq_len + 4].problem_number == 1
            and g.iloc[-max_seq_len + 4].testId != g.iloc[-max_seq_len - 1 + 4].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 4 :])
        elif (
            g.iloc[-max_seq_len + 5].problem_number == 1
            and g.iloc[-max_seq_len + 5].testId != g.iloc[-max_seq_len - 1 + 5].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 5 :])
        elif (
            g.iloc[-max_seq_len + 6].problem_number == 1
            and g.iloc[-max_seq_len + 6].testId != g.iloc[-max_seq_len - 1 + 6].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 6 :])
        elif (
            g.iloc[-max_seq_len + 7].problem_number == 1
            and g.iloc[-max_seq_len + 7].testId != g.iloc[-max_seq_len - 1 + 7].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 7 :])
        elif (
            g.iloc[-max_seq_len + 8].problem_number == 1
            and g.iloc[-max_seq_len + 8].testId != g.iloc[-max_seq_len - 1 + 8].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 8 :])
        elif (
            g.iloc[-max_seq_len + 9].problem_number == 1
            and g.iloc[-max_seq_len + 9].testId != g.iloc[-max_seq_len - 1 + 9].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 9 :])
        elif (
            g.iloc[-max_seq_len + 10].problem_number == 1
            and g.iloc[-max_seq_len + 10].testId != g.iloc[-max_seq_len - 1 + 10].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 10 :])
        elif (
            g.iloc[-max_seq_len + 11].problem_number == 1
            and g.iloc[-max_seq_len + 11].testId != g.iloc[-max_seq_len - 1 + 11].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 11 :])
        elif (
            g.iloc[-max_seq_len + 12].problem_number == 1
            and g.iloc[-max_seq_len + 12].testId != g.iloc[-max_seq_len - 1 + 12].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 12 :])
        elif (
            g.iloc[-max_seq_len + 13].problem_number == 1
            and g.iloc[-max_seq_len + 13].testId != g.iloc[-max_seq_len - 1 + 13].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 14 :])
        elif (
            g.iloc[-max_seq_len + 14].problem_number == 1
            and g.iloc[-max_seq_len + 14].testId != g.iloc[-max_seq_len - 1 + 14].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 15 :])
        elif (
            g.iloc[-max_seq_len + 15].problem_number == 1
            and g.iloc[-max_seq_len + 15].testId != g.iloc[-max_seq_len - 1 + 15].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 16 :])
        elif (
            g.iloc[-max_seq_len + 16].problem_number == 1
            and g.iloc[-max_seq_len + 16].testId != g.iloc[-max_seq_len - 1 + 16].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 17 :])
        elif (
            g.iloc[-max_seq_len + 17].problem_number == 1
            and g.iloc[-max_seq_len + 17].testId != g.iloc[-max_seq_len - 1 + 17].testId
        ):
            new = new.append(g.iloc[-max_seq_len + 18 :])
        else:
            # ERROR 에러 뜬다면 조건문 더 추가해줘야함
            print(g.iloc[-201:-190])
            print("no")
    return new
