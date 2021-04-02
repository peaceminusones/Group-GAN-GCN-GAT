import os
import pandas as pd          # dataframe
from scipy.signal import savgol_filter

df = pd.read_csv("datasets_group/1939_1/out_1939_2.mp4.csv")
f = open('datasets_group/1939_1/test.txt', mode='w')

# 平滑
F = df.values
pedestrain = sorted(set(F[:, 1].astype(int)))
for p in pedestrain:
    print("1:=========================\n")
    print(df[df['track_id']==p])
    choosebytrackid = df[df['track_id']==p]
    x = choosebytrackid.iloc[:,2].values
    y = choosebytrackid.iloc[:,3].values
    if len(x) < 19:
        continue
    sx = savgol_filter(x, 19, 3)
    sy = savgol_filter(y, 19, 3)
    # print(x,'\n',sx)
    list_choosebytrackid = choosebytrackid.index.tolist()
    for i in range(len(list_choosebytrackid)):
        index = list_choosebytrackid[i]
        df.iloc[[index],[2]] = sx[i]
        df.iloc[[index],[3]] = sy[i]

    print("2:=========================\n")
    print(df[df['track_id']==p])
    # break


for i in range(df.shape[0]):
    f.writelines(['{}\t'.format(df.iloc[i][0]), '{}\t'.format(df.iloc[i][1]),'{}\t'.format(df.iloc[i][2]/100),
                  '{}\t'.format(df.iloc[i][3]/100), '{}\n'.format(df.iloc[i][4])])

f.close()