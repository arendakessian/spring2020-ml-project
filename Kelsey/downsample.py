#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def downsample(df, pct_pos):
    ''' 
    Borrowed heavily from earlier project: https://github.com/kelseymarkey/
    cook-county-mental-health-prediction/blob/master/Final_Data_Prep.py

    takes in df and a percentage from 1 to 50
    samples all label==1 cases, then samples from label==0 cases 
    until downsampled_df has pct_pos % positive cases
    '''
    # split into df by label
    label_1 = df[df['label'] == 1]
    label_0 = df[df['label'] == 0]

    #count number of pos
    count_label_1 = len(label_1)

    #compute number of negative cases to sample
    num_label_0 = count_label_1 * int(round((100 - pct_pos) / pct_pos))

    #sample from negative cases
    label_0_sample = label_0.sample(n=num_label_0, random_state=22)

    #append sampled negative cases to all positive cases
    downsampled_df = label_1.append(label_0_sample)

    return downsampled_df