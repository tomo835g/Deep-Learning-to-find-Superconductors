from __future__ import print_function
import argparse
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser(
    description='to predict the critical temperature')
parser.add_argument('--fill', '-fill', type=int, default=0,
                    help='fill empty value in Tc by 0')
parser.add_argument('--year_until', '-year_until', type=int, default=-1,
                    help='supercon data until the year')
parser.add_argument('--dealing_tn', '-dealing_tn', type=float, default=-1,
                    help='to use tn below the temperature or not')
parser.add_argument('--skip_hydride', '-skip_hydride', type=int, default=0,
                    help='skip hydride or not')
args = parser.parse_args()

print("-----------------------------------------------------------------------------")
if args.year_until <= 0:
    print(args.year_until)
    print('the year is not specified')
    # exit()
print('the year is :', args.year_until)

filepath = os.path.join(
    os.getcwd(), 'OXIDEMETALLICSearchR_1_{}.csv'.format(args.year_until))
if args.year_until <= 0:
    filepath = os.path.join(
        os.getcwd(), 'OXIDEMETALLICSearchR_1.csv')

df = pd.read_csv(filepath)
if (args.year_until >= 2018 or args.year_until == -1) and args.skip_hydride == 0:
    hydride = pd.read_csv('./hydride.csv')
    df = pd.concat([hydride, df])
    del hydride
variables = ['element', 'str3', 'Tc', 'tcn']
df = df[variables]
df.columns = ["element", "str3", "Tc", "Tcn"]
# variables = ["element", "str3", "Tc"]
# df = df[variables]
df["element"] = df["element"].astype(str)
print(df.loc[df['Tc'] > 205])
# df = df.loc[df['Tc'] < 400] # must do the last
print(df.loc[df['Tc'] > 205])

print("# Original Data", len(df))
# drop completely duplicated data
# about 3000 data are deleted
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print("remove complete duplications", len(df))

index_list = []
count = 0
filepath = os.path.join(os.getcwd(), 'ending_with_x_y_z_d.csv')

"""Drop if a material ends with x, y, z, d, and . """
# this is 6271 data
for index, row in df.iterrows():
    if row["element"][-1] in {"X", "Y", "Z", "x", "y", "z", "d", "D", "."}:
        # print(row["element"])
        index_list.append(index)
        count += 1
    # if row['Tc'] >= 160:
    #     index_list.append(index)
        # print(row["element"])  # row["Tc"])
# df2 = df[list]
# df2.to_csv(filepath)
# del df2
df = df.drop(index_list)
df = df.reset_index(drop=True)
# print(df)

# print(index_list)
print("The length of the element ending with x,y,z,d is", count)

prior = ""
count = 0
df = df.sort_values(by=["element"])
df = df.reset_index(drop=True)
# print(df)
''' check duplicated element by my way, this will be deleted after duplicated element is replaced by onelement and one median. '''
for index, row in df.iterrows():
    if index > 0 and prior == row["element"] and row["element"][-1] not in {"X", "Y", "Z", "x", "y", "z", "d", "D", "."}:
        '''Do not delete the duplicated element for now'''
        # index_list.append(index)
        # index_list.append(index-1)
        count += 1
        # print(row["element"], row["Tc"], row["str3"])
    prior = row["element"]

print("The number of the duplications", count)

""" what follows is the list of duplicated materials"""
duplicated_element = df.duplicated("element", keep=False)
# print(df[duplicated_element])
filepath = os.path.join(os.getcwd(), 'duplicated_elements.csv')
df2 = df[duplicated_element]
df2 = df2.reset_index()
df2.to_csv(filepath)
del df2

# this is to know Tc is NaN
count = 0
for index, row in df.iterrows():
    if pd.isnull(row["Tc"]):
        # print(row["element"])
        # index_list.append(index)

        count += 1
# at this point 4729 matters has no Tc data reported.
print("# Tc is NaN:", count)

# df = df.drop(index_list)
# print(len(df))
# df = df.reset_index(drop=True)
# print(df)

# this is to know how many Tc are 0
# only 60 data reports Tc=0
count = 0
for index, row in df.iterrows():
    if 0 == row["Tc"]:
        count += 1
print("The number of Tc=0 is\n", count)
# df = df.reset_index(drop=True)
# print(df)

# index_list = []
# for index, row in df.iterrows():
#     if row["element"][-1] in {"X", "Y", "Z", "x", "y", "z"}:
#         print(row["element"])
#         index_list.append(index)
# print(len(index_list))
# df = df.drop(index_list)
# df = df.reset_index(drop=True)
# print(df)


'''If Tc is null and Tn is blow the threshold, then replace Tc with Tn.'''
count_fill_tc_with_tn = 0
tmp = 1e-2
if args.dealing_tn > 0 or args.dealing_tn == True:
    for index, row in df.iterrows():
        if pd.isnull(row["Tc"]) and row["Tcn"] <= args.dealing_tn + tmp:
            row['Tc'] = row['Tcn']
            count_fill_tc_with_tn += 1
print('Replaced Tc with Tcn for {} elements'.format(count_fill_tc_with_tn))
del tmp

if args.fill:
    '''If Tc is NaN, then replace it by 0'''
    df = df.fillna({'Tc': 0})
else:
    '''If Tc is NaN, then drop it'''
    df = df.dropna(subset=['Tc'])


u = df['str3'].unique()
vc = df['str3'].value_counts(dropna=False)
print(vc)
""" what follows is the list of str3"""
filepath = os.path.join(os.getcwd(), 'dup_str3.csv')
vc.to_csv(filepath)

vc = df['element'].value_counts(dropna=False)
print(vc)

""" what follows is the list of duplicated materials"""
filepath = os.path.join(os.getcwd(), 'dup_element.csv')
vc.to_csv(filepath)
del vc

# df = df.drop(index_list)
# df = df.reset_index(drop=True)
u = df['element'].unique().tolist()
# print(u)
count = 0
for index, row in df.iterrows():
    if row["element"][-1] not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}:
        print(row["element"])
        # index_list.append(index)
        row["element"] += "1"
        print(row['element'])
        count += 1
print("check", count)

print("the number of the rest list is", len(df))
print()
if args.fill:
    filepath = os.path.join(os.getcwd(), 'non_organic_supercon_1.csv')
else:
    filepath = os.path.join(os.getcwd(), 'non_organic_supercon_1_not_fill.csv')
df.to_csv(filepath)

'''if the duplication is found, then take one with median Tc. but the structure information is missing in the following method '''
df2 = df.groupby('element')['Tc'].median().reset_index()
# df2["str3"] = ""
# df2 = df2.reset_index()
count = 0
''' structure information for df2 is not consider in the following block. I may go gack to the issue later when I must deal with structure information '''
''' the following block does not work. The following break means this.'''
for index_2, row_2 in df2.iterrows():
    break
    df_temp = df[(df["element"] == row_2["element"])]
    # df_temp["abs"] = ""
    df_temp["abs"] = np.abs(row_2["Tc"] - df_temp["Tc"])
    df_temp = df_temp[[df_temp["abs"].min() == df_temp["abs"][0]]]
    print(df_temp)

    row_2["str3"] = df_temp["str3"]
    if index_2 % 100 == 0:
        print("{0}/{1}".format(index_2, index))
        # print("row_2[str3]:", row_2["str3"])
        print("row_2[Tc]:", row_2["Tc"])
    if len(df_temp.index) == 0 or len(df_temp.index) > 2:
        print("error")
        print(df_temp)
        count += 1
        print("something happens", count)
        # exit()

list = []
from material_to_dict import Material_to_dict
material_to_dict = Material_to_dict()
for index, row in df2.iterrows():
    element = row['element']
    # print(element)
    dict = material_to_dict.formula_to_dict(
        formula=element)  # original function
    # print(2dict)
    list.append(dict)  # this is a list of dictiornary
df2['dict'] = list
# print(df['dict'])
del element, dict

index_list = []
for index, row in df2.iterrows():
    elements_dict = row['dict']
    for key in elements_dict.keys():
        if elements_dict[key] > 1000:
            print(row["element"])
            print(elements_dict)
            elements_dict[key] /= 1000
            print(elements_dict)
            print(index)
            break
        elif elements_dict[key] > 100 and len(elements_dict):
            print(elements_dict)
            print(row["element"])
            elements_dict[key] /= 100
            print(elements_dict)
            print(index)
            break
del df2['dict']
if args.dealing_tn < 0:
    if args.fill:
        filepath = os.path.join(os.getcwd(), 'non_organic_supercon_2.csv')
    else:
        filepath = os.path.join(
            os.getcwd(), 'non_organic_supercon_2_not_fill.csv')
else:
    filepath = os.path.join(
        os.getcwd(), 'non_organic_supercon_2_dealing_tn_not_fill.csv')
variables = ["element", "Tc"]

df2 = df2[variables]
print(df2.loc[df2['Tc'] <= 400])
df2 = df2.loc[df2['Tc'] < 400]
df2.to_csv(filepath)
'''the information about the structure is deleted in the list'''
# for index, row in df.iterrows():


print(df2)
duplicated_element = df2.duplicated("element", keep=False)
# print(df[duplicated_element])
filepath = os.path.join(os.getcwd(), 'duplicated_elements_2.csv')

print('the number of data rest is', len(df2))
print('Replaced Tc with Tcn for {} elements'.format(count_fill_tc_with_tn))
