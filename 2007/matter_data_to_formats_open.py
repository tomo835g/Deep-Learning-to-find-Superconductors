from __future__ import print_function
import keras
import argparse
import numpy as np
import os
import pandas as pd
from material_to_dict import Material_to_dict
import json
import time
import collections

'''
Eliminate organic-materials from the begining
'''

parser = argparse.ArgumentParser(
    description='preprocessing data made by first project')
parser.add_argument('--save_dir', '-sd', type=str)
parser.add_argument('--data_filepath', '-dfp', type=str)
parser.add_argument('--is_Tc', '-is_Tc', type=int, default=0)
parser.add_argument('--data_name', '-dn', type=str)  # cod, first
parser.add_argument('--thresh_hold_tc', '-tht',
                    type=int, default=10)  # cod, first
parser.add_argument('--eliminate_non_organic', '-eno', type=int, default=1)
parser.add_argument('--with_Fe', type=int, default=1,
                    help='with Fe or without Fe')


global args
args = parser.parse_args()
since = time.time()
'''This file is to make inputs data formats

NOTE: elements are used in confusing way, it sometimes mean material also, not only atoms!
'''

if args.save_dir:
    save_dir = args.save_dir


if args.data_filepath:
    data_filepath = os.path.abspath(args.data_filepath)
    print('args.data_filepath is loaded')
else:
    data_filepath = os.path.join(
        os.getcwd(), 'non_organic_supercon_2.csv')

# args.data_name='cod'

if args.data_name == 'first':
    ''' First Data '''
    save_dir = 'first_data'
    if args.with_Fe:
        data_filepath = os.path.join(
            os.getcwd(), save_dir,  'first_concat_list_dup_over_training_removed.csv')
    else:
        data_filepath = os.path.join(
            os.getcwd(), save_dir, 'first_concat_list_dup_over_training_removed_without_Fe.csv')
    args.is_Tc = 1
    print('First Data')

elif args.data_name == 'cod':
    '''Cod Data'''
    save_dir = 'cod_data'
    data_filepath = os.path.join(
        os.getcwd(), save_dir, 'chemical_formula_sum_overlap_removed_2.csv')
    args.is_Tc = 0
    print('Cod Data')
elif args.data_name == 'training_data':
    data_filepath = os.path.join(
        os.getcwd(), 'non_organic_supercon_2.csv')
    args.is_Tc = True
    print('Training data')
    save_dir = 'training_data'
elif args.data_name == 'training_data_not_fill':
    data_filepath = os.path.join(
        os.getcwd(), 'non_organic_supercon_2_not_fill.csv')
    args.is_Tc = True
    print('Training data NOT fill')
    save_dir = 'training_data_not_fill'
elif args.data_name == 'training_data_dealing_tn_not_fill':
    data_filepath = os.path.join(
        os.getcwd(), 'non_organic_supercon_2_dealing_tn_not_fill.csv')
    args.is_Tc = True
    print('Training data NOT fill and dealing tn')
    save_dir = 'training_data_dealing_tn_not_fill'
elif args.data_name == 'band_gap':
    data_filepath = os.path.join(
        os.getcwd(), "band_gap", 'band_gap_data_2.csv')
    save_dir = os.path.join(os.getcwd(), "band_gap")

else:
    print('error no data_name matched')
    exit()


# cod: cod_data
save_dir = os.path.join(os.getcwd(), save_dir)

if os.path.isdir(save_dir) == False:
    os.mkdir(save_dir)
    print('save directory does not exists')
    # exit()

'''To make the directory ./$save_dir/made_data'''
path_to_made_data = os.path.join(save_dir, 'made_data')
if os.path.isdir(path_to_made_data) == False:
    os.mkdir(path_to_made_data)
del path_to_made_data


# cod ../data/cod/made_data/


print(save_dir)
print(data_filepath)

start_dir = os.getcwd()
period_table_num_only_filepath = os.path.join(
    os.getcwd(), 'period_table_num_only.csv')
period = pd.read_csv(period_table_num_only_filepath)
''' first raw: element, number 
second raw H, 0
He, 1
...
'''
num_element = len(period)
del period, period_table_num_only_filepath
print(num_element)

'''this is to correct Tc for material consists of single element like H, He, Li, Be, Bo, C, No'''
Tc_filepath = os.path.join(os.getcwd(), 'single_element_tc.csv')
single_element_tc = pd.read_csv(Tc_filepath)
single_element_tc = single_element_tc.set_index('element')


def period_table_electron():
    os.chdir(start_dir)
    filepath = os.path.join(os.getcwd(), 'period_table.csv')
    data = pd.read_csv(filepath)
    # data = data.set_index('element')
    # data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    dict = {}
    data = data.values
    # print(data)
    for i in range(len(data)):
        key = data[i, 0]
        # print(key, data[i, 1:])
        dict[key] = data[i, 1:].tolist()
    # this is because there are wrong element as TI and LI in the supercon dataset. By making wrong-named element, we can deal with such wrong-nambe elements properly
    if 'Tl' in dict.keys():
        dict['TI'] = dict['Tl']
    if 'Li' in dict.keys():
        dict['LI'] = dict['Li']
    return dict


period_table_electron = period_table_electron()
''' code block for making period table with electron period
        'H':[1,0,0,.....]
        'B':[2,2,1,0,.....]
        'element':[1s,2s,....], #electrons in each orbit is writen
        the length of [2,2,1,0,....] is 19
        '''

'''change to save dir to save data'''


# np.save('./made_data/period_table_electron', period_table_electron)
# file = open('period_table_electron.json', 'w')
# json.dump(period_table_electron, file)
# file.close()


''' I will make a dictionary from element to the number of the outmost shell electron number'''
dict_element_to_outmost_e_num = {}
for key in period_table_electron.keys():
    element_electron_orbit = period_table_electron[key]
    for i in element_electron_orbit[::-1]:
        if i > 0:
            dict_element_to_outmost_e_num[key] = i
if 'Tl' in dict_element_to_outmost_e_num.keys():
    dict_element_to_outmost_e_num['TI'] = dict_element_to_outmost_e_num['Tl']
if 'Li' in dict_element_to_outmost_e_num.keys():
    dict_element_to_outmost_e_num['LI'] = dict_element_to_outmost_e_num['Li']


def period_table_as_rpt_left_step():
    ''' to make the period table of left step as an rpt'''
    os.chdir(start_dir)

    def channel(x):
        if x <= 14:
            channel = 3
        elif x <= 14 + 10:
            channel = 2
        elif x <= 24 + 6:
            channel = 1
        elif x <= 30 + 2:
            channel = 0
        else:
            print("error in making channel")
        return channel

    ''' code block for making left stepperiod table as rpt. this is not a data itself. this is what for making the data.'''
    filepath = os.path.join(os.getcwd(
    ), 'period_table_num_only.csv')  # element and number like H 0, He 1, Li 2
    data = pd.read_csv(filepath)
    data = data.values  # [element, number] like [H,0], [He,1],...
    dict = {}
    data[:, 1] += 1  # suppose element of H is 0 orignally in the data

    for i in range(len(data)):
        key = data[i, 0]  # it is the name of element like H, He, Li, Be
        # print(key)
        num = i+1  # which is element number as H of num is 1
        coordinate = np.zeros(
            [4, 7, 18 + 14], dtype=np.float32)  # (channel, y,x) channel, y=vertical, x=horizontals
        '''18+14=32 # 4 channel that means s,p,d,and f, the second 7 is y-coordinate, and the third 32 is x-coordinate'''
        if num <= 2:  # for H, and He
            # the -1 after num is because x coord. start with 0
            coordinate[0, 0, 30 + num-1] = 1
        elif num <= 4:  # for Li, and Be
            coordinate[0, 0, 30 + num - 2 - 1] = 1  # the end of s

        elif num <= 4 + 6 + 2:  # for B to Mg
            coordinate[channel(14 + 10 + num - 4), 1, 14 + 10 + num - 4-1] = 1
        elif num <= 12 + (6 + 2):  # for Al to Ca
            coordinate[channel(14 + 10 + num - 12), 2,
                       14 + 10 + num - 12-1] = 1
        elif num <= 20 + (10 + 6 + 2):  # for Sc to Sr
            coordinate[channel(14 + num - 20), 3, 14 + num - 20-1] = 1
        elif num <= 38 + (10 + 6 + 2):  # for Y to Ba
            coordinate[channel(14 + num - 38), 4, 14 + num - 38-1] = 1
        elif num <= 56 + (14 + 10 + 6 + 2):  # La to Ra
            coordinate[channel(num - 56), 5, num - 56-1] = 1
        elif num <= 88 + 32:  # Ac to the end
            coordinate[channel(num-88), 6, num-88-1] = 1
        else:
            print('error in period to rpt-like')
            exit()
        dict[key] = coordinate
    if 'Tl' in dict.keys():
        dict['TI'] = dict['Tl']
    if 'Li' in dict.keys():
        dict['LI'] = dict['Li']
    return dict


print('here')
os.chdir(save_dir)
print(os.getcwd())
period_table_as_rpt_left_step = period_table_as_rpt_left_step()
if os.path.isdir('./made_data') == False:
    os.mkdir('./made_data')
np.save('./made_data/period_table_as_rpt_left_step',
        period_table_as_rpt_left_step)
# print(period_table_as_rpt)


def period_table_as_rpt():
    os.chdir(start_dir)

    def channel(x, y):
        ''' input: number of the element. e.g. for H, the number is 1
            output: 0 if the element is in s, and 1 if it is in p, and so forth for d:2 and f:3
        '''
        # f 14, d 10, p 6
        x, y = x+1, y+1  # changing to start from 1. 1 is the origin
        if y == 1 or x <= 2:
            channel = 0  # s
        elif 27 <= x:
            channel = 1  # p
        elif x == 3 or (3 + 14+1 <= x and x <= 17 + 9):
            channel = 2  # d
        elif 4 <= x and x <= 17:
            channel = 3  # f
        else:
            print("error in making channel in period_table_as_rpt")
        return channel

    ''' code block for making period table as rpt. this is not a data itself. this is what for making the data.'''
    filepath = os.path.join(os.getcwd(
    ), 'period_table_num_only.csv')  # element and number like H 0, He 1, Li 2
    data = pd.read_csv(filepath)
    data = data.values
    dict = {}
    data[:, 1] += 1  # suppose element of H is 0 originally in the data
    for i in range(len(data)):
        key = data[i, 0]
        # print(key)
        num = i+1  # which is element number as H of num is 1
        # 18+14=32 # channel mens s,p,d, and f
        coordinate = np.zeros([4, 7, 18+14], dtype=np.float32)
        if num == 1:  # H
            coordinate[channel(0, 0), 0, 0] = 1
        elif num == 2:  # He
            coordinate[channel(0, 32-1), 0, 32-1] = 1

        elif num <= 18:
            y, x = divmod(num-2, 8)  # if q, mod=divmod(10,3) then q=3, mod=1
            if x == 1 or x == 2:
                coordinate[channel(y+1, x-1), y+1, x-1] = 1
            else:
                if x == 0:
                    x = 8
                    y -= 1
                x = x+10+14
                coordinate[channel(y+1, x-1), y + 1, x - 1] = 1

        elif num <= 54:  # from K to Xe, which are from 4th and 5th period
            y, x = divmod(num-18, 18)
            if x == 0:
                x = 18
                y -= 1
            if x == 1 or x == 2 or x == 3:
                coordinate[channel(y+3, x-1), y+3, x-1] = 1
            else:
                x = x+14
                coordinate[channel(y+3, x-1), y + 3, x - 1] = 1

        elif num <= 118:
            y, x = divmod(num-54, 32)
            if x == 0:
                x = 32
                y -= 1
            coordinate[channel(y+5, x-1), y+5, x-1] = 1
        else:
            print('error in period to rpt-like')
            exit()
        dict[key] = coordinate
    if 'Tl' in dict.keys():
        dict['TI'] = dict['Tl']
    if 'Li' in dict.keys():
        dict['LI'] = dict['Li']
    return dict


os.chdir(start_dir)
period_table_as_rpt = period_table_as_rpt()
np.save('./made_data/period_table_as_rpt', period_table_as_rpt)


''' returns where the electron in the outmost shell resides'''
dict_element_spdf = {}  # returns s,p,d,or f.
# returns (1,0,0,0) if s, (0,1,0,0) if p, and so forth.
dict_element_spdf_vector = {}
'''retuns the number 1 if it is in 1s, 2 if it is in 2p, and so forth'''
dict_element_num_before_spdf = {}

for element, element_table_as_rpt in period_table_as_rpt.items():
    if element == 'He':
        dict_element_spdf[element] = 's'
    index = np.where(element_table_as_rpt == 1)
    y, x = index[0], index[1]  # down vertical:y right horizontal:x
    if x == 0 or x == 1:
        dict_element_spdf[element] = 's'
        dict_element_spdf_vector[element] = [1, 0, 0, 0]
    elif x == 3-1 or (14 + 4-1 <= x and x <= 14 + 12-1):
        dict_element_spdf[element] = 'd'
        dict_element_spdf_vector[element] = [0, 0, 1, 0]
    elif 4-1 <= x and x <= 17-1:
        dict_element_spdf[element] = 'f'
        dict_element_spdf_vector[element] = [0, 0, 0, 1]
    elif 27-1 <= x and x <= 32-1 and y != 0:
        dict_element_spdf[element] = 'p'
        dict_element_spdf_vector[element] = [0, 1, 0, 0]
    elif x == 31 and y == 0:  # this is for He.
        dict_element_spdf[element] = 's'
        dict_element_spdf_vector[element] = [1, 0, 0, 0]
    else:
        print("error in making dict_element_spdf")
        print('x', x)
        print('y', y)
    dict_element_num_before_spdf[element] = y + 1
    if dict_element_spdf[element] == 'd':
        dict_element_num_before_spdf[element] -= 1
    elif dict_element_spdf[element] == 'f':
        dict_element_num_before_spdf[element] -= 2
    dict_element_spdf_vector[element] = np.array(
        dict_element_spdf_vector[element])

if 'TI' in dict_element_spdf.keys():
    dict_element_spdf['TI'] = dict_element_spdf['Tl']
    dict_element_num_before_spdf['TI'] = dict_element_num_before_spdf['Tl']
    dict_element_spdf_vector['TI'] = dict_element_spdf_vector['Tl']

if 'LI' in dict_element_spdf.keys():
    dict_element_spdf['LI'] = dict_element_spdf['Li']
    dict_element_spdf_vector['LI'] = dict_element_spdf_vector['Li']
    dict_element_num_before_spdf['LI'] = dict_element_num_before_spdf['Li']

'''data_filepath is defined in the begining'''
df = pd.read_csv(data_filepath)
# drop completely duplicated data
print('before removing duplication', len(df))
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print('after removing duplication', len(df))

if args.is_Tc:
    pass
else:
    df['Tc'] = np.nan
print(df.head())
if args.data_name == 'cod' or args.data_name == 'first':
    df.columns = ['element', 'Tc']
variables = ['element', 'Tc']
df = df[variables]
data = df.values
print(df.head())
if args.data_name == 'cod':
    print('Sorted')
    df = df.sort_values('element')
    print(df.head())

# data = np.genfromtxt(filepath, dtype=None, delimiter=',')
# print(data)
# print(data.shape)
# print(data[3])
# print(len(data))

filepath = os.path.join(start_dir, 'period_table_num_only.csv')
period = pd.read_csv(filepath)
element_number = np.arange(len(period))
# element_number += 1
period['number'] = element_number
''' period (dataframe)
    H 0
    He 1
    Li 2,
    ...
    '''
dict_element_number = {}  # like {'H':0,'He':1,'Li':2,...}
# this is above dict. swaped between keys and valuess {0:'H',1:'He',2:'Li',...}
dict_number_element = {}
for index, row in period.iterrows():
    dict_element_number[row['element']] = index
    dict_number_element[index] = row['element']
print(dict_element_number)
print(dict_number_element)

# del element_number
# period.to_csv(filepath)
# for index, row in period.iterrows():
list = []
# print(period)
# exit()

'''
input: a chemical formula like H2B, H2.01B3.4
output: a dictionary like {'H':2, 'B':1}, {'H':2.01, 'B':3.4}
it will be elements_dict later for each row of the dataframe, which is the acutual material
'''
material_to_dict = Material_to_dict()

for index, row in df.iterrows():
    temp = row['element']
    if temp == '1T-TaS2':
        temp = 'TaS2'
    dict = material_to_dict.formula_to_dict(
        formula=temp)  # original function
    list.append(dict)  # this is a list of dictiornary
df['dict'] = list
# print(df['dict'])
del temp, dict

dict_period_simple = {}
print(len(period))
for index, row in period.iterrows():
    temp = keras.utils.to_categorical(row['number'], len(period))
    dict_period_simple[row['element']] = temp
del temp

if 'Tl' in dict_period_simple.keys():
    dict_period_simple['TI'] = dict_period_simple['Tl']
if 'Li' in dict_period_simple.keys():
    dict_period_simple['LI'] = dict_period_simple['Li']
''' dict_period_simple is element to one-hot-vector
dict_period_simple['He]=[0,1,0,0,....,0]
dict_period_simple['H']=[1,0,0,....0]
 '''


# max_len is 8 in this case
'''
 frequencies of the length of composit elements in materias in SuperCon are 
[  79. 3510. 4297. 3403. 4293. 1488.  212.   28.]
they i.e. 79 means that the materials of 1 element exist 79
the max number of elements materials have is 8, and it is not anormal.
'''
max_len_matter_dict = 0

problematic_materials = []
problematic_elements = []
problematic_tc = []
tc_list = np.zeros([len(df), 1])
pn_list = np.zeros([len(df), 1])
super_conductor_family_index = np.arange(
    len(df)).astype(str)  # 'CuO', 'Fe',or 'others'
# print(len(df)) gives the length of the matrial list df
organic_index = np.arange(len(df)).astype(str)
count = 0
count_non_organic = 0
matter_name_to_use = []
matter_tc_to_use = []
index_to_use = []
max_len_matter_dict_training_data = 8
len_matter_within_trainig_data_list = []
count_Fe = 0
count_CuO = 0
for index, row in df.iterrows():
    '''looking materias in supercon one by one here'''
    elements_dict = row['dict']  # {'H':1.09, 'He':2.09,'Li':1.09,...}, etc
    '''this is a decomposition of materials into the dictionaries of element and composition number. {'Cu':1.09, 'O':2.08,}'''
    vector_element = np.zeros(len(
        period))  # len(period) is possibly 118, whcih is the number of elements I use.
    flag_Cu = 0
    flag_O = 0
    flag_CuO = 0
    flag_Fe = 0
    flag_material_not_problematic = True
    flag_H = 0
    flag_C = 0
    flag_Fe_only = 0
    flag_Fe_partner = 0
    flag_organic = 0
    flag_len_4 = 0
    flag_La = 0
    if len(elements_dict) > 3:
        flag_len_4 = 1
    # start looking one material
    for key in elements_dict.keys():
        if key == 'Cu':
            flag_Cu = 1
        if key == 'O':
            flag_O = 1
        if key == 'La':
            flag_La = 1
        if key == 'Fe':
            flag_Fe_only = 1  # Fe family super-conductor
        if key in ['As', 'Se', 'S', 'P']:
            flag_Fe_partner = 1
        if key == 'H':
            flag_H = 1
        if key == 'C':
            flag_C = 1
        if flag_C == 1 and flag_H == 1:
            flag_organic = 1
        if flag_Fe_only == 1 and flag_Fe_partner == 1:
            flag_Fe = 1
        if (flag_Cu == 1 and flag_O == 1 and flag_len_4 == 1) or (flag_Cu == 1 and flag_O == 1 and flag_La == 1):
            flag_CuO = 1  # CuO super conductor
        if key not in dict_period_simple.keys():
            '''If wrong 'element' is found we skip it'''
            print('------------------')
            print(row['element'], row['Tc'], key)
            print(key)
            print(elements_dict)
            problematic_elements.append(key)
            problematic_materials.append(row['element'])
            problematic_tc.append(row['Tc'])
            count += 1
            flag_material_not_problematic = False
            break  # goes to 1
        '''for replacing single element material Tc with correct Tc'''
        if len(elements_dict) == 1:
            if key in single_element_tc.index:
                row['Tc'] = single_element_tc.ix[key, 'Tc']
            else:
                row['Tc'] = 0
        vector_element += dict_period_simple[key] * elements_dict[key]
        # start appending
    # 1
    if flag_organic == 1 and args.eliminate_non_organic == 1:
        continue
    count_non_organic += 1
    tc_list[index] = row['Tc']
    pn_list[index] = (row['Tc'] > args.thresh_hold_tc)

    if flag_material_not_problematic:
        matter_name_to_use.append(row['element'])
        matter_tc_to_use.append(row['Tc'])
        index_to_use.append(index)
    if flag_material_not_problematic == False:  # problematic data. Can the loop come here?
        continue
    max_len_matter_dict = max(max_len_matter_dict, len(elements_dict))
    if flag_CuO == 1:  # if CuO family
        super_conductor_family_index[index] = 'CuO'
        count_CuO += 1
    elif flag_Fe == 1:  # if Fe family
        super_conductor_family_index[index] = 'Fe'
        # print(row['element'])
        count_Fe += 1
    else:  # if neither CuO nor Fe famility super conductor
        super_conductor_family_index[index] = 'others'
    ''' Because it is annoying to check if the material is organic or non_organic, and then deal with it, I take only non_organic materials.'''
    # if flag_organic == 1:
    #     organic_index[index]='organic'
    # else:
    #     organic_index[index]='non_organic'
    #     count_non_organic += 1
    len_matter_within_trainig_data_list.append(len(elements_dict))
del vector_element
tc_list = tc_list[index_to_use]
pn_list = pn_list[index_to_use]
super_conductor_family_index = super_conductor_family_index[index_to_use]
exit()
# organic_index=organic_index[index_to_use]


if len(tc_list) != len(super_conductor_family_index):
    print('the length of two lists is not equal.')
    exit()
print('the length of the list is ', len(tc_list))

os.chdir(save_dir)
print('problemetic element', np.unique(problematic_elements))
print('frequency', collections.Counter(problematic_elements))

np.save('./made_data/super_conductor_family_index',
        super_conductor_family_index)
print('super_conductor_family_index.shape', super_conductor_family_index.shape)
# np.save('./made_data/organic_index', organic_index)
print('the length of non_organic is', count_non_organic)


# problematic_materials_list = pd.DataFrame(
#     {'material': problematic_materials, 'element': problematic_elements, 'Tc': problematic_tc})
# filepath = os.path.join(save_dir,'./made_data' ,'problematic_elements.csv')
# problematic_materials_list.to_csv(filepath,index=False)

matter_to_use_list = pd.DataFrame(
    {'element': matter_name_to_use, 'Tc': matter_tc_to_use})

if args.data_name == 'first':
    if args.with_Fe:
        file_name = 'use_for_prediction_with_Fe.csv'
    else:
        file_name = 'use_for_prediction_without_Fe.csv'
elif args.data_name == 'cod':
    file_name = 'use_for_prediction.csv'
    index_Fe = np.where(super_conductor_family_index == 'Fe')
    index_CuO = np.where(super_conductor_family_index == 'CuO')
    tmp_path = os.path.join(save_dir, './made_data', 'Fe_in_COD.csv')
    matter_to_use_list.iloc[index_Fe].to_csv(tmp_path)
    tmp_path = os.path.join(save_dir, './made_data', 'CuO_in_COD.csv')
    matter_to_use_list.iloc[index_CuO].to_csv(tmp_path)
    del tmp_path
else:
    print('error')
    print('If you are dealing with supercon data, it is OK')
    exit()
filepath = os.path.join(save_dir, './made_data', file_name)
matter_to_use_list.to_csv(filepath, index=False)
del filepath, file_name

print('the length of the original dataset', len(df))
print('The number of materials with unknow element:', count)
print('the length of the rest dataset', len(matter_to_use_list))
print('the ratio of problematic data that will be delted to original one', count/len(df))
# del count, key, flag_Cu, flag_CuO, flag_Fe, flag_O
del count

# just saving including everything
np.save('./made_data/tc_list', tc_list)
np.save('./made_data/pn_list', pn_list)

'''making combined list of tc and pn, which means if the material trainsit to superconductor or not'''
# tc_pn = np.zeros([len(tc_list,2])
# for i in range(len(tc_list)):
tc_pn = np.hstack([tc_list, pn_list])
np.save('./made_data/tc_pn', tc_pn)
# if args.data_name == 'cod':
#     print('Sorted')
#     print(df.head())

print('# CuO', count_CuO)
print('# Fe', count_Fe)


def rpts_materials(period_table_as_rpt, df):
    '''
    inputs:
        period_table_as_rpt: period table reped by rpts, which are normal period table and left step one
        df: panda dataframe of the list of materials
    outputs:
        list of materials represented by period table as rpts (left step or not)
        elements_dict_list: {'H':1.09, 'He':2.09,'Li':1.09,...}, etc for each material
    '''
    elements_dict_list = []
    count = 0
    input_as_rpt = np.zeros([len(tc_list), 4, 7, 32])
    elements_dict_list = {}
    for index, row in df.iterrows():
        if index not in index_to_use:
            continue
        elements_dict = row['dict']  # {'H':1.09, 'He':2.09,'Li':1.09,...}, etc
        '''the above is a decomposition of materials into the dictionaries of element and number. {'Cu':1.09, 'O':2.08,}'''
        f_problematic_material = False
        vector_element_as_rpt = np.zeros([4, 7, 18+14])
        for key in elements_dict.keys():
            if key not in dict_period_simple.keys():
                f_problematic_material = True
                break
            # vector_element_electron += dict_period_simple[key] * elements_dict[key]
            vector_element_as_rpt += period_table_as_rpt[key] * \
                elements_dict[key]
        if f_problematic_material:
            continue
        elements_dict_list[count] = elements_dict
        input_as_rpt[count, :, :, :] = vector_element_as_rpt
        # print(input_as_rpt.shape)
        count += 1  # useless
    del key, count
    return input_as_rpt, elements_dict_list


input_as_rpt, elements_dict_list = rpts_materials(
    period_table_as_rpt=period_table_as_rpt, df=df)
''' this is a 7x32 period table like rep. of one material
the first element corresponds to ID.If we have N materials, the length of the first raw is N '''

os.chdir(save_dir)
input_as_rpt = np.array(input_as_rpt, dtype=np.float32)
np.save('./made_data/input_as_rpt', input_as_rpt)
print('the shape of input_as_rpt', input_as_rpt.shape)

''' this is a 7x32 period table left step like rep. of one material
the first element corresponds to ID.If we have N materials, the length of the first raw is N '''
input_as_rpt_left_step, _ = rpts_materials(
    period_table_as_rpt=period_table_as_rpt_left_step, df=df)
input_as_rpt_left_step = np.array(input_as_rpt_left_step, dtype=np.float32)
os.chdir(save_dir)
np.save('./made_data/input_as_rpt_left_step', input_as_rpt_left_step)

# print(input_as_rpt[0])
file = open('elements_dict_list.json', 'w')
'''this is just a json form of input_as_rpt'''
json.dump(elements_dict_list, file)
file.close()
print('')
# print(elements_dict_list)
print(input_as_rpt[1].shape)


print('the maximum length of the matter_dict is', max_len_matter_dict)
