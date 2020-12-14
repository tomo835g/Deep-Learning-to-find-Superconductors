import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from pymatgen import Composition


class TransformReadingPeriodicTable():
    def __init__(self, formula=None, rel_cif_file_path='write cif file path', data_dir='../data'):
        self.formula = formula
        self.allowed_elements_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                                      'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    def formula_to_periodic_table(self):
        def channel(x, y):
            '''                 
            x: horizontal
            y: vertical
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
                print("error in making channel in period_table_as_img")
            return channel

        dict_formula = Composition(self.formula).as_dict()
        coordinate = np.zeros([4, 7, 18 + 14], dtype=np.float32)
        for key, value in dict_formula.items():
            i = self.allowed_elements_list.index(key)
            # print(key)
            num = i+1  # which is element number as H of num is 1
            # 18+14=32 # channel mens s,p,d, and f
            if num == 1:  # H
                coordinate[channel(0, 0), 0, 0] = value
            elif num == 2:  # He
                coordinate[channel(0, 32-1), 0, 32-1] = value

            elif num <= 18:
                # if q, mod=divmod(10,3) then q=3, mod=1
                y, x = divmod(num-2, 8)
                if x == 1 or x == 2:
                    coordinate[channel(y+1, x-1), y+1, x-1] = value
                else:
                    if x == 0:
                        x = 8
                        y -= 1
                    x = x+10+14
                    coordinate[channel(y+1, x-1), y + 1, x - 1] = value

            elif num <= 54:  # from K to Xe, which are from 4th and 5th period
                y, x = divmod(num-18, 18)
                if x == 0:
                    x = 18
                    y -= 1
                if x == 1 or x == 2 or x == 3:
                    coordinate[channel(y+3, x-1), y+3, x-1] = value
                else:
                    x = x+14
                    coordinate[channel(y+3, x-1), y + 3, x - 1] = value

            elif num <= 118:
                y, x = divmod(num-54, 32)
                if x == 0:
                    x = 32
                    y -= 1
                coordinate[channel(y+5, x-1), y+5, x-1] = value
            else:
                raise ValueError('error in period to image-like')

        #     dict[key] = coordinate
        # if 'Tl' in dict.keys():
        #     dict['TI'] = dict['Tl']
        # if 'Li' in dict.keys():
        #     dict['LI'] = dict['Li']
        return coordinate

    def from_periodic_table_form_to_dict_form(self, periodic_table_form):
        ''' input: periodic_table_form,  basically [4,7,32]  the first is for 4 channels s,p,d, and f. but the channel number can be arbitrary if we have more orbitals
        '''
        periodic_table_form = np.sum(periodic_table_form, axis=0)
        dict_form = {}
        element_num = 0  # it is like H is 0,He is 1
        vertical_len, horizontal_len = periodic_table_form.shape

        def add_element_to_dict(y, x, element_num, dic_form, decimal_num=4):
            if periodic_table_form[y, x] > 0:
                key = self.allowed_elements_list[element_num]
                val = periodic_table_form[y, x]
                dict_form[key] = np.round(val, decimal_num)
            return dict_form

        for y in range(vertical_len):
            for x in range(horizontal_len):
                if y == 0 and (x == 0 or x == 31):  # 2 first row
                    dict_form = add_element_to_dict(
                        y, x, element_num, dict_form)
                    element_num += 1
                elif (y == 1 or y == 2) and (x <= 1 or 26 <= x):  # 2+6=8 (16) 2nd and 3rd row
                    dict_form = add_element_to_dict(
                        y, x, element_num, dict_form)
                    element_num += 1
                elif (y == 3 or y == 4) and (x <= 2 or 17 <= x):  # 2+16=18 (36)
                    dict_form = add_element_to_dict(
                        y, x, element_num, dict_form)
                    element_num += 1
                elif (y == 5 or y == 6):  # 32 (64)
                    dict_form = add_element_to_dict(
                        y, x, element_num, dict_form)
                    element_num += 1
        if element_num != 118:
            print('error1090okc')
            exit()
        return dict_form

    def dict_form_to_chemical_formula(self, dict_form):
        return Composition.from_dict(dict_form).reduced_formula

    def periodic_table_form_to_chemical_formula(self, periodic_table_form):
        dict_form = self.from_periodic_table_form_to_dict_form(
            periodic_table_form)
        return self.dict_form_to_chemical_formula(dict_form)


'''here is an example'''
test_formula = 'H2He5'
reading_periodic_table = TransformReadingPeriodicTable(formula=test_formula)
reading_periodic_table_form_data = reading_periodic_table.formula_to_periodic_table()
print(reading_periodic_table_form_data)
formula_dict_form = reading_periodic_table.from_periodic_table_form_to_dict_form(reading_periodic_table_form_data)
print(formula_dict_form)