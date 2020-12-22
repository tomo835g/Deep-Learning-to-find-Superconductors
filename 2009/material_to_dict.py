import numpy as np
import os
import pandas as pd
from pymatgen import Composition


class Material_to_dict():
    def __init__(self, show_strange=False):
        self.num_list = ["1", "2", "3", "4", "5", "6", "7",
                         "8", "9", "0", "."]
        filepath = os.path.join(os.getcwd(), 'period_table_num_only.csv')
        period = pd.read_csv(filepath)
        element_number = np.arange(len(period))
# element_number += 1
        period["number"] = element_number
        self.element_list = period["element"].tolist()
        # print(self.element_list)
        self.show_strange = show_strange

    def correct_formula(self, formula):
        formula = formula.replace(" ", "")
        formula = formula.replace('?', '')
        formula = formula.replace('\'', '')
        formula = formula.replace('\"', '')
        formula = formula.replace('~', '')
        formula = formula.replace('\'\'', '')
        formula = formula.replace('_', '')
        formula = formula.replace(',', '.')
        formula = formula.replace('TI', 'Tl')
        formula = formula.replace('LI', 'Li')
        formula = formula.replace('RE', 'Re')
        '''specific supercon'''
        if formula == 'Ge0.15Nn3Sn0.85':
            formula = 'Ge0.15Nb3Sn0.85'
        elif formula == 'Nb1Ps1':
            formula = 'Nb1Pt1'
        elif formula == 'P4Ph5':
            formula = 'P4Pd5'
        elif formula == 'Sm1Ba-1Cu3O6.94':
            formula = 'Sm1Ba1Cu3O6.94'
        elif formula == 'Ag7Bf4O8':
            formula = 'Ag7B4F4O8'

            '''specific to first'''
        elif formula == '1T-TaS2':
            formula = 'T1Ta1S2'
        # elif formula == 'LaIrPn':
        #     formula='LaIrP'
        return formula

    def formula_to_dict(self, formula):
        """
        input: a chemical formula like H2B, H2.01B3.4
        output: a dictionary like {"H":2, "B":1}, {"H":2.01, "B":3.4}
         """
        number = ''
        element = ''
        diff = 0
        dict = {}
        strange_data = 0
        # print(length)
        formula = formula.replace(" ", "")
        formula = formula.replace('?', '')
        formula = formula.replace('\'', '')
        formula = formula.replace('\"', '')
        formula = formula.replace('~', '')
        formula = formula.replace('\'\'', '')
        formula = formula.replace('_', '')

        for i in formula:
            if i == ",":
                i = "."
            if i == "-":
                continue
            # if i == " " or "\'":  # skip space and signle quote ' , it does not work at all!
            #     continue
            if i not in self.num_list:  # letter
                if diff == 1:  # this case happens only when previous letter is in num list
                    # 数と文字の切り替わり
                    if number == '.':
                        continue
                    element = element.lower().title()
                    dict[element] = float(number)
                    element = ''
                    number = ''
                # such as BaCu the max length of elements is just two
                element += i
                diff = 0
                if element == 'TI':
                    element = 'Tl'
                elif element == 'LI':
                    element = 'Li'
                elif element == "RE":
                    element = "Re"

                if len(element) == 3:
                    temp_ele = element[:2].lower().title()
                    dict[temp_ele] = 1
                    element = element[2]
                    number = ""
                elif len(element) == 2:
                    if element[1].isupper():
                        dict[element[0]] = 1
                        element = element[1]
                        number = ""

            else:  # if i is in self.num_list 0,1,2,
                number += i
                diff = 1

        # read through the final character
        if diff == 0:  # If the last character is a letter
            dict[element] = 1
        elif number == '.' or number == ".":  # If number is just . ist is interpreted as 1
            dict[element] = 1
        else:  # if the last character is a number
            dict[element] = float(number)
            if float(number) > 100:
                strange_data = 1
                if self.show_strange:
                    print("!!!!!!!!!!!!!!!!!!!")
        if strange_data:
            if self.show_strange:
                print(formula)
                print(dict)
        if "n" in dict.keys():
            dict["N"] = dict["n"]
            del dict["n"]
        if "u" in dict.keys():
            dict['U'] = dict['u']
            del dict['u']
        if "re" in dict.keys():
            dict['Re'] = dict['re']
            del dict['re']

        # keys = dict.keys()
        # for key in keys:
        #     if key == '':
        #         continue
        #     if key[0].islower():
        #         tmp_key = key.capitalize()
        #         dict[tmp_key] = dict[key]
        #         del dict[key]

        return dict

    def from_dict_to_estimated_mother_dict(self, dict):
        '''
        args: dict like {'X':1.9,'Y':0.1,'Z':2}
        output: dict {'X':2,'Z':2}
        '''
        flag = False
        for key_1, value_1 in dict.items():
            if flag:
                break
            value_1 = float(value_1)
            if float(value_1).is_integer():
                continue
            for k, v in dict.items():
                if key_1 == k:
                    continue
                if (value_1 + v).is_integer():
                    if value_1 > v:
                        l_k, s_k = key_1, k
                    else:
                        l_k, s_k = k, key_1
                    dict[l_k] = int(value_1 + v)
                    flag = True
                    break
        if flag:
            del dict[s_k]

        for k1, v1 in dict.items():
            v1 = float(v1)
            if v1.is_integer():
                continue
            else:
                decimal = v1 % 1
                if decimal >= 0.9 or decimal <= 0.1:
                    v1 = np.round(v1)
                    dict[k1] = v1
        return dict

    def from_dict_to_reduced_formula(self, dict):
        '''
        args: dict like {'H':2,'O':1}
        output: string H2O
        '''
        return Composition(dict).reduced_formula

        formula = ''
        for k, v in dict.items():
            formula += k
            if v != 1:
                formula += str(v)
        comp = Composition(formula)
        formula = comp.reduced_formula
        return formula
        '''
        may be
        Compotition(dict).reduced_formula may work I think
        '''


'''
test = 'Y1Ba2Cu2.7In0.3O6.9'
test = 'Y1Ba2Cu2.82Fe0.18O6.895'
mtc = Material_to_dict()
correct_formula = mtc.correct_formula(test)
print(correct_formula)
dict_formula = mtc.formula_to_dict(test)
print(dict_formula)
'''
