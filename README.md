# Introduction

The data and the code for ***''Deep Learning Model for Finding New Superconductors'' by Tomohiko Konno, Hodaka Kurokawa, Fuyuki Nabeshima, Yuki Sakishita, Ryo Ogawa, Iwao Hosako, and Atsutaka Maeda, [Physical Review B](https://doi.org/10.1103/PhysRevB.103.014509)

The paper is [open access](https://doi.org/10.1103/PhysRevB.103.014509), and Arxiv version is found [here](https://arxiv.org/abs/1812.01995).


# Condition
The data and the codes can be used under the condition that you cite the following two papers. Also see Licence.

```
%\cite{PhysRevB.103.014509}
@article{PhysRevB.103.014509,
  title = {Deep learning model for finding new superconductors},
  author = {Konno, Tomohiko and Kurokawa, Hodaka and Nabeshima, Fuyuki and Sakishita, Yuki and Ogawa, Ryo and Hosako, Iwao and Maeda, Atsutaka},
  journal = {Phys. Rev. B},
  volume = {103},
  issue = {1},
  pages = {014509},
  numpages = {6},
  year = {2021},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.103.014509},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.103.014509}
}

```

# The list for candidate materials for superconductors with our comment
The candidate materials list is found in the following file.
`` go_open_candidate_materials_list.xlsx``

The column `num_elements` denotes the number of elements the material has, `O_contained` denotes whether the material contains oxide or nor, and so forth.
# Data handling
The data handling is explained in detail in the supplementary materials of our paper.

# Directories
The data as of 2018, 2009, and 2007 are under `./2018`, `./2009`, and `./2007`, respectively.
The data and the codes are separate according to the years in order not to be mixed. You must enter the directories.
# Test data for identifying superconductors
We collected the data of superconductors and non-superconductors from Ref. by Hosono et al., and entered missing values by investigating each original paper cited in the reference (see our paper for detail). The data becomes the test data for identifying superconductors and are in` ./[year]/first_data/ first_concat_list_dup_over_training_removed.csv `for each year. The [year] is for 2018, 2009, and 2007. The data itself had been collected since 2010. From this point, to use the data of 2009 is the best way to test the model. For each year, the data in the test data that overlap the data of SuperCon and COD as of the year are removed. This is what is meant by the year. For years other than 2009, these data are used to exclude wrong models as explained in our paper.

# The data for "garbage-in"
The data of inorganic materials from [COD](http://www.crystallography.net/cod/) for the method of "garbage-in" is in ` ./[year]/cod_data/chemical_formula_sum_overlap_removed_2.csv` The [year] is for 2018, 2009, and 2007. The data reported until the [year] is collected for each [year].

# The data of superconductors
Because National Institute for Materials Science (NIMS) prohibits us from opening the data of superconductors for you, instead of opening the data and the data in reading periodic table data format, we provide the code for preprocessing the data from the database of superconductors (SuperCon).

If you use the code, you can get the preprocessed data of superconductors and the data of superconductors in reading periodic table data format.
## How to prepare the data of superconductors 
First of all, you must get the data of superconductors from SuperCon. 
The file name must be `OXIDEMETALLICSearchR_1_[year].csv`, where year is 2009 or 2007.
An example is `OXIDEMETALLICSearchR_1_2009.csv`, if the year is 2009. Or if the data year is 2018, the file name must be `OXIDEMETALLICSearchR_1.csv`.  We recommend to save the data in separate directories according to the year in order not to overwrite.


*Caution !*
If you want to get the data before 2010 from SuperCon, you must search for the data with putting 'before 2009'! If you put 'before 2010', the data contains the data of 2010 in SuperCon. SuperCon goes in this way.


The data must contain the rows of ['element', 'str3', 'Tc', 'tcn'].
The 'element' means the name of material like H<sub>2</sub>O contrary to the meaning of the term of 'element'.

You must delete two Fe-based materials, LaFePO and LaFePFO, from the data of 2007, if you want to use the data of 2007, because the two Fe-based materials are similar to Fe-based high-Tc superconductors. See our paper in detail to get more explanation.

# How to make the reading periodic table (rpt) type data 
You must enter each [year] directory. The codes are under each [year] directory.
## For superconductors
1. `python preprocessing_open.py --year_until 2007, 2009, or -1 --fill 0`
 year_until==-1 means the year_until is 2018.  -1 is default, which means 2018.
2. `python matter_data_to_formats_open.py --data_name training_data_not_fill`
   
Then you must have several data formats in `./made_data/`
The filename of the data reading periodic table type is `./made_data/period_table_as_rpt.npy`
The list of Tc is in `./made_data/tc_list` in the same order.
`./made_data/super_conductor_family_index` has indexes 'others', 'Fe', or 'CuO' in the same order as the materials in periodic_table_as_rpt.npy and so forth. The 'others' basically mean conventional superconductors, 'Fe' and 'CuO' mean Fe-based and CuO-based superconductors respectively.

## For test data for identifying superconductors
The test data is transformed into reading periodic table format by doing
`python matter_data_to_formats.py --data_name 'first' --with_Fe 1`
If --with_Fe 1, then the data contains the materials with Fe. If --with_Fe 0, then the materials that contain Fe are removed. The default is --with_Fe 1. The data in the form of reading periodic table are made in `./first_data/made_data/period_table_as_rpt.npy`.

## For "garbage-in" by COD
`python matter_data_to_formats.py --data_name 'cod'` 
The data of COD in the form of reading periodic table are made in `./cod_data/made_data/period_table_as_rpt.npy`
# Alternative way to make reading periodic table data format
You can use the class and methods in ```chemical_formula_to_reading_periodic_table.py``` for each chemical formula to transform the one into reading periodic table data format. The example to use is written in the file and below. The class is useful to use our methods to your own problems.

The code transforms chemical formula like H<sub>2</sub>O into **reading periodic table type data format**.
In the code, formula or chemical formula is like H<sub>2</sub>O, periodic_table is the data form of reading periodic table, the shape of which is (4,7,32), and dict or dict_form is like {"H":2,"O":1}.

An example
```
test_formula = 'H2He5'
reading_periodic_table = TransformReadingPeriodicTable(formula=test_formula)
reading_periodic_table_form_data = reading_periodic_table.formula_to_periodic_table()
print(reading_periodic_table_form_data)
>> must print 4*7*32 data. (rpt)
formula_dict_form=reading_periodic_table.from_periodic_table_form_to_dict_form(reading_periodic_table_form_data)
print(formula_dict_form)
>> must print {'H':2,'He':5}
```
# The code for model
The model for reading periodic table is the class *ModelReadingPeriodicTable* in `networks_go_open.py` in each [year] directory.

Due to the scarce human resource, we can only provide above codes.
## Requirement
torch, keras, numpy, pandas,pymatgen

