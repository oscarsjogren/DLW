import numpy as np
################
### Plotting ###
################

def plot(y_data, x_data, index=None, title=None, xlabel=None, ylabel=None):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(x_data, y_data)
	if index is not None:
		ax.set_xtocklabels(index)
	if title is not None:
		ax.set_title(title, fontsize='large')
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	plt.show()

def plot_dict(dictionary, title, xlabel, ylabel):
	y_data, x_data = dictionary.items()
	plot(y_data, x_data, title=title, xlabel=xlabel, ylabel=ylabel)

###########
### I/O ###
###########

import csv

def find_path(file_name):
	import os
	cwd = os.getcwd()
	if not os.path.exists('data'):
		os.makedirs('data')
	d = os.path.join(cwd, os.path.join('data',file_name+'.csv'))
	return d

def create_file(file_name):
	import os
	d = find_path(file_name)
	if not os.path.isfile(d):
		open(d, 'w').close()
	return d

def file_exists(file_name):
	import os
	d = find_path(file_name)
	return os.path.isfile(d)

def append_to_csv_1D(array, file_name, delimiter=';', end_char=None):
	write_csv_1D(array, file_name, delimiter, end_char, open_as='a')

def append_to_csv_2D(array, file_name, delimiter=';', end_char=None):
	write_csv_2D(array, file_name, delimiter, end_char, open_as='a')

def write_csv_dict(d, file_name, key_items=None, delimiter=';'):
	import collections
	od = collections.OrderedDict(sorted(d.items()))
	if key_items is None:
		keys = od.keys()
		vals = od.values()
		write_csv_1D(od[keys[0]], file_name, delimiter=delimiter)
		for val in vals[1:]:
			append_to_csv_1D(val, file_name, delimiter=delimiter)
	else:
		write_csv_1D(od[key_items[0]], file_name, delimiter=delimiter)
		for key in key_items[1:]:
			append_to_csv_1D(od[key], file_name, delimiter=delimiter)

def write_csv_1D(array, file_name, delimiter=';', end_char=None, open_as='wb'):
	assert len(array.shape) == 1, "Array need to be 1D"
	d = find_path(file_name)
	with open(d, open_as) as f:
		writer = csv.writer(f, delimiter=delimiter)
		writer.writerow(array)
		if end_char is not None:
			writer.writerow(end_char)

def write_csv_2D(array, file_name, delimiter=';', end_char=None, open_as='wb'):
	d = find_path(file_name)
	with open(d, open_as) as f:
		writer = csv.writer(f, delimiter=delimiter)
		for row in array:
			writer.writerow(row)
		if end_char is not None:
			writer.writerow(end_char)

def load_csv(file_name, delimiter=';', comment=None):
	d = find_path(file_name)
	pass

def write_columns_csv(lst, file_name, header=[], index=None, start_char=None, delimiter=';', open_as='wb'):
	d = find_path(file_name)
	if index is not None:
		index.extend(lst)
		output_lst = zip(*index)
	else:
		output_lst = zip(*lst)

	with open(d, open_as) as f:
		writer = csv.writer(f, delimiter=delimiter)
		if start_char is not None:
			writer.writerow([start_char])
		writer.writerow(header)
		for row in output_lst:
			writer.writerow(row)

def write_columns_to_existing(lst, file_name, header="", delimiter=';'):
	d = find_path(file_name)
	with open(d, 'r') as finput:
			reader = csv.reader(finput, delimiter=delimiter)
			all_lst = []
			row = next(reader)
			nested_list = isinstance(lst[0], list) or isinstance(lst[0], np.ndarray)
			if nested_list:
				lst = zip(*lst)
				row.extend(header)	
			else:
				row.append(header)
			all_lst.append(row)
			n = len(lst)
			i = 0
			for row in reader:
				if nested_list:
					row.extend(lst[i])
				else:
					row.append(lst[i])
				all_lst.append(row)
				i += 1
	with open(d, 'w') as foutput:
			writer = csv.writer(foutput, delimiter=delimiter)
			writer.writerows(all_lst)
			
def append_to_existing(lst, file_name, header="", index=None, delimiter=';'):
	write_columns_csv(lst, file_name, header, index, start_char='\n', delimiter=delimiter, open_as='a')



##########
### MP ###
##########


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)