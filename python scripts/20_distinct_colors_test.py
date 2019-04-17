import re
import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Loading data' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
colors = pdObj.colors
print(colors)


ddir = pdir + '/data/'



#
# from ast import literal_eval as make_tuple
# colors_20 = []
# with open(ddir + '20_distinct_colors.txt', 'r') as file:
#     for line in file:
#         f = line.replace('\n', '')
#         g = '(%s)' % f
#         h = make_tuple(g)
#         colors_20.append(h)
#         print(h)
# print(colors_20)
