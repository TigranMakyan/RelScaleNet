
# file1 = open('/home/powerx/computer_vision/RelScale/list_mega.txt', 'r')
# a = [line.split(', ') for line in file1.readlines()]
# cfg = a[0][:-1]
# import os
# for item in cfg:




# p = os.path.join('/media/storagedrive/Megadepth/pairs', item, 'sparse-txt/balanced_pairs.txt')

cfg_path = './list_mega.txt'
import os
file1 = open(cfg_path, 'r')
a = [line.split(', ') for line in file1.readlines()]
cfg = a[0][:-1]

for item in cfg:
    p = os.path.join('/media/storagedrive/Megadepth/pairs/', item, 'sparse-txt/balanced_pairs2.txt')
    with open(p, 'r') as file:
        lines = file.readlines()
    print(item)
    # remove the first line
    lines.pop(0)
    new_p = p = os.path.join('/media/storagedrive/Megadepth/pairs/', item, 'sparse-txt/result_pairs1.txt')
    with open(new_p, 'w') as file:
        file.writelines(lines)