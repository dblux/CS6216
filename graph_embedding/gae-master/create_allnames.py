#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

pathways = os.listdir("info/kegg_human-edgelist/large/")

with open("all_files.txt", "w") as output_file:
    for file in pathways:
        fname = file[:-4]
        for i in range(2):
            if i == 0:
                for j in range(1,180):
                    output_file.write("{}_{}-{}.npy\n".format(i, j, fname))
            else:
                for k in range(1,108):
                    output_file.write("{}_{}-{}.npy\n".format(i, k, fname))
