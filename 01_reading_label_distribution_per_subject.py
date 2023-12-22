############
# MIT License
#
# Copyright (c) 2023 Minwoo Seong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import h5py
from collections import defaultdict, Counter

# NOTE: HDFView is a helpful program for exploring HDF5 contents.
#   The official download page is at https://www.hdfgroup.org/downloads/hdfview.
#   It can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Specify the downloaded file to parse.
filepath = './data_processed/data_processed_allStreams_60hz_allActs.hdf5'

# Open the file.
h5_file = h5py.File(filepath, 'r')
print(h5_file.keys())

example_label_indexes = h5_file["example_label_indexes"][:]
example_labels = h5_file["example_labels"][:]
example_matrices = h5_file["example_matrices"][:]
example_score_annot_3_hori = h5_file["example_score_annot_3_hori"][:]
example_score_annot_3_ver = h5_file["example_score_annot_3_ver"][:]
example_score_annot_4 = h5_file["example_score_annot_4"][:]
example_score_annot_5 = h5_file["example_score_annot_5"][:]
example_subject_ids = h5_file["example_subject_ids"][:]
print(example_label_indexes)

subject_ids = example_subject_ids
label_indexes = example_label_indexes
subject_label_distribution = defaultdict(Counter)

for subject_id, label_index in zip(subject_ids, label_indexes):
    subject_id_str = subject_id.decode('utf-8') if isinstance(subject_id, bytes) else str(subject_id)
    subject_label_distribution[subject_id_str][label_index] += 1

for subject_id, label_counts in subject_label_distribution.items():
    print(f"Subject ID: {subject_id}, Label Distribution: {dict(label_counts)}")
