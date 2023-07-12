############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

try:
  import pyperclip
except ModuleNotFoundError:
  pass
  
# A helper to insert new entries into the end of an array,
#  shifting back and deleting earlier entries.
# Do it in-place instead of using append() and then trimming,
#  to avoid making inefficient copies.
# Assumes arr and new_entries are 1D arrays.
def add_to_rolling_array(arr, new_entries):
  # Determine start/end indexes of the new entries such that it
  #  can fit in the array to update.
  new_entries_end_index = new_entries.shape[0] - 1
  new_entries_start_index = new_entries_end_index - arr.shape[0] + 1
  new_entries_start_index = max(0, new_entries_start_index)
  # Shift original entries back and add the new entries.
  num_new_elements = new_entries_end_index - new_entries_start_index + 1
  arr[0:-num_new_elements] = arr[num_new_elements:]
  arr[-num_new_elements:] = new_entries[new_entries_start_index:(new_entries_end_index+1)]
  return arr

# Get a matrix as a string that can be printed/copied.
def get_matrix_str(matrix, column_delim='\t', row_delim='\n'):
  return row_delim.join([column_delim.join([str(x) for x in matrix[row_index,:]])
                         for row_index in range(matrix.shape[0])])

# Copy a matrix as a string so it can be pasted into Excel or a similar program.
def copy_matrix(matrix, column_delim='\t', row_delim='\n'):
  pyperclip.copy(get_matrix_str(matrix, column_delim=column_delim, row_delim=row_delim))
  











