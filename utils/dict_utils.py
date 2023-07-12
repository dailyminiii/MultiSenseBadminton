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

from collections import OrderedDict


# Rename a key of a dictionary.
# If it's an OrderedDict, keep the order the same.
def rename_dict_key(d, old_key, new_key):
  # Check if the rename operation makes sense.
  if old_key not in d:
    raise KeyError('Cannot rename key %s -> %s since the old key does not exist' % (old_key, new_key))
  if old_key == new_key:
    return d
  if new_key in d:
    raise AssertionError('Cannot rename key %s -> %s since new key already exists' % (old_key, new_key))
  
  # If it's not an ordered dictionary, simply pop and reassign the value.
  if not isinstance(d, OrderedDict):
    d[new_key] = d.pop(old_key)
    return d
  
  # Rename the key while preserving key order.
  keys = list(d.keys())
  keys[keys.index(old_key)] = new_key
  return OrderedDict(zip(keys, d.values()))

# Cast all values in a (possibly nested) dictionary to strings.
# Will remove key-value pairs for values that cannot be easily converted to a string.
# If preserve_nested_dicts is True, will preserve nested structure but recursively convert their values.
#   Otherwise, will simply stringify the whole nested dictionary.
def convert_dict_values_to_str(d, preserve_nested_dicts=True):
  # Create a new dictionary that will be populated
  if isinstance(d, OrderedDict):
    d_converted = OrderedDict()
  else:
    d_converted = {}
  
  for (key, value) in d.items():
    # Recurse if the value is a dictionary
    if isinstance(value, dict) and preserve_nested_dicts:
      d_converted[key] = convert_dict_values_to_str(value, preserve_nested_dicts=preserve_nested_dicts)
    else:
      # Add the item to the new dictionary if its value is convertible to a string
      try:
        d_converted[key] = str(value)
      except:
        pass
  return d_converted

# Flatten a dictionary.
# Will bring items of nested dictionaries up to the root level.
# Keys from nested dictionaries will have the parent key prepended to it.
def flatten_dict(d):
  d_flattened_items = _get_flattened_dict_items(d)
  if isinstance(d, OrderedDict):
    return OrderedDict(d_flattened_items)
  else:
    return dict(d_flattened_items)
# Worker method for the above, which will return a list of items for the flattened dictionary.
def _get_flattened_dict_items(d, parent_key=None, parent_key_joiner='|'):
  d_items = []
  for (key, value) in d.items():
    if parent_key is not None:
      key = '%s%s%s' % (parent_key, parent_key_joiner, key)
    if isinstance(value, dict):
      d_items.extend(_get_flattened_dict_items(value, parent_key=key, parent_key_joiner=parent_key_joiner))
    else:
      d_items.append((key, value))
  return d_items
















