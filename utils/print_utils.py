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
  import numpy as np
except:
  pass
from utils.time_utils import *

# Print a dictionary (recursively as appropriate).
def print_dict(d):
  print(get_dict_str(d))

# Get a string to display a dictionary (recursively as appropriate).
def get_dict_str(d, level=0):
  indent_root = ' '*level
  indent_keys =  indent_root + ' '
  msg = '%s{\n' % indent_root
  for (key, value) in d.items():
    msg += '%s %s: ' % (indent_keys, key)
    if isinstance(value, dict):
      msg += '\n'
      msg += get_dict_str(value, level+2) # one level for the key indent, one for advancing the level
    else:
      msg += '%s\n' % str(value)
  msg += '%s}\n' % indent_root
  return msg

# Print a variable and its type.
def print_var(var, name=None):
  print(get_var_str(var, name=name))

# Get a string to display a variable and its type.
def get_var_str(var, name=None):
  msg = ''
  if name is not None:
    msg += 'Variable "%s" of ' % name
  msg += 'Type %s: ' % type(var)
  msg += ''
  processed_var = False
  # Dictionary
  if isinstance(var, dict):
    msg += get_dict_str(var, level=3)
    processed_var = True
  # String
  if isinstance(var, str):
    msg += '"%s"' % var
    processed_var = True
  # Numpy array, if numpy has been imported
  try:
    if isinstance(var, np.ndarray):
      msg += '\n shape: %s' % str(var.shape)
      msg += '\n data type: %s' % str(var.dtype)
      msg += '\n %s' % str(var)
      processed_var = True
  except NameError:
    pass
  # Lists and tuples
  if isinstance(var, (list, tuple)):
    contains_non_numbers = False in [isinstance(x, (int, float)) for x in var]
    if contains_non_numbers:
      msg += '['
      for (i, x) in enumerate(var):
        msg += '\n %d: %s' % (i, get_var_str(x))
      msg += '\n ]'
    else:
      msg += '%s' % str(var)
    processed_var = True
  # Everything else
  if not processed_var:
    msg += '%s' % str(var)
    processed_var = True
  # Done!
  return msg.strip()


# Format (and optionally print) a timestamped log message.
def format_log_message(msg, *extra_msgs, source_tag=None,
                       debug=False, warning=False, error=False, userAction=False,
                       print_message=True, **kwargs):
  # Add a timestamp.
  timestamp_str = get_time_str(format='%Y-%m-%d %H:%M:%S.%f')
  # timestamp_str = get_time_str(format='%H:%M:%S.%f')
  # Add an indicator of the source.
  source_str = '[%s]' % source_tag
  source_str = source_str.ljust(10)
  # Add an indicator of the importance level.
  levels_str = {
    'normal'    : '[normal]',
    'debug'     : '[debug]',
    'warning'   : '[warn]',
    'error'     : '[error]',
    'userAction': '[prompt]',
  }
  level_str = levels_str['normal']
  if debug     : level_str = levels_str['debug']
  if warning   : level_str = levels_str['warning']
  if error     : level_str = levels_str['error']
  if userAction: level_str = levels_str['userAction']
  level_str = level_str.ljust(max([len(x) for x in levels_str.values()]))
  # Add the message prefix to each line of the message.
  msg_prefix = '%s %s %s' % (timestamp_str, source_str, level_str)
  msg = '%s: %s' % (msg_prefix, msg.replace('\n', '\n%s: ' % msg_prefix))
  # Mimic how print() would print multiple arguments.
  for extra_msg in extra_msgs:
    msg += ' %s' % extra_msg
  # Print the entry if desired.
  if print_message:
    print(msg, **kwargs)
  # Return the formatted message.
  return msg

# Format, optionally print, and optionally write to file a timestamped log message.
def write_log_message(msg, *extra_msgs, source_tag=None,
                      debug=False, warning=False, error=False, userAction=False,
                      print_message=True, filepath=None, **kwargs):
  msg = format_log_message(msg, *extra_msgs, source_tag=source_tag,
                           debug=debug, warning=warning, error=error, userAction=userAction,
                           print_message=print_message, **kwargs)
  if filepath is not None:
    # Note that it seems multiple file handles can be opened at the same time,
    #  so it shouldn't get an error in the case of multiple threads calling
    #  this method simultaneously.  But surround with try/except just in case.
    try:
      fout = open(filepath, 'a')
      fout.write(msg)
      fout.write('\n')
      fout.close()
    except:
      print('\n\nWARNING: error trying to write a log message to the following file: ', filepath, '\n\n')
  return msg







