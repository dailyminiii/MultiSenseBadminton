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

import time
from datetime import datetime
from dateutil import tz


# Get a date string from seconds since epoch.
# If time_s is None, will use the current time.
def get_time_str(time_s=None, format='%Y-%m-%d_%H-%M-%S', return_time_s=False):
  time_s = time_s or time.time()
  time_datetime = datetime.fromtimestamp(time_s)
  time_str = time_datetime.strftime(format)
  if return_time_s:
    return (time_str, time_s)
  else:
    return time_str

# Given a UTC time string in the format %H:%M:%S.%f,
#  add the current UTC date then convert it to local time and return seconds since epoch.
def get_time_s_from_utc_timeNoDate_str(time_utc_str, input_time_format='%H:%M:%S.%f',
                                       date_utc_str=None, input_date_format='%Y-%m-%d'):
  from_zone = tz.tzutc()
  to_zone = tz.tzlocal()
  
  # Get the current UTC date if no date was provided.
  if date_utc_str is None:
    now_utc_datetime = datetime.utcnow()
    date_utc_str = now_utc_datetime.strftime(input_date_format)
  
  # Combine the date and time.
  utc_str = '%s %s' % (date_utc_str, time_utc_str)
  utc_datetime = datetime.strptime(utc_str, input_date_format + ' ' + input_time_format)
  
  # Convert to local time, then to seconds since epoch.
  utc_datetime = utc_datetime.replace(tzinfo=from_zone)
  local_datetime = utc_datetime.astimezone(to_zone)
  return local_datetime.timestamp()

# Given a local time string in the format %H:%M:%S.%f,
#  add the current local date then return seconds since epoch.
def get_time_s_from_local_str(time_local_str, input_time_format='%H:%M:%S.%f',
                              date_local_str=None, input_date_format='%Y-%m-%d'):
  # Get the current local date if no date was provided.
  if date_local_str is None:
    now_local_datetime = datetime.now()
    date_local_str = now_local_datetime.strftime(input_date_format)
  
  # Combine the date and time.
  local_str = '%s %s' % (date_local_str, time_local_str)
  local_datetime = datetime.strptime(local_str, input_date_format + ' ' + input_time_format)
  
  return local_datetime.timestamp()






