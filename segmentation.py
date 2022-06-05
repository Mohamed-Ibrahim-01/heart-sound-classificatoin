import math


def segment_sound(record,label, n_cycles=5, samplerate=4000):
  segement_duration = n_cycles*0.8
  record_duration = len(record)/samplerate #the dataset is sampled at 4K
  segments_num = math.floor(record_duration/segement_duration)
  segment_pts = math.floor(segement_duration*samplerate)
  segs_arr = []
  single_seg = []
  for i in range(segments_num):
      single_seg= record[i*segment_pts : segment_pts*(i+1)]
      segs_arr.append([single_seg,label])
  return segs_arr


def build_segements(data_arr):
  segments = []
  for record in data_arr:
    segments.extend(segment_sound(record[0], record[1]))
  return segments