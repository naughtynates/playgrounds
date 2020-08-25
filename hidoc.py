import pandas as pd
import numpy as np
import os
import re
import copy
import string
from toolz import sliding_window, take_nth

class HierarchicalDoc(object):
  def __init__(self, text, patterns=None, strip_section_id=False):
    self.text = text
    self.strip_section_id = strip_section_id
    self.patterns = {
      'Section 1.': re.compile('Section *[0-9]+\.'),
      '1.': re.compile('[0-9]+\. '),
      'A.': re.compile('[A-Z]+\. '),
      '(1)': re.compile('\((\d+)\) '),
      '(a)': re.compile('\([a-h]+\) '),
      '(A)': re.compile('\([A-H]+\) '),
      '(i)': re.compile('\([ivx]+\) '),
    }
    if patterns is not None:
      self.patterns = patterns
    self.refresh()

  def get_hierarchy(self, text, patterns):
    h = []
    for line in text.split('\n'):
      first_words = ' '.join(line.split()[:2])
      for p in patterns:
        if p not in h:
          if patterns[p].match(first_words) is not None:
            h.append(p)
    return h

  def simple(self):
    data = self.levels
    data['text'] = data.apply(lambda x: x[str(x['level'])], axis=1)
    data = data[['level', 'section', 'text']]
    return data

  def get_levels(self, text, h, patterns, strip_section_id):
    data = []
    current_index = [0] * len(h)
    index = []
    row = {'level':0}
    for line in text.split('\n'):
      level = 0
      for p in h:
        first_words = ' '.join(line.split()[:2])
        m = patterns[p].match(first_words)
        if m is not None:
          if strip_section_id:
            clean_line = line.replace(m.group(), '').strip()
          else:
            clean_line = line
          level = h.index(p) + 1
          data.append(copy.deepcopy(row))
          index.append(copy.deepcopy(current_index))
          if level <= row['level']:
            for k in list(row.keys()):
              if k not in ['level', 'section'] and int(k) > level:
                del row[k]
            row['section'] = ' / '.join(row['section'].split(' / ')[:level-1])
            for i in range(level, len(current_index)):
              current_index[i] = 0
          current_index[level - 1] += 1
          if level == 1:
            row['section'] = m.group()
          else:
            row['section'] += ' / ' + m.group()
          row['level'] = level
          row[str(level)] = clean_line
      if level == 0 and row['level'] != 0:
        row[str(row['level'])] += '\n' + line
    data.append(copy.deepcopy(row))
    index.append(copy.deepcopy(current_index))
    index = pd.MultiIndex.from_tuples(index, names=[i+1 for i in range(len(current_index))])
    data = pd.DataFrame(data, index=index)
    data  = data[data['level'] != 0].fillna('')
    return data

  def get(self, level, ancestors=0, children=0):
    def apply_f(data, level, ancestors, children):
      cols = [x for x in data.columns if x in map(str, range(level - ancestors, level+1))]
      level_only = data[data['level'] == level]
      if len(level_only) == 0:
        return None
      text = ' '.join(level_only[cols].values[0])
      data = data[data['level'].isin(range(level + 1, level + 1 + children))]
      for _, row in data.iterrows():
        text += ' ' + row[str(row['level'])]
      return pd.Series({'section': level_only['section'].values[0], 'text':text})
    lvl = self.levels
    lvl = lvl[lvl['level'] >= level]
    lvl = lvl.groupby(level=list(range(level))).apply(lambda x: apply_f(x, level, ancestors, children))
    return lvl

  def get_context(self, level, min_context=0, max_context=None):
      if max_context is None:
        max_context = len(self.hierarchy) - (len(self.hierarchy) - (level - 1))

      data = pd.DataFrame()
      for i in range(min_context, max_context+1):
        data = data.append(self.get(level, ancestors=i), ignore_index=True)
      return data

  def context(self, level=None, min_context=0, max_context=None):
    if level is None:
      data = pd.DataFrame()
      for i in range(len(self.hierarchy)):
        data = data.append(self.get_context(i+1), ignore_index=True)
      return data
    else:
      return self.get_context(level, min_context, max_context)
  
  def descriptions(self, max_length=256):
    data = self.context()
    data = data.dropna()
    descriptions = data[data['text'].apply(lambda x: len(x.split()) < max_length)]
    descriptions = descriptions.groupby('section').last().reset_index()
    simple = self.simple()
    for _, row in simple.iterrows():
      if row['section'] not in descriptions['section'].values:
        descriptions = descriptions.append(pd.DataFrame([{
            'section': row['section'],
          'text': ' '.join(row['text'].split()[:max_length]) + ' ...'
        }]), ignore_index=True)
    descriptions.columns = ['section', 'description']
    return descriptions

  def refresh(self):
    self.hierarchy = self.get_hierarchy(self.text, self.patterns)
    self.levels = self.get_levels(self.text, self.hierarchy, self.patterns, self.strip_section_id)


