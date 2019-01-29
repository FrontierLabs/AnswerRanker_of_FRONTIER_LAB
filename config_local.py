# -*- coding: utf-8 -*-

machine = 'GPU'
# machine = 'shareGPU'
# machine = 'vm302'
# machine = 'other'

if machine == 'GPU':
	base_model_folder = '/mnt/sdb/bowen/datas/model_context_lasagne/'
elif machine == 'shareGPU':
	base_model_folder = '/home/bowen/model/'
elif machine == 'vm302':
	base_model_folder = './model/'
elif machine == 'other':
	base_model_folder = './model/'
else:
	raise Exception('not support for machine ' + machine + ' now')
