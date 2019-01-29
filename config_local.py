# -*- coding: utf-8 -*-

machine = 'GPU'

if machine == 'GPU':
	base_model_folder = '/mnt/sdb/bowen/datas/model_context_lasagne/'
else:
	raise Exception('not support for machine ' + machine + ' now')
