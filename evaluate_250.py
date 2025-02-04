import argparse
import os
import numpy as np
import json
import pickle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import h5py
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import torch

import models
from models import EnsembleEvaluateModel
import loader
import util
import plots

from util import generator_from_config

EARTH_RADIUS = 6371

sns.set(font_scale=1.5)
sns.set_style('ticks')

from torch.backends import cudnn
cudnn.benchmark = True 


def calculate_warning_times(config, model_list, data, event_metadata, batch_size, sampling_rate=100,
                            times=np.arange(0.5, 25, 0.2), alpha=(0.3, 0.4, 0.5, 0.6, 0.7), use_multiprocessing=True,
                            dataset_id=None, device='cuda'):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']

    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None

    alpha = np.array(alpha)
    stations_table = json.load(open('./stations.json', 'r'))
    
    if isinstance(training_params['data_path'], list):
        if dataset_id is not None:
            training_params['data_path'] = training_params['data_path'][dataset_id]
        else:
            training_params['data_path'] = training_params['data_path'][0]

    f = h5py.File(training_params['data_path'], 'r')
    g_data = f['data']
    thresholds = f['metadata']['pga_thresholds'][()]
    time_before = f['metadata']['time_before'][()]

    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')

    if 'KiK_File' in event_metadata.columns:
        event_key = 'KiK_File'
    else:
        event_key = '#EventID'

    full_predictions = []
    coord_keys = util.detect_location_keys(event_metadata.columns)

    for i, _ in tqdm(enumerate(event_metadata.iterrows()), total=len(event_metadata)): 
        event = event_metadata.iloc[i]
        event_metadata_tmp = event_metadata.iloc[i:i+1]
        data_tmp = {key: val[i:i+1] for key, val in data.items()}
        generator_params['translate'] = False
        generator = util.PreloadedEventGenerator(data=data_tmp,
                                                 stations_table=stations_table,
                                                 event_metadata=event_metadata_tmp,
                                                 coords_target=True,
                                                 cutout=(0, 3000),
                                                 pga_targets=n_pga_targets,
                                                 max_stations=max_stations,
                                                 sampling_rate=sampling_rate,
                                                 select_first=True,
                                                 shuffle=False,
                                                 pga_mode=True,
                                                 **generator_params)

        cutout_generator = util.CutoutGenerator(generator, times, sampling_rate=sampling_rate)

        workers = 1
        if use_multiprocessing:
            workers = 10
            
        predictions = model_list.predict_generator(cutout_generator, workers=workers, use_multiprocessing=use_multiprocessing) 
        predictions = predictions.reshape((len(times), -1) + predictions.shape[2:])
        pga_pred = torch.Tensor(predictions[:,:250,:,:])
        
        
        pga_times_pre = np.zeros((pga_pred.shape[1], thresholds.shape[0], alpha.shape[0]), dtype=int)

        alpha = torch.Tensor(alpha)
        for j, log_level in enumerate(np.log10(thresholds * 9.81)):
            
            prob = torch.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                dim=-1)
            
            prob = prob.reshape(prob.shape + (1,))
            
            exceedance = torch.gt(prob,alpha)  
            exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode='constant')
            pga_times_pre[:, j] = np.argmax(exceedance, axis=0)

        pga_times_pre -= 1
        pga_times_pred = np.zeros_like(pga_times_pre, dtype=float)
        pga_times_pred[pga_times_pre == -1] = np.nan
        pga_times_pred[pga_times_pre > -1] = times[pga_times_pre[pga_times_pre > -1]]

        g_event = g_data[str(event[event_key])]
        pga_times_true_pre = g_event['pga_times'][()]
        coords = g_event['coords'][()]
        coords_event = event[coord_keys]
        
        pga_times_true = np.zeros(pga_times_pred.shape[:2], dtype=float)
        stations_table = json.load(open('./stations.json', 'r'))
        dist = np.zeros(len(stations_table))
        for station_index in range(coords.shape[0]):
            station = coords[station_index]
            station_key = f"{station[0]},{station[1]},{station[2]}"
            postion = stations_table[station_key]
            pga_times_true[postion] = pga_times_true_pre[station_index]
            dist[postion] = geodesic([station[0],station[1]], coords_event[:2]).km

        pga_times_true[pga_times_true == 0] = np.nan
        pga_times_true[pga_times_true != 0] = pga_times_true[pga_times_true != 0] / sampling_rate - time_before
        
        dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)  

        print(f"pga_times_pred:{pga_times_pred.shape}")
        print(f"pga_times_true:{pga_times_true.shape}")
    
        full_predictions += [(pga_times_pred, pga_times_true, dist)]

    return full_predictions
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')  # Has only performance implications
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val', action='store_true')  # Evaluate on val set
    parser.add_argument('--n_pga_targets', type=int)  # Overwrite number of PGA targets
    parser.add_argument('--head_times', action='store_true')  # Evaluate warning times
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    parser.add_argument('--alpha', type=str, default='0.3,0.4,0.5,0.6,0.7')  # 機率大於alpha以上就會發布警報 Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--dataset_id', type=int)  # ID of dataset to evaluate on, in case of joint training
    parser.add_argument('--wait_file', type=str)  # Wait for this file to exist before starting evaluation
    parser.add_argument('--ensemble_member', action='store_true')  # Task to evaluate is an ensemble member
                                                                   # (not the full ensembel)
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    # A combination of tensorflow multiprocessing for generators and pandas dataframes causes the code to deadlock
    # sometimes. This flag provides a workaround.
    parser.add_argument('--no_multiprocessing', action='store_true')
    args = parser.parse_args()

    if args.wait_file is not None:
        util.wait_for_file(args.wait_file)

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    
    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    
    
    if (args.dataset_id is None) and (isinstance(training_params['data_path'], list) and
                                      len(training_params['data_path']) > 1):
        raise ValueError('dataset_id needs to be set for experiments with multiple input data sets.')
    if (args.dataset_id is not None) and not (isinstance(training_params['data_path'], list) and
                                              len(training_params['data_path']) > 1):
        raise ValueError('dataset_id may only be set for experiments with multiple input data sets.')

    if args.dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[args.dataset_id]
        data_path = training_params['data_path'][args.dataset_id]
        n_datasets = len(training_params['data_path'])
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]
        data_path = training_params['data_path']
        n_datasets = 1
        
    batch_size = generator_params['batch_size']
    key = generator_params.get('key', 'MA')
    pos_offset = generator_params.get('pos_offset', (-21, -69))
    pga_key = generator_params.get('pga_key', 'pga')

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.dataset_id is not None:
        suffix += f'_{args.dataset_id}'

    if args.val:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'val_6')
        training_params['data_path'] = "./taiwan_val_v3_small.hdf5"
        test_set = False
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test_6')
        training_params['data_path'] = "./taiwan_test_v3_small.hdf5"
        test_set = True

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    shuffle_train_dev = generator_params.get('shuffle_train_dev', False)
    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(training_params['data_path'],
                                                        shuffle_train_dev=shuffle_train_dev,
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate)

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if pga_key in data:
        pga_true = data[pga_key]
    else:
        pga_true = None

    if 'max_stations' not in config['model_params']:
        config['model_params']['max_stations'] = data['waveforms'].shape[1]
    if args.max_stations is not None:
        config['model_params']['max_stations'] = args.max_stations

    if args.n_pga_targets is not None:
        if config['model_params'].get('n_pga_targets', 0) > 0:
            print('Overwriting number of PGA targets')
            config['model_params']['n_pga_targets'] = args.n_pga_targets
        else:
            print('PGA flag is set, but model does not support PGA')

    
    ensemble = config.get('ensemble', 1)
    print(ensemble)
    if ensemble > 1 and not args.ensemble_member:
        experiment_path = args.experiment_path
        model_list = EnsembleEvaluateModel(config, experiment_path, loss_limit=args.loss_limit, batch_size=batch_size, device=device)

    else: 
        if 'n_datasets' in config['model_params']:
            del config['model_params']['n_datasets']
        model_list = models.build_transformer_model(**config['model_params'], trace_length=data['waveforms'][0].shape[1], n_datasets=n_datasets).to(device)

        if args.weight_file is not None:
            weight_file = os.path.join(args.experiment_path, args.weight_file)
        else:
            weight_file = sorted([x for x in os.listdir(args.experiment_path) if x[:11] == 'checkpoint-'])[-1]
            weight_file = os.path.join(args.experiment_path, weight_file)
        
        print(weight_file)
        model_list.load_state_dict(torch.load(weight_file)['model_weights'])

    mag_stats = []
    loc_stats = []
    pga_stats = []
    mag_pred_full = []
    loc_pred_full = []
    pga_pred_full = []
    
    
   
    
    results = {'times': times,
               'pga_stats': np.array(pga_stats).tolist()}

    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    if args.head_times:
        times_pga = np.arange(args.blind_time, 25, 0.2)
        alpha = [float(x) for x in args.alpha.split(',')]
        warning_time_information = calculate_warning_times(config, model_list, data, event_metadata,
                                                           times=times_pga,
                                                           alpha=alpha,
                                                           batch_size=batch_size,
                                                           use_multiprocessing=not args.no_multiprocessing,
                                                           dataset_id=args.dataset_id, device=device)
    else:
        warning_time_information = None
        alpha = None


    with open(os.path.join(output_dir, 'head_times_predictions_250_fill.pkl'), 'wb') as pred_file:
        pickle.dump((times, mag_pred_full, loc_pred_full, pga_pred_full, warning_time_information, alpha), pred_file)
        