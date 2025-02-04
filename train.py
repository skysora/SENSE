import numpy as np
import h5py
import torch
from tqdm import tqdm 
import os
import pickle
import argparse
import json
import time
from scipy.stats import norm

import util
import loader
import models
import logging
import sys

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.backends import cudnn

def seed_np_pt(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU seed
    torch.cuda.manual_seed(seed) #GPU seed


def transfer_weights(model, weights_path, ensemble_load=False, wait_for_load=False, ens_id=None, sleeptime=600):
    
    print("weights_path")
    if ensemble_load:
        weights_path = os.path.join(weights_path, f'{ens_id}')

    # If weight file does not exists, wait until it exists. Intended for ensembles. Warning: Can deadlock program.
    if wait_for_load:
        if os.path.isfile(weights_path):
            target_object = weights_path
        else:
            target_object = os.path.join(weights_path, 'train.log')

        while not os.path.exists(target_object):
            print(f'File {target_object} for weight transfer missing. Sleeping for {sleeptime} seconds.')
            time.sleep(sleeptime)

    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x[:11] == 'checkpoint_'])[-1] 
        weights_path = os.path.join(weights_path, last_weight)
        
    print(weights_path)
    own_state = model.state_dict()
    state_dict = torch.load(weights_path)['model_weights']
    
    for name, param in state_dict.items():
        if name not in own_state.keys():
            print(f"{name} is not load weight")
            continue
        else:
            own_state[name].copy_(param)
            
    full_model.load_state_dict(own_state)
    return full_model

def training(model, optimizer, loader, epoch,epochs, device,training_params ,pga_loss,train_loss_record,logger):
    
    
    train_loop = tqdm(loader)
    model.train()
    total_train_loss = 0.0
    for x,y in train_loop:
        inputs_waveforms, inputs_coords, targets_coords, targets_pga = x[0].to(device).to(non_blocking=True), x[1].to(device).to(non_blocking=True),x[2].to(device).to(non_blocking=True), y[0].to(device).to(non_blocking=True)
        s = torch.unsqueeze(torch.arange(0,250), 0).to(device).to(non_blocking=True)
        s = s.repeat(inputs_waveforms.shape[0], 1)
        pred = model(inputs_waveforms, inputs_coords,s)     # Forward Pass
        train_loss = pga_loss(targets_pga, pred)     # Find the Loss
        total_train_loss = train_loss.item() + total_train_loss
        
        train_loss.backward()      # Calculate gradients 
        
        clip_grad_norm_(model.parameters(), training_params['clipnorm'])
        optimizer.step()     # Update Weights     
        optimizer.zero_grad(set_to_none=True)     # Clear the gradients

        train_loop.set_description(f"[Train Epoch {epoch+1}/{epochs}]")
        train_loop.set_postfix(loss=train_loss.detach().cpu().item())
        
    train_loss_record.append(total_train_loss / len(train_loop))  
    
    logger.info('[Train] epoch: %d -> loss: %.4f' %(epoch, total_train_loss / len(train_loop)))
    return model, optimizer,train_loss_record

def validating(model, optimizer, loader, epoch,epochs, device,pga_loss,scheduler,val_loss_record,logger):
    
    valid_loop = tqdm(loader)
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x,y in valid_loop:
            inputs_waveforms, inputs_coords, targets_coords, targets_pga = x[0].to(device).to(non_blocking=True), x[1].to(device).to(non_blocking=True),x[2].to(device).to(non_blocking=True), y[0].to(device).to(non_blocking=True)
            s = torch.unsqueeze(torch.arange(0,250), 0).to(device).to(non_blocking=True)
            s = s.repeat(inputs_waveforms.shape[0], 1)
        
            pred = model(inputs_waveforms, inputs_coords,s)     # Forward Pass   
            val_loss = pga_loss(targets_pga, pred)
            
            total_val_loss = val_loss.item() + total_val_loss
            valid_loop.set_description(f"[Eval Epoch {epoch+1}/{epochs}]")
            valid_loop.set_postfix(loss=val_loss.detach().cpu().item())

    val_loss_record.append((total_val_loss / len(valid_loop)))
    scheduler.step(total_val_loss/len(valid_loop))
    
    logger.info('[Eval] epoch: %d -> loss: %.4f' %(epoch, total_val_loss/ len(valid_loop)))
    logger.info('======================================================')
    return val_loss_record,scheduler


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                    "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    seed_np_pt(config.get('seed', 42))
    stations_table = json.load(open('./stations.json', 'r'))
    
    
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(training_params['weight_path']):
        os.mkdir(training_params['weight_path'])
    listdir = os.listdir(training_params['weight_path'])

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print('Loading data')
    if args.test_run:
        limit = 300
    else:
        limit = None

    if not isinstance(training_params['data_path'], list):
        training_params['data_path'] = [training_params['data_path']]

    assert len(generator_params) == len(training_params['data_path'])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)

    full_data_train = [loader.load_events("./taiwan_train_v3_small.hdf5", limit=limit,
                                          shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                          custom_split=generator.get('custom_split', None),
                                          min_mag=generator.get('min_mag', None),
                                          mag_key=generator.get('key', 'MA'),
                                          overwrite_sampling_rate=overwrite_sampling_rate,
                                          decimate_events=generator.get('decimate_events', None))
                            for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_dev = [loader.load_events("./taiwan_val_v3_small.hdf5", limit=limit,
                                        shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                        custom_split=generator.get('custom_split', None),
                                        min_mag=generator.get('min_mag', None),
                                        mag_key=generator.get('key', 'MA'),
                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                        decimate_events=generator.get('decimate_events', None))
                            for data_path, generator in zip(training_params['data_path'], generator_params)]
    
    event_metadata_train = [d[0] for d in full_data_train]
    data_train = [d[1] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    data_dev = [d[1] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev)
    waveforms = data_train[0]['waveforms']

    max_stations = config['model_params']['max_stations']
    ensemble = config.get('ensemble', 1)

    super_config = config.copy()
    super_training_params = training_params.copy()
    super_model_params = config['model_params'].copy()

    for ens_id in [6]:
        if ensemble > 1:
            seed_np_pt(ens_id)

            config = super_config.copy()
            config['ens_id'] = ens_id
            training_params = super_training_params.copy()
            training_params['weight_path'] = os.path.join(training_params['weight_path'], f'{ens_id}')
            config['training_params'] = training_params
            config['model_params'] = super_model_params.copy()

            if training_params.get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                config['model_params']['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)
            
            if args.continue_ensemble and os.path.isdir(training_params['weight_path']):
                hist_path = os.path.join(training_params['weight_path'], 'hist.pkl')
                if os.path.isfile(hist_path):
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. Checking for {hist_path} failed.')

            if not os.path.isdir(training_params['weight_path']):
                os.mkdir(training_params['weight_path'])

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)


        #print('Building model')
        full_model = models.build_transformer_model(**config['model_params'], device=device, trace_length=data_train[0]['waveforms'][0].shape[1])

        full_model = torch.nn.DataParallel(full_model).to(device)   

        key = generator_params[0]['key']
    
        
        
        noise_seconds = generator_params[0].get('noise_seconds', 5)
        cutout = (
            sampling_rate * (noise_seconds + generator_params[0]['cutout_start']), sampling_rate * (noise_seconds + generator_params[0]['cutout_end']))
        sliding_window = generator_params[0].get('sliding_window', False)
        n_pga_targets = config['model_params'].get('n_pga_targets', 0)
        
        if 'load_model_path' in training_params:
            print('Loading full model')
            full_model.load_weights(training_params['load_model_path'])

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            transfer_weights(full_model, training_params['transfer_model_path'],
                            ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)

        train_datas = []
        val_datas = []               
        
        for i, generator_param_set in enumerate(generator_params):
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (sampling_rate * (noise_seconds + generator_param_set['cutout_start']), sampling_rate * (noise_seconds + generator_param_set['cutout_end']))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)
            train_datas += [util.PreloadedEventGenerator(data=data_train[i],
                                                        event_metadata=event_metadata_train[i],
                                                        stations_table=stations_table,
                                                        coords_target=True,
                                                        label_smoothing=True,
                                                        station_blinding=True,
                                                        cutout=cutout,
                                                        pga_targets=n_pga_targets,
                                                        max_stations=max_stations,
                                                        sampling_rate=sampling_rate,
                                                        **generator_param_set)]
            old_oversample = generator_param_set.get('oversample', 1)
            val_datas += [util.PreloadedEventGenerator(data=data_dev[i],
                                                            event_metadata=event_metadata_dev[i],
                                                            stations_table=stations_table,
                                                            coords_target=True,
                                                            station_blinding=True,
                                                            cutout=cutout,
                                                            pga_targets=n_pga_targets,
                                                            max_stations=max_stations,
                                                            sampling_rate=sampling_rate,
                                                            **generator_param_set)]
            generator_param_set['oversample'] = old_oversample
            
        filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
        workers = training_params.get('workers', 10)
        
        
        optimizer = torch.optim.Adam(full_model.parameters(), lr=training_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=4)
        
        if n_pga_targets:
            def pga_loss(y_true, y_pred):
                return models.time_distributed_loss(y_true, y_pred, models.mixture_density_loss, device, mean=True, kwloss={'mean': False})
        
        losses = {}
        losses['pga'] = pga_loss
            
            
        num_epochs = training_params['epochs_full_model']
        metrics_record = {}
        train_loss_record = []
        val_loss_record = []
        lr_record = []
        log_path = training_params['weight_path']+'/train.log'
        logger = my_custom_logger(log_path)
        logger.info('start training')
        
        train_generators = DataLoader(train_datas[0], shuffle=True, batch_size=None, collate_fn=models.my_collate, pin_memory=True, num_workers=workers)
        val_generators = DataLoader(val_datas[0], shuffle=False, batch_size=None, collate_fn=models.my_collate, pin_memory=True, num_workers=workers)
            
        for epoch in range(num_epochs):
            
            full_model, optimizer,train_loss_record = training(full_model, optimizer, train_generators, epoch,num_epochs, device,training_params ,pga_loss,train_loss_record,logger)
            val_loss_record,scheduler = validating(full_model, optimizer, val_generators, epoch,num_epochs, device,pga_loss,scheduler,val_loss_record,logger)
            lr_record.append(scheduler.optimizer.param_groups[0]['lr'])
            
            #save model
            
            if (epoch>=1) and (val_loss_record[-1] < min(val_loss_record[:-1])):
                metrics_record['train_loss'] = train_loss_record
                metrics_record['val_loss'] = val_loss_record
                metrics_record['lr_record'] = lr_record
                with open (os.path.join(training_params['weight_path'], 'metrics.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(metrics_record))

                print("-----Saving checkpoint-----")
                torch.save({
                    'model_weights' : full_model.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                            }, 
                os.path.join(training_params['weight_path'], f'checkpoint_{epoch:02d}.pth'))
        
            
        
              