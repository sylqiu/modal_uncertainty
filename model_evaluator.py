import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Process, Queue
from importlib.machinery import SourceFileLoader
import logging
import pickle
import imp
import matplotlib.pyplot as plt
from PIL import Image
import collections



def get_array_of_modes(cf, seg):
    """
    Assemble an array holding all label modes.
    :param cf: config module
    :param seg: 4D integer array
    :return: 4D integer array
    """
    mode_stats = get_mode_statistics(cf.label_switches, exp_modes=cf.exp_modes)
    switch = mode_stats['switch']

    # construct ground-truth modes
    gt_seg_modes = np.zeros(shape=(cf.num_modes,) + seg.shape, dtype=np.uint8)
    for mode in range(cf.num_modes):
        switched_seg = seg.copy()
        for i, c in enumerate(cf.label_switches.keys()):
            if switch[mode, i]:
                init_id = cf.name2trainId[c]
                final_id = cf.name2trainId[c + '_2']
                switched_seg[switched_seg == init_id] = final_id
        gt_seg_modes[mode] = switched_seg

    return gt_seg_modes


def get_array_of_samples(cf, img_key):
    """
    Assemble an array holding all segmentation samples for a given image.
    :param cf: config module
    :param img_key: string
    :return: 5D integer array
    """
    seg_samples = np.zeros(shape=(cf.num_samples,1,1) + tuple(cf.patch_size), dtype=np.uint8)
    for i in range(cf.num_samples):
        sample_path = os.path.join(cf.out_dir, '{}_sample{}_labelIds.npy'.format(img_key, i))
        try:
            seg_samples[i] = np.load(sample_path)
        except:
            print('Could not load {}'.format(sample_path))

    return seg_samples

def get_array_of_samples_combined(cf, img_key):
    """
    Assemble an array holding all segmentation samples for a given image.
    :param cf: config module
    :param img_key: string
    :return: 5D integer array
    """
    seg_samples = np.zeros(shape=(cf.num_samples,1) + tuple(cf.patch_size), dtype=np.uint8)
    sample_path = os.path.join(cf.out_dir, '{}_16sample_labelIds.npy'.format(img_key))
    try:
        total_seg_samples = np.load(sample_path)
    except:
        print('Could not load {}'.format(sample_path))
        
    for i in range(cf.num_samples):
        seg_samples[i] = total_seg_samples[i]

    return seg_samples

def get_mode_counts(d_matrix_YS):
    """
    Calculate image-level mode counts.
    :param d_matrix_YS: 3D array
    :return: numpy array
    """
    # assign each sample to a mode
    mean_d = np.nanmean(d_matrix_YS, axis=-1)
    sampled_modes = np.argmin(mean_d, axis=-2)

    # count the modes
    num_modes = d_matrix_YS.shape[0]
    mode_count = np.zeros(shape=(num_modes,), dtype=np.int)
    for sampled_mode in sampled_modes:
        mode_count[sampled_mode] += 1

    return mode_count


def get_pixelwise_mode_counts(data_loader, seg, seg_samples):
    """
    Calculate pixel-wise mode counts.
    :param data_loader: data loader used for the model, must has swticher implemented
    :param seg: 4D array of integer labeled segmentations
    :param seg_samples: 5D array of integer labeled segmentations
    :return: array of shape (switchable classes, 3)
    """
    assert seg.shape == seg_samples.shape[1:]
    num_samples = seg_samples.shape[0]
    pixel_counts = np.zeros(shape=(len(data_loader.switcher._label_switches),3), dtype=np.int)

    # iterate all switchable classes
    for i,c in enumerate(data_loader.switcher._label_switches.keys()):
        c_id = data_loader.switcher._name2id[c]
        alt_c_id = data_loader.switcher.name2id[c+'_2']
        c_ixs = np.where(seg == c_id)

        total_num_pixels = np.sum((seg == c_id).astype(np.uint8)) * num_samples
        pixel_counts[i,0] = total_num_pixels

        # count the pixels of original class|original class and alternative class|original class
        for j in range(num_samples):
            sample = seg_samples[j]
            sampled_original_pixels = np.sum((sample[c_ixs] == c_id).astype(np.uint8))
            sampled_alternative_pixels = np.sum((sample[c_ixs] == alt_c_id).astype(np.uint8))
            pixel_counts[i,1] += sampled_original_pixels
            pixel_counts[i,2] += sampled_alternative_pixels

    return pixel_counts


def get_mode_statistics(label_switches, exp_modes=5):
    """
    Calculate a binary matrix of switches as well as a vector of mode probabilities.
    :param label_switches: dict specifying class names and their individual sampling probabilities
    :param exp_modes: integer, number of independently switchable classes
    :return: dict
    """
    num_modes = 2 ** exp_modes

    # assemble a binary matrix of switch decisions
    switch = np.zeros(shape=(num_modes, 5), dtype=np.uint8)
    for i in range(exp_modes):
        switch[:,i] = 2 ** i * (2 ** (exp_modes - 1 - i) * [0] + 2 ** (exp_modes - 1 - i) * [1])

    # calculate the probability for each individual mode
    mode_probs = np.zeros(shape=(num_modes,), dtype=np.float32)
    for mode in range(num_modes):
        prob = 1.
        for i, c in enumerate(label_switches.keys()):
            if switch[mode, i]:
                prob *= label_switches[c]
            else:
                prob *= 1. - label_switches[c]
        mode_probs[mode] = prob
    assert np.sum(mode_probs) == 1.

    return {'switch': switch, 'mode_probs': mode_probs}

def get_energy_distance_components(gt_seg_modes, seg_samples, eval_class_ids, ignore_mask=None):
    """
    Calculates the components for the IoU-based generalized energy distance given an array holding all segmentation
    modes and an array holding all sampled segmentations.
    :param gt_seg_modes: N-D array in format (num_modes,[...],H,W)
    :param seg_samples: N-D array in format (num_samples,[...],H,W)
    :param eval_class_ids: integer or list of integers specifying the classes to encode, if integer range() is applied
    :param ignore_mask: N-D array in format ([...],H,W)
    :return: dict
    """
    num_modes = gt_seg_modes.shape[0]
    num_samples = seg_samples.shape[0]

    if isinstance(eval_class_ids, int):
        eval_class_ids = list(range(eval_class_ids))

    d_matrix_YS = np.zeros(shape=(num_modes, num_samples, len(eval_class_ids)), dtype=np.float32)
    d_matrix_YY = np.zeros(shape=(num_modes, num_modes, len(eval_class_ids)), dtype=np.float32)
    d_matrix_SS = np.zeros(shape=(num_samples, num_samples, len(eval_class_ids)), dtype=np.float32)

    # iterate all ground-truth modes
    for mode in range(num_modes):

        ##########################################
        #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
        #   with S ~ P_pred, Y ~ P_gt  			 #
        ##########################################

        # iterate the samples S
        for i in range(num_samples):
            conf_matrix = calc_confusion(gt_seg_modes[mode], seg_samples[i],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YS[mode, i] = 1. - iou

        ###########################################
        #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
        #   with Y,Y' ~ P_gt  	   				  #
        ###########################################

        # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
        for mode_2 in range(mode, num_modes):
            conf_matrix = calc_confusion(gt_seg_modes[mode], gt_seg_modes[mode_2],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YY[mode, mode_2] = 1. - iou
            d_matrix_YY[mode_2, mode] = 1. - iou

    #########################################
    #   Calculate d(S,S') = 1 - IoU(S,S'),  #
    #   with S,S' ~ P_pred        			#
    #########################################

    # iterate all samples S
    for i in range(num_samples):
        # iterate all samples S'
        for j in range(i, num_samples):
            conf_matrix = calc_confusion(seg_samples[i], seg_samples[j],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}


def calc_energy_distances(d_matrices, num_samples=None, source_probability_weighted=None, target_probability_weighted=None):
    """
    Calculate the energy distance for each image based on matrices holding the combinatorial distances.
    :param d_matrices: dict holding 4D arrays of shape \
    (num_images, num_modes/num_samples, num_modes/num_samples, num_classes)
    :param num_samples: integer or None
    :param source_probability_weighted: probability vector (num_testing_sample, num_samples)
    :param target_probability_weighted: probability vector (num_testing_sample, num_modes)
    :param label_switches: None or dict
    :param exp_mode: integer
    :return: numpy array
    """
    d_matrices = d_matrices.copy() # (num_testing_sample, num_modes, num_source, num_class)

    if num_samples is None:
        num_samples = d_matrices['SS'].shape[1]

    d_matrices['YS'] = d_matrices['YS'][:,:,:num_samples]
    d_matrices['SS'] = d_matrices['SS'][:,:num_samples,:num_samples]

    # perform a nanmean over the class axis so as to not factor in classes that are not present in
    # both the ground-truth mode as well as the sampled prediction
    if (target_probability_weighted is not None) and (source_probability_weighted is None):
        
        mode_probs = target_probability_weighted

        mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1) # average over classes
        mean_d_YS = np.mean(mean_d_YS, axis=2) # average over source i.e. samples, since no source probability is provided
        mean_d_YS = mean_d_YS * mode_probs
        d_YS = np.sum(mean_d_YS, axis=1)

        mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
        d_SS = np.mean(mean_d_SS, axis=(1, 2))

        mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
        mean_d_YY = mean_d_YY * mode_probs[:, :, np.newaxis] * mode_probs[:, np.newaxis, :]
        d_YY = np.sum(mean_d_YY, axis=(1, 2))

    elif (target_probability_weighted is None) and (source_probability_weighted is not None):
        mode_probs = source_probability_weighted

        mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1) 
        mean_d_YS = np.mean(mean_d_YS, axis=1) # average over target
        mean_d_YS = mean_d_YS * mode_probs
        d_YS = np.sum(mean_d_YS, axis=1)

        mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
        d_YY = np.mean(mean_d_YY, axis=(1, 2))

        mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
        mean_d_SS = mean_d_SS * mode_probs[:, :, np.newaxis] * mode_probs[:, np.newaxis, :]
        d_SS = np.sum(mean_d_SS, axis=(1, 2))
        

    elif (target_probability_weighted is not None) and (source_probability_weighted is not None):
        mode_probs_target = target_probability_weighted
        mode_probs_source = source_probability_weighted

        mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
        mean_d_YS = mean_d_YS * mode_probs_target[:, :, np.newaxis] * mode_probs_source[:, np.newaxis, :] 

        d_YS = np.sum(mean_d_YS, axis=[1,2])

        mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
        mean_d_SS = mean_d_SS * mode_probs_source[:, :, np.newaxis] * mode_probs_source[:, np.newaxis, :]
        d_SS = np.sum(mean_d_SS, axis=(1, 2))

        mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
        mean_d_YY = mean_d_YY * mode_probs_target[:, :, np.newaxis] * mode_probs_target[:, np.newaxis, :]
        d_YY = np.sum(mean_d_YY, axis=(1, 2))


    else:
        mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
        d_YS = np.mean(mean_d_YS, axis=(1,2))

        mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
        d_SS = np.mean(mean_d_SS, axis=(1, 2))

        mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
        d_YY = np.nanmean(mean_d_YY, axis=(1, 2))

    return 2 * d_YS - d_SS - d_YY


def eval_city(cf, file_list, data_loader, queue=None):
    """
    Perform evaluation w.r.t the generalized energy distance based on the IoU as well as image-level and pixel-level
    mode frequencies (using samples written to file).
    :param cf: config module
    :param cities: string or list of strings
    :param queue: instance of multiprocessing.Queue
    :param ixs: None or 2-tuple of ints
    :return: NoneType or numpy array
    """
    

    num_modes = cf.num_modes
    num_samples = cf.num_samples

    # evaluate only switchable classes, so a total of 10 here
    eval_class_names = list(data_loader.switcher._label_switches.keys()) + list(data_loader.switcher._switched_name2Id.keys())
    eval_class_ids = [data_loader.switcher._name2id[n] for n in eval_class_names]

    d_matrices = {'YS': np.zeros(shape=(len(file_list), num_modes, num_samples, len(eval_class_ids)),
                                 dtype=np.float32),
                  'YY': np.ones(shape=(len(file_list), num_modes, num_modes, len(eval_class_ids)),
                                dtype=np.float32),
                  'SS': np.ones(shape=(len(file_list), num_samples, num_samples, len(eval_class_ids)),
                                dtype=np.float32)}
    sampled_mode_counts = np.zeros(shape=(num_modes,), dtype=np.int)
    sampled_pixel_counts = np.zeros(shape=(len(cf.label_switches), 3), dtype=np.int)

    logging.info('Evaluating class names: {} (corresponding to labels {})'.format(eval_class_names, eval_class_ids))

    # iterate all validation images
    for img_n, file in enumerate(tqdm(file_list)):
        img_key = os.path.basename(file).replace("_16prob.npy", "")
        data_dict = data_loader.get_gt_modes(img_key)
        
        seg = data_dict['seg']
        seg = seg[np.newaxis, np.newaxis]
        ignore_mask = (seg == cf.ignore_label).astype(np.uint8)

        seg_samples = np.load(file.replace("prob.npy", "sample_labelIds.npy"))[:,None,...]
        gt_seg_modes = data_dict['gt_modes']

        energy_dist = get_energy_distance_components(gt_seg_modes=gt_seg_modes, seg_samples=seg_samples,
                                                     eval_class_ids=eval_class_ids, ignore_mask=ignore_mask)

        sampled_mode_counts += get_mode_counts(energy_dist['YS'])
        sampled_pixel_counts += get_pixelwise_mode_counts(data_loader, seg, seg_samples)

        for k in d_matrices.keys():
            d_matrices[k][img_n] = energy_dist[k]

    results = {'d_matrices': d_matrices, 'sampled_pixel_counts': sampled_pixel_counts,
               'sampled_mode_counts': sampled_mode_counts, 'total_num_samples': len(file_list) * num_samples}

    if queue is not None:
        queue.put(results)
        return
    else:
        return results


def eval_LIDC(cf, file_list, data_loader, queue=None):
    # print(file_list)
    num_modes = 4
    num_samples = cf.num_samples

    # 
    eval_class_ids = [1]
    d_matrices = {'YS': np.zeros(shape=(len(file_list), num_modes, num_samples, len(eval_class_ids)),
                                 dtype=np.float32),
                  'YY': np.ones(shape=(len(file_list), num_modes, num_modes, len(eval_class_ids)),
                                dtype=np.float32),
                  'SS': np.ones(shape=(len(file_list), num_samples, num_samples, len(eval_class_ids)),
                                dtype=np.float32)}



    # iterate all validation images
    for img_n, file in enumerate(tqdm(file_list)):
        img_key = os.path.basename(file).replace("_16prob.npy", "")
        seg_samples = np.load(file.replace("prob.npy", "sample_labelIds.npy"))[:,None,...]
        # print(seg_samples.shape)
        gt_seg_modes = data_loader.get_gt_modes(img_key)['gt_modes'][:,None, None, ...]

        energy_dist = get_energy_distance_components(gt_seg_modes=gt_seg_modes, seg_samples=seg_samples,
                                                     eval_class_ids=eval_class_ids, ignore_mask=None)

        # print(energy_dist)
        for k in d_matrices.keys():
            d_matrices[k][img_n] = energy_dist[k]

    results = {'d_matrices': d_matrices}

    if queue is not None:
        queue.put(results)
        return
    else:
        return results



#### parallel running ####

def runInParallel(fns_args, queue):
    """Run functions in parallel.
    :param fns_args: list of tuples containing functions and a tuple of arguments each
    :param queue: instance of multiprocessing.Queue()
    :return: list of queue results
    """
    proc = []
    for fn in fns_args:
        p = Process(target=fn[0], args=fn[1])
        p.start()
        proc.append(p)
    return [queue.get() for p in proc]


def multiprocess_evaluation_city(cf):
    """Evaluate the energy distance in multiprocessing.
    :param cf: config module"""
    q = Queue()
    results = runInParallel([(eval_city, (cf, cf.file_list[0:100], cf.data_loader, q)),
                             (eval_city, (cf, cf.file_list[100:200], cf.data_loader, q)),
                             (eval_city, (cf, cf.file_list[200:300], cf.data_loader, q)),
                             (eval_city, (cf, cf.file_list[300:400], cf.data_loader, q)),
                             (eval_city, (cf, cf.file_list[400:500], cf.data_loader, q)),
                             ],
                             queue=q)
    total_num_samples = 0
    sampled_mode_counts = np.zeros(shape=(cf.num_modes,), dtype=np.int)
    sampled_pixel_counts = np.zeros(shape=(len(cf.label_switches), 3), dtype=np.int)
    d_matrices = {'YS':[], 'SS':[], 'YY':[]}

    # aggregate results from the queue
    for result_dict in results:
        for key in d_matrices.keys():
            d_matrices[key].append(result_dict['d_matrices'][key])

        sampled_pixel_counts += result_dict['sampled_pixel_counts']
        sampled_mode_counts += result_dict['sampled_mode_counts']
        total_num_samples += result_dict['total_num_samples']

    for key in d_matrices.keys():
        d_matrices[key] = np.concatenate(d_matrices[key], axis=0)

    # calculate frequencies
    print('pixel frequencies', sampled_pixel_counts)
    sampled_pixelwise_mode_per_class = sampled_pixel_counts[:,1:]
    total_num_pixels_per_class = sampled_pixel_counts[:,0:1]
    sampled_pixel_frequencies = sampled_pixelwise_mode_per_class / total_num_pixels_per_class
    sampled_mode_frequencies = sampled_mode_counts / total_num_samples

    print('sampled pixel frequencies', sampled_pixel_frequencies)
    print('sampled_mode_frequencies', sampled_mode_frequencies)

    results_dict = {'d_matrices': d_matrices, 'pixel_frequencies': sampled_pixel_frequencies,
               'mode_frequencies': sampled_mode_frequencies}

    results_file = os.path.join(cf.out_dir, 'eval_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Wrote to {}'.format(results_file))


def multiprocess_evaluation_LIDC(cf):
    """Evaluate the energy distance in multiprocessing.
    :param cf: config module"""
    q = Queue()
    results = runInParallel([
                            (eval_LIDC, (cf, cf.file_list[0:250], cf.data_loader, q)),\
                            (eval_LIDC, (cf, cf.file_list[250:500], cf.data_loader, q)), \
                            (eval_LIDC, (cf, cf.file_list[500:750], cf.data_loader, q)), \
                            (eval_LIDC, (cf, cf.file_list[750:1000], cf.data_loader, q)), \
                            (eval_LIDC, (cf, cf.file_list[1000:1250], cf.data_loader, q)), \
                            (eval_LIDC, (cf, cf.file_list[1250:1450], cf.data_loader, q)),
                            (eval_LIDC, (cf, cf.file_list[1450:1700], cf.data_loader, q)), \
                            (eval_LIDC, (cf, cf.file_list[1700:], cf.data_loader, q)), 
                             ],
                             queue=q)
    # results = runInParallel([(eval_LIDC, (cf, cf.file_list[0:1], cf.data_loader, q)),
    #                          ],
    #                          queue=q)
    # print(results)
   
    d_matrices = {'YS':[], 'SS':[], 'YY':[]}

    # aggregate results from the queue
    for result_dict in results:
        for key in d_matrices.keys():
            d_matrices[key].append(result_dict['d_matrices'][key])


    for key in d_matrices.keys():
        d_matrices[key] = np.concatenate(d_matrices[key], axis=0)



    results_dict = {'d_matrices': d_matrices}

    results_file = os.path.join(cf.out_dir, 'eval_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Wrote to {}'.format(results_file))


##### utility functions ##### 
def softmax_2_onehot(arr):
    """Transform a numpy array of softmax values into a one-hot encoded array. Assumes classes are encoded in axis 1.
    :param arr: ND array
    :return: ND array
    """
    num_classes = arr.shape[1]
    arr_argmax = np.argmax(arr, axis=1)

    for c in range(num_classes):
        arr[:,c] = (arr_argmax == c).astype(np.uint8)
    return arr


def numpy_one_hot(label_arr, num_classes):
    """One-hotify an integer-labeled numpy array. One-hot encoding is encoded in additional last axis.
    :param label_arr: ND array
    :param num_classes: integer
    :return: (N+1)D array
    """
    # replace labels >= num_classes with 0
    label_arr[label_arr >= num_classes] = 0

    res = np.eye(num_classes)[np.array(label_arr).reshape(-1)]
    return res.reshape(list(label_arr.shape)+[num_classes])


def calc_confusion(labels, samples, class_ixs, loss_mask=None):
    """
    Compute confusion matrix for each class across the given arrays.
    Assumes classes are given in integer-valued encoding.
    :param labels: 4/5D array (1, num_class, h, w)
    :param samples: 4/5D array
    :param class_ixs: integer or list of integers specifying the classes to evaluate
    :param loss_mask: 4/5D array
    :return: 2D array
    """
    try:
        assert labels.shape == samples.shape
    except:
        raise AssertionError('shape mismatch {} vs. {}'.format(labels.shape, samples.shape))

    if isinstance(class_ixs, int):
        num_classes = class_ixs
        class_ixs = range(class_ixs)
    elif isinstance(class_ixs, list):
        num_classes = len(class_ixs)
    else:
        raise TypeError('arg class_ixs needs to be int or list, not {}.'.format(type(class_ixs)))

    if loss_mask is None:
        shp = labels.shape
        loss_mask = np.zeros(shape=(shp[0], 1, shp[2], shp[3]))

    conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)
    for i,c in enumerate(class_ixs):

        pred_ = (samples == c).astype(np.uint8)
        labels_ = (labels == c).astype(np.uint8)

        conf_matrix[i,0] = int(((pred_ != 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # TP
        conf_matrix[i,1] = int(((pred_ != 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # FP
        conf_matrix[i,2] = int(((pred_ == 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # TN
        conf_matrix[i,3] = int(((pred_ == 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # FN

    return conf_matrix


def metrics_from_conf_matrix(conf_matrix):
    """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    fns = conf_matrix[:,3]

    metrics = {}
    metrics['iou'] = np.zeros_like(tps, dtype=np.float32)

    # iterate classes
    for c in range(tps.shape[0]):
        # unless both the prediction and the ground-truth is empty, calculate a finite IoU
        if tps[c] + fps[c] + fns[c] != 0:
            metrics['iou'][c] = tps[c] / (tps[c] + fps[c] + fns[c])
        else:
            metrics['iou'][c] = 1
            # metrics['iou'][c] = np.nan

    return metrics



if __name__ == '__main__':
    cf = imp.load_source('cf', 'LIDC_eval_config.py')
    multiprocess_evaluation_LIDC(cf)
    # conf_matrix = calc_confusion(np.concatenate([np.zeros([1,1, 10, 10]),np.ones([1,1, 10, 10])], axis=2),\
    #         np.concatenate([np.zeros([1,1, 15, 10]),np.ones([1,1, 5, 10])], axis=2), \
    #          [1], loss_mask=None)
    # print(conf_matrix)
    # print(metrics_from_conf_matrix(conf_matrix)['iou'])