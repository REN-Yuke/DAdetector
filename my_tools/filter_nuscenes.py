import os
import json

from nuscenes.nuscenes import NuScenes
import mmcv


def filter_scenes(root_path,
                  version='v1.0-trainval',
                  include=None,
                  exclude=None,
                  output_path=None):
    """select the scenes whose explanation contain some specific strings and
    don't contain some other specific strings.

    :param root_path: (str) Path of the data root.
    :param version: (str) Version of the data. Defaults to 'v1.0-trainval'.
    :param include: (list(str)) the keyword(s) that contained in scenes,
                    is not case-sensitive. Defaults to None.
    :param exclude: (list(str)) the keyword(s) that don't contained in scenes,
                    is not case-sensitive. Defaults to None.
    :param output_path: (str) Path of the output json file.
                        Defaults to f'{version}_include_{include}_exclude_{exclude}.json'.
    :return: filtered_scenes: (list[str]) the names of filtered scenes.
    """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    # get the name list of the unfiltered scenes according to the nuScenes official splits
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        unfiltered_scenes = list(sorted(set(splits.train + splits.val)))
    elif version == 'v1.0-test':
        unfiltered_scenes = splits.test
    elif version == 'v1.0-mini':
        unfiltered_scenes = list(sorted(set(splits.mini_train + splits.mini_val)))
    else:
        raise ValueError('unknown')

    #  check the format of the parameter include
    if include is not None:
        if isinstance(include, str):
            include = [include]
        elif isinstance(include, list):
            assert mmcv.is_list_of(include, str), 'include should be a list of strings.'
        else:
            raise ValueError('include must be either str or list of str')

    #  check the format of the parameter exclude
    if exclude is not None:
        if isinstance(exclude, str):
            assert exclude != '', 'exclude should not be '', but can be None.'
            exclude = [exclude]
        elif isinstance(exclude, list):
            assert mmcv.is_list_of(exclude, str), 'exclude should be a list of strings.'
            assert all([key_word != '' for key_word in exclude]), 'exclude should not be ''.'
        else:
            raise ValueError('exclude must be either str or list of str')

    # get available scenes and check if the unfiltered_scenes is equal to available_scenes
    available_scenes = []
    for s in nusc.scene:
        available_scenes.append(s['name'])
    assert sorted(available_scenes) == unfiltered_scenes, 'loaded scenes is not equal to unfiltered_scenes.'

    # filter the scenes
    filtered_scenes = []
    for s in unfiltered_scenes:
        scene_token = nusc.scene[available_scenes.index(s)]['token']
        scene_description = nusc.get('scene', scene_token)['description']

        # filter the scenes whose description contain specific keyword(include)
        if include is not None:
            if all([key_word.lower() in scene_description.lower()
                    for key_word in include]):
                include_flag = True
            else:
                include_flag = False
        else:
            include_flag = True

        # filter the scenes whose description don't contain specific keyword(exclude)
        if exclude is not None:
            if all([key_word.lower() not in scene_description.lower()
                    for key_word in exclude]):
                exclude_flag = True
            else:
                exclude_flag = False
        else:
            exclude_flag = True

        if include_flag and exclude_flag:
            filtered_scenes.append(s)

    saved_data = dict(version=version,
                      include=include,
                      exclude=exclude,
                      num_unfiltered_scenes=len(unfiltered_scenes),
                      num_filtered_scenes=len(filtered_scenes),
                      filtered_scenes=filtered_scenes)

    # path of output file
    if output_path is None:
        output_path = os.path.join(root_path,
                                   '{}_include_{}_exclude_{}.json'.
                                   format(version, include, exclude))
        # output_path = os.path.join(root_path,
        #                            f'{version}_include_{include}_exclude_{exclude}.json')

    # bump to .json file by mmcv
    mmcv.dump(saved_data, output_path)
    # bump to .json file by json
    # with open(output_path, 'w') as json_file:
    #     json.dump(saved_data, json_file)


def subtract_scenes(minuend_json,
                    subtrahend_json,
                    output_path):
    """Subtract some part of scenes from another part of scenes.
    Note that num_minuend_scenes subtracts num_subtrahend_scenes may not be
    num_filtered_scenes, unless minuend_scenes contains subtrahend_scenes.

    :param minuend_json: (str) Path of json file of minuend_scenes,
                         generated by def filter_scenes.
    :param subtrahend_json: (str) Path of json file of subtrahend_scenes,
                            generated by def filter_scenes.
    :param output_path: (str) Path of json file of output file, in which
                        filtered_scenes is the result scenes.
    :return: filtered_scenes: (list[str]) the names of filtered scenes.
    """
    # load .json file that contains filtered scenes list by mmcv
    minuend_data = mmcv.load(minuend_json)
    # load .json file that contains filtered scenes list by json
    # with open(minuend_json) as json_file:
    #     minuend_data = json.load(json_file)

    # load .json file that contains filtered scenes list by mmcv
    subtrahend_data = mmcv.load(subtrahend_json)
    # load .json file that contains filtered scenes list by json
    # with open(subtrahend_json) as json_file:
    #     subtrahend_data = json.load(json_file)

    minuend_scenes = set(minuend_data['filtered_scenes'])
    subtrahend_scenes = set(subtrahend_data['filtered_scenes'])
    filtered_scenes = list(sorted(minuend_scenes - subtrahend_scenes))

    saved_data = dict(num_minuend_scenes=len(minuend_scenes),
                      num_subtrahend_scenes=len(subtrahend_scenes),
                      num_filtered_scenes=len(filtered_scenes),
                      filtered_scenes=filtered_scenes)

    # bump to .json file by mmcv
    mmcv.dump(saved_data, output_path)
    # bump to .json file by json
    # with open(output_path, 'w') as json_file:
    #     json.dump(saved_data, json_file)


if __name__ == '__main__':
    # daytime scenes
    filter_scenes(root_path='E:/datasets/nuScenes/Full dataset (v1.0)/trainval',
                  version='v1.0-trainval',
                  include=None,
                  exclude=['night'])

    # daytime raining scenes
    filter_scenes(root_path='E:/datasets/nuScenes/Full dataset (v1.0)/trainval',
                  version='v1.0-trainval',
                  include=['rain'],
                  exclude=['after rain', 'night'])

    # daytime no rain scenes = daytime scenes - daytime raining scenes
    # transform the list to set format, then use subtraction of set.
    subtract_scenes(minuend_json="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                 "v1.0-trainval_include_None_exclude_['night'].json",
                    subtrahend_json="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                    "v1.0-trainval_include_['rain']_exclude_['after rain', 'night'].json",
                    output_path="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                "v1.0-trainval_daytime_no_rain.json")

    # nighttime scenes
    filter_scenes(root_path='E:/datasets/nuScenes/Full dataset (v1.0)/trainval',
                  version='v1.0-trainval',
                  include=['night'],
                  exclude=None)

    # nighttime raining scenes
    filter_scenes(root_path='E:/datasets/nuScenes/Full dataset (v1.0)/trainval',
                  version='v1.0-trainval',
                  include=['night', 'rain'],
                  exclude=['after rain'])

    # nighttime no rain scenes = nighttime scenes - nighttime raining scenes
    # transform the list to set format, then use subtraction of set.
    subtract_scenes(minuend_json="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                 "v1.0-trainval_include_['night']_exclude_None.json",
                    subtrahend_json="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                    "v1.0-trainval_include_['night', 'rain']_exclude_['after rain'].json",
                    output_path="E:/datasets/nuScenes/Full dataset (v1.0)/trainval/"
                                "v1.0-trainval_nighttime_no_rain.json")
