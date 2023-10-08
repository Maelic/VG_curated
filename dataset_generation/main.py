from utils import *
import argparse
import os

def main(args):
    # Setup logger
    logger = setup_logger("VG_dataset_creation", verbose=True)
    logger.info("Using arguments: {}".format(args))

    base_path = os.path.dirname(args.h5_file)
    input_path = os.path.dirname(args.metadata_input)
    # creating cache dir
    cache_dir = os.path.join(base_path, '.cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Preprocessing and cleaning
    obj_data, rel_data, img_heights, img_widths = preprocess_data(args, logger)

    preprocess_labels(obj_data, rel_data, args.object_alias, args.pred_alias, logger)

    if args.part_whole_path is not None:
        rel_data = filter_part_whole(rel_data, obj_data, logger, args.part_whole_path)

    bounding_boxes_process(obj_data, args.min_box_area_frac, img_heights, img_widths, logger, base_dir=base_path, save=args.save_bboxes)

    obj_list, pred_list = build_class_lists(args.object_list, args.pred_list, logger)

    img_list = []
    if args.images_list:
        logger.info("Using images list from {}".format(args.images_list))
        img_list = load_images_list(args.images_list)
    if args.class_selection == 'connectivity':
        logger.info("Selecting object and predicate classes based on their inter-connectivity")
        object_token_counter, predicate_token_counter = find_most_connected_objects(rel_data, obj_data, logger, obj_list, pred_list, num_objects=args.num_objects, num_predicates=args.num_predicates, img_list=img_list, use_negative_lists=args.use_negative_lists, input_dir=input_path)
    elif args.class_selection == 'frequency':
        object_token_counter = extract_object_token_no_rel(obj_data, args.num_objects, args.object_list, img_list=img_list)
        predicate_token_counter = extract_predicate_token(rel_data, args.num_predicates, args.pred_list, img_list=img_list)
    else:
        raise ValueError('Class_selection should be either connectivity or frequency')
    
    # postprocess and save to path
    postprocess_data(args, logger, predicate_token_counter, object_token_counter, obj_data, rel_data, img_list=img_list, heights=img_heights, widths=img_widths, base_path=base_path)

    print('Saving final annotations to %s and %s' % (args.h5_file, args.json_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Global options ###
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--save_bboxes', default=True, type=bool)
    parser.add_argument('--attributes', default=False, type=bool, help="Generate dataset with attributes")

    ### Paths and files ###
    # Inputs
    parser.add_argument('--imdb', default='inputs/imdb_1024.h5', type=str)
    parser.add_argument('--object_input', default='inputs/objects_v1_2/objects.json', type=str)
    parser.add_argument('--relationship_input', default='inputs/relationships.json', type=str)
    parser.add_argument('--metadata_input', default='inputs/image_data.json', type=str)
    parser.add_argument('--object_alias', default='inputs/object_alias_refined_indoor.txt', type=str)
    parser.add_argument('--pred_alias', default='inputs/predicate_alias_curated.txt', type=str)
    parser.add_argument('--object_list', default='', type=str)
    parser.add_argument('--pred_list', default='', type=str)
    parser.add_argument('--merged_boxes', type=str, help="use merged box file")
    parser.add_argument('--part_whole_path', type=str, help="set to none to disable filtering part-whole relationships")
    parser.add_argument('--images_list', type=str, help="list of images to process")
    parser.add_argument('--attributes_input', default='inputs/attributes.json', type=str)
    parser.add_argument('--attributes_synsets', default='inputs/attributes_synsets.json', type=str)
    # Outputs
    parser.add_argument('--dest_path', default="output", type=str)
    parser.add_argument('--json_file', default='VG-dicts.json')
    parser.add_argument('--h5_file', default='VG.h5')
    parser.add_argument('--yolo_anno' , default=False, type=bool)

    ### Curation options ###
    parser.add_argument('--num_objects', default=150, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--num_predicates', default=50, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--min_box_area_frac', default=0.002, type=float)
    parser.add_argument('--use_input_split', default=False, type=bool)
    parser.add_argument('--train_frac', default=0.65, type=float)
    parser.add_argument('--val_frac', default=0.7, type=float)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--class_selection', default='connectivity', type=str, help="How to select the object and predicate classes, options are connectivity or frequency")
    parser.add_argument('--use_negative_lists', default=False, type=bool)

    args = parser.parse_args()
    main(args)
