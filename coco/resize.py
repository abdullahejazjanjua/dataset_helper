import os
import json
import cv2

def resize_image_and_annotations(image_dir, coco_json_path, output_dir, target_size, downscale_factor):
    """
        Resizes an image to target_size or downscales it by downscale factor and saves to specified
        path
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image {image_filename}. Skipping...")
            continue

        height, width = image.shape[:2]
        if target_size:
            pass
        elif downscale_factor:
            target_size = (width //  downscale_factor, height // downscale_factor)
        else:
            raise ValueError("Either target size or downscale factor must be provided")

        image_resized = cv2.resize(image, target_size)

        resized_image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(resized_image_path, image_resized)

        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                bbox = annotation['bbox']
                x, y, w, h = bbox

                scale_x = target_size[0] / width
                scale_y = target_size[1] / height
                new_bbox = [
                    x * scale_x,
                    y * scale_y,
                    w * scale_x,
                    h * scale_y
                ]

                annotation['bbox'] = new_bbox

        image_info['width'] = target_size[0]
        image_info['height'] = target_size[1]

    base_name = os.path.splitext(os.path.basename(coco_json_path))[0]
    output_json_path = os.path.join(output_dir, f'{base_name}_resized.json')
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"    Resized images and annotations saved to {output_dir}")


base_image_path = "path/to/image_splits"
base_ann_path= '/path/to/annotations/'
output_base_path = '/path/to/output/'
target_size = (640, 640) # W, H

splits = ["train", "val", "test"]

for split in splits:
    print(f"Processing {split}...")
    images_directory_path = os.path.join(base_image_path, split)
    coco_json_path = os.path.join(base_ann_path, f"{split}.json")
    output_path = os.path.join(output_base_path, split)

    resize_image_and_annotations(images_directory_path, coco_json_path, output_path, None, 4)