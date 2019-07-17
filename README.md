![Version 1.0](https://img.shields.io/badge/iouKMeans-v1.0-blue.svg)
# IOU K-Means
IOU K-means implements mini batch K-means using IOU distance to compute anchor boxes used in object detection task.
Command Line Utilities are provided to generate data and compute anchor boxes. However repository also contains Jupyter
Notebook which runs K-means and plots the results. Please see the k-means-run.ipynb notebook.

## Installation
Clone this repository

## Usage
### 1. Generate CSV file from annotation data
This utility assumes that object detection data is properly annotated using PASCAL VOC format. All the XML files are
parsed to generate one CSV file which serves as input to compute anchor boxes. CSV file is generated as follows
```
python generate_csv.py input_data_path output_file_name
```
Example
```
python generate_csv.py data/pascal_voc/train/ annotation_data/csv_filename.csv
```
CSV file contains file_path, label, xmin, ymin, xmax, ymax, xc, yc, w, h.<br>
### 2. Compute anchor boxes
CSV file generated in previous step is used to compute anchor boxes. Only w and h from CSV files are used to compute
anchor boxes. To generate anchor boxes.
```
Usage: compute_anchors.py [OPTIONS] FILENAME K

  Utility generates K anchor boxes using input CSV file FILENAME.

Options:
  -n, --max-iteration INTEGER  Max iteration  [default: 1000]
  -b, --sample-size INTEGER    Mini batch size for K-means  [default: -1]
  -s, --scale INTEGER          Scaling factor  [default: 1]
  -o, --output TEXT            Output file name  [default: anchor.txt]
  --help                       Show this message and exit.

```
Example
```
python compute_anchors.py -n 1000 -b 1000 -s 13 -o anchor.txt annotation_data/csv_filename.csv 3
```
This utility will show average IOU and error at the end along with anchor box dimensions as w and h.
By default anchor box dimensions are stored in the anchor.txt unless specified.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors
1. [Siddharth Gawas](https://www.linkedin.com/in/siddharth-gawas-b24418133/)

## References 
The implementation is based on idea discussed in [YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

1. [YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
2. [Anchor Boxes](https://medium.com/@andersasac/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9)
3. [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf)

## License

MIT License

Copyright (c) 2019 Siddharth Gawas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.